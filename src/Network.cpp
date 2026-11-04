#include "Network.h"
#include "Random.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
constexpr const char* kMagic = "NEURACRAFT";
constexpr int kFormatVersion = 1;
} // namespace

float accuracyScore(const Matrix& predictions, const Matrix& targets) {
    if (!predictions.sameShape(targets)) throw std::invalid_argument("accuracy: shape mismatch");
    if (predictions.rows() == 0) return 0.0f;
    std::size_t correct = 0;
    for (std::size_t i = 0; i < predictions.rows(); ++i) {
        if (predictions.cols() == 1) {
            correct += (predictions(i, 0) >= 0.5f) == (targets(i, 0) >= 0.5f);
        } else {
            correct += predictions.argmaxRow(i) == targets.argmaxRow(i);
        }
    }
    return static_cast<float>(correct) / static_cast<float>(predictions.rows());
}

Network& Network::add(std::unique_ptr<Layer> layer) {
    if (!layer) throw std::invalid_argument("Network::add: null layer");
    layers_.push_back(std::move(layer));
    return *this;
}

Matrix Network::forward(const Matrix& input, bool training) {
    if (layers_.empty()) throw std::logic_error("Network has no layers");
    Matrix out = input;
    for (auto& l : layers_) out = l->forward(out, training);
    return out;
}

Matrix Network::backward(const Matrix& gradOutput) {
    Matrix grad = gradOutput;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) grad = (*it)->backward(grad);
    return grad;
}

float Network::trainBatch(const Matrix& x, const Matrix& y, Loss& loss, Optimizer& optimizer) {
    const Matrix out = forward(x, true);
    const float value = loss.compute(out, y);
    backward(loss.gradient());
    optimizer.step(parameters());
    return value;
}

History Network::fit(const Matrix& x, const Matrix& y, Loss& loss, Optimizer& optimizer,
                     const FitOptions& options) {
    if (x.rows() != y.rows()) throw std::invalid_argument("fit: x and y have different numbers of rows");
    if (x.rows() == 0) throw std::invalid_argument("fit: empty dataset");
    if (options.batchSize == 0) throw std::invalid_argument("fit: batchSize must be > 0");
    if ((options.validationX == nullptr) != (options.validationY == nullptr))
        throw std::invalid_argument("fit: set both validationX and validationY, or neither");

    History history;
    std::vector<std::size_t> order(x.rows());
    std::iota(order.begin(), order.end(), 0);
    std::size_t step = options.startStep;
    float lastLr = optimizer.learningRate();
    bool stop = options.maxSteps > 0 && step >= options.maxSteps;

    for (std::size_t epoch = 1; epoch <= options.epochs && !stop; ++epoch) {
        // Shuffle from the identity each epoch, so the order depends only on the
        // random state (which checkpoints save) and not on earlier epochs.
        std::iota(order.begin(), order.end(), 0);
        if (options.shuffle) std::shuffle(order.begin(), order.end(), rng());

        double lossSum = 0.0, normSum = 0.0;
        std::size_t correct = 0, seen = 0, batches = 0;
        for (std::size_t start = 0; start < order.size(); start += options.batchSize) {
            const std::size_t end = std::min(start + options.batchSize, order.size());
            const std::vector<std::size_t> idx(order.begin() + start, order.begin() + end);
            const Matrix bx = x.gatherRows(idx);
            const Matrix by = y.gatherRows(idx);

            const Matrix out = forward(bx, true);
            lossSum += static_cast<double>(loss.compute(out, by)) * static_cast<double>(idx.size());
            if (options.trackAccuracy) {
                correct += static_cast<std::size_t>(accuracyScore(out, by) * static_cast<float>(idx.size()) + 0.5f);
            }
            backward(loss.gradient());
            const auto params = parameters();
            normSum += options.clipGradNorm > 0.0f ? clipGradientNorm(params, options.clipGradNorm) : gradientNorm(params);
            if (options.learningRate) optimizer.setLearningRate(options.learningRate(step));
            lastLr = optimizer.learningRate();
            optimizer.step(params);
            ++step;
            ++batches;
            seen = end;
            if (options.onStep) options.onStep(step);

            const std::size_t batchIndex = start / options.batchSize + 1;
            if (options.verbose && options.printEveryBatches > 0 && batchIndex % options.printEveryBatches == 0) {
                const std::size_t total = (order.size() + options.batchSize - 1) / options.batchSize;
                std::cout << "  epoch " << epoch << " batch " << batchIndex << "/" << total << "  loss "
                          << std::fixed << std::setprecision(5) << lossSum / static_cast<double>(end) << "  lr "
                          << std::scientific << std::setprecision(2) << lastLr << std::defaultfloat << std::endl;
            }
            if (options.maxSteps > 0 && step >= options.maxSteps) {
                stop = true;
                break;
            }
        }

        EpochStats stats;
        stats.epoch = epoch;
        stats.loss = static_cast<float>(lossSum / static_cast<double>(seen));
        if (options.trackAccuracy) stats.accuracy = static_cast<float>(correct) / static_cast<float>(seen);
        stats.step = step;
        stats.learningRate = lastLr;
        stats.gradientNorm = static_cast<float>(normSum / static_cast<double>(std::max<std::size_t>(1, batches)));
        if (options.validationX) {
            stats.validationLoss = evaluate(*options.validationX, *options.validationY, loss);
            if (options.trackAccuracy) stats.validationAccuracy = accuracy(*options.validationX, *options.validationY);
        }
        history.push_back(stats);

        const std::size_t every = std::max<std::size_t>(1, options.printEvery);
        if (options.verbose && (epoch % every == 0 || epoch == 1 || epoch == options.epochs || stop)) {
            std::cout << "epoch " << std::setw(4) << epoch << "/" << options.epochs
                      << "  loss " << std::fixed << std::setprecision(5) << stats.loss;
            if (stats.accuracy >= 0) std::cout << "  acc " << std::setprecision(4) << stats.accuracy;
            if (stats.validationLoss >= 0) std::cout << "  val_loss " << std::setprecision(5) << stats.validationLoss;
            if (stats.validationAccuracy >= 0) std::cout << "  val_acc " << std::setprecision(4) << stats.validationAccuracy;
            std::cout << std::defaultfloat << std::endl;
        }
        if (options.onEpochEnd) options.onEpochEnd(stats);
    }
    return history;
}

float Network::evaluate(const Matrix& x, const Matrix& y, Loss& loss, std::size_t batchSize) {
    if (x.rows() != y.rows()) throw std::invalid_argument("evaluate: x and y have different numbers of rows");
    if (x.rows() == 0) throw std::invalid_argument("evaluate: empty dataset");
    batchSize = std::max<std::size_t>(1, batchSize);
    double total = 0.0;
    for (std::size_t start = 0; start < x.rows(); start += batchSize) {
        const std::size_t end = std::min(start + batchSize, x.rows());
        const Matrix out = predict(x.sliceRows(start, end));
        total += static_cast<double>(loss.compute(out, y.sliceRows(start, end))) * static_cast<double>(end - start);
    }
    return static_cast<float>(total / static_cast<double>(x.rows()));
}

float Network::accuracy(const Matrix& x, const Matrix& y, std::size_t batchSize) {
    if (x.rows() != y.rows()) throw std::invalid_argument("accuracy: x and y have different numbers of rows");
    if (x.rows() == 0) return 0.0f;
    batchSize = std::max<std::size_t>(1, batchSize);
    double correct = 0.0;
    for (std::size_t start = 0; start < x.rows(); start += batchSize) {
        const std::size_t end = std::min(start + batchSize, x.rows());
        correct += static_cast<double>(accuracyScore(predict(x.sliceRows(start, end)), y.sliceRows(start, end))) *
                   static_cast<double>(end - start);
    }
    return static_cast<float>(correct / static_cast<double>(x.rows()));
}

std::vector<Param> Network::parameters() {
    std::vector<Param> params;
    for (auto& l : layers_) {
        auto lp = l->parameters();
        params.insert(params.end(), lp.begin(), lp.end());
    }
    return params;
}

std::size_t Network::parameterCount() {
    std::size_t n = 0;
    for (const auto& p : parameters()) n += p.value->size();
    return n;
}

void Network::summary(std::ostream& os) const {
    os << "Network (" << layers_.size() << " layers)\n";
    std::size_t total = 0;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        std::size_t count = 0;
        for (const auto& p : layers_[i]->parameters()) count += p.value->size();
        total += count;
        os << "  [" << i << "] " << std::left << std::setw(24) << layers_[i]->describe() << std::right
           << std::setw(10) << count << " params\n";
    }
    os << "  total trainable parameters: " << total << '\n';
}

void Network::summary() const { summary(std::cout); }

void Network::saveParameters(std::ostream& out) {
    const auto params = parameters();
    out << kMagic << ' ' << kFormatVersion << '\n' << params.size() << '\n';
    out << std::setprecision(9);
    for (const auto& p : params) {
        out << p.value->rows() << ' ' << p.value->cols() << '\n';
        const auto& values = p.value->values();
        for (std::size_t i = 0; i < values.size(); ++i) out << values[i] << (i + 1 < values.size() ? ' ' : '\n');
    }
}

void Network::loadParameters(std::istream& in, const std::string& source) {
    std::string magic;
    int version = 0;
    std::size_t count = 0;
    if (!(in >> magic >> version >> count) || magic != kMagic)
        throw std::runtime_error("Network::load: " + source + " is not a NeuraCraft weights file");
    if (version != kFormatVersion)
        throw std::runtime_error("Network::load: unsupported format version " + std::to_string(version));

    auto params = parameters();
    if (count != params.size()) {
        throw std::runtime_error("Network::load: file has " + std::to_string(count) +
                                 " parameter tensors, network has " + std::to_string(params.size()));
    }
    // Read everything first so a bad file leaves the network untouched.
    std::vector<Matrix> loaded;
    loaded.reserve(count);
    for (std::size_t k = 0; k < count; ++k) {
        std::size_t rows = 0, cols = 0;
        if (!(in >> rows >> cols)) throw std::runtime_error("Network::load: truncated file");
        if (rows != params[k].value->rows() || cols != params[k].value->cols()) {
            std::ostringstream msg;
            msg << "Network::load: tensor " << k << " is " << rows << "x" << cols << ", expected "
                << params[k].value->rows() << "x" << params[k].value->cols();
            throw std::runtime_error(msg.str());
        }
        Matrix m(rows, cols);
        for (auto& v : m.values())
            if (!(in >> v)) throw std::runtime_error("Network::load: truncated file");
        loaded.push_back(std::move(m));
    }
    for (std::size_t k = 0; k < count; ++k) *params[k].value = std::move(loaded[k]);
}

void Network::save(const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Network::save: cannot open " + path);
    saveParameters(out);
    if (!out) throw std::runtime_error("Network::save: write failed for " + path);
}

void Network::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Network::load: cannot open " + path);
    loadParameters(in, path);
}

namespace {
constexpr const char* kCheckpointMagic = "NEURACRAFT-CHECKPOINT";
} // namespace

void saveCheckpoint(const std::string& path, Network& net, const Optimizer& optimizer, std::size_t step) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) throw std::runtime_error("saveCheckpoint: cannot open " + tmp);
        out << kCheckpointMagic << " 1\n" << step << '\n' << rng() << '\n';
        net.saveParameters(out);
        optimizer.saveState(out, net.parameters());
        if (!out) throw std::runtime_error("saveCheckpoint: write failed for " + tmp);
    }
    // Replace the old checkpoint only once the new one is complete.
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("saveCheckpoint: cannot write " + path);
}

std::size_t loadCheckpoint(const std::string& path, Network& net, Optimizer& optimizer) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("loadCheckpoint: cannot open " + path);
    std::string magic;
    int version = 0;
    std::size_t step = 0;
    if (!(in >> magic >> version >> step) || magic != kCheckpointMagic || version != 1)
        throw std::runtime_error("loadCheckpoint: " + path + " is not a NeuraCraft checkpoint");
    std::mt19937 generator;
    if (!(in >> generator)) throw std::runtime_error("loadCheckpoint: bad random state in " + path);
    net.loadParameters(in, path);
    optimizer.loadState(in, net.parameters());
    rng() = generator;
    return step;
}

} // namespace nc
