// NeuraCraft demo CLI.
//
//   NeuraCraft                 run the xor, spiral and sine demos
//   NeuraCraft xor|spiral|sine run one demo
//   NeuraCraft mnist <dir> [epochs] [mlp|cnn]
//                              train a digit classifier on MNIST IDX files in <dir>
//   NeuraCraft eval <dir> [mlp|cnn]
//                              evaluate the saved weights on the MNIST test set
//                              (see scripts/download_mnist.sh)
//   NeuraCraft gpt <text-file> [epochs] [bpe|char]
//                              train a GPT on a text file (scripts/download_shakespeare.sh)
//   NeuraCraft generate <text-file> [prompt] [length]
//                              sample from the trained GPT
//   NeuraCraft bench [--quick] matrix-multiply throughput of each backend
//
// Global options: --gpu (Metal on Apple silicon), --threads N

#include "NeuraCraft.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace nc;

namespace {

void banner(const std::string& title) {
    std::cout << "\n=== " << title << " " << std::string(title.size() < 60 ? 60 - title.size() : 0, '=') << '\n';
}

int runXor() {
    banner("XOR (2-8-1, tanh + sigmoid, binary cross-entropy)");
    const Dataset data = makeXor();

    Network net;
    net.add<Dense>(2, 8);
    net.add<Tanh>();
    net.add<Dense>(8, 1);
    net.add<Sigmoid>();
    net.summary();

    BinaryCrossEntropy loss;
    Adam optimizer(0.05f);
    FitOptions opts;
    opts.epochs = 1000;
    opts.batchSize = 4;
    opts.printEvery = 250;
    opts.trackAccuracy = true;
    net.fit(data.x, data.y, loss, optimizer, opts);

    const Matrix out = net.predict(data.x);
    for (std::size_t i = 0; i < data.x.rows(); ++i) {
        std::cout << "  " << data.x(i, 0) << " XOR " << data.x(i, 1) << " = " << std::fixed
                  << std::setprecision(3) << out(i, 0) << std::defaultfloat << "  (target " << data.y(i, 0) << ")\n";
    }
    const float acc = net.accuracy(data.x, data.y);
    std::cout << "accuracy: " << acc * 100.0f << "%\n";
    return acc == 1.0f ? 0 : 1;
}

int runSpiral() {
    banner("Spiral, 3 classes (2-64-64-3, ReLU, softmax cross-entropy)");
    const Dataset data = makeSpiral(100, 3, 0.2f);
    const Split split = trainTestSplit(data.x, data.y, 0.2f);

    Network net;
    net.add<Dense>(2, 64, Init::He);
    net.add<ReLU>();
    net.add<Dense>(64, 64, Init::He);
    net.add<ReLU>();
    net.add<Dense>(64, 3);
    net.summary();

    SoftmaxCrossEntropy loss;
    Adam optimizer(0.01f);
    FitOptions opts;
    opts.epochs = 300;
    opts.batchSize = 32;
    opts.printEvery = 50;
    opts.trackAccuracy = true;
    opts.validationX = &split.test.x;
    opts.validationY = &split.test.y;
    net.fit(split.train.x, split.train.y, loss, optimizer, opts);

    // ASCII map of the learned decision regions; training points shown in caps.
    const int w = 56, h = 24;
    const char region[] = {'.', '-', '~'};
    const char point[] = {'A', 'B', 'C'};
    std::vector<std::string> canvas(h, std::string(w, ' '));
    Matrix grid(static_cast<std::size_t>(w * h), 2);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            grid(static_cast<std::size_t>(r * w + c), 0) = -1.1f + 2.2f * static_cast<float>(c) / (w - 1);
            grid(static_cast<std::size_t>(r * w + c), 1) = 1.1f - 2.2f * static_cast<float>(r) / (h - 1);
        }
    const auto labels = argmaxRows(net.predict(grid));
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) canvas[r][c] = region[labels[static_cast<std::size_t>(r * w + c)]];
    for (std::size_t i = 0; i < data.x.rows(); i += 3) {
        const int c = static_cast<int>(std::lround((data.x(i, 0) + 1.1f) / 2.2f * (w - 1)));
        const int r = static_cast<int>(std::lround((1.1f - data.x(i, 1)) / 2.2f * (h - 1)));
        if (r >= 0 && r < h && c >= 0 && c < w) canvas[r][c] = point[data.y.argmaxRow(i)];
    }
    std::cout << "decision regions (. - ~) and samples (A B C):\n";
    for (const auto& line : canvas) std::cout << "  |" << line << "|\n";

    const float acc = net.accuracy(split.test.x, split.test.y);
    std::cout << "test accuracy: " << std::fixed << std::setprecision(1) << acc * 100.0f << std::defaultfloat << "%\n";
    return acc >= 0.9f ? 0 : 1;
}

int runSine() {
    banner("Sine regression (1-32-32-1, tanh, MSE)");
    const Dataset data = makeSine(200, 0.05f);

    Network net;
    net.add<Dense>(1, 32);
    net.add<Tanh>();
    net.add<Dense>(32, 32);
    net.add<Tanh>();
    net.add<Dense>(32, 1);
    net.summary();

    MSE loss;
    Adam optimizer(0.01f);
    FitOptions opts;
    opts.epochs = 400;
    opts.batchSize = 32;
    opts.printEvery = 100;
    net.fit(data.x, data.y, loss, optimizer, opts);

    const Dataset clean = makeSine(9);
    const Matrix out = net.predict(clean.x);
    std::cout << "        x    sin(x)   predicted\n";
    for (std::size_t i = 0; i < clean.x.rows(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << std::setw(9) << clean.x(i, 0) << std::setw(10)
                  << clean.y(i, 0) << std::setw(12) << out(i, 0) << '\n';
    }
    MSE eval;
    const float mse = net.evaluate(clean.x, clean.y, eval);
    std::cout << "MSE vs. noiseless sine: " << std::setprecision(5) << mse << std::defaultfloat << '\n';
    return mse < 0.01f ? 0 : 1;
}

struct Mnist {
    Matrix trainX, trainY, testX, testY;
    std::vector<int> testLabels;
};

bool loadMnist(const std::string& dir, bool withTraining, Mnist& m) {
    try {
        if (withTraining) {
            m.trainX = loadIdxImages(dir + "/train-images-idx3-ubyte");
            m.trainY = oneHot(loadIdxLabels(dir + "/train-labels-idx1-ubyte"), 10);
        }
        m.testX = loadIdxImages(dir + "/t10k-images-idx3-ubyte");
        m.testLabels = loadIdxLabels(dir + "/t10k-labels-idx1-ubyte");
        m.testY = oneHot(m.testLabels, 10);
    } catch (const std::exception& e) {
        std::cerr << "could not load MNIST: " << e.what() << "\n"
                  << "download it with: scripts/download_mnist.sh " << dir << '\n';
        return false;
    }
    return true;
}

// The architectures must match between training and `eval`, since weight files
// store parameter values only.
bool buildMnistModel(Network& net, const std::string& model) {
    if (model == "mlp") {
        net.add<Dense>(784, 128, Init::He);
        net.add<ReLU>();
        net.add<Dropout>(0.2f);
        net.add<Dense>(128, 64, Init::He);
        net.add<ReLU>();
        net.add<Dense>(64, 10);
        return true;
    }
    if (model == "cnn") {
        net.add<Conv2D>(ImageShape{1, 28, 28}, 16, 3, 1, 1);
        net.add<ReLU>();
        net.add<MaxPool2D>(ImageShape{16, 28, 28}, 2);
        net.add<Conv2D>(ImageShape{16, 14, 14}, 32, 3, 1, 1);
        net.add<ReLU>();
        net.add<MaxPool2D>(ImageShape{32, 14, 14}, 2);
        net.add<Dense>(32 * 7 * 7, 128, Init::He);
        net.add<ReLU>();
        net.add<Dropout>(0.3f);
        net.add<Dense>(128, 10);
        return true;
    }
    std::cerr << "unknown model '" << model << "' (use mlp or cnn)\n";
    return false;
}

std::string mnistWeightsPath(const std::string& dir, const std::string& model) {
    return dir + (model == "mlp" ? "/mnist.weights" : "/mnist_" + model + ".weights");
}

int runMnist(const std::string& dir, std::size_t epochs, const std::string& model) {
    banner("MNIST " + model + " (Adam, batch 64)");
    Network net;
    if (!buildMnistModel(net, model)) return 2;
    Mnist m;
    if (!loadMnist(dir, true, m)) return 2;
    std::cout << "train: " << m.trainX.rows() << " images, test: " << m.testX.rows() << " images\n";
    net.summary();

    SoftmaxCrossEntropy loss;
    Adam optimizer(0.001f);
    FitOptions opts;
    opts.epochs = epochs;
    opts.batchSize = 64;
    opts.trackAccuracy = true;
    opts.validationX = &m.testX;
    opts.validationY = &m.testY;
    const auto start = std::chrono::steady_clock::now();
    net.fit(m.trainX, m.trainY, loss, optimizer, opts);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const std::string weights = mnistWeightsPath(dir, model);
    net.save(weights);
    std::cout << "trained in " << std::fixed << std::setprecision(1) << seconds << " s ("
              << seconds / static_cast<double>(epochs) << " s/epoch)\n"
              << "weights saved to " << weights << '\n'
              << "test accuracy: " << std::setprecision(2) << net.accuracy(m.testX, m.testY) * 100.0f
              << std::defaultfloat << "%\n";
    return 0;
}

void printDigit(const Matrix& images, std::size_t row) {
    static const char shades[] = " .:-=+*#%@";
    for (std::size_t y = 0; y < 28; y += 2) {  // halve rows: terminal cells are tall
        std::cout << "    ";
        for (std::size_t x = 0; x < 28; ++x) {
            const float v = std::max(images(row, y * 28 + x), images(row, (y + 1) * 28 + x));
            std::cout << shades[static_cast<int>(v * 9.0f + 0.5f)];
        }
        std::cout << '\n';
    }
}

int runEval(const std::string& dir, const std::string& model) {
    banner("Evaluate saved MNIST " + model);
    Network net;
    if (!buildMnistModel(net, model)) return 2;
    Mnist m;
    if (!loadMnist(dir, false, m)) return 2;
    const std::string weights = mnistWeightsPath(dir, model);
    try {
        net.load(weights);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ntrain first with: NeuraCraft mnist " << dir << " 5 " << model << '\n';
        return 2;
    }
    std::cout << "loaded " << weights << " (" << net.parameterCount() << " parameters)\n";

    std::vector<int> predicted;
    for (std::size_t start = 0; start < m.testX.rows(); start += 500) {
        const std::size_t end = std::min(m.testX.rows(), start + 500);
        const auto batch = argmaxRows(net.predict(m.testX.sliceRows(start, end)));
        predicted.insert(predicted.end(), batch.begin(), batch.end());
    }

    std::size_t correct = 0, perTotal[10] = {}, perCorrect[10] = {};
    std::vector<std::size_t> mistakes;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        const int label = m.testLabels[i];
        ++perTotal[label];
        if (predicted[i] == label) {
            ++correct;
            ++perCorrect[label];
        } else {
            mistakes.push_back(i);
        }
    }
    std::cout << "test accuracy: " << std::fixed << std::setprecision(2)
              << 100.0 * static_cast<double>(correct) / static_cast<double>(predicted.size()) << "%  ("
              << mistakes.size() << " mistakes)\nper digit:";
    for (int d = 0; d < 10; ++d)
        std::cout << "  " << d << ": " << std::setprecision(1)
                  << 100.0 * static_cast<double>(perCorrect[d]) / static_cast<double>(std::max<std::size_t>(1, perTotal[d]))
                  << "%";
    std::cout << std::defaultfloat << "\n";

    for (std::size_t k = 0; k < std::min<std::size_t>(3, mistakes.size()); ++k) {
        const std::size_t i = mistakes[k];
        std::cout << "\n  test image #" << i << ": label " << m.testLabels[i] << ", predicted " << predicted[i] << '\n';
        printDigit(m.testX, i);
    }
    return 0;
}

GPTConfig textModelConfig(std::size_t vocab) {
    GPTConfig cfg;
    cfg.vocabSize = vocab;
    cfg.contextLength = 64;
    cfg.dim = 128;
    cfg.heads = 4;
    cfg.layers = 4;
    cfg.dropout = 0.1f;
    return cfg;
}

std::string gptWeightsPath(const std::string& textPath) { return textPath + ".gpt.weights"; }
std::string gptTokenizerPath(const std::string& textPath) { return textPath + ".gpt.tokenizer"; }
std::string gptConfigPath(const std::string& textPath) { return textPath + ".gpt.config"; }
std::string gptCheckpointPath(const std::string& textPath) { return textPath + ".gpt.checkpoint"; }

// Model built for generation/resume: the saved config wins over the defaults.
GPTConfig savedOrDefaultConfig(const std::string& textPath, std::size_t vocab) {
    std::ifstream probe(gptConfigPath(textPath));
    GPTConfig cfg = probe ? loadGPTConfig(gptConfigPath(textPath)) : textModelConfig(vocab);
    cfg.vocabSize = vocab;
    return cfg;
}

struct GptOptions {
    std::size_t epochs = 1;
    std::string tokenizer = "bpe";
    bool resume = false;
    bool cosine = false;  // warmup + cosine decay (helps long runs; constant is better for a few epochs)
    float learningRate = 1e-3f;
    std::size_t dim = 0, layers = 0, heads = 0, context = 0;  // 0 = default / saved
    std::string arch = "classic";  // classic (GPT-2 style) or modern (RoPE, RMSNorm, SwiGLU, tied)
};
constexpr std::size_t kBpeVocab = 512;

// The tokenizer file records which tokenizer the weights were trained with:
// BPE merges, or a marker meaning "character vocabulary of the text file".
void saveTokenizer(const std::string& textPath, const Tokenizer& tok) {
    if (const auto* bpe = dynamic_cast<const BpeTokenizer*>(&tok)) {
        bpe->save(gptTokenizerPath(textPath));
    } else {
        std::ofstream out(gptTokenizerPath(textPath));
        out << "NEURACRAFT-CHAR 1\n";
        if (!out) throw std::runtime_error("cannot write " + gptTokenizerPath(textPath));
    }
}

std::unique_ptr<Tokenizer> loadTokenizer(const std::string& textPath, const std::string& text) {
    std::ifstream in(gptTokenizerPath(textPath));
    std::string magic;
    if (!in || !(in >> magic) || magic == "NEURACRAFT-CHAR") return std::make_unique<CharTokenizer>(text);
    return std::make_unique<BpeTokenizer>(BpeTokenizer::load(gptTokenizerPath(textPath)));
}

void printSample(Network& net, const SequenceContextPtr& ctx, const Tokenizer& tok, const std::string& prompt,
                 std::size_t length, std::size_t context) {
    SamplingOptions sampling;
    sampling.temperature = 0.8f;
    sampling.topK = 20;
    std::cout << "----- sample -----\n"
              << prompt << tok.decode(generate(net, tok.encode(prompt), length, context, sampling, ctx))
              << "\n------------------" << std::endl;
}

int runGpt(const std::string& path, const GptOptions& o) {
    const std::string& tokenizerKind = o.tokenizer;
    banner("GPT language model (" + tokenizerKind + " tokens)");
    if (tokenizerKind != "char" && tokenizerKind != "bpe") {
        std::cerr << "unknown tokenizer '" << tokenizerKind << "' (use char or bpe)\n";
        return 2;
    }
    std::string text;
    try {
        text = loadText(path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ndownload a corpus with: scripts/download_shakespeare.sh\n";
        return 2;
    }
    std::unique_ptr<Tokenizer> tok;
    if (o.resume) {
        tok = loadTokenizer(path, text);  // must match the checkpoint
    } else if (tokenizerKind == "bpe") {
        const auto t0 = std::chrono::steady_clock::now();
        tok = std::make_unique<BpeTokenizer>(BpeTokenizer::train(text, kBpeVocab));
        std::cout << "trained BPE (" << tok->vocabSize() << " tokens) in " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n"
                  << std::defaultfloat;
    } else {
        tok = std::make_unique<CharTokenizer>(text);
    }
    const std::vector<int> ids = tok->encode(text);
    const std::size_t split = ids.size() * 9 / 10;
    const std::vector<int> trainIds(ids.begin(), ids.begin() + static_cast<long>(split));
    const std::vector<int> valIds(ids.begin() + static_cast<long>(split), ids.end());
    // Loss per character makes runs with different tokenizers comparable.
    const double tokensPerChar = static_cast<double>(valIds.size()) / static_cast<double>(tok->decode(valIds).size());

    GPTConfig cfg = o.resume ? savedOrDefaultConfig(path, tok->vocabSize()) : textModelConfig(tok->vocabSize());
    if (!o.resume) {
        if (o.arch == "modern") cfg.modern();
        if (o.dim) cfg.dim = o.dim;
        if (o.layers) cfg.layers = o.layers;
        if (o.heads) cfg.heads = o.heads;
        if (o.context) cfg.contextLength = o.context;
    }
    const Dataset train = makeNextTokenDataset(trainIds, cfg.contextLength);
    const Dataset val = makeNextTokenDataset(valIds, cfg.contextLength);
    std::cout << path << ": " << text.size() << " characters -> " << ids.size() << " tokens ("
              << std::setprecision(3) << 1.0 / tokensPerChar << std::defaultfloat << " chars/token), vocabulary "
              << tok->vocabSize() << ", " << train.x.rows() << " training / " << val.x.rows()
              << " validation sequences of " << cfg.contextLength << " tokens\n";

    Network net;
    const SequenceContextPtr ctx = buildGPT(net, cfg);
    net.summary();
    SparseSoftmaxCrossEntropy loss;
    Adam optimizer(o.learningRate);

    const std::size_t batchSize = 32;
    const std::size_t stepsPerEpoch = (train.x.rows() + batchSize - 1) / batchSize;
    std::size_t startStep = 0;
    if (o.resume) {
        try {
            startStep = loadCheckpoint(gptCheckpointPath(path), net, optimizer);
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n(start a new run without --resume)\n";
            return 2;
        }
        std::cout << "resumed from step " << startStep << " (epoch " << startStep / stepsPerEpoch << ")\n";
    } else {
        saveTokenizer(path, *tok);
        saveGPTConfig(gptConfigPath(path), cfg);
    }
    // The schedule spans all epochs of the run, counted from the first step, so a
    // resumed run continues the same curve (--resume with N epochs extends it).
    const std::size_t doneEpochs = startStep / stepsPerEpoch;
    CosineSchedule schedule;
    schedule.base = o.learningRate;
    schedule.totalSteps = (doneEpochs + o.epochs) * stepsPerEpoch;
    schedule.warmupSteps = std::min<std::size_t>(200, schedule.totalSteps / 10);
    schedule.minRatio = 0.1f;

    FitOptions opts;
    opts.epochs = o.epochs;
    opts.batchSize = batchSize;
    opts.printEveryBatches = 50;
    opts.validationX = &val.x;
    opts.validationY = &val.y;
    if (o.cosine) opts.learningRate = schedule;
    opts.clipGradNorm = 1.0f;
    opts.startStep = startStep;
    const std::string prompt(1, text[0]);
    opts.onEpochEnd = [&](const EpochStats& s) {
        std::cout << "validation loss per character: " << std::fixed << std::setprecision(4)
                  << s.validationLoss * tokensPerChar << "  (step " << s.step << ", lr " << std::scientific
                  << std::setprecision(2) << s.learningRate << ", grad norm " << std::fixed << std::setprecision(2)
                  << s.gradientNorm << ")" << std::defaultfloat << '\n';
        saveCheckpoint(gptCheckpointPath(path), net, optimizer, s.step);
        printSample(net, ctx, *tok, prompt, 300 / (tokenizerKind == "bpe" ? 2 : 1), cfg.contextLength);
    };
    const auto start = std::chrono::steady_clock::now();
    net.fit(train.x, train.y, loss, optimizer, opts);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    net.save(gptWeightsPath(path));
    std::cout << "trained in " << std::fixed << std::setprecision(1) << seconds << " s ("
              << seconds / static_cast<double>(o.epochs) << " s/epoch, incl. samples)\n" << std::defaultfloat
              << "saved " << gptWeightsPath(path) << ", " << gptConfigPath(path) << ", " << gptTokenizerPath(path)
              << " and " << gptCheckpointPath(path) << "\ncontinue training with: NeuraCraft gpt " << path
              << " N --resume\ngenerate text with: NeuraCraft generate " << path << " \"PROMPT\" 500\n";
    return 0;
}

int runGenerate(const std::string& path, const std::string& prompt, std::size_t length) {
    std::string text;
    std::unique_ptr<Tokenizer> tok;
    try {
        text = loadText(path);
        tok = loadTokenizer(path, text);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    const GPTConfig cfg = savedOrDefaultConfig(path, tok->vocabSize());
    Network net;
    const SequenceContextPtr ctx = buildGPT(net, cfg);
    try {
        net.load(gptWeightsPath(path));
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ntrain first with: NeuraCraft gpt " << path << '\n';
        return 2;
    }
    std::vector<int> promptIds;
    try {
        promptIds = tok->encode(prompt.empty() ? std::string(1, text[0]) : prompt);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    SamplingOptions sampling;
    sampling.temperature = 0.8f;
    sampling.topK = 20;
    // `length` counts tokens; with BPE a token is ~2 characters.
    const auto start = std::chrono::steady_clock::now();
    const std::vector<int> out = generate(net, promptIds, length, cfg.contextLength, sampling, ctx);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << prompt << tok->decode(out) << '\n';
    std::cerr << out.size() << " tokens in " << std::fixed << std::setprecision(2) << seconds << " s ("
              << std::setprecision(0) << static_cast<double>(out.size()) / seconds << " tokens/s, KV cache)\n";
    return 0;
}

// Times `run` (one n x n x n product); returns GFLOPS.
double timeGemm(std::size_t n, const std::function<void()>& run) {
    run();  // warm-up
    const double flops = 2.0 * static_cast<double>(n) * n * n;
    int reps = 0;
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    do {
        run();
        ++reps;
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    } while (elapsed < 0.25);
    return flops * reps / elapsed / 1e9;
}

int runBench(bool quick) {
    banner("Matrix multiply benchmark");
    std::cout << "CPU backend: " << cpuBackendName() << ", " << threadCount() << " threads\n"
              << "GPU: " << (gpuAvailable() ? gpuName() + " (Metal)" : std::string("not available")) << "\n\n";

    using RawGemm = void (*)(bool, bool, std::size_t, std::size_t, std::size_t, const float*, std::size_t,
                             const float*, std::size_t, float*, std::size_t, bool);
    struct Column {
        std::string name;
        RawGemm raw;        // raw-pointer implementation, or
        bool gpuMatmul;     // Matrix-level GPU path (page-backed operands mapped in place)
    };
    std::vector<Column> columns{{"blocked", gemmBlocked, false}};
    if (cpuBackendName() != "NeuraCraft blocked kernel") columns.push_back({"accelerate", gemmCpu, false});
    if (gpuAvailable()) {
        columns.push_back({"metal copy", gemmGpu, false});
        columns.push_back({"metal 0-copy", nullptr, true});
    }

    const std::vector<std::size_t> sizes = quick ? std::vector<std::size_t>{64, 129}
                                                 : std::vector<std::size_t>{128, 256, 512, 1024, 2048};
    std::cout << "     n";
    for (const auto& col : columns) std::cout << std::setw(14) << col.name;
    std::cout << "   (GFLOPS)\n";

    int mismatches = 0;
    for (std::size_t n : sizes) {
        const Matrix a = Matrix::uniform(n, n, -1, 1), b = Matrix::uniform(n, n, -1, 1);
        Matrix expected(n, n);
        gemmBlocked(false, false, n, n, n, a.data(), n, b.data(), n, expected.data(), n, false);
        std::cout << std::setw(6) << n;
        for (const auto& col : columns) {
            Matrix c(n, n);
            double gflops = 0.0;
            if (col.gpuMatmul) {
                const Device savedDevice = currentDevice();
                const std::size_t savedMin = gpuMinWork();
                setDevice(Device::GPU);
                setGpuMinWork(0);
                gflops = timeGemm(n, [&] { matmul(a, false, b, false, c); });
                setGpuMinWork(savedMin);
                setDevice(savedDevice);
            } else {
                gflops = timeGemm(n, [&] { col.raw(false, false, n, n, n, a.data(), n, b.data(), n, c.data(), n, false); });
            }
            const bool ok = c.approxEquals(expected, 1e-3f * static_cast<float>(n));
            mismatches += !ok;
            std::cout << std::setw(13) << std::fixed << std::setprecision(1) << gflops << (ok ? ' ' : '!');
        }
        std::cout << std::defaultfloat << '\n';
    }
    if (mismatches) std::cout << "\n" << mismatches << " result(s) disagreed with the reference ('!')\n";
    return mismatches == 0 ? 0 : 1;
}

void usage() {
    std::cout << "NeuraCraft " << NEURACRAFT_VERSION << " - a small neural network library in C++17\n\n"
              << "usage:\n"
              << "  NeuraCraft                   run the xor, spiral and sine demos\n"
              << "  NeuraCraft xor|spiral|sine   run one demo\n"
              << "  NeuraCraft mnist <dir> [epochs] [mlp|cnn]\n"
              << "                               train on MNIST IDX files in <dir>, save weights\n"
              << "  NeuraCraft eval <dir> [mlp|cnn]\n"
              << "                               evaluate saved MNIST weights on the test set\n"
              << "  NeuraCraft gpt <text-file> [epochs] [bpe|char] [--resume] [--lr X] [--cosine]\n"
              << "                 [--arch classic|modern] [--dim N] [--layers N] [--heads N] [--context N]\n"
              << "                               train a GPT on a text file (BPE or character tokens)\n"
              << "  NeuraCraft generate <text-file> [prompt] [length]\n"
              << "                               sample text from the trained GPT\n"
              << "  NeuraCraft bench [--quick]   matrix-multiply speed of each backend\n\n"
              << "options:\n"
              << "  --gpu          run large matrix products on the GPU (Metal)\n"
              << "  --threads N    number of CPU threads (default: all cores)\n";
}

} // namespace

int main(int argc, char** argv) {
    setSeed(42);
    std::vector<std::string> args;
    bool wantGpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gpu") {
            wantGpu = true;
        } else if (a == "--threads") {
            if (i + 1 >= argc) {
                std::cerr << "--threads needs a number\n";
                return 2;
            }
            setThreadCount(static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10)));
        } else {
            args.push_back(a);
        }
    }
    if (wantGpu && !setDevice(Device::GPU)) {
        std::cerr << "warning: no GPU available in this build/machine, using the CPU\n";
    }
    std::cout << "device: " << deviceDescription() << '\n';

    const std::string cmd = args.empty() ? "all" : args[0];
    try {
        if (cmd == "all") {
            int failed = 0;
            failed += runXor() != 0;
            failed += runSpiral() != 0;
            failed += runSine() != 0;
            return failed == 0 ? 0 : 1;
        }
        if (cmd == "xor") return runXor();
        if (cmd == "spiral") return runSpiral();
        if (cmd == "sine") return runSine();
        if (cmd == "bench") return runBench(args.size() > 1 && args[1] == "--quick");
        if (cmd == "mnist") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            const std::size_t epochs = args.size() > 2 ? static_cast<std::size_t>(std::strtoul(args[2].c_str(), nullptr, 10)) : 5;
            return runMnist(args[1], epochs == 0 ? 5 : epochs, args.size() > 3 ? args[3] : "mlp");
        }
        if (cmd == "eval") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            return runEval(args[1], args.size() > 2 ? args[2] : "mlp");
        }
        if (cmd == "gpt") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            GptOptions o;
            std::size_t positional = 0;
            for (std::size_t i = 2; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "--resume") o.resume = true;
                else if (a == "--cosine") o.cosine = true;
                else if (a == "--arch") {
                    o.arch = value();
                    if (o.arch != "classic" && o.arch != "modern") {
                        std::cerr << "--arch must be classic or modern\n";
                        return 2;
                    }
                }
                else if (a == "--lr") o.learningRate = std::stof(value());
                else if (a == "--dim") o.dim = std::stoul(value());
                else if (a == "--layers") o.layers = std::stoul(value());
                else if (a == "--heads") o.heads = std::stoul(value());
                else if (a == "--context") o.context = std::stoul(value());
                else if (positional == 0 && std::isdigit(static_cast<unsigned char>(a[0]))) {
                    o.epochs = std::max<std::size_t>(1, std::stoul(a));
                    ++positional;
                } else if (a == "bpe" || a == "char") {
                    o.tokenizer = a;
                } else {
                    std::cerr << "unknown gpt option: " << a << "\n";
                    return 2;
                }
            }
            return runGpt(args[1], o);
        }
        if (cmd == "generate") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            const std::size_t length = args.size() > 3 ? static_cast<std::size_t>(std::strtoul(args[3].c_str(), nullptr, 10)) : 500;
            return runGenerate(args[1], args.size() > 2 ? args[2] : "", length == 0 ? 500 : length);
        }
        if (cmd == "-h" || cmd == "--help" || cmd == "help") {
            usage();
            return 0;
        }
        std::cerr << "unknown command: " << cmd << "\n\n";
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
