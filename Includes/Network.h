#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"
#include "Loss.h"
#include "Optimizer.h"

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nc {

struct EpochStats {
    std::size_t epoch = 0;          // 1-based
    float loss = 0.0f;              // mean training loss over the epoch
    float accuracy = -1.0f;         // training accuracy, -1 if not tracked
    float validationLoss = -1.0f;   // -1 if no validation set
    float validationAccuracy = -1.0f;
    std::size_t step = 0;           // global optimizer steps completed so far
    float learningRate = 0.0f;      // learning rate of the last step
    float gradientNorm = 0.0f;      // mean global gradient norm (before clipping) over the epoch
};

using History = std::vector<EpochStats>;

struct FitOptions {
    std::size_t epochs = 10;
    std::size_t batchSize = 32;
    bool shuffle = true;
    bool trackAccuracy = false;     // classification accuracy per epoch
    bool verbose = true;            // print a line every `printEvery` epochs
    std::size_t printEvery = 1;
    std::size_t printEveryBatches = 0;  // also print running loss every N batches (0 = off)
    const Matrix* validationX = nullptr;
    const Matrix* validationY = nullptr;
    std::function<void(const EpochStats&)> onEpochEnd;  // optional callback

    // Learning rate for each global step (e.g. a CosineSchedule); empty keeps
    // the optimizer's own learning rate.
    std::function<float(std::size_t step)> learningRate;
    float clipGradNorm = 0.0f;      // clip the global gradient norm to this (0 = off)
    std::size_t startStep = 0;      // global step to continue from when resuming
    std::size_t maxSteps = 0;       // stop once this global step is reached (0 = no limit)
    std::function<void(std::size_t step)> onStep;  // after every optimizer step
};

// A feed-forward stack of layers, trained with mini-batch gradient descent.
class Network {
public:
    Network() = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;

    // net.add<Dense>(2, 8); net.add<ReLU>(); ...
    template <typename L, typename... Args>
    L& add(Args&&... args) {
        auto layer = std::make_unique<L>(std::forward<Args>(args)...);
        L& ref = *layer;
        layers_.push_back(std::move(layer));
        return ref;
    }
    Network& add(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix& input, bool training = false);
    Matrix predict(const Matrix& input) { return forward(input, false); }
    Matrix backward(const Matrix& gradOutput);

    // One optimization step on a single batch; returns the batch loss.
    float trainBatch(const Matrix& x, const Matrix& y, Loss& loss, Optimizer& optimizer);
    History fit(const Matrix& x, const Matrix& y, Loss& loss, Optimizer& optimizer,
                const FitOptions& options = FitOptions{});

    float evaluate(const Matrix& x, const Matrix& y, Loss& loss, std::size_t batchSize = 256);
    // Accuracy of predict(x) against y. Multi-column y: argmax vs argmax (one-hot).
    // Single-column y: prediction >= 0.5 vs target >= 0.5 (sigmoid output).
    float accuracy(const Matrix& x, const Matrix& y, std::size_t batchSize = 256);

    std::vector<Param> parameters();
    std::size_t parameterCount();
    // Frees every gradient buffer (they are recreated by the next backward
    // pass). Halves the memory of an inference-only model.
    void releaseGradients();
    std::size_t layerCount() const { return layers_.size(); }
    Layer& layer(std::size_t i) { return *layers_.at(i); }
    const Layer& layer(std::size_t i) const { return *layers_.at(i); }

    void summary(std::ostream& os) const;
    void summary() const;

    // Saves/loads parameter values only (like a state dict). Load into a
    // network with the same architecture. Throws std::runtime_error on failure.
    void save(const std::string& path);
    void load(const std::string& path);

    void saveParameters(std::ostream& out);
    void loadParameters(std::istream& in, const std::string& source);

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

// A checkpoint holds the weights, the optimizer state, the global step and the
// random generator state, so `fit` with options.startStep = the returned step
// continues exactly as if training had never stopped.
void saveCheckpoint(const std::string& path, Network& net, const Optimizer& optimizer, std::size_t step);
std::size_t loadCheckpoint(const std::string& path, Network& net, Optimizer& optimizer);

// Fraction of rows where predictions and targets agree (see Network::accuracy).
float accuracyScore(const Matrix& predictions, const Matrix& targets);

} // namespace nc

#endif // NETWORK_H
