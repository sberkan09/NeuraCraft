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
    std::size_t layerCount() const { return layers_.size(); }
    Layer& layer(std::size_t i) { return *layers_.at(i); }
    const Layer& layer(std::size_t i) const { return *layers_.at(i); }

    void summary(std::ostream& os) const;
    void summary() const;

    // Saves/loads parameter values only (like a state dict). Load into a
    // network with the same architecture. Throws std::runtime_error on failure.
    void save(const std::string& path);
    void load(const std::string& path);

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

// Fraction of rows where predictions and targets agree (see Network::accuracy).
float accuracyScore(const Matrix& predictions, const Matrix& targets);

} // namespace nc

#endif // NETWORK_H
