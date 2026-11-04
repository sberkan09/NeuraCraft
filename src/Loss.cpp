#include "Loss.h"
#include "Activations.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nc {

namespace {
constexpr float kEps = 1e-7f;

float clampProb(float p) { return std::min(std::max(p, kEps), 1.0f - kEps); }
} // namespace

void Loss::checkShapes(const Matrix& predictions, const Matrix& targets, const char* who) {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument(std::string(who) + ": predictions (" + std::to_string(predictions.rows()) +
                                    "x" + std::to_string(predictions.cols()) + ") and targets (" +
                                    std::to_string(targets.rows()) + "x" + std::to_string(targets.cols()) +
                                    ") differ in shape");
    }
    if (predictions.empty()) throw std::invalid_argument(std::string(who) + ": empty batch");
}

// ---- MSE --------------------------------------------------------------------

float MSE::compute(const Matrix& predictions, const Matrix& targets) {
    checkShapes(predictions, targets, "MSE");
    diff_ = predictions - targets;
    double acc = 0.0;
    for (float d : diff_.values()) acc += static_cast<double>(d) * d;
    return static_cast<float>(acc / static_cast<double>(diff_.size()));
}

Matrix MSE::gradient() const {
    return diff_ * (2.0f / static_cast<float>(diff_.size()));
}

// ---- BinaryCrossEntropy -------------------------------------------------------

float BinaryCrossEntropy::compute(const Matrix& predictions, const Matrix& targets) {
    checkShapes(predictions, targets, "BinaryCrossEntropy");
    predictions_ = predictions;
    targets_ = targets;
    double acc = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        const float p = clampProb(predictions.data()[i]);
        const float t = targets.data()[i];
        acc -= t * std::log(p) + (1.0f - t) * std::log(1.0f - p);
    }
    return static_cast<float>(acc / static_cast<double>(predictions.size()));
}

Matrix BinaryCrossEntropy::gradient() const {
    Matrix grad(predictions_.rows(), predictions_.cols());
    const float n = static_cast<float>(predictions_.size());
    for (std::size_t i = 0; i < grad.size(); ++i) {
        const float p = clampProb(predictions_.data()[i]);
        const float t = targets_.data()[i];
        grad.data()[i] = (p - t) / (p * (1.0f - p)) / n;
    }
    return grad;
}

// ---- CategoricalCrossEntropy --------------------------------------------------

float CategoricalCrossEntropy::compute(const Matrix& predictions, const Matrix& targets) {
    checkShapes(predictions, targets, "CategoricalCrossEntropy");
    predictions_ = predictions;
    targets_ = targets;
    double acc = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        const float t = targets.data()[i];
        if (t != 0.0f) acc -= t * std::log(clampProb(predictions.data()[i]));
    }
    return static_cast<float>(acc / static_cast<double>(predictions.rows()));
}

Matrix CategoricalCrossEntropy::gradient() const {
    Matrix grad(predictions_.rows(), predictions_.cols());
    const float n = static_cast<float>(predictions_.rows());
    for (std::size_t i = 0; i < grad.size(); ++i) {
        grad.data()[i] = -targets_.data()[i] / clampProb(predictions_.data()[i]) / n;
    }
    return grad;
}

// ---- SoftmaxCrossEntropy ------------------------------------------------------

float SoftmaxCrossEntropy::compute(const Matrix& logits, const Matrix& targets) {
    checkShapes(logits, targets, "SoftmaxCrossEntropy");
    probabilities_ = softmax(logits);
    targets_ = targets;
    double acc = 0.0;
    for (std::size_t i = 0; i < logits.rows(); ++i) {
        // log-softmax computed from logits directly for stability.
        float maxVal = logits(i, 0);
        for (std::size_t j = 1; j < logits.cols(); ++j) maxVal = std::max(maxVal, logits(i, j));
        double sumExp = 0.0;
        for (std::size_t j = 0; j < logits.cols(); ++j) sumExp += std::exp(logits(i, j) - maxVal);
        const double logSumExp = maxVal + std::log(sumExp);
        for (std::size_t j = 0; j < logits.cols(); ++j) {
            const float t = targets(i, j);
            if (t != 0.0f) acc -= t * (logits(i, j) - logSumExp);
        }
    }
    return static_cast<float>(acc / static_cast<double>(logits.rows()));
}

Matrix SoftmaxCrossEntropy::gradient() const {
    return (probabilities_ - targets_) / static_cast<float>(probabilities_.rows());
}

} // namespace nc
