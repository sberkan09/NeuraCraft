#include "Activations.h"
#include "Random.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nc {

float sigmoid(float x) {
    // Branch keeps exp() from overflowing for large |x|.
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

Matrix softmax(const Matrix& logits) {
    Matrix out(logits.rows(), logits.cols());
    for (std::size_t i = 0; i < logits.rows(); ++i) {
        float maxVal = logits(i, 0);
        for (std::size_t j = 1; j < logits.cols(); ++j) maxVal = std::max(maxVal, logits(i, j));
        float total = 0.0f;
        for (std::size_t j = 0; j < logits.cols(); ++j) {
            out(i, j) = std::exp(logits(i, j) - maxVal);
            total += out(i, j);
        }
        for (std::size_t j = 0; j < logits.cols(); ++j) out(i, j) /= total;
    }
    return out;
}

// ---- ReLU -------------------------------------------------------------------

Matrix ReLU::forward(const Matrix& input, bool) {
    input_ = input;
    return input.apply([](float x) { return x > 0.0f ? x : 0.0f; });
}

Matrix ReLU::backward(const Matrix& gradOutput) {
    Matrix grad(gradOutput);
    for (std::size_t i = 0; i < grad.size(); ++i)
        if (input_.data()[i] <= 0.0f) grad.data()[i] = 0.0f;
    return grad;
}

// ---- LeakyReLU --------------------------------------------------------------

Matrix LeakyReLU::forward(const Matrix& input, bool) {
    input_ = input;
    const float a = alpha_;
    return input.apply([a](float x) { return x > 0.0f ? x : a * x; });
}

Matrix LeakyReLU::backward(const Matrix& gradOutput) {
    Matrix grad(gradOutput);
    for (std::size_t i = 0; i < grad.size(); ++i)
        if (input_.data()[i] <= 0.0f) grad.data()[i] *= alpha_;
    return grad;
}

// ---- Sigmoid ----------------------------------------------------------------

Matrix Sigmoid::forward(const Matrix& input, bool) {
    output_ = input.apply(sigmoid);
    return output_;
}

Matrix Sigmoid::backward(const Matrix& gradOutput) {
    Matrix grad(gradOutput);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        const float s = output_.data()[i];
        grad.data()[i] *= s * (1.0f - s);
    }
    return grad;
}

// ---- Tanh -------------------------------------------------------------------

Matrix Tanh::forward(const Matrix& input, bool) {
    output_ = input.apply([](float x) { return std::tanh(x); });
    return output_;
}

Matrix Tanh::backward(const Matrix& gradOutput) {
    Matrix grad(gradOutput);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        const float t = output_.data()[i];
        grad.data()[i] *= 1.0f - t * t;
    }
    return grad;
}

// ---- Softmax ----------------------------------------------------------------

Matrix Softmax::forward(const Matrix& input, bool) {
    output_ = softmax(input);
    return output_;
}

Matrix Softmax::backward(const Matrix& gradOutput) {
    // Jacobian-vector product per row: dx = s * (g - <g, s>)
    Matrix grad(gradOutput.rows(), gradOutput.cols());
    for (std::size_t i = 0; i < grad.rows(); ++i) {
        float dot = 0.0f;
        for (std::size_t j = 0; j < grad.cols(); ++j) dot += gradOutput(i, j) * output_(i, j);
        for (std::size_t j = 0; j < grad.cols(); ++j) grad(i, j) = output_(i, j) * (gradOutput(i, j) - dot);
    }
    return grad;
}

// ---- Dropout ----------------------------------------------------------------

Dropout::Dropout(float rate) : rate_(rate) {
    if (rate < 0.0f || rate >= 1.0f) throw std::invalid_argument("Dropout: rate must be in [0, 1)");
}

Matrix Dropout::forward(const Matrix& input, bool training) {
    if (!training || rate_ == 0.0f) {
        mask_ = Matrix();
        return input;
    }
    mask_ = Matrix(input.rows(), input.cols());
    std::bernoulli_distribution keep(1.0 - rate_);
    const float scale = 1.0f / (1.0f - rate_);
    for (auto& m : mask_.values()) m = keep(rng()) ? scale : 0.0f;
    return input.hadamard(mask_);
}

Matrix Dropout::backward(const Matrix& gradOutput) {
    return mask_.empty() ? gradOutput : gradOutput.hadamard(mask_);
}

std::string Dropout::describe() const {
    std::ostringstream os;
    os << "Dropout(" << std::setprecision(3) << rate_ << ")";
    return os.str();
}

} // namespace nc
