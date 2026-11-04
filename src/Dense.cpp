#include "Dense.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace nc {

Dense::Dense(std::size_t inputSize, std::size_t outputSize, Init init)
    : inputSize_(inputSize), outputSize_(outputSize) {
    if (inputSize == 0 || outputSize == 0) throw std::invalid_argument("Dense: sizes must be > 0");
    if (SkipParameterInit::active()) return;  // a loader assigns the parameters
    bias_ = Matrix(1, outputSize, 0.0f);
    weightGrad_ = Matrix(inputSize, outputSize);
    biasGrad_ = Matrix(1, outputSize);

    switch (init) {
    case Init::Xavier: {
        const float limit = std::sqrt(6.0f / static_cast<float>(inputSize + outputSize));
        weights_ = Matrix::uniform(inputSize, outputSize, -limit, limit);
        break;
    }
    case Init::He:
        weights_ = Matrix::normal(inputSize, outputSize, 0.0f, std::sqrt(2.0f / static_cast<float>(inputSize)));
        break;
    case Init::Zeros:
        weights_ = Matrix::zeros(inputSize, outputSize);
        break;
    }
}

Matrix Dense::forward(const Matrix& input, bool training) {
    if (input.cols() != inputSize_) {
        throw std::invalid_argument("Dense: expected " + std::to_string(inputSize_) +
                                    " input features, got " + std::to_string(input.cols()));
    }
    if (quantized()) return quantized_.multiply(input).addRowVector(bias_);
    // Inference keeps nothing for backward (large models would hold every activation).
    input_ = training ? input : Matrix();
    return (input * weights_).addRowVector(bias_);
}

std::size_t Dense::quantize() {
    if (quantized() || weights_.empty()) return 0;
    quantized_ = QuantizedWeights::fromInOut(weights_);
    weights_ = Matrix();
    weightGrad_ = Matrix();
    biasGrad_ = Matrix();
    input_ = Matrix();
    return 1;
}

void Dense::setQuantized(QuantizedWeights weights) {
    if (weights.inputs() != inputSize_ || weights.outputs() != outputSize_)
        throw std::invalid_argument("Dense::setQuantized: weights are " + std::to_string(weights.inputs()) + " -> " +
                                    std::to_string(weights.outputs()) + ", layer is " + std::to_string(inputSize_) +
                                    " -> " + std::to_string(outputSize_));
    quantized_ = std::move(weights);
    weights_ = Matrix();
    weightGrad_ = Matrix();
    biasGrad_ = Matrix();
    if (bias_.empty()) bias_ = Matrix(1, outputSize_);
}

Matrix Dense::backward(const Matrix& gradOutput) {
    if (quantized()) throw std::logic_error("Dense: quantized layers are inference-only");
    if (input_.empty()) throw std::logic_error("Dense::backward called before forward");
    weightGrad_ = input_.transposeMatmul(gradOutput);  // dL/dW = x^T g
    biasGrad_ = gradOutput.sumRows();                    // dL/db = sum over batch
    return gradOutput.matmulTranspose(weights_);         // dL/dx = g W^T
}

std::vector<Param> Dense::parameters() {
    if (quantized()) return {};
    return {{&weights_, &weightGrad_}, {&bias_, &biasGrad_}};
}

std::string Dense::describe() const {
    return "Dense(" + std::to_string(inputSize_) + " -> " + std::to_string(outputSize_) + (quantized() ? ", int8" : "") + ")";
}

void Dense::printWeight() const {
    weights_.print();
}

} // namespace nc
