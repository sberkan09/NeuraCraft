#include "Dense.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace nc {

Dense::Dense(std::size_t inputSize, std::size_t outputSize, Init init)
    : inputSize_(inputSize), outputSize_(outputSize),
      bias_(1, outputSize, 0.0f), weightGrad_(inputSize, outputSize), biasGrad_(1, outputSize) {
    if (inputSize == 0 || outputSize == 0) throw std::invalid_argument("Dense: sizes must be > 0");

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

Matrix Dense::forward(const Matrix& input, bool /*training*/) {
    if (input.cols() != inputSize_) {
        throw std::invalid_argument("Dense: expected " + std::to_string(inputSize_) +
                                    " input features, got " + std::to_string(input.cols()));
    }
    input_ = input;
    return (input * weights_).addRowVector(bias_);
}

Matrix Dense::backward(const Matrix& gradOutput) {
    if (input_.empty()) throw std::logic_error("Dense::backward called before forward");
    weightGrad_ = input_.transposeMatmul(gradOutput);  // dL/dW = x^T g
    biasGrad_ = gradOutput.sumRows();                    // dL/db = sum over batch
    return gradOutput.matmulTranspose(weights_);         // dL/dx = g W^T
}

std::vector<Param> Dense::parameters() {
    return {{&weights_, &weightGrad_}, {&bias_, &biasGrad_}};
}

std::string Dense::describe() const {
    return "Dense(" + std::to_string(inputSize_) + " -> " + std::to_string(outputSize_) + ")";
}

void Dense::printWeight() const {
    weights_.print();
}

} // namespace nc
