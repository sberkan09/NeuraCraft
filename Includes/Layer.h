#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"

#include <string>
#include <vector>

namespace nc {

// A trainable tensor and the gradient accumulated for it by backward().
struct Param {
    Matrix* value;
    Matrix* grad;
};

// Base class for every layer. Inputs/outputs are (batch x features) matrices.
//
// forward() caches whatever backward() needs. backward() receives dL/d(output),
// stores dL/d(param) in each parameter's grad, and returns dL/d(input) so the
// gradient can flow to the previous layer.
class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input, bool training) = 0;
    virtual Matrix backward(const Matrix& gradOutput) = 0;

    virtual std::vector<Param> parameters() { return {}; }
    virtual std::string name() const = 0;
    // Human-readable description, e.g. "Dense(784 -> 128)".
    virtual std::string describe() const { return name(); }
    // Number of output features for a given number of input features.
    virtual std::size_t outputSize(std::size_t inputSize) const { return inputSize; }
};

} // namespace nc

#endif // LAYER_H
