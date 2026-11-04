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

// While a SkipParameterInit object is alive on the current thread, layers are
// constructed without allocating or initializing their weights and gradients.
// Model loaders use this so a large checkpoint never needs random weights,
// gradient buffers and the loaded weights in memory at the same time; every
// parameter must then be assigned before use.
class SkipParameterInit {
public:
    SkipParameterInit();
    ~SkipParameterInit();
    SkipParameterInit(const SkipParameterInit&) = delete;
    SkipParameterInit& operator=(const SkipParameterInit&) = delete;
    static bool active();
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
    // Converts large weight matrices to int8 for inference (see Quantized.h).
    // Returns the number of matrices converted; layers without any return 0.
    virtual std::size_t quantize() { return 0; }
};

} // namespace nc

#endif // LAYER_H
