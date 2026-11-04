#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"
#include "Quantized.h"

namespace nc {

enum class Init {
    Xavier,  // Glorot uniform; good for tanh / sigmoid / linear outputs
    He,      // He normal; good for ReLU-family activations
    Zeros,
};

// Fully connected layer: y = x W + b, W is (in x out), b is (1 x out).
class Dense : public Layer {
public:
    Dense(std::size_t inputSize, std::size_t outputSize, Init init = Init::Xavier);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "Dense"; }
    std::string describe() const override;
    std::size_t outputSize(std::size_t) const override { return outputSize_; }
    // Int8 weights for inference: forward uses them, backward then throws and
    // parameters() lists nothing (save a network before quantizing it).
    std::size_t quantize() override;
    bool quantized() const { return !quantized_.empty(); }
    // Installs ready-made int8 weights (inputs x outputs must match).
    void setQuantized(QuantizedWeights weights);

    std::size_t inputSize() const { return inputSize_; }
    std::size_t outputSize() const { return outputSize_; }
    Matrix& weights() { return weights_; }
    Matrix& bias() { return bias_; }
    const Matrix& weightGrad() const { return weightGrad_; }
    const Matrix& biasGrad() const { return biasGrad_; }

    void printWeight() const;

private:
    std::size_t inputSize_;
    std::size_t outputSize_;
    Matrix weights_;
    Matrix bias_;
    Matrix weightGrad_;
    Matrix biasGrad_;
    Matrix input_;
    QuantizedWeights quantized_;
};

} // namespace nc

#endif // DENSE_H
