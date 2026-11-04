#ifndef NEURACRAFT_QUANTIZED_H
#define NEURACRAFT_QUANTIZED_H

#include "Matrix.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nc {

// Int8 weight matrix for inference: symmetric quantization with one float
// scale per group of `group` consecutive inputs of each output feature
// (w[j][k] ~= scale[j][k / group] * q[j][k]). Small groups keep a single
// outlier weight from coarsening a whole row. About a quarter of the memory of
// float weights, and products read a quarter of the bytes, which is what
// limits token-by-token generation.
class QuantizedWeights {
public:
    static constexpr std::size_t kDefaultGroup = 64;

    QuantizedWeights() = default;
    // From a (in x out) weight matrix, the layout Dense uses.
    static QuantizedWeights fromInOut(const Matrix& w, std::size_t group = kDefaultGroup);
    // From an (out x in) matrix, the layout of PyTorch / Hugging Face Linear layers.
    static QuantizedWeights fromOutIn(const Matrix& w, std::size_t group = kDefaultGroup);
    std::size_t group() const { return group_; }

    std::size_t inputs() const { return in_; }
    std::size_t outputs() const { return out_; }
    bool empty() const { return q_.empty(); }
    std::size_t bytes() const { return q_.size() + scale_.size() * sizeof(float); }

    // x (N x in) -> N x out. Each weight row is expanded once and applied to
    // all N inputs, so a forward pass reads the weights once whatever N is.
    Matrix multiply(const Matrix& x) const;
    Matrix dequantize() const;  // back to (in x out) floats

private:
    std::size_t in_ = 0, out_ = 0, group_ = kDefaultGroup, groupsPerRow_ = 0;
    std::vector<std::int8_t> q_;  // out x in, row-major
    std::vector<float> scale_;    // out x groupsPerRow
};

} // namespace nc

#endif // NEURACRAFT_QUANTIZED_H
