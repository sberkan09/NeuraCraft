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

    // x (N x in) -> N x out. Up to kGemmRows rows (token-by-token generation,
    // prompts) the inputs are quantized too, per 32 columns (as llama.cpp's
    // q8_0 x q8_0), so every block is an integer dot product: AVX-VNNI / AVX2
    // on x86, NEON dot product on ARM, plain C++ elsewhere; each weight row is
    // read once and applied to all N inputs while it is in cache. With more
    // rows (image models: thousands of pixels) tiles of weights are expanded
    // to float and multiplied with the float GEMM, which is then faster and
    // keeps the inputs exact.
    Matrix multiply(const Matrix& x) const;
    static constexpr std::size_t kGemmRows = 256;
    // Rows j0..j1 as floats, (j1 - j0) x inputs(), into dst.
    void expandRows(std::size_t j0, std::size_t j1, float* dst) const;
    // Same product with float inputs (each weight row expanded to float):
    // slower, without the input rounding. For checks and comparisons.
    Matrix multiplyFloat(const Matrix& x) const;
    static const char* kernelName();  // which integer kernel multiply() uses
    Matrix dequantize() const;  // back to (in x out) floats
    // Output feature j's weights as floats (dst holds inputs() values), e.g. an
    // embedding row when the table is stored quantized.
    void expandRow(std::size_t j, float* dst) const;

private:
    std::size_t in_ = 0, out_ = 0, group_ = kDefaultGroup, groupsPerRow_ = 0;
    std::vector<std::int8_t> q_;  // out x in, row-major
    std::vector<float> scale_;    // out x groupsPerRow
};

// Whether QuantizedWeights::multiply quantizes its inputs (default: yes).
// Integer products are several times faster; models whose activations have
// extreme outliers lose precision (GPT-2: perplexity -3.7 % instead of
// +0.5 %; Llama 3.2: +0.05 %), so the gpt2 command turns this off.
void setInt8Inputs(bool enabled);
bool int8Inputs();

} // namespace nc

#endif // NEURACRAFT_QUANTIZED_H
