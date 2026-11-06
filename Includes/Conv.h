#ifndef NEURACRAFT_CONV_H
#define NEURACRAFT_CONV_H

#include "Dense.h"  // Init
#include "Layer.h"
#include "Quantized.h"

#include <vector>

namespace nc {

// Images are stored one per row, flattened channel-major (C, H, W):
// value (c, y, x) of sample b is at m(b, (c * H + y) * W + x).
struct ImageShape {
    std::size_t channels = 1;
    std::size_t height = 1;
    std::size_t width = 1;
    std::size_t size() const { return channels * height * width; }
};

// 2-D convolution (cross-correlation) with square kernels, implemented as
// im2col + GEMM, so it runs on whichever backend is active. Training keeps the
// whole im2col matrix for backward; inference works in strips of output rows
// (bounded memory, output written in place in channel-major order), and 1x1
// convolutions are a single GEMM on the image itself.
// Stride-1 convolutions as k*k shifted GEMMs on a padded copy of the image
// instead of im2col (off by default; NEURACRAFT_SHIFTED_CONV=1 turns it on;
// `bench` compares both).
void setShiftedConvolutions(bool enabled);
bool shiftedConvolutions();

class Conv2D : public Layer {
public:
    Conv2D(ImageShape input, std::size_t outChannels, std::size_t kernelSize,
           std::size_t stride = 1, std::size_t padding = 0, Init init = Init::He);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "Conv2D"; }
    std::string describe() const override;
    std::size_t outputSize(std::size_t) const override { return outputShape().size(); }

    ImageShape inputShape() const { return in_; }
    ImageShape outputShape() const { return {outChannels_, outH_, outW_}; }
    // Weights are (inChannels * k * k) x outChannels; bias is 1 x outChannels.
    Matrix& weights() { return weights_; }
    Matrix& bias() { return bias_; }

    std::size_t kernelSize() const { return kernel_; }
    // Int8 weights for inference (integer products on im2col rows); backward
    // then throws and parameters() lists nothing.
    std::size_t quantize() override;
    bool quantized() const { return !quantized_.empty(); }

private:
    Matrix forwardInference(const Matrix& input) const;
    // Strip convolution with weights either (patch x out) (outByPatch = false) or (out x patch).
    // weights == nullptr: products come from quantized_ (multiplyLeft, the GPU int8 kernel).
    Matrix stripConvolution(const Matrix& input, const float* weights, bool outByPatch) const;
    // Stride-1 convolutions without im2col: k*k GEMMs on shifted views of a padded copy of the image.
    Matrix shiftedConvolution(const Matrix& input, const float* weights, bool outByPatch) const;
    // Stride 1, 16+ channels, blocked CPU kernel: taps read in place from a padded channels-last copy.
    Matrix directConvolution(const Matrix& input, const float* weights, bool outByPatch) const;
    Matrix forwardQuantized(const Matrix& input) const;
    QuantizedWeights quantized_;

    ImageShape in_;
    std::size_t outChannels_, kernel_, stride_, padding_, outH_, outW_;
    Matrix weights_, bias_, weightGrad_, biasGrad_;
    Matrix columns_;  // cached im2col of the last forward input
    std::size_t batch_ = 0;
};

// Max pooling over square windows, per channel.
class MaxPool2D : public Layer {
public:
    MaxPool2D(ImageShape input, std::size_t poolSize, std::size_t stride = 0);  // stride 0 = poolSize

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "MaxPool2D"; }
    std::string describe() const override;
    std::size_t outputSize(std::size_t) const override { return outputShape().size(); }

    ImageShape outputShape() const { return {in_.channels, outH_, outW_}; }

private:
    ImageShape in_;
    std::size_t pool_, stride_, outH_, outW_;
    std::vector<std::size_t> argmax_;  // input column of each output's maximum
    std::size_t batch_ = 0;
};

} // namespace nc

#endif // NEURACRAFT_CONV_H
