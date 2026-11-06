#ifndef NEURACRAFT_IMAGE_H
#define NEURACRAFT_IMAGE_H

// Layers for convolutional image models (VAE, diffusion UNet) and image files.
// Every layer works on the library's image layout: one image per row,
// flattened channel-major (C, H, W) — see ImageShape in Conv.h.

#include "Conv.h"
#include "Layer.h"
#include "Network.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nc {

// x * sigmoid(x) (also called swish), element-wise.
class SiLU : public Layer {
public:
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "SiLU"; }

private:
    Matrix input_;
};

// Group normalization: channels are split into `groups` groups, each
// normalized over its channels and all pixels, then scaled and shifted per
// channel (gamma, beta).
class GroupNorm : public Layer {
public:
    GroupNorm(ImageShape shape, std::size_t groups, float epsilon = 1e-5f);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override { return {{&gamma_, &gammaGrad_}, {&beta_, &betaGrad_}}; }
    std::string name() const override { return "GroupNorm"; }
    std::string describe() const override;

    Matrix& gamma() { return gamma_; }  // 1 x channels
    Matrix& beta() { return beta_; }
    ImageShape shape() const { return shape_; }
    std::size_t groups() const { return groups_; }
    float epsilon() const { return epsilon_; }

private:
    ImageShape shape_;
    std::size_t groups_;
    float epsilon_;
    Matrix gamma_, beta_, gammaGrad_, betaGrad_;
    Matrix normalized_;              // cached for backward
    std::vector<float> invStd_;      // per (sample, group)
};

// Nearest-neighbour upsampling by an integer factor.
class Upsample2D : public Layer {
public:
    Upsample2D(ImageShape input, std::size_t factor = 2);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "Upsample2D"; }
    std::string describe() const override;
    std::size_t outputSize(std::size_t) const override { return outputShape().size(); }
    ImageShape outputShape() const { return {in_.channels, in_.height * factor_, in_.width * factor_}; }
    ImageShape inputShape() const { return in_; }
    std::size_t factor() const { return factor_; }

private:
    ImageShape in_;
    std::size_t factor_;
};

// Zero padding with separate amounts per side (e.g. the (0, 1, 0, 1) padding
// before Stable Diffusion's stride-2 downsampling convolutions).
class Pad2D : public Layer {
public:
    Pad2D(ImageShape input, std::size_t top, std::size_t bottom, std::size_t left, std::size_t right);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "Pad2D"; }
    std::string describe() const override;
    std::size_t outputSize(std::size_t) const override { return outputShape().size(); }
    ImageShape outputShape() const { return {in_.channels, in_.height + top_ + bottom_, in_.width + left_ + right_}; }
    ImageShape inputShape() const { return in_; }
    std::size_t top() const { return top_; }
    std::size_t bottom() const { return bottom_; }
    std::size_t left() const { return left_; }
    std::size_t right() const { return right_; }

private:
    ImageShape in_;
    std::size_t top_, bottom_, left_, right_;
};

// y = main(x) + shortcut(x), with an identity shortcut when it has no layers:
// the residual block of ResNets and of Stable Diffusion's VAE and UNet.
class ResnetBlock2D : public Layer {
public:
    // The standard pre-activation block: GroupNorm, SiLU, 3x3 conv, GroupNorm,
    // SiLU, 3x3 conv; a 1x1 conv shortcut when the channel count changes.
    ResnetBlock2D(ImageShape input, std::size_t outChannels, std::size_t groups = 32, float epsilon = 1e-6f);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "ResnetBlock2D"; }
    std::string describe() const override;
    std::size_t outputSize(std::size_t) const override { return outputShape().size(); }
    ImageShape outputShape() const { return {outChannels_, in_.height, in_.width}; }
    ImageShape inputShape() const { return in_; }

    Network& main() { return main_; }          // norm1, SiLU, conv1, norm2, SiLU, conv2
    Network& shortcut() { return shortcut_; }  // empty, or one 1x1 Conv2D

private:
    ImageShape in_;
    std::size_t outChannels_;
    Network main_, shortcut_;
};

// Self-attention over the pixels of a feature map (the attention block of
// Stable Diffusion's VAE): GroupNorm, then q/k/v projections of each pixel's
// channel vector, multi-head attention across all H*W pixels, an output
// projection and a residual connection. Queries are processed in chunks, so
// memory grows with H*W, not (H*W)^2. Inference only (backward throws).
class SpatialSelfAttention : public Layer {
public:
    SpatialSelfAttention(ImageShape shape, std::size_t heads = 1, std::size_t groups = 32, float epsilon = 1e-6f);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "SpatialSelfAttention"; }
    std::string describe() const override;

    GroupNorm& norm() { return norm_; }
    ImageShape shape() const { return shape_; }
    std::size_t heads() const { return heads_; }
    // Projections in PyTorch Linear layout: (out x in) = channels x channels.
    Matrix& query() { return wq_; }
    Matrix& key() { return wk_; }
    Matrix& value() { return wv_; }
    Matrix& out() { return wo_; }
    Matrix& queryBias() { return bq_; }
    Matrix& keyBias() { return bk_; }
    Matrix& valueBias() { return bv_; }
    Matrix& outBias() { return bo_; }

private:
    ImageShape shape_;
    std::size_t heads_;
    GroupNorm norm_;
    Matrix wq_, wk_, wv_, wo_, bq_, bk_, bv_, bo_;
    Matrix grads_[8];  // never filled (no backward), so optimizers and savers see a consistent list
};

// An 8-bit image, pixels interleaved (H, W, C), C = 1 (grey), 3 (RGB) or 4 (RGBA).
struct Image {
    std::size_t width = 0, height = 0, channels = 3;
    std::vector<std::uint8_t> pixels;
};

// PNG with adaptive per-row filters and its own deflate compressor (LZ77 +
// dynamic Huffman blocks).
void savePng(const std::string& path, const Image& image);
// Animated PNG (APNG): full-colour frames of one size, no palette or dither;
// browsers and most viewers play it, others show the first frame.
void saveApng(const std::string& path, const std::vector<Image>& frames, float framesPerSecond = 8.0f, bool loop = true);
// Baseline JPEG (4:2:0, standard Huffman tables, libjpeg's quality scaling).
std::vector<std::uint8_t> encodeJpeg(const Image& image, int quality = 90);
void saveJpeg(const std::string& path, const Image& image, int quality = 90);
// A QuickTime movie of JPEG frames (Photo-JPEG / Motion-JPEG): QuickTime,
// macOS Finder previews, VLC and ffmpeg play it; frames of one size.
void saveMjpegMov(const std::string& path, const std::vector<Image>& frames, float framesPerSecond = 8.0f, int quality = 90);
// A zlib stream (RFC 1950 / 1951) of `data`.
std::vector<std::uint8_t> deflateCompress(const std::vector<std::uint8_t>& data);
// PNG reader: 8-bit grey, grey + alpha, RGB, RGBA and palette images (also
// 1/2/4-bit palette and grey, 16-bit by keeping the high byte), all filters,
// with its own inflate; interlaced files are refused. Returns the stored
// channels (1, 2 -> grey kept, 3 or 4); use toRgb() for 3 channels.
Image loadPng(const std::string& path);
// The same from bytes in memory (name: for messages), and a PNG file's bytes.
Image decodePng(const std::vector<std::uint8_t>& data, const std::string& name = "PNG data");
std::vector<std::uint8_t> encodePng(const Image& image);

// The largest picture (width x height) the readers decode: a bigger file is refused before anything
// is allocated for it. Per thread; the default is 2^28 pixels.
void setMaxDecodePixels(std::size_t pixels);
std::size_t maxDecodePixels();
// PNG or PPM/PGM by file contents.
Image loadImage(const std::string& path);
// Drops alpha (composited on white) and expands grey to RGB.
Image toRgb(const Image& image);
// Bilinear resampling to width x height.
Image resizeImage(const Image& image, std::size_t width, std::size_t height);
// The largest centred crop with the aspect ratio of width : height, resized to width x height.
Image fitImage(const Image& image, std::size_t width, std::size_t height);
// Canny edges as OpenCV's cv2.Canny(image, low, high) computes them (the
// ControlNet canny preprocessor): grey, 3x3 Sobel, L1 gradient magnitude,
// non-maximum suppression, hysteresis between the two thresholds. Returns a
// 3-channel image, edges white on black.
Image cannyEdges(const Image& image, float low = 100.0f, float high = 200.0f);
// Affine resampling about the image centre: scale > 1 zooms in, (dx, dy)
// shifts the content by that many pixels, angle rotates it (degrees,
// counter-clockwise). Bilinear, edges clamped. Same size as the input.
Image transformImage(const Image& image, float scale, float dx = 0.0f, float dy = 0.0f, float angleDegrees = 0.0f);

// Animated GIF (RGB frames of one size): one 256-colour palette shared by all
// frames (median cut over all of them, so colours do not flicker), a fixed
// 4x4 ordered dither, LZW compression, looping forever when loop is set.
void saveGif(const std::string& path, const std::vector<Image>& frames, float framesPerSecond = 8.0f, bool loop = true);

// Binary PPM (P6) / PGM (P5), 8-bit.
void savePpm(const std::string& path, const Image& image);
Image loadPpm(const std::string& path);

// Between images and channel-major float rows: pixel value v maps to
// lo + (hi - lo) * v / 255 (Stable Diffusion uses [-1, 1]).
Matrix imageToTensor(const Image& image, float lo = -1.0f, float hi = 1.0f);
Image tensorToImage(const float* chw, ImageShape shape, float lo = -1.0f, float hi = 1.0f);

} // namespace nc

#endif // NEURACRAFT_IMAGE_H
