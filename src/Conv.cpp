#include "Conv.h"
#include "Backend.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nc {

namespace {
std::string shapeString(const ImageShape& s) {
    return std::to_string(s.channels) + "x" + std::to_string(s.height) + "x" + std::to_string(s.width);
}

void checkInput(const Matrix& input, const ImageShape& shape, const char* who) {
    if (input.cols() != shape.size()) {
        throw std::invalid_argument(std::string(who) + ": expected " + std::to_string(shape.size()) +
                                    " input features (" + shapeString(shape) + "), got " +
                                    std::to_string(input.cols()));
    }
}
} // namespace

// ---- Conv2D -----------------------------------------------------------------------

Conv2D::Conv2D(ImageShape input, std::size_t outChannels, std::size_t kernelSize, std::size_t stride,
               std::size_t padding, Init init)
    : in_(input), outChannels_(outChannels), kernel_(kernelSize), stride_(stride), padding_(padding) {
    if (in_.size() == 0 || outChannels == 0 || kernelSize == 0 || stride == 0)
        throw std::invalid_argument("Conv2D: sizes and stride must be > 0");
    if (in_.height + 2 * padding < kernelSize || in_.width + 2 * padding < kernelSize)
        throw std::invalid_argument("Conv2D: kernel larger than the padded input");
    outH_ = (in_.height + 2 * padding - kernelSize) / stride + 1;
    outW_ = (in_.width + 2 * padding - kernelSize) / stride + 1;

    const std::size_t fanIn = in_.channels * kernelSize * kernelSize;
    if (SkipParameterInit::active()) return;  // a loader assigns the parameters
    switch (init) {
    case Init::He:
        weights_ = Matrix::normal(fanIn, outChannels, 0.0f, std::sqrt(2.0f / static_cast<float>(fanIn)));
        break;
    case Init::Xavier: {
        const float fanOut = static_cast<float>(outChannels * kernelSize * kernelSize);
        const float limit = std::sqrt(6.0f / (static_cast<float>(fanIn) + fanOut));
        weights_ = Matrix::uniform(fanIn, outChannels, -limit, limit);
        break;
    }
    case Init::Zeros:
        weights_ = Matrix::zeros(fanIn, outChannels);
        break;
    }
    bias_ = Matrix(1, outChannels);
    weightGrad_ = Matrix(fanIn, outChannels);
    biasGrad_ = Matrix(1, outChannels);
}

Matrix Conv2D::forwardInference(const Matrix& input) const {
    const std::size_t B = input.rows(), P = outH_ * outW_, C = in_.channels, H = in_.height, W = in_.width;
    const std::size_t patch = C * kernel_ * kernel_;
    Matrix out(B, outChannels_ * P);
    for (std::size_t b = 0; b < B; ++b) {
        const float* img = input.data() + b * input.cols();
        float* dst = out.data() + b * out.cols();
        if (kernel_ == 1 && stride_ == 1 && padding_ == 0) {
            // The image already is a (C x HW) matrix: out = W^T x image.
            gemm(true, false, outChannels_, P, C, weights_.data(), outChannels_, img, P, dst, P, false);
        } else {
            // Strips of output rows: columns (patch x strip positions), then
            // out[:, strip] = W^T x columns, written straight into place.
            const std::size_t budget = std::size_t(1) << 23;  // floats per strip of columns (32 MB)
            const std::size_t rows = std::max<std::size_t>(1, std::min(outH_, budget / std::max<std::size_t>(1, patch * outW_)));
            std::vector<float> cols(patch * rows * outW_);
            for (std::size_t oy0 = 0; oy0 < outH_; oy0 += rows) {
                const std::size_t oy1 = std::min(outH_, oy0 + rows), n = (oy1 - oy0) * outW_;
                parallelFor(0, patch, 8, [&](std::size_t r0, std::size_t r1) {
                    for (std::size_t r = r0; r < r1; ++r) {
                        const std::size_t c = r / (kernel_ * kernel_), ky = r / kernel_ % kernel_, kx = r % kernel_;
                        float* row = cols.data() + r * n;
                        for (std::size_t oy = oy0; oy < oy1; ++oy) {
                            const long iy = static_cast<long>(oy * stride_ + ky) - static_cast<long>(padding_);
                            float* o = row + (oy - oy0) * outW_;
                            if (iy < 0 || iy >= static_cast<long>(H)) {
                                std::fill(o, o + outW_, 0.0f);
                                continue;
                            }
                            const float* src = img + (c * H + static_cast<std::size_t>(iy)) * W;
                            for (std::size_t ox = 0; ox < outW_; ++ox) {
                                const long ix = static_cast<long>(ox * stride_ + kx) - static_cast<long>(padding_);
                                o[ox] = ix >= 0 && ix < static_cast<long>(W) ? src[ix] : 0.0f;
                            }
                        }
                    }
                });
                gemm(true, false, outChannels_, n, patch, weights_.data(), outChannels_, cols.data(), n, dst + oy0 * outW_, P,
                     false);
            }
        }
        parallelFor(0, outChannels_, 4, [&](std::size_t c0, std::size_t c1) {
            for (std::size_t oc = c0; oc < c1; ++oc) {
                const float bias = bias_(0, oc);
                float* plane = dst + oc * P;
                for (std::size_t p = 0; p < P; ++p) plane[p] += bias;
            }
        });
    }
    return out;
}

Matrix Conv2D::forward(const Matrix& input, bool training) {
    checkInput(input, in_, "Conv2D");
    if (!training) {
        columns_ = Matrix();  // nothing kept for backward
        return forwardInference(input);
    }
    batch_ = input.rows();
    const std::size_t positions = outH_ * outW_;
    const std::size_t patch = in_.channels * kernel_ * kernel_;
    const std::size_t H = in_.height, W = in_.width;

    // im2col: one row per output position, one column per (channel, ky, kx).
    columns_ = Matrix(batch_ * positions, patch);
    parallelFor(0, batch_, 1, [&](std::size_t b0, std::size_t b1) {
        for (std::size_t b = b0; b < b1; ++b) {
            const float* img = input.data() + b * input.cols();
            for (std::size_t oy = 0; oy < outH_; ++oy)
                for (std::size_t ox = 0; ox < outW_; ++ox) {
                    float* row = columns_.data() + (b * positions + oy * outW_ + ox) * patch;
                    for (std::size_t c = 0; c < in_.channels; ++c)
                        for (std::size_t ky = 0; ky < kernel_; ++ky) {
                            const long iy = static_cast<long>(oy * stride_ + ky) - static_cast<long>(padding_);
                            for (std::size_t kx = 0; kx < kernel_; ++kx) {
                                const long ix = static_cast<long>(ox * stride_ + kx) - static_cast<long>(padding_);
                                const bool inside = iy >= 0 && ix >= 0 && iy < static_cast<long>(H) &&
                                                    ix < static_cast<long>(W);
                                *row++ = inside ? img[(c * H + static_cast<std::size_t>(iy)) * W +
                                                      static_cast<std::size_t>(ix)]
                                                : 0.0f;
                            }
                        }
                }
        }
    });

    const Matrix y = (columns_ * weights_).addRowVector(bias_);  // (B*positions) x outC

    Matrix out(batch_, outChannels_ * positions);
    parallelFor(0, batch_, 1, [&](std::size_t b0, std::size_t b1) {
        for (std::size_t b = b0; b < b1; ++b)
            for (std::size_t p = 0; p < positions; ++p)
                for (std::size_t oc = 0; oc < outChannels_; ++oc)
                    out(b, oc * positions + p) = y(b * positions + p, oc);
    });
    return out;
}

Matrix Conv2D::backward(const Matrix& gradOutput) {
    if (columns_.empty()) throw std::logic_error("Conv2D::backward called before forward");
    const std::size_t positions = outH_ * outW_;
    const std::size_t patch = in_.channels * kernel_ * kernel_;
    if (gradOutput.rows() != batch_ || gradOutput.cols() != outChannels_ * positions)
        throw std::invalid_argument("Conv2D::backward: gradient shape does not match the last forward");

    Matrix g(batch_ * positions, outChannels_);
    for (std::size_t b = 0; b < batch_; ++b)
        for (std::size_t oc = 0; oc < outChannels_; ++oc)
            for (std::size_t p = 0; p < positions; ++p) g(b * positions + p, oc) = gradOutput(b, oc * positions + p);

    weightGrad_ = columns_.transposeMatmul(g);
    biasGrad_ = g.sumRows();
    const Matrix dColumns = g.matmulTranspose(weights_);  // (B*positions) x patch

    // col2im: scatter-add patch gradients back to the input pixels.
    const std::size_t H = in_.height, W = in_.width;
    Matrix dInput(batch_, in_.size());
    parallelFor(0, batch_, 1, [&](std::size_t b0, std::size_t b1) {
        for (std::size_t b = b0; b < b1; ++b) {
            float* img = dInput.data() + b * in_.size();
            for (std::size_t oy = 0; oy < outH_; ++oy)
                for (std::size_t ox = 0; ox < outW_; ++ox) {
                    const float* row = dColumns.data() + (b * positions + oy * outW_ + ox) * patch;
                    for (std::size_t c = 0; c < in_.channels; ++c)
                        for (std::size_t ky = 0; ky < kernel_; ++ky) {
                            const long iy = static_cast<long>(oy * stride_ + ky) - static_cast<long>(padding_);
                            for (std::size_t kx = 0; kx < kernel_; ++kx, ++row) {
                                const long ix = static_cast<long>(ox * stride_ + kx) - static_cast<long>(padding_);
                                if (iy >= 0 && ix >= 0 && iy < static_cast<long>(H) && ix < static_cast<long>(W))
                                    img[(c * H + static_cast<std::size_t>(iy)) * W + static_cast<std::size_t>(ix)] += *row;
                            }
                        }
                }
        }
    });
    return dInput;
}

std::vector<Param> Conv2D::parameters() {
    return {{&weights_, &weightGrad_}, {&bias_, &biasGrad_}};
}

std::string Conv2D::describe() const {
    std::string s = "Conv2D(" + shapeString(in_) + " -> " + shapeString(outputShape()) + ", k" +
                    std::to_string(kernel_);
    if (stride_ != 1) s += " s" + std::to_string(stride_);
    if (padding_ != 0) s += " p" + std::to_string(padding_);
    return s + ")";
}

// ---- MaxPool2D ----------------------------------------------------------------------

MaxPool2D::MaxPool2D(ImageShape input, std::size_t poolSize, std::size_t stride)
    : in_(input), pool_(poolSize), stride_(stride == 0 ? poolSize : stride) {
    if (in_.size() == 0 || poolSize == 0) throw std::invalid_argument("MaxPool2D: sizes must be > 0");
    if (poolSize > in_.height || poolSize > in_.width) throw std::invalid_argument("MaxPool2D: pool larger than input");
    outH_ = (in_.height - pool_) / stride_ + 1;
    outW_ = (in_.width - pool_) / stride_ + 1;
}

Matrix MaxPool2D::forward(const Matrix& input, bool) {
    checkInput(input, in_, "MaxPool2D");
    batch_ = input.rows();
    const ImageShape o = outputShape();
    Matrix out(batch_, o.size());
    argmax_.assign(batch_ * o.size(), 0);
    parallelFor(0, batch_, 1, [&](std::size_t b0, std::size_t b1) {
        for (std::size_t b = b0; b < b1; ++b) {
            const float* img = input.data() + b * in_.size();
            for (std::size_t c = 0; c < in_.channels; ++c)
                for (std::size_t oy = 0; oy < outH_; ++oy)
                    for (std::size_t ox = 0; ox < outW_; ++ox) {
                        float best = -std::numeric_limits<float>::infinity();
                        std::size_t bestIdx = 0;
                        for (std::size_t py = 0; py < pool_; ++py)
                            for (std::size_t px = 0; px < pool_; ++px) {
                                const std::size_t idx =
                                    (c * in_.height + oy * stride_ + py) * in_.width + ox * stride_ + px;
                                if (img[idx] > best) {
                                    best = img[idx];
                                    bestIdx = idx;
                                }
                            }
                        const std::size_t o_idx = (c * outH_ + oy) * outW_ + ox;
                        out(b, o_idx) = best;
                        argmax_[b * o.size() + o_idx] = bestIdx;
                    }
        }
    });
    return out;
}

Matrix MaxPool2D::backward(const Matrix& gradOutput) {
    const std::size_t outSize = outputShape().size();
    if (gradOutput.rows() != batch_ || gradOutput.cols() != outSize || argmax_.size() != batch_ * outSize)
        throw std::invalid_argument("MaxPool2D::backward: gradient shape does not match the last forward");
    Matrix dInput(batch_, in_.size());
    for (std::size_t b = 0; b < batch_; ++b)
        for (std::size_t i = 0; i < outSize; ++i) dInput(b, argmax_[b * outSize + i]) += gradOutput(b, i);
    return dInput;
}

std::string MaxPool2D::describe() const {
    return "MaxPool2D(" + shapeString(in_) + " -> " + shapeString(outputShape()) + ", " + std::to_string(pool_) + ")";
}

} // namespace nc
