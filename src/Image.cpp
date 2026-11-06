#include "Image.h"
#include "Profile.h"
#include "VectorMath.h"
#include "Backend.h"
#include "Parallel.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace nc {

namespace {
std::string shapeString(const ImageShape& s) {
    return std::to_string(s.channels) + "x" + std::to_string(s.height) + "x" + std::to_string(s.width);
}

void expectColumns(const Matrix& m, std::size_t cols, const char* who) {
    if (m.cols() != cols)
        throw std::invalid_argument(std::string(who) + ": expected " + std::to_string(cols) + " features per row, got " +
                                    std::to_string(m.cols()));
}
} // namespace

// ---- SiLU ---------------------------------------------------------------------------------

Matrix SiLU::forward(const Matrix& input, bool training) {
    input_ = training ? input : Matrix();
    Matrix out(input.rows(), input.cols());
    const float* x = input.data();
    float* y = out.data();
    parallelFor(0, input.size(), 1 << 14, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) y[i] = x[i] / (1.0f + std::exp(-x[i]));
    });
    return out;
}

Matrix SiLU::backward(const Matrix& gradOutput) {
    if (input_.empty()) throw std::logic_error("SiLU::backward called before a training forward");
    Matrix out(gradOutput.rows(), gradOutput.cols());
    const float* x = input_.data();
    const float* g = gradOutput.data();
    float* d = out.data();
    parallelFor(0, out.size(), 1 << 14, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            const float s = 1.0f / (1.0f + std::exp(-x[i]));
            d[i] = g[i] * (s + x[i] * s * (1.0f - s));
        }
    });
    return out;
}

// ---- GroupNorm ----------------------------------------------------------------------------

GroupNorm::GroupNorm(ImageShape shape, std::size_t groups, float epsilon)
    : shape_(shape), groups_(groups), epsilon_(epsilon) {
    if (shape.size() == 0 || groups == 0 || shape.channels % groups != 0)
        throw std::invalid_argument("GroupNorm: channels (" + std::to_string(shape.channels) + ") must split into " +
                                    std::to_string(groups) + " groups");
    gammaGrad_ = Matrix(1, shape.channels);
    betaGrad_ = Matrix(1, shape.channels);
    if (SkipParameterInit::active()) return;
    gamma_ = Matrix(1, shape.channels, 1.0f);
    beta_ = Matrix(1, shape.channels);
}

Matrix GroupNorm::forward(const Matrix& input, bool training) {
    NC_PROFILE("group norm");
    expectColumns(input, shape_.size(), "GroupNorm");
    const std::size_t B = input.rows(), HW = shape_.height * shape_.width, perGroup = shape_.channels / groups_;
    const std::size_t n = perGroup * HW;  // a group's values are contiguous in (C, H, W) order
    Matrix out(B, input.cols());
    if (training) normalized_ = Matrix(B, input.cols());
    else normalized_ = Matrix();
    invStd_.assign(B * groups_, 0.0f);
    parallelFor(0, B * groups_, 1, [&](std::size_t t0, std::size_t t1) {
        for (std::size_t t = t0; t < t1; ++t) {
            const std::size_t b = t / groups_, g = t % groups_;
            const float* x = input.data() + b * input.cols() + g * n;
            double mean = 0.0, var = 0.0;
            meanAndVariance(x, n, mean, var);
            const float inv = static_cast<float>(1.0 / std::sqrt(var + epsilon_));
            invStd_[t] = inv;
            const float m = static_cast<float>(mean);
            float* y = out.data() + b * out.cols() + g * n;
            float* xn = training ? normalized_.data() + b * out.cols() + g * n : nullptr;
            for (std::size_t c = 0; c < perGroup; ++c) {
                const std::size_t ch = g * perGroup + c;
                const float gm = gamma_(0, ch), bt = beta_(0, ch);
                const float a = inv * gm, bb = bt - m * a;
                if (xn) {
                    for (std::size_t p = c * HW; p < (c + 1) * HW; ++p) {
                        xn[p] = (x[p] - m) * inv;
                        y[p] = xn[p] * gm + bt;
                    }
                } else {
                    for (std::size_t p = c * HW; p < (c + 1) * HW; ++p) y[p] = x[p] * a + bb;
                }
            }
        }
    });
    return out;
}

Matrix GroupNorm::backward(const Matrix& gradOutput) {
    if (normalized_.empty() || !normalized_.sameShape(gradOutput))
        throw std::logic_error("GroupNorm::backward called before a training forward of the same shape");
    const std::size_t B = gradOutput.rows(), C = shape_.channels, HW = shape_.height * shape_.width, perGroup = C / groups_;
    const std::size_t n = perGroup * HW;
    Matrix dx(B, gradOutput.cols());
    gammaGrad_ = Matrix(1, C);
    betaGrad_ = Matrix(1, C);
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t c = 0; c < C; ++c) {
            const float* g = gradOutput.data() + b * gradOutput.cols() + c * HW;
            const float* xn = normalized_.data() + b * gradOutput.cols() + c * HW;
            double sg = 0.0, sgx = 0.0;
            for (std::size_t p = 0; p < HW; ++p) {
                sg += g[p];
                sgx += g[p] * xn[p];
            }
            betaGrad_(0, c) += static_cast<float>(sg);
            gammaGrad_(0, c) += static_cast<float>(sgx);
        }
    parallelFor(0, B * groups_, 1, [&](std::size_t t0, std::size_t t1) {
        for (std::size_t t = t0; t < t1; ++t) {
            const std::size_t b = t / groups_, grp = t % groups_;
            const std::size_t base = b * gradOutput.cols() + grp * n;
            // dxhat = dy * gamma; dx = inv / n * (n dxhat - sum(dxhat) - xhat sum(dxhat xhat))
            double s1 = 0.0, s2 = 0.0;
            for (std::size_t c = 0; c < perGroup; ++c) {
                const float gm = gamma_(0, grp * perGroup + c);
                for (std::size_t p = c * HW; p < (c + 1) * HW; ++p) {
                    const double d = gradOutput.data()[base + p] * gm;
                    s1 += d;
                    s2 += d * normalized_.data()[base + p];
                }
            }
            const double inv = invStd_[t], N = static_cast<double>(n);
            for (std::size_t c = 0; c < perGroup; ++c) {
                const float gm = gamma_(0, grp * perGroup + c);
                for (std::size_t p = c * HW; p < (c + 1) * HW; ++p) {
                    const double d = gradOutput.data()[base + p] * gm;
                    dx.data()[base + p] = static_cast<float>(inv / N * (N * d - s1 - normalized_.data()[base + p] * s2));
                }
            }
        }
    });
    return dx;
}

std::string GroupNorm::describe() const {
    return "GroupNorm(" + shapeString(shape_) + ", " + std::to_string(groups_) + " groups)";
}

// ---- Upsample2D / Pad2D -------------------------------------------------------------------------

Upsample2D::Upsample2D(ImageShape input, std::size_t factor) : in_(input), factor_(factor) {
    if (input.size() == 0 || factor == 0) throw std::invalid_argument("Upsample2D: sizes must be > 0");
}

Matrix Upsample2D::forward(const Matrix& input, bool) {
    expectColumns(input, in_.size(), "Upsample2D");
    const ImageShape o = outputShape();
    Matrix out(input.rows(), o.size());
    parallelFor(0, input.rows() * in_.channels, 1, [&](std::size_t t0, std::size_t t1) {
        for (std::size_t t = t0; t < t1; ++t) {
            const std::size_t b = t / in_.channels, c = t % in_.channels;
            const float* src = input.data() + b * input.cols() + c * in_.height * in_.width;
            float* dst = out.data() + b * out.cols() + c * o.height * o.width;
            for (std::size_t y = 0; y < in_.height; ++y) {
                const float* s = src + y * in_.width;
                float* d = dst + y * factor_ * o.width;
                for (std::size_t x = 0; x < in_.width; ++x)
                    for (std::size_t f = 0; f < factor_; ++f) d[x * factor_ + f] = s[x];
                for (std::size_t f = 1; f < factor_; ++f) std::copy(d, d + o.width, d + f * o.width);  // repeated rows
            }
        }
    });
    return out;
}

Matrix Upsample2D::backward(const Matrix& gradOutput) {
    const ImageShape o = outputShape();
    expectColumns(gradOutput, o.size(), "Upsample2D::backward");
    Matrix dx(gradOutput.rows(), in_.size());
    for (std::size_t b = 0; b < gradOutput.rows(); ++b)
        for (std::size_t c = 0; c < in_.channels; ++c)
            for (std::size_t y = 0; y < o.height; ++y)
                for (std::size_t x = 0; x < o.width; ++x)
                    dx(b, (c * in_.height + y / factor_) * in_.width + x / factor_) += gradOutput(b, (c * o.height + y) * o.width + x);
    return dx;
}

std::string Upsample2D::describe() const {
    return "Upsample2D(" + shapeString(in_) + " -> " + shapeString(outputShape()) + ")";
}

Pad2D::Pad2D(ImageShape input, std::size_t top, std::size_t bottom, std::size_t left, std::size_t right)
    : in_(input), top_(top), bottom_(bottom), left_(left), right_(right) {
    if (input.size() == 0) throw std::invalid_argument("Pad2D: empty input shape");
}

Matrix Pad2D::forward(const Matrix& input, bool) {
    expectColumns(input, in_.size(), "Pad2D");
    const ImageShape o = outputShape();
    Matrix out(input.rows(), o.size());
    for (std::size_t b = 0; b < input.rows(); ++b)
        for (std::size_t c = 0; c < in_.channels; ++c)
            for (std::size_t y = 0; y < in_.height; ++y) {
                const float* s = input.data() + b * input.cols() + (c * in_.height + y) * in_.width;
                std::copy(s, s + in_.width, out.data() + b * out.cols() + (c * o.height + y + top_) * o.width + left_);
            }
    return out;
}

Matrix Pad2D::backward(const Matrix& gradOutput) {
    const ImageShape o = outputShape();
    expectColumns(gradOutput, o.size(), "Pad2D::backward");
    Matrix dx(gradOutput.rows(), in_.size());
    for (std::size_t b = 0; b < gradOutput.rows(); ++b)
        for (std::size_t c = 0; c < in_.channels; ++c)
            for (std::size_t y = 0; y < in_.height; ++y) {
                const float* s = gradOutput.data() + b * gradOutput.cols() + (c * o.height + y + top_) * o.width + left_;
                std::copy(s, s + in_.width, dx.data() + b * dx.cols() + (c * in_.height + y) * in_.width);
            }
    return dx;
}

std::string Pad2D::describe() const {
    return "Pad2D(" + shapeString(in_) + " -> " + shapeString(outputShape()) + ")";
}

// ---- ResnetBlock2D --------------------------------------------------------------------------

ResnetBlock2D::ResnetBlock2D(ImageShape input, std::size_t outChannels, std::size_t groups, float epsilon)
    : in_(input), outChannels_(outChannels) {
    const ImageShape mid{outChannels, input.height, input.width};
    main_.add<GroupNorm>(input, groups, epsilon);
    main_.add<SiLU>();
    main_.add<Conv2D>(input, outChannels, 3, 1, 1);
    main_.add<GroupNorm>(mid, groups, epsilon);
    main_.add<SiLU>();
    main_.add<Conv2D>(mid, outChannels, 3, 1, 1);
    if (input.channels != outChannels) shortcut_.add<Conv2D>(input, outChannels, 1);
}

Matrix ResnetBlock2D::forward(const Matrix& input, bool training) {
    Matrix h = main_.forward(input, training);
    const Matrix projected = shortcut_.layerCount() ? shortcut_.forward(input, training) : Matrix();
    const float* s = shortcut_.layerCount() ? projected.data() : input.data();
    float* y = h.data();
    parallelFor(0, h.size(), 1 << 15, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) y[i] += s[i];
    });
    return h;
}

Matrix ResnetBlock2D::backward(const Matrix& gradOutput) {
    Matrix dx = main_.backward(gradOutput);
    if (shortcut_.layerCount() == 0) return dx + gradOutput;
    return dx + shortcut_.backward(gradOutput);
}

std::vector<Param> ResnetBlock2D::parameters() {
    auto p = main_.parameters();
    for (const auto& q : shortcut_.parameters()) p.push_back(q);
    return p;
}

std::string ResnetBlock2D::describe() const {
    return "ResnetBlock2D(" + shapeString(in_) + " -> " + std::to_string(outChannels_) + " channels)";
}

// ---- SpatialSelfAttention ---------------------------------------------------------------------

SpatialSelfAttention::SpatialSelfAttention(ImageShape shape, std::size_t heads, std::size_t groups, float epsilon)
    : shape_(shape), heads_(heads), norm_(shape, groups, epsilon) {
    if (heads == 0 || shape.channels % heads != 0)
        throw std::invalid_argument("SpatialSelfAttention: channels must split evenly into heads");
    if (SkipParameterInit::active()) return;
    const std::size_t C = shape.channels;
    const float limit = std::sqrt(3.0f / static_cast<float>(C));
    wq_ = Matrix::uniform(C, C, -limit, limit);
    wk_ = Matrix::uniform(C, C, -limit, limit);
    wv_ = Matrix::uniform(C, C, -limit, limit);
    wo_ = Matrix::uniform(C, C, -limit, limit);
    bq_ = Matrix(1, C);
    bk_ = Matrix(1, C);
    bv_ = Matrix(1, C);
    bo_ = Matrix(1, C);
}

Matrix SpatialSelfAttention::forward(const Matrix& input, bool) {
    expectColumns(input, shape_.size(), "SpatialSelfAttention");
    const std::size_t C = shape_.channels, HW = shape_.height * shape_.width, dh = C / heads_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    const Matrix normed = norm_.forward(input, false);
    Matrix out(input.rows(), input.cols());
    std::vector<float> q(C * HW), k(C * HW), v(C * HW), o(C * HW);
    // Scores for a chunk of queries at a time: at most ~4M floats.
    const std::size_t chunk = std::max<std::size_t>(1, std::min(HW, (std::size_t(1) << 22) / HW));
    std::vector<float> scores(chunk * HW);
    auto project = [&](const Matrix& w, const Matrix& bias, const float* x, float* y) {
        gemm(false, false, C, HW, C, w.data(), C, x, HW, y, HW, false);  // (C x C) x (C x HW)
        parallelFor(0, C, 8, [&](std::size_t c0, std::size_t c1) {
            for (std::size_t c = c0; c < c1; ++c)
                for (std::size_t p = 0; p < HW; ++p) y[c * HW + p] += bias(0, c);
        });
    };
    for (std::size_t b = 0; b < input.rows(); ++b) {
        const float* x = normed.data() + b * normed.cols();
        project(wq_, bq_, x, q.data());
        project(wk_, bk_, x, k.data());
        project(wv_, bv_, x, v.data());
        for (std::size_t h = 0; h < heads_; ++h) {
            const float* qh = q.data() + h * dh * HW;
            const float* kh = k.data() + h * dh * HW;
            const float* vh = v.data() + h * dh * HW;
            for (std::size_t q0 = 0; q0 < HW; q0 += chunk) {
                const std::size_t rows = std::min(chunk, HW - q0);
                // scores (rows x HW) = Q[:, q0..]^T K
                gemm(true, false, rows, HW, dh, qh + q0, HW, kh, HW, scores.data(), HW, false);
                parallelFor(0, rows, 4, [&](std::size_t r0, std::size_t r1) {
                    for (std::size_t r = r0; r < r1; ++r) {
                        float* s = scores.data() + r * HW;
                        float m = -std::numeric_limits<float>::infinity();
                        for (std::size_t j = 0; j < HW; ++j) m = std::max(m, s[j] *= scale);
                        float sum = 0.0f;
                        for (std::size_t j = 0; j < HW; ++j) sum += s[j] = std::exp(s[j] - m);
                        const float inv = 1.0f / sum;
                        for (std::size_t j = 0; j < HW; ++j) s[j] *= inv;
                    }
                });
                // O[:, q0..] = V scores^T
                gemm(false, true, dh, rows, HW, vh, HW, scores.data(), HW, o.data() + h * dh * HW + q0, HW, false);
            }
        }
        float* y = out.data() + b * out.cols();
        project(wo_, bo_, o.data(), y);
        const float* residual = input.data() + b * input.cols();
        for (std::size_t i = 0; i < C * HW; ++i) y[i] += residual[i];
    }
    return out;
}

Matrix SpatialSelfAttention::backward(const Matrix&) {
    throw std::logic_error("SpatialSelfAttention: inference only (backward is not implemented)");
}

std::vector<Param> SpatialSelfAttention::parameters() {
    auto p = norm_.parameters();
    Matrix* values[8] = {&wq_, &wk_, &wv_, &wo_, &bq_, &bk_, &bv_, &bo_};
    for (int i = 0; i < 8; ++i) p.push_back({values[i], &grads_[i]});
    return p;
}

std::string SpatialSelfAttention::describe() const {
    return "SpatialSelfAttention(" + shapeString(shape_) + ", " + std::to_string(heads_) + " head" + (heads_ == 1 ? "" : "s") + ")";
}

// ---- Image files ------------------------------------------------------------------------------

namespace {
std::uint32_t crc32(const std::uint8_t* data, std::size_t n, std::uint32_t crc = 0) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void putBE32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) out.push_back(static_cast<std::uint8_t>(v >> s));
}

void writeChunk(std::ofstream& f, const char* type, const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> chunk;
    putBE32(chunk, static_cast<std::uint32_t>(data.size()));
    chunk.insert(chunk.end(), type, type + 4);
    chunk.insert(chunk.end(), data.begin(), data.end());
    putBE32(chunk, crc32(chunk.data() + 4, chunk.size() - 4));
    f.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
}

void checkImage(const Image& image, const char* who) {
    if (image.width == 0 || image.height == 0 || image.pixels.size() != image.width * image.height * image.channels)
        throw std::invalid_argument(std::string(who) + ": image size and pixel count disagree");
}
} // namespace

namespace {
// ---- Deflate compressor (LZ77 + dynamic Huffman blocks) -----------------------------------------

struct BitWriter {
    std::vector<std::uint8_t>& out;
    std::uint32_t acc = 0;
    int n = 0;
    void put(std::uint32_t bits, int count) {  // LSB first
        acc |= bits << n;
        n += count;
        while (n >= 8) {
            out.push_back(static_cast<std::uint8_t>(acc & 0xFF));
            acc >>= 8;
            n -= 8;
        }
    }
    void flush() {
        if (n > 0) out.push_back(static_cast<std::uint8_t>(acc & 0xFF));
        acc = 0;
        n = 0;
    }
};

// Code lengths (at most maxLen bits) for the given frequencies by package-merge;
// unused symbols get 0. At least two symbols get a code, as decoders expect.
std::vector<int> limitedLengths(std::vector<std::uint32_t> freq, int maxLen) {
    std::vector<int> used;
    for (std::size_t i = 0; i < freq.size(); ++i)
        if (freq[i]) used.push_back(static_cast<int>(i));
    for (std::size_t i = 0; used.size() < 2 && i < freq.size(); ++i)
        if (!freq[i]) {
            freq[i] = 1;
            used.push_back(static_cast<int>(i));
        }
    struct Item {
        std::uint64_t weight;
        std::vector<int> symbols;
    };
    std::vector<Item> leaves;
    for (int s : used) leaves.push_back({freq[static_cast<std::size_t>(s)], {s}});
    std::sort(leaves.begin(), leaves.end(), [](const Item& a, const Item& b) { return a.weight < b.weight; });
    std::vector<Item> current = leaves;
    for (int level = 1; level < maxLen; ++level) {
        std::vector<Item> packages;
        for (std::size_t i = 0; i + 1 < current.size(); i += 2) {
            Item p{current[i].weight + current[i + 1].weight, current[i].symbols};
            p.symbols.insert(p.symbols.end(), current[i + 1].symbols.begin(), current[i + 1].symbols.end());
            packages.push_back(std::move(p));
        }
        std::vector<Item> merged;
        std::merge(leaves.begin(), leaves.end(), packages.begin(), packages.end(), std::back_inserter(merged),
                   [](const Item& a, const Item& b) { return a.weight < b.weight; });
        current = std::move(merged);
    }
    std::vector<int> len(freq.size(), 0);
    for (std::size_t i = 0; i < 2 * used.size() - 2; ++i)
        for (int s : current[i].symbols) ++len[static_cast<std::size_t>(s)];
    return len;
}

// Canonical codes, bit-reversed for LSB-first output.
std::vector<std::uint32_t> canonicalCodes(const std::vector<int>& len) {
    int maxLen = 0;
    for (int l : len) maxLen = std::max(maxLen, l);
    std::vector<int> count(static_cast<std::size_t>(maxLen) + 1, 0);
    for (int l : len)
        if (l) ++count[static_cast<std::size_t>(l)];
    std::vector<std::uint32_t> next(static_cast<std::size_t>(maxLen) + 2, 0);
    std::uint32_t code = 0;
    for (int b = 1; b <= maxLen; ++b) {
        code = (code + static_cast<std::uint32_t>(count[static_cast<std::size_t>(b) - 1])) << 1;
        next[static_cast<std::size_t>(b)] = code;
    }
    std::vector<std::uint32_t> codes(len.size(), 0);
    for (std::size_t s = 0; s < len.size(); ++s) {
        const int l = len[s];
        if (!l) continue;
        std::uint32_t c = next[static_cast<std::size_t>(l)]++, r = 0;
        for (int i = 0; i < l; ++i) r |= ((c >> i) & 1u) << (l - 1 - i);
        codes[s] = r;
    }
    return codes;
}

const std::uint16_t kLenBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const std::uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const std::uint16_t kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
const std::uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

struct Token {
    std::uint16_t lenOrLiteral;  // < 256 literal, else match length (3..258) + 256
    std::uint16_t distance;
};

int lengthCode(int len) {
    int c = 0;
    while (c < 28 && kLenBase[c + 1] <= len) ++c;
    return c;
}
int distanceCode(int d) {
    int c = 0;
    while (c < 29 && kDistBase[c + 1] <= d) ++c;
    return c;
}

void writeBlock(BitWriter& bw, const std::vector<Token>& tokens, bool last) {
    std::vector<std::uint32_t> lf(286, 0), df(30, 0);
    for (const Token& t : tokens) {
        if (t.lenOrLiteral < 256) ++lf[t.lenOrLiteral];
        else {
            ++lf[257 + static_cast<std::size_t>(lengthCode(t.lenOrLiteral - 256))];
            ++df[static_cast<std::size_t>(distanceCode(t.distance))];
        }
    }
    lf[256] = 1;
    const std::vector<int> ll = limitedLengths(lf, 15), dl = limitedLengths(df, 15);
    std::size_t hlit = 286, hdist = 30;
    while (hlit > 257 && ll[hlit - 1] == 0) --hlit;
    while (hdist > 1 && dl[hdist - 1] == 0) --hdist;
    std::vector<int> all(ll.begin(), ll.begin() + static_cast<long>(hlit));
    all.insert(all.end(), dl.begin(), dl.begin() + static_cast<long>(hdist));
    // Run-length encode the code lengths (16: repeat previous, 17 / 18: zeros).
    std::vector<std::pair<int, int>> rle;  // (symbol, extra value)
    for (std::size_t i = 0; i < all.size();) {
        std::size_t run = 1;
        while (i + run < all.size() && all[i + run] == all[i]) ++run;
        if (all[i] == 0 && run >= 3) {
            const std::size_t r = std::min<std::size_t>(run, 138);
            rle.push_back(r >= 11 ? std::make_pair(18, static_cast<int>(r - 11)) : std::make_pair(17, static_cast<int>(r - 3)));
            i += r;
        } else if (all[i] != 0 && run >= 4) {
            rle.push_back({all[i], 0});
            const std::size_t r = std::min<std::size_t>(run - 1, 6);
            rle.push_back({16, static_cast<int>(r - 3)});
            i += 1 + r;
        } else {
            rle.push_back({all[i], 0});
            ++i;
        }
    }
    std::vector<std::uint32_t> cf(19, 0);
    for (const auto& r : rle) ++cf[static_cast<std::size_t>(r.first)];
    const std::vector<int> cl = limitedLengths(cf, 7);
    static const int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    std::size_t hclen = 19;
    while (hclen > 4 && cl[static_cast<std::size_t>(order[hclen - 1])] == 0) --hclen;
    const std::vector<std::uint32_t> lc = canonicalCodes(ll), dc = canonicalCodes(dl), cc = canonicalCodes(cl);
    bw.put(last ? 1 : 0, 1);
    bw.put(2, 2);
    bw.put(static_cast<std::uint32_t>(hlit - 257), 5);
    bw.put(static_cast<std::uint32_t>(hdist - 1), 5);
    bw.put(static_cast<std::uint32_t>(hclen - 4), 4);
    for (std::size_t i = 0; i < hclen; ++i) bw.put(static_cast<std::uint32_t>(cl[static_cast<std::size_t>(order[i])]), 3);
    for (const auto& r : rle) {
        bw.put(cc[static_cast<std::size_t>(r.first)], cl[static_cast<std::size_t>(r.first)]);
        if (r.first == 16) bw.put(static_cast<std::uint32_t>(r.second), 2);
        else if (r.first == 17) bw.put(static_cast<std::uint32_t>(r.second), 3);
        else if (r.first == 18) bw.put(static_cast<std::uint32_t>(r.second), 7);
    }
    for (const Token& t : tokens) {
        if (t.lenOrLiteral < 256) {
            bw.put(lc[t.lenOrLiteral], ll[t.lenOrLiteral]);
            continue;
        }
        const int len = t.lenOrLiteral - 256, c = lengthCode(len), d = distanceCode(t.distance);
        bw.put(lc[257 + static_cast<std::size_t>(c)], ll[257 + static_cast<std::size_t>(c)]);
        if (kLenExtra[c]) bw.put(static_cast<std::uint32_t>(len - kLenBase[c]), kLenExtra[c]);
        bw.put(dc[static_cast<std::size_t>(d)], dl[static_cast<std::size_t>(d)]);
        if (kDistExtra[d]) bw.put(static_cast<std::uint32_t>(t.distance - kDistBase[d]), kDistExtra[d]);
    }
    bw.put(lc[256], ll[256]);
}

// zlib stream: LZ77 with hash chains and one-step lazy matching, dynamic Huffman blocks.
std::vector<std::uint8_t> zlibCompress(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out{0x78, 0x9C};
    BitWriter bw{out};
    constexpr std::size_t kWindow = 32768, kMaxChain = 48, kNice = 128, kBlock = 1 << 16;
    std::vector<std::int32_t> head(1 << 15, -1), prev(kWindow, -1);
    const std::size_t n = data.size();
    auto hash = [&](std::size_t i) {
        return ((static_cast<std::uint32_t>(data[i]) << 10) ^ (static_cast<std::uint32_t>(data[i + 1]) << 5) ^ data[i + 2]) & 0x7FFF;
    };
    auto insert = [&](std::size_t i) {
        if (i + 2 >= n) return;
        const std::uint32_t h = hash(i);
        prev[i % kWindow] = head[h];
        head[h] = static_cast<std::int32_t>(i);
    };
    auto longest = [&](std::size_t i, std::size_t& dist) {
        std::size_t best = 0;
        if (i + 2 >= n) return best;
        std::int32_t cand = head[hash(i)];
        const std::size_t maxLen = std::min<std::size_t>(258, n - i);
        for (std::size_t chain = 0; cand >= 0 && chain < kMaxChain; ++chain) {
            const std::size_t c = static_cast<std::size_t>(cand);
            if (i - c > kWindow - 1 || c >= i) break;
            if (data[c + best] == data[i + best]) {
                std::size_t l = 0;
                while (l < maxLen && data[c + l] == data[i + l]) ++l;
                if (l > best) {
                    best = l;
                    dist = i - c;
                    if (l >= kNice || l >= maxLen) break;  // nothing longer is possible (and data[i + best] would be past the end)
                }
            }
            const std::int32_t p = prev[c % kWindow];
            if (p >= cand) break;  // stale entry from an older window
            cand = p;
        }
        return best >= 3 ? best : 0;
    };
    std::vector<Token> tokens;
    std::size_t i = 0;
    while (i < n) {
        std::size_t dist = 0, len = longest(i, dist);
        if (len && len < kNice && i + 1 < n) {  // lazy: a longer match one byte later wins
            std::size_t d2 = 0;
            insert(i);
            const std::size_t l2 = longest(i + 1, d2);
            if (l2 > len) {
                tokens.push_back({data[i], 0});
                ++i;
                len = l2;
                dist = d2;
            } else {
                for (std::size_t k = 1; k < len; ++k) insert(i + k);
                tokens.push_back({static_cast<std::uint16_t>(256 + len), static_cast<std::uint16_t>(dist)});
                i += len;
                len = 0;
            }
            if (len) {
                for (std::size_t k = 0; k < len; ++k) insert(i + k);
                tokens.push_back({static_cast<std::uint16_t>(256 + len), static_cast<std::uint16_t>(dist)});
                i += len;
            }
        } else if (len) {
            for (std::size_t k = 0; k < len; ++k) insert(i + k);
            tokens.push_back({static_cast<std::uint16_t>(256 + len), static_cast<std::uint16_t>(dist)});
            i += len;
        } else {
            insert(i);
            tokens.push_back({data[i], 0});
            ++i;
        }
        if (tokens.size() >= kBlock) {
            writeBlock(bw, tokens, false);
            tokens.clear();
        }
    }
    writeBlock(bw, tokens, true);
    bw.flush();
    if (out.size() > n + n / 65535 * 5 + 7) {  // incompressible: stored blocks are smaller
        out.assign({0x78, 0x01});
        for (std::size_t pos = 0;;) {
            const std::size_t len = std::min<std::size_t>(65535, n - pos);
            const bool last = pos + len == n;
            out.insert(out.end(), {static_cast<std::uint8_t>(last ? 1 : 0), static_cast<std::uint8_t>(len & 0xFF),
                                   static_cast<std::uint8_t>(len >> 8), static_cast<std::uint8_t>(~len & 0xFF),
                                   static_cast<std::uint8_t>((~len >> 8) & 0xFF)});
            out.insert(out.end(), data.begin() + static_cast<long>(pos), data.begin() + static_cast<long>(pos + len));
            pos += len;
            if (last) break;
        }
    }
    std::uint32_t a = 1, b = 0;  // Adler-32
    for (std::size_t k = 0; k < n;) {
        const std::size_t end = std::min(n, k + 5552);
        for (; k < end; ++k) {
            a += data[k];
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    putBE32(out, (b << 16) | a);
    return out;
}

// Scanlines with a per-row filter (the one with the smallest sum of absolute
// signed bytes, the usual heuristic), compressed.
std::vector<std::uint8_t> pngImageData(const Image& image) {
    const std::size_t C = image.channels, stride = image.width * C;
    std::vector<std::uint8_t> raw;
    raw.reserve((stride + 1) * image.height);
    std::vector<std::uint8_t> zero(stride, 0), best(stride), trial(stride);
    for (std::size_t y = 0; y < image.height; ++y) {
        const std::uint8_t* row = image.pixels.data() + y * stride;
        const std::uint8_t* up = y ? row - stride : zero.data();
        std::uint64_t bestScore = ~std::uint64_t{0};
        std::uint8_t bestType = 0;
        for (std::uint8_t type = 0; type < 5; ++type) {
            std::uint64_t score = 0;
            for (std::size_t x = 0; x < stride; ++x) {
                const int a = x >= C ? row[x - C] : 0, b = up[x], c = x >= C ? up[x - C] : 0;
                int pred = 0;
                switch (type) {
                case 1: pred = a; break;
                case 2: pred = b; break;
                case 3: pred = (a + b) / 2; break;
                case 4: {
                    const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                    pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
                    break;
                }
                default: break;
                }
                trial[x] = static_cast<std::uint8_t>(row[x] - pred);
                score += static_cast<std::uint64_t>(std::abs(static_cast<int>(static_cast<std::int8_t>(trial[x]))));
            }
            if (score < bestScore) {
                bestScore = score;
                bestType = type;
                best.swap(trial);
            }
        }
        raw.push_back(bestType);
        raw.insert(raw.end(), best.begin(), best.end());
    }
    return zlibCompress(raw);
}

std::uint8_t pngColorType(const Image& image, const char* who) {
    switch (image.channels) {
    case 1: return 0;
    case 3: return 2;
    case 4: return 6;
    default: throw std::invalid_argument(std::string(who) + ": 1, 3 or 4 channels");
    }
}

void writePngHeader(std::ofstream& f, const Image& image, std::uint8_t colorType) {
    const std::uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    f.write(reinterpret_cast<const char*>(signature), 8);
    std::vector<std::uint8_t> ihdr;
    putBE32(ihdr, static_cast<std::uint32_t>(image.width));
    putBE32(ihdr, static_cast<std::uint32_t>(image.height));
    ihdr.insert(ihdr.end(), {8, colorType, 0, 0, 0});
    writeChunk(f, "IHDR", ihdr);
}
} // namespace

std::vector<std::uint8_t> deflateCompress(const std::vector<std::uint8_t>& data) { return zlibCompress(data); }

void savePng(const std::string& path, const Image& image) {
    checkImage(image, "savePng");
    const std::uint8_t colorType = pngColorType(image, "savePng");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("savePng: cannot open " + path);
    writePngHeader(f, image, colorType);
    writeChunk(f, "IDAT", pngImageData(image));
    writeChunk(f, "IEND", {});
    if (!f) throw std::runtime_error("savePng: write failed for " + path);
}

void saveApng(const std::string& path, const std::vector<Image>& frames, float framesPerSecond, bool loop) {
    if (frames.empty()) throw std::invalid_argument("saveApng: no frames");
    if (framesPerSecond <= 0.0f) throw std::invalid_argument("saveApng: frames per second must be > 0");
    const Image& first = frames.front();
    checkImage(first, "saveApng");
    const std::uint8_t colorType = pngColorType(first, "saveApng");
    for (const Image& im : frames)
        if (im.width != first.width || im.height != first.height || im.channels != first.channels)
            throw std::invalid_argument("saveApng: frames must have one size and channel count");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("saveApng: cannot open " + path);
    writePngHeader(f, first, colorType);
    std::vector<std::uint8_t> actl;
    putBE32(actl, static_cast<std::uint32_t>(frames.size()));
    putBE32(actl, loop ? 0 : 1);
    writeChunk(f, "acTL", actl);
    // Delay as a fraction: round(1000 / fps) / 1000 s.
    const std::uint16_t delayNum = static_cast<std::uint16_t>(std::max(1L, std::lround(1000.0 / framesPerSecond))), delayDen = 1000;
    std::uint32_t sequence = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        std::vector<std::uint8_t> fctl;
        putBE32(fctl, sequence++);
        putBE32(fctl, static_cast<std::uint32_t>(first.width));
        putBE32(fctl, static_cast<std::uint32_t>(first.height));
        putBE32(fctl, 0);
        putBE32(fctl, 0);
        fctl.insert(fctl.end(), {static_cast<std::uint8_t>(delayNum >> 8), static_cast<std::uint8_t>(delayNum & 0xFF),
                                 static_cast<std::uint8_t>(delayDen >> 8), static_cast<std::uint8_t>(delayDen & 0xFF), 0, 0});
        writeChunk(f, "fcTL", fctl);
        const std::vector<std::uint8_t> z = pngImageData(frames[i]);
        if (i == 0) writeChunk(f, "IDAT", z);  // the first frame is also the still image
        else {
            std::vector<std::uint8_t> fdat;
            putBE32(fdat, sequence++);
            fdat.insert(fdat.end(), z.begin(), z.end());
            writeChunk(f, "fdAT", fdat);
        }
    }
    writeChunk(f, "IEND", {});
    if (!f) throw std::runtime_error("saveApng: write failed for " + path);
}

void savePpm(const std::string& path, const Image& image) {
    checkImage(image, "savePpm");
    if (image.channels != 1 && image.channels != 3) throw std::invalid_argument("savePpm: 1 or 3 channels");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("savePpm: cannot open " + path);
    f << (image.channels == 3 ? "P6" : "P5") << '\n' << image.width << ' ' << image.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (!f) throw std::runtime_error("savePpm: write failed for " + path);
}

namespace {
thread_local std::size_t decodeLimit = std::size_t(1) << 28;
void checkDecodeSize(std::size_t w, std::size_t h, const std::string& path) {
    if (w == 0 || h == 0) return;
    if (w > decodeLimit / h) throw std::runtime_error(path + ": " + std::to_string(w) + " x " + std::to_string(h) + " is too big to decode");
}
} // namespace
void setMaxDecodePixels(std::size_t pixels) { decodeLimit = pixels ? pixels : std::size_t(1) << 28; }
std::size_t maxDecodePixels() { return decodeLimit; }

Image loadPpm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadPpm: cannot open " + path);
    auto token = [&]() {
        std::string t;
        char c;
        while (f.get(c)) {
            if (c == '#') {
                std::string skip;
                std::getline(f, skip);
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                if (!t.empty()) break;
            } else {
                t += c;
            }
        }
        return t;
    };
    const std::string magic = token();
    if (magic != "P6" && magic != "P5") throw std::runtime_error("loadPpm: " + path + " is not a binary PPM/PGM");
    Image image;
    image.channels = magic == "P6" ? 3 : 1;
    try {
        image.width = std::stoul(token());
        image.height = std::stoul(token());
        if (std::stoul(token()) != 255) throw std::runtime_error("only 8-bit");
    } catch (const std::exception&) {
        throw std::runtime_error("loadPpm: bad header in " + path);
    }
    checkDecodeSize(image.width, image.height, path);
    image.pixels.resize(image.width * image.height * image.channels);
    f.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (!f) throw std::runtime_error("loadPpm: truncated " + path);
    return image;
}

// ---- PNG reading ------------------------------------------------------------------------------

namespace {
// Minimal inflate (RFC 1951): stored, fixed and dynamic Huffman blocks.
class Inflater {
public:
    // maxOut: the most bytes the stream may expand to (a "zip bomb" is refused).
    Inflater(const std::uint8_t* data, std::size_t size, std::size_t maxOut = std::numeric_limits<std::size_t>::max()) : d_(data), n_(size), max_(maxOut) {}

    std::vector<std::uint8_t> run() {
        std::vector<std::uint8_t> out;
        for (bool last = false; !last;) {
            last = bits(1);
            const unsigned type = bits(2);
            if (type == 0) {
                bitPos_ = (bitPos_ + 7) & ~std::size_t(7);
                const unsigned len = bits(16), nlen = bits(16);
                if ((len ^ 0xFFFF) != nlen) throw std::runtime_error("loadPng: corrupt stored block");
                if (out.size() + len > max_) throw std::runtime_error("loadPng: image data longer than the image");
                for (unsigned i = 0; i < len; ++i) out.push_back(static_cast<std::uint8_t>(bits(8)));
            } else if (type == 1) {
                std::array<unsigned, 288> lit{};
                for (int i = 0; i < 144; ++i) lit[i] = 8;
                for (int i = 144; i < 256; ++i) lit[i] = 9;
                for (int i = 256; i < 280; ++i) lit[i] = 7;
                for (int i = 280; i < 288; ++i) lit[i] = 8;
                std::array<unsigned, 30> dist;
                dist.fill(5);
                block(Huffman(lit.data(), 288), Huffman(dist.data(), 30), out);
            } else if (type == 2) {
                const unsigned hlit = bits(5) + 257, hdist = bits(5) + 1, hclen = bits(4) + 4;
                static const int order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
                unsigned clen[19] = {};
                for (unsigned i = 0; i < hclen; ++i) clen[order[i]] = bits(3);
                const Huffman codeLengths(clen, 19);
                std::vector<unsigned> lengths;
                while (lengths.size() < hlit + hdist) {
                    const unsigned sym = decode(codeLengths);
                    if (sym < 16) lengths.push_back(sym);
                    else if (sym == 16) {
                        if (lengths.empty()) throw std::runtime_error("loadPng: corrupt code lengths");
                        const unsigned prev = lengths.back();
                        for (unsigned r = 3 + bits(2); r > 0; --r) lengths.push_back(prev);
                    } else {
                        for (unsigned r = sym == 17 ? 3 + bits(3) : 11 + bits(7); r > 0; --r) lengths.push_back(0);
                    }
                }
                if (lengths.size() != hlit + hdist) throw std::runtime_error("loadPng: corrupt code lengths");
                block(Huffman(lengths.data(), hlit), Huffman(lengths.data() + hlit, hdist), out);
            } else {
                throw std::runtime_error("loadPng: invalid deflate block");
            }
        }
        return out;
    }

private:
    struct Huffman {
        std::vector<unsigned> counts, symbols;  // canonical code: counts per length, symbols by code
        Huffman(const unsigned* lengths, std::size_t n) : counts(16, 0) {
            for (std::size_t i = 0; i < n; ++i) ++counts[lengths[i]];
            counts[0] = 0;
            std::vector<unsigned> offsets(16, 0);
            for (int l = 1; l < 16; ++l) offsets[l] = offsets[l - 1] + counts[l - 1];
            symbols.resize(n);
            for (std::size_t i = 0; i < n; ++i)
                if (lengths[i]) symbols[offsets[lengths[i]]++] = static_cast<unsigned>(i);
        }
    };

    unsigned bits(unsigned count) {
        unsigned v = 0;
        for (unsigned i = 0; i < count; ++i, ++bitPos_) {
            if ((bitPos_ >> 3) >= n_) throw std::runtime_error("loadPng: truncated image data");
            v |= static_cast<unsigned>((d_[bitPos_ >> 3] >> (bitPos_ & 7)) & 1) << i;
        }
        return v;
    }
    unsigned decode(const Huffman& h) {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len < 16; ++len) {
            code |= static_cast<int>(bits(1));
            const int count = static_cast<int>(h.counts[static_cast<std::size_t>(len)]);
            if (code - count < first) return h.symbols[static_cast<std::size_t>(index + (code - first))];
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        throw std::runtime_error("loadPng: invalid Huffman code");
    }
    void block(const Huffman& lit, const Huffman& dist, std::vector<std::uint8_t>& out) {
        static const unsigned lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
        static const unsigned lextra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
        static const unsigned dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
        static const unsigned dextra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
        for (;;) {
            const unsigned sym = decode(lit);
            if (out.size() >= max_) throw std::runtime_error("loadPng: image data longer than the image");
            if (sym < 256) {
                out.push_back(static_cast<std::uint8_t>(sym));
            } else if (sym == 256) {
                return;
            } else {
                const unsigned li = sym - 257;
                if (li >= 29) throw std::runtime_error("loadPng: invalid length code");
                const unsigned len = lbase[li] + bits(lextra[li]);
                const unsigned di = decode(dist);
                if (di >= 30) throw std::runtime_error("loadPng: invalid distance code");
                const std::size_t back = dbase[di] + bits(dextra[di]);
                if (back > out.size()) throw std::runtime_error("loadPng: distance beyond the output");
                if (out.size() + len > max_) throw std::runtime_error("loadPng: image data longer than the image");
                const std::size_t from = out.size() - back;
                for (unsigned i = 0; i < len; ++i) out.push_back(out[from + i]);
            }
        }
    }

    const std::uint8_t* d_;
    std::size_t n_, max_, bitPos_ = 0;
};

std::uint32_t be32(const std::vector<std::uint8_t>& d, std::size_t at) {
    return (std::uint32_t(d[at]) << 24) | (std::uint32_t(d[at + 1]) << 16) | (std::uint32_t(d[at + 2]) << 8) | d[at + 3];
}
} // namespace

Image loadPng(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadPng: cannot open " + path);
    const std::vector<std::uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    static const std::uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    if (d.size() < 8 || !std::equal(sig, sig + 8, d.begin())) throw std::runtime_error("loadPng: " + path + " is not a PNG");
    std::size_t width = 0, height = 0;
    unsigned depth = 0, colorType = 0, interlace = 0;
    std::vector<std::uint8_t> idat, palette, trns;
    for (std::size_t at = 8; at + 12 <= d.size();) {
        const std::uint32_t len = be32(d, at);
        if (at + 12 + len > d.size()) throw std::runtime_error("loadPng: truncated chunk in " + path);
        const std::string type(d.begin() + static_cast<long>(at + 4), d.begin() + static_cast<long>(at + 8));
        const std::uint8_t* body = d.data() + at + 8;
        if (type == "IHDR") {
            if (len < 13) throw std::runtime_error("loadPng: short IHDR in " + path);
            width = be32(d, at + 8);
            height = be32(d, at + 12);
            depth = body[8];
            colorType = body[9];
            interlace = body[12];
        } else if (type == "PLTE") {
            palette.assign(body, body + len);
        } else if (type == "tRNS") {
            trns.assign(body, body + len);
        } else if (type == "IDAT") {
            idat.insert(idat.end(), body, body + len);
        } else if (type == "IEND") {
            break;
        }
        at += 12 + len;
    }
    if (width == 0 || height == 0) throw std::runtime_error("loadPng: no IHDR in " + path);
    checkDecodeSize(width, height, path);
    if (interlace) throw std::runtime_error("loadPng: interlaced PNGs are not supported");
    const unsigned samples = colorType == 0 ? 1 : colorType == 2 ? 3 : colorType == 3 ? 1 : colorType == 4 ? 2 : colorType == 6 ? 4 : 0;
    if (!samples || (depth != 8 && depth != 16 && !((colorType == 0 || colorType == 3) && depth < 8)))
        throw std::runtime_error("loadPng: unsupported colour type / bit depth in " + path);
    if (idat.size() < 2 || (idat[0] & 0x0F) != 8) throw std::runtime_error("loadPng: bad zlib stream in " + path);
    const std::size_t bitsPerPixel = samples * depth, stride = (width * bitsPerPixel + 7) / 8, bpp = std::max<std::size_t>(1, bitsPerPixel / 8);
    // (a little slack past the pixel rows for encoders that pad)
    const std::vector<std::uint8_t> raw = Inflater(idat.data() + 2, idat.size() - 2, height * (stride + 1) + 65536).run();
    if (raw.size() < height * (stride + 1)) throw std::runtime_error("loadPng: image data too short in " + path);
    // Undo the per-scanline filters.
    std::vector<std::uint8_t> px(height * stride), prev(stride, 0);
    for (std::size_t y = 0; y < height; ++y) {
        const std::uint8_t filter = raw[y * (stride + 1)];
        const std::uint8_t* in = raw.data() + y * (stride + 1) + 1;
        std::uint8_t* row = px.data() + y * stride;
        for (std::size_t x = 0; x < stride; ++x) {
            const int a = x >= bpp ? row[x - bpp] : 0, b = prev[x], c = x >= bpp ? prev[x - bpp] : 0;
            int v = in[x];
            switch (filter) {
            case 0: break;
            case 1: v += a; break;
            case 2: v += b; break;
            case 3: v += (a + b) / 2; break;
            case 4: {
                const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
                break;
            }
            default: throw std::runtime_error("loadPng: invalid filter in " + path);
            }
            row[x] = static_cast<std::uint8_t>(v);
        }
        std::copy(row, row + stride, prev.begin());
    }
    Image img;
    img.width = width;
    img.height = height;
    auto sample = [&](std::size_t y, std::size_t i) -> unsigned {  // i-th sample of row y, scaled to 8 bits
        const std::uint8_t* row = px.data() + y * stride;
        if (depth == 8) return row[i];
        if (depth == 16) return row[2 * i];
        const std::size_t bit = i * depth;
        const unsigned v = (row[bit / 8] >> (8 - depth - bit % 8)) & ((1u << depth) - 1);
        return colorType == 3 ? v : v * 255 / ((1u << depth) - 1);
    };
    if (colorType == 3) {
        const bool alpha = !trns.empty();
        img.channels = alpha ? 4 : 3;
        img.pixels.resize(width * height * img.channels);
        for (std::size_t y = 0; y < height; ++y)
            for (std::size_t x = 0; x < width; ++x) {
                const unsigned idx = sample(y, x);
                if (3 * idx + 2 >= palette.size()) throw std::runtime_error("loadPng: palette index out of range in " + path);
                std::uint8_t* o = img.pixels.data() + (y * width + x) * img.channels;
                o[0] = palette[3 * idx];
                o[1] = palette[3 * idx + 1];
                o[2] = palette[3 * idx + 2];
                if (alpha) o[3] = idx < trns.size() ? trns[idx] : 255;
            }
        return img;
    }
    img.channels = samples;
    img.pixels.resize(width * height * samples);
    for (std::size_t y = 0; y < height; ++y)
        for (std::size_t i = 0; i < width * samples; ++i) img.pixels[y * width * samples + i] = static_cast<std::uint8_t>(sample(y, i));
    return img;
}

Image loadImage(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadImage: cannot open " + path);
    char head[8] = {};
    f.read(head, 8);
    if (static_cast<unsigned char>(head[0]) == 0x89 && head[1] == 'P' && head[2] == 'N' && head[3] == 'G') return loadPng(path);
    if (head[0] == 'P' && (head[1] == '5' || head[1] == '6')) return loadPpm(path);
    throw std::runtime_error("loadImage: " + path + " is neither PNG nor binary PPM/PGM");
}

Image toRgb(const Image& image) {
    checkImage(image, "toRgb");
    if (image.channels == 3) return image;
    Image out;
    out.width = image.width;
    out.height = image.height;
    out.channels = 3;
    out.pixels.resize(image.width * image.height * 3);
    for (std::size_t p = 0; p < image.width * image.height; ++p) {
        const std::uint8_t* s = image.pixels.data() + p * image.channels;
        const bool grey = image.channels <= 2;
        const unsigned alpha = image.channels == 2 ? s[1] : image.channels == 4 ? s[3] : 255;
        for (std::size_t c = 0; c < 3; ++c) {
            const unsigned v = grey ? s[0] : s[c];
            out.pixels[p * 3 + c] = static_cast<std::uint8_t>((v * alpha + 255 * (255 - alpha) + 127) / 255);  // over white
        }
    }
    return out;
}

Image resizeImage(const Image& image, std::size_t width, std::size_t height) {
    checkImage(image, "resizeImage");
    if (width == 0 || height == 0) throw std::invalid_argument("resizeImage: empty size");
    Image out;
    out.width = width;
    out.height = height;
    out.channels = image.channels;
    out.pixels.resize(width * height * image.channels);
    const double sx = static_cast<double>(image.width) / width, sy = static_cast<double>(image.height) / height;
    // Bilinear, with a box prefilter when shrinking by more than 2x (average of the covered source pixels).
    parallelFor(0, height, 4, [&](std::size_t y0, std::size_t y1) {
        for (std::size_t y = y0; y < y1; ++y)
            for (std::size_t x = 0; x < width; ++x)
                for (std::size_t c = 0; c < image.channels; ++c) {
                    double v;
                    if (sx > 2.0 || sy > 2.0) {
                        const std::size_t ax = static_cast<std::size_t>(x * sx), bx = std::max(ax + 1, static_cast<std::size_t>((x + 1) * sx));
                        const std::size_t ay = static_cast<std::size_t>(y * sy), by = std::max(ay + 1, static_cast<std::size_t>((y + 1) * sy));
                        double sum = 0;
                        for (std::size_t yy = ay; yy < std::min(by, image.height); ++yy)
                            for (std::size_t xx = ax; xx < std::min(bx, image.width); ++xx)
                                sum += image.pixels[(yy * image.width + xx) * image.channels + c];
                        v = sum / static_cast<double>((std::min(by, image.height) - ay) * (std::min(bx, image.width) - ax));
                    } else {
                        const double fx = std::max(0.0, (x + 0.5) * sx - 0.5), fy = std::max(0.0, (y + 0.5) * sy - 0.5);
                        const std::size_t x0 = std::min(static_cast<std::size_t>(fx), image.width - 1), y0_ = std::min(static_cast<std::size_t>(fy), image.height - 1);
                        const std::size_t x1 = std::min(x0 + 1, image.width - 1), y1_ = std::min(y0_ + 1, image.height - 1);
                        const double ax = fx - x0, ay = fy - y0_;
                        auto at = [&](std::size_t xx, std::size_t yy) { return static_cast<double>(image.pixels[(yy * image.width + xx) * image.channels + c]); };
                        v = (at(x0, y0_) * (1 - ax) + at(x1, y0_) * ax) * (1 - ay) + (at(x0, y1_) * (1 - ax) + at(x1, y1_) * ax) * ay;
                    }
                    out.pixels[(y * width + x) * image.channels + c] = static_cast<std::uint8_t>(std::lround(std::min(255.0, std::max(0.0, v))));
                }
    });
    return out;
}

Image fitImage(const Image& image, std::size_t width, std::size_t height) {
    checkImage(image, "fitImage");
    // Largest centred crop with the target aspect ratio.
    std::size_t cw = image.width, ch = image.width * height / width;
    if (ch > image.height) {
        ch = image.height;
        cw = image.height * width / height;
    }
    Image crop;
    crop.width = std::max<std::size_t>(1, cw);
    crop.height = std::max<std::size_t>(1, ch);
    crop.channels = image.channels;
    crop.pixels.resize(crop.width * crop.height * crop.channels);
    const std::size_t ox = (image.width - crop.width) / 2, oy = (image.height - crop.height) / 2;
    for (std::size_t y = 0; y < crop.height; ++y)
        std::copy(image.pixels.begin() + static_cast<long>(((oy + y) * image.width + ox) * image.channels),
                  image.pixels.begin() + static_cast<long>(((oy + y) * image.width + ox + crop.width) * image.channels),
                  crop.pixels.begin() + static_cast<long>(y * crop.width * crop.channels));
    return crop.width == width && crop.height == height ? crop : resizeImage(crop, width, height);
}

Image cannyEdges(const Image& input, float low, float high) {
    checkImage(input, "cannyEdges");
    const Image im = input.channels == 1 ? input : toRgb(input);
    const long W = static_cast<long>(im.width), H = static_cast<long>(im.height), C = static_cast<long>(im.channels);
    auto at = [&](long x, long y, long c) {  // replicated border, as OpenCV's Canny
        x = std::max(0L, std::min(W - 1, x));
        y = std::max(0L, std::min(H - 1, y));
        return static_cast<float>(im.pixels[static_cast<std::size_t>((y * W + x) * C + c)]);
    };
    // Per pixel, the channel with the largest L1 gradient (OpenCV's rule for colour images).
    std::vector<float> mag(static_cast<std::size_t>(W * H)), gx(mag.size()), gy(mag.size());
    for (long y = 0; y < H; ++y)
        for (long x = 0; x < W; ++x) {
            const std::size_t i = static_cast<std::size_t>(y * W + x);
            mag[i] = -1.0f;
            for (long c = 0; c < C; ++c) {
                const float dx = (at(x + 1, y - 1, c) + 2 * at(x + 1, y, c) + at(x + 1, y + 1, c)) -
                                 (at(x - 1, y - 1, c) + 2 * at(x - 1, y, c) + at(x - 1, y + 1, c));
                const float dy = (at(x - 1, y + 1, c) + 2 * at(x, y + 1, c) + at(x + 1, y + 1, c)) -
                                 (at(x - 1, y - 1, c) + 2 * at(x, y - 1, c) + at(x + 1, y - 1, c));
                const float m = std::abs(dx) + std::abs(dy);
                if (m > mag[i]) {
                    mag[i] = m;
                    gx[i] = dx;
                    gy[i] = dy;
                }
            }
        }
    // Non-maximum suppression along the gradient direction (4 sectors, as OpenCV with tan(22.5°)).
    std::vector<std::uint8_t> state(mag.size(), 0);  // 0 none, 1 weak, 2 strong
    const float tan22 = 0.4142135623730951f;
    for (long y = 1; y + 1 < H; ++y)
        for (long x = 1; x + 1 < W; ++x) {
            const std::size_t i = static_cast<std::size_t>(y * W + x);
            const float m = mag[i];
            if (m <= low) continue;
            const float ax = std::abs(gx[i]), ay = std::abs(gy[i]);
            float a, b;
            if (ay <= ax * tan22) {  // horizontal gradient: compare left / right
                a = mag[i - 1];
                b = mag[i + 1];
            } else if (ay >= ax / tan22) {  // vertical
                a = mag[i - static_cast<std::size_t>(W)];
                b = mag[i + static_cast<std::size_t>(W)];
            } else {  // diagonal: both neighbours strictly smaller
                const bool same = (gx[i] < 0) == (gy[i] < 0);
                a = same ? mag[i - static_cast<std::size_t>(W) - 1] : mag[i - static_cast<std::size_t>(W) + 1];
                b = same ? mag[i + static_cast<std::size_t>(W) + 1] : mag[i + static_cast<std::size_t>(W) - 1];
                if (!(m > b)) continue;
            }
            if (m > a && m >= b) state[i] = m > high ? 2 : 1;
        }
    // Hysteresis: weak pixels connected (8-neighbourhood) to strong ones become edges.
    std::vector<std::size_t> stack;
    for (std::size_t i = 0; i < state.size(); ++i)
        if (state[i] == 2) stack.push_back(i);
    while (!stack.empty()) {
        const std::size_t i = stack.back();
        stack.pop_back();
        const long x = static_cast<long>(i) % W, y = static_cast<long>(i) / W;
        for (long dy = -1; dy <= 1; ++dy)
            for (long dx = -1; dx <= 1; ++dx) {
                const long xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
                const std::size_t j = static_cast<std::size_t>(yy * W + xx);
                if (state[j] == 1) {
                    state[j] = 2;
                    stack.push_back(j);
                }
            }
    }
    Image out;
    out.width = im.width;
    out.height = im.height;
    out.pixels.resize(im.width * im.height * 3);
    for (std::size_t i = 0; i < state.size(); ++i) std::fill_n(out.pixels.data() + i * 3, 3, state[i] == 2 ? 255 : 0);
    return out;
}

Image transformImage(const Image& image, float scale, float dx, float dy, float angleDegrees) {
    checkImage(image, "transformImage");
    if (scale <= 0.0f) throw std::invalid_argument("transformImage: scale must be > 0");
    Image out = image;
    const double cx = (image.width - 1) / 2.0, cy = (image.height - 1) / 2.0;
    const double a = angleDegrees * 3.14159265358979323846 / 180.0, ca = std::cos(a), sa = std::sin(a);
    const std::size_t C = image.channels;
    parallelFor(0, image.height, 4, [&](std::size_t y0, std::size_t y1) {
        for (std::size_t y = y0; y < y1; ++y)
            for (std::size_t x = 0; x < image.width; ++x) {
                // Output pixel -> source: undo the shift, rotation and zoom about the centre.
                const double ux = (static_cast<double>(x) - cx - dx) / scale, uy = (static_cast<double>(y) - cy - dy) / scale;
                const double sx = std::min(std::max(cx + ca * ux + sa * uy, 0.0), image.width - 1.0);
                const double sy = std::min(std::max(cy - sa * ux + ca * uy, 0.0), image.height - 1.0);
                const std::size_t x0 = static_cast<std::size_t>(sx), y0_ = static_cast<std::size_t>(sy);
                const std::size_t x1 = std::min(x0 + 1, image.width - 1), y1_ = std::min(y0_ + 1, image.height - 1);
                const double fx = sx - x0, fy = sy - y0_;
                for (std::size_t c = 0; c < C; ++c) {
                    auto at = [&](std::size_t xx, std::size_t yy) { return static_cast<double>(image.pixels[(yy * image.width + xx) * C + c]); };
                    const double v = (at(x0, y0_) * (1 - fx) + at(x1, y0_) * fx) * (1 - fy) + (at(x0, y1_) * (1 - fx) + at(x1, y1_) * fx) * fy;
                    out.pixels[(y * image.width + x) * C + c] = static_cast<std::uint8_t>(std::lround(v));
                }
            }
    });
    return out;
}

// ---- Animated GIF -----------------------------------------------------------------------------

namespace {
struct Box {
    std::vector<std::array<std::uint8_t, 3>> colors;
};

// Median cut: split the box with the widest channel range at its median until there are `n` boxes.
std::vector<std::array<std::uint8_t, 3>> medianCut(std::vector<std::array<std::uint8_t, 3>> colors, std::size_t n) {
    std::vector<std::vector<std::array<std::uint8_t, 3>>> boxes{std::move(colors)};
    auto range = [](const std::vector<std::array<std::uint8_t, 3>>& b, int& channel) {
        int best = -1;
        for (int c = 0; c < 3; ++c) {
            int lo = 255, hi = 0;
            for (const auto& p : b) {
                lo = std::min<int>(lo, p[static_cast<std::size_t>(c)]);
                hi = std::max<int>(hi, p[static_cast<std::size_t>(c)]);
            }
            if (hi - lo > best) {
                best = hi - lo;
                channel = c;
            }
        }
        return best;
    };
    while (boxes.size() < n) {
        std::size_t pick = boxes.size();
        int widest = 0, channel = 0;
        for (std::size_t i = 0; i < boxes.size(); ++i) {
            int c = 0;
            const int r = boxes[i].size() > 1 ? range(boxes[i], c) : 0;
            if (r > widest) {
                widest = r;
                pick = i;
                channel = c;
            }
        }
        if (pick == boxes.size()) break;  // every box is a single colour
        auto& b = boxes[pick];
        const std::size_t ch = static_cast<std::size_t>(channel);
        std::nth_element(b.begin(), b.begin() + static_cast<long>(b.size() / 2), b.end(),
                         [ch](const auto& x, const auto& y) { return x[ch] < y[ch]; });
        std::vector<std::array<std::uint8_t, 3>> upper(b.begin() + static_cast<long>(b.size() / 2), b.end());
        b.resize(b.size() / 2);
        boxes.push_back(std::move(upper));
    }
    std::vector<std::array<std::uint8_t, 3>> palette;
    for (const auto& b : boxes) {
        if (b.empty()) continue;
        unsigned long s[3] = {0, 0, 0};
        for (const auto& p : b)
            for (std::size_t c = 0; c < 3; ++c) s[c] += p[c];
        palette.push_back({static_cast<std::uint8_t>(s[0] / b.size()), static_cast<std::uint8_t>(s[1] / b.size()),
                           static_cast<std::uint8_t>(s[2] / b.size())});
    }
    return palette;
}

// GIF LZW: variable-width codes (min code size 8), a clear code whenever the table is full.
std::vector<std::uint8_t> lzw(const std::vector<std::uint8_t>& indices) {
    const int clear = 256, end = 257;
    std::vector<std::uint8_t> out;
    std::uint32_t buffer = 0;
    int bits = 0, width = 9;
    auto put = [&](int code) {
        buffer |= static_cast<std::uint32_t>(code) << bits;
        bits += width;
        while (bits >= 8) {
            out.push_back(static_cast<std::uint8_t>(buffer & 0xFF));
            buffer >>= 8;
            bits -= 8;
        }
    };
    std::unordered_map<std::uint32_t, int> table;  // (prefix code << 8 | byte) -> code
    int next = 258;
    put(clear);
    int prefix = -1;
    for (std::uint8_t v : indices) {
        if (prefix < 0) {
            prefix = v;
            continue;
        }
        const std::uint32_t key = (static_cast<std::uint32_t>(prefix) << 8) | v;
        auto it = table.find(key);
        if (it != table.end()) {
            prefix = it->second;
            continue;
        }
        put(prefix);
        if (next < 4096) {
            table.emplace(key, next++);
            if (next > (1 << width) && width < 12) ++width;
        } else {
            put(clear);
            table.clear();
            next = 258;
            width = 9;
        }
        prefix = v;
    }
    if (prefix >= 0) put(prefix);
    put(end);
    if (bits > 0) out.push_back(static_cast<std::uint8_t>(buffer & 0xFF));
    return out;
}
} // namespace

void saveGif(const std::string& path, const std::vector<Image>& frames, float fps, bool loop) {
    if (frames.empty()) throw std::invalid_argument("saveGif: no frames");
    const std::size_t W = frames[0].width, H = frames[0].height;
    if (W > 65535 || H > 65535) throw std::invalid_argument("saveGif: frames are too large");
    for (const auto& f : frames) {
        checkImage(f, "saveGif");
        if (f.width != W || f.height != H || f.channels != 3) throw std::invalid_argument("saveGif: frames must be RGB and of one size");
    }
    // Shared palette from a sample of all frames.
    std::vector<std::array<std::uint8_t, 3>> sample;
    const std::size_t total = frames.size() * W * H, step = std::max<std::size_t>(1, total / 200000);
    for (std::size_t i = 0; i < total; i += step) {
        const Image& f = frames[i / (W * H)];
        const std::size_t p = i % (W * H);
        sample.push_back({f.pixels[p * 3], f.pixels[p * 3 + 1], f.pixels[p * 3 + 2]});
    }
    auto palette = medianCut(std::move(sample), 256);
    palette.resize(256, {0, 0, 0});
    // Nearest palette entry per 5-bit-per-channel colour cell, filled on demand.
    std::vector<int> cache(32 * 32 * 32, -1);
    auto nearest = [&](int r, int g, int b) {
        const std::size_t cell = static_cast<std::size_t>(((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3));
        if (cache[cell] < 0) {
            const int cr = (r & ~7) + 4, cg = (g & ~7) + 4, cb = (b & ~7) + 4;
            int best = 0, bestD = 1 << 30;
            for (int i = 0; i < 256; ++i) {
                const int d = (palette[static_cast<std::size_t>(i)][0] - cr) * (palette[static_cast<std::size_t>(i)][0] - cr) * 2 +
                              (palette[static_cast<std::size_t>(i)][1] - cg) * (palette[static_cast<std::size_t>(i)][1] - cg) * 4 +
                              (palette[static_cast<std::size_t>(i)][2] - cb) * (palette[static_cast<std::size_t>(i)][2] - cb) * 3;
                if (d < bestD) {
                    bestD = d;
                    best = i;
                }
            }
            cache[cell] = best;
        }
        return static_cast<std::uint8_t>(cache[cell]);
    };
    static const int bayer[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("saveGif: cannot open " + path);
    std::vector<std::uint8_t> out{'G', 'I', 'F', '8', '9', 'a'};
    auto u16 = [&](std::size_t v) {
        out.push_back(static_cast<std::uint8_t>(v & 0xFF));
        out.push_back(static_cast<std::uint8_t>(v >> 8));
    };
    u16(W);
    u16(H);
    out.insert(out.end(), {0xF7, 0, 0});  // global colour table of 2^(7+1) entries
    for (const auto& c : palette) out.insert(out.end(), c.begin(), c.end());
    if (loop) {
        const std::uint8_t netscape[] = {0x21, 0xFF, 0x0B, 'N', 'E', 'T', 'S', 'C', 'A', 'P', 'E', '2', '.', '0', 3, 1, 0, 0, 0};
        out.insert(out.end(), netscape, netscape + sizeof(netscape));
    }
    const std::size_t delay = static_cast<std::size_t>(std::lround(100.0 / std::max(0.1f, fps)));
    std::vector<std::uint8_t> indices(W * H);
    for (const auto& frame : frames) {
        out.insert(out.end(), {0x21, 0xF9, 4, 0x04});  // graphic control: disposal "leave in place"
        u16(delay);
        out.insert(out.end(), {0, 0});
        out.push_back(0x2C);  // image descriptor
        u16(0);
        u16(0);
        u16(W);
        u16(H);
        out.push_back(0);
        for (std::size_t y = 0; y < H; ++y)
            for (std::size_t x = 0; x < W; ++x) {
                const int d = (bayer[y & 3][x & 3] - 8) / 2;  // +-4 levels
                const std::uint8_t* p = frame.pixels.data() + (y * W + x) * 3;
                auto clamp = [](int v) { return std::min(255, std::max(0, v)); };
                indices[y * W + x] = nearest(clamp(p[0] + d), clamp(p[1] + d), clamp(p[2] + d));
            }
        const std::vector<std::uint8_t> data = lzw(indices);
        out.push_back(8);  // LZW minimum code size
        for (std::size_t at = 0; at < data.size(); at += 255) {
            const std::size_t n = std::min<std::size_t>(255, data.size() - at);
            out.push_back(static_cast<std::uint8_t>(n));
            out.insert(out.end(), data.begin() + static_cast<long>(at), data.begin() + static_cast<long>(at + n));
        }
        out.push_back(0);
    }
    out.push_back(0x3B);
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!f) throw std::runtime_error("saveGif: write failed for " + path);
}

Matrix imageToTensor(const Image& image, float lo, float hi) {
    checkImage(image, "imageToTensor");
    const std::size_t HW = image.width * image.height;
    Matrix m(1, image.channels * HW);
    for (std::size_t p = 0; p < HW; ++p)
        for (std::size_t c = 0; c < image.channels; ++c)
            m(0, c * HW + p) = lo + (hi - lo) * static_cast<float>(image.pixels[p * image.channels + c]) / 255.0f;
    return m;
}

Image tensorToImage(const float* chw, ImageShape shape, float lo, float hi) {
    Image image;
    image.width = shape.width;
    image.height = shape.height;
    image.channels = shape.channels;
    const std::size_t HW = shape.width * shape.height;
    image.pixels.resize(HW * shape.channels);
    for (std::size_t p = 0; p < HW; ++p)
        for (std::size_t c = 0; c < shape.channels; ++c) {
            const float v = (chw[c * HW + p] - lo) / (hi - lo) * 255.0f;
            image.pixels[p * shape.channels + c] = static_cast<std::uint8_t>(std::lround(std::min(255.0f, std::max(0.0f, v))));
        }
    return image;
}

// ---- JPEG (baseline) and Motion-JPEG QuickTime movies ------------------------------------------

namespace {
const std::uint8_t kZigzag[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                                  41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                                  30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
// ITU T.81 Annex K tables (natural order).
const std::uint8_t kLumaQ[64] = {16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,  14, 13, 16, 24, 40,  57,
                                 69, 56, 14, 17, 22,  29,  51,  87,  80, 62, 18, 22, 37,  56,  68,  109, 103, 77, 24, 35, 55, 64,
                                 81, 104, 113, 92, 49, 64, 78, 87,  103, 121, 120, 101, 72, 92, 95,  98,  112, 100, 103, 99};
const std::uint8_t kChromaQ[64] = {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99,
                                   99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                                   99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};
const std::uint8_t kDcLumaBits[16] = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const std::uint8_t kDcChromaBits[16] = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
const std::uint8_t kDcValues[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const std::uint8_t kAcLumaBits[16] = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
const std::uint8_t kAcLumaValues[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81,
    0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18,
    0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5,
    0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
const std::uint8_t kAcChromaBits[16] = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
const std::uint8_t kAcChromaValues[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08,
    0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25,
    0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74,
    0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4,
    0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

struct HuffTable {
    std::uint16_t code[256] = {};
    std::uint8_t size[256] = {};
    HuffTable(const std::uint8_t* bits, const std::uint8_t* values) {
        std::uint16_t c = 0;
        std::size_t k = 0;
        for (int len = 1; len <= 16; ++len) {
            for (int i = 0; i < bits[len - 1]; ++i, ++k) {
                code[values[k]] = c++;
                size[values[k]] = static_cast<std::uint8_t>(len);
            }
            c <<= 1;
        }
    }
};

struct JpegBits {
    std::vector<std::uint8_t>& out;
    std::uint32_t acc = 0;
    int n = 0;
    void put(std::uint32_t bits, int count) {  // MSB first, with 0xFF stuffing
        acc = (acc << count) | (bits & ((1u << count) - 1));
        n += count;
        while (n >= 8) {
            const std::uint8_t b = static_cast<std::uint8_t>((acc >> (n - 8)) & 0xFF);
            out.push_back(b);
            if (b == 0xFF) out.push_back(0);
            n -= 8;
        }
        acc &= (1u << n) - 1;
    }
    void flush() {
        if (n > 0) put(0x7F, 8 - n);  // pad with ones
    }
};

void fdct8x8(float* b) {  // separable float DCT-II, scaled as JPEG expects
    static const std::array<float, 64> cosTable = [] {
        std::array<float, 64> c{};
        for (int u = 0; u < 8; ++u)
            for (int x = 0; x < 8; ++x)
                c[static_cast<std::size_t>(u * 8 + x)] = static_cast<float>((u == 0 ? std::sqrt(0.125) : 0.5) * std::cos((2 * x + 1) * u * 3.14159265358979323846 / 16));
        return c;
    }();
    float tmp[64];
    for (int y = 0; y < 8; ++y)
        for (int u = 0; u < 8; ++u) {
            float s = 0;
            for (int x = 0; x < 8; ++x) s += cosTable[static_cast<std::size_t>(u * 8 + x)] * b[y * 8 + x];
            tmp[y * 8 + u] = s;
        }
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float s = 0;
            for (int y = 0; y < 8; ++y) s += cosTable[static_cast<std::size_t>(v * 8 + y)] * tmp[y * 8 + u];
            b[v * 8 + u] = s;
        }
}

void putMarker(std::vector<std::uint8_t>& o, std::uint8_t m, const std::vector<std::uint8_t>& payload) {
    o.insert(o.end(), {0xFF, m, static_cast<std::uint8_t>((payload.size() + 2) >> 8), static_cast<std::uint8_t>((payload.size() + 2) & 0xFF)});
    o.insert(o.end(), payload.begin(), payload.end());
}
} // namespace

std::vector<std::uint8_t> encodeJpeg(const Image& input, int quality) {
    checkImage(input, "encodeJpeg");
    if (quality < 1 || quality > 100) throw std::invalid_argument("encodeJpeg: quality must be in 1..100");
    if (input.width > 65535 || input.height > 65535) throw std::invalid_argument("encodeJpeg: image too large");
    const Image im = input.channels == 3 ? input : toRgb(input);
    const std::size_t W = im.width, H = im.height;
    // Quantization tables, scaled as libjpeg does.
    const int scale = quality < 50 ? 5000 / quality : 200 - 2 * quality;
    std::uint8_t q[2][64];
    for (int i = 0; i < 64; ++i) {
        q[0][i] = static_cast<std::uint8_t>(std::min(255, std::max(1, (kLumaQ[i] * scale + 50) / 100)));
        q[1][i] = static_cast<std::uint8_t>(std::min(255, std::max(1, (kChromaQ[i] * scale + 50) / 100)));
    }
    // Planes: Y full size, Cb / Cr averaged over 2x2 (4:2:0), edges replicated to whole 16x16 MCUs.
    const std::size_t mw = (W + 15) / 16, mh = (H + 15) / 16, PW = mw * 16, PH = mh * 16;
    std::vector<float> Y(PW * PH), Cb(PW / 2 * PH / 2), Cr(PW / 2 * PH / 2);
    for (std::size_t y = 0; y < PH; ++y)
        for (std::size_t x = 0; x < PW; ++x) {
            const std::uint8_t* p = im.pixels.data() + (std::min(y, H - 1) * W + std::min(x, W - 1)) * 3;
            Y[y * PW + x] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2] - 128.0f;
        }
    for (std::size_t y = 0; y < PH / 2; ++y)
        for (std::size_t x = 0; x < PW / 2; ++x) {
            float cb = 0, cr = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const std::uint8_t* p = im.pixels.data() + (std::min(2 * y + dy, H - 1) * W + std::min(2 * x + dx, W - 1)) * 3;
                    cb += -0.168736f * p[0] - 0.331264f * p[1] + 0.5f * p[2];
                    cr += 0.5f * p[0] - 0.418688f * p[1] - 0.081312f * p[2];
                }
            Cb[y * (PW / 2) + x] = cb / 4;
            Cr[y * (PW / 2) + x] = cr / 4;
        }
    std::vector<std::uint8_t> o{0xFF, 0xD8};
    putMarker(o, 0xE0, {'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0});
    for (int t = 0; t < 2; ++t) {
        std::vector<std::uint8_t> dqt{static_cast<std::uint8_t>(t)};
        for (int i = 0; i < 64; ++i) dqt.push_back(q[t][kZigzag[i]]);
        putMarker(o, 0xDB, dqt);
    }
    putMarker(o, 0xC0, {8, static_cast<std::uint8_t>(H >> 8), static_cast<std::uint8_t>(H & 0xFF), static_cast<std::uint8_t>(W >> 8),
                        static_cast<std::uint8_t>(W & 0xFF), 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1});
    auto dht = [&](std::uint8_t cls, const std::uint8_t* bits, const std::uint8_t* values) {
        std::vector<std::uint8_t> d{cls};
        int count = 0;
        for (int i = 0; i < 16; ++i) {
            d.push_back(bits[i]);
            count += bits[i];
        }
        d.insert(d.end(), values, values + count);
        putMarker(o, 0xC4, d);
    };
    dht(0x00, kDcLumaBits, kDcValues);
    dht(0x10, kAcLumaBits, kAcLumaValues);
    dht(0x01, kDcChromaBits, kDcValues);
    dht(0x11, kAcChromaBits, kAcChromaValues);
    putMarker(o, 0xDA, {3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0});
    static const HuffTable dcL(kDcLumaBits, kDcValues), acL(kAcLumaBits, kAcLumaValues), dcC(kDcChromaBits, kDcValues),
        acC(kAcChromaBits, kAcChromaValues);
    JpegBits bits{o};
    int prevDc[3] = {0, 0, 0};
    auto block = [&](const std::vector<float>& plane, std::size_t stride, std::size_t bx, std::size_t by, int comp) {
        float b[64];
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) b[y * 8 + x] = plane[(by + static_cast<std::size_t>(y)) * stride + bx + static_cast<std::size_t>(x)];
        fdct8x8(b);
        const std::uint8_t* qt = q[comp == 0 ? 0 : 1];
        int z[64];
        for (int i = 0; i < 64; ++i) z[i] = static_cast<int>(std::lround(b[kZigzag[i]] / qt[kZigzag[i]]));
        const HuffTable& dc = comp == 0 ? dcL : dcC;
        const HuffTable& ac = comp == 0 ? acL : acC;
        auto magnitude = [](int v, int& sizeOut) {
            int a = std::abs(v), s = 0;
            while (a) {
                ++s;
                a >>= 1;
            }
            sizeOut = s;
            return static_cast<std::uint32_t>(v < 0 ? v + (1 << s) - 1 : v);
        };
        int s;
        const int diff = z[0] - prevDc[comp];
        prevDc[comp] = z[0];
        std::uint32_t m = magnitude(diff, s);
        bits.put(dc.code[s], dc.size[s]);
        if (s) bits.put(m, s);
        int run = 0;
        for (int i = 1; i < 64; ++i) {
            if (z[i] == 0) {
                ++run;
                continue;
            }
            while (run > 15) {
                bits.put(ac.code[0xF0], ac.size[0xF0]);
                run -= 16;
            }
            m = magnitude(z[i], s);
            const int sym = (run << 4) | s;
            bits.put(ac.code[sym], ac.size[sym]);
            bits.put(m, s);
            run = 0;
        }
        if (run) bits.put(ac.code[0], ac.size[0]);
    };
    for (std::size_t my = 0; my < mh; ++my)
        for (std::size_t mx = 0; mx < mw; ++mx) {
            for (int k = 0; k < 4; ++k) block(Y, PW, mx * 16 + static_cast<std::size_t>(k % 2) * 8, my * 16 + static_cast<std::size_t>(k / 2) * 8, 0);
            block(Cb, PW / 2, mx * 8, my * 8, 1);
            block(Cr, PW / 2, mx * 8, my * 8, 2);
        }
    bits.flush();
    o.insert(o.end(), {0xFF, 0xD9});
    return o;
}

void saveJpeg(const std::string& path, const Image& image, int quality) {
    const std::vector<std::uint8_t> d = encodeJpeg(image, quality);
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("saveJpeg: cannot open " + path);
    f.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size()));
    if (!f) throw std::runtime_error("saveJpeg: write failed for " + path);
}

namespace {
struct Atom {  // QuickTime / MP4 box writer
    std::vector<std::uint8_t> d;
    void u8(std::uint8_t v) { d.push_back(v); }
    void u16(std::uint32_t v) { d.insert(d.end(), {static_cast<std::uint8_t>(v >> 8), static_cast<std::uint8_t>(v)}); }
    void u32(std::uint32_t v) { putBE32(d, v); }
    void tag(const char* s) { d.insert(d.end(), s, s + 4); }
    void zeros(std::size_t n) { d.insert(d.end(), n, 0); }
    void add(const Atom& a) { d.insert(d.end(), a.d.begin(), a.d.end()); }
    static Atom box(const char* type, const std::vector<Atom>& children = {}, const std::vector<std::uint8_t>& body = {}) {
        Atom a;
        std::size_t size = 8 + body.size();
        for (const Atom& c : children) size += c.d.size();
        a.u32(static_cast<std::uint32_t>(size));
        a.tag(type);
        a.d.insert(a.d.end(), body.begin(), body.end());
        for (const Atom& c : children) a.add(c);
        return a;
    }
};
void matrixIdentity(Atom& a) {
    for (std::uint32_t v : {0x00010000u, 0u, 0u, 0u, 0x00010000u, 0u, 0u, 0u, 0x40000000u}) a.u32(v);
}
} // namespace

void saveMjpegMov(const std::string& path, const std::vector<Image>& frames, float framesPerSecond, int quality) {
    if (frames.empty()) throw std::invalid_argument("saveMjpegMov: no frames");
    if (framesPerSecond <= 0.0f) throw std::invalid_argument("saveMjpegMov: frames per second must be > 0");
    const std::size_t W = frames[0].width, H = frames[0].height;
    std::vector<std::vector<std::uint8_t>> jpegs;
    for (const Image& f : frames) {
        if (f.width != W || f.height != H) throw std::invalid_argument("saveMjpegMov: frames must have one size");
        jpegs.push_back(encodeJpeg(f, quality));
    }
    // Timescale with whole-number frame durations: 1000 * fps rounded, delta 1000.
    const std::uint32_t timescale = static_cast<std::uint32_t>(std::lround(framesPerSecond * 1000.0f)), delta = 1000;
    const std::uint32_t n = static_cast<std::uint32_t>(frames.size()), duration = n * delta;
    Atom ftyp = Atom::box("ftyp", {}, {'q', 't', ' ', ' ', 0, 0, 2, 0, 'q', 't', ' ', ' '});
    std::size_t mdatSize = 8;
    for (const auto& j : jpegs) mdatSize += j.size();
    std::vector<std::uint32_t> offsets;
    std::size_t at = ftyp.d.size() + 8;
    for (const auto& j : jpegs) {
        offsets.push_back(static_cast<std::uint32_t>(at));
        at += j.size();
    }
    if (at > 0xFFFFFFFFu) throw std::runtime_error("saveMjpegMov: movie larger than 4 GB");

    Atom mvhd;
    mvhd.u32(0);  // version, flags
    mvhd.u32(0);
    mvhd.u32(0);  // creation / modification time
    mvhd.u32(timescale);
    mvhd.u32(duration);
    mvhd.u32(0x00010000);  // rate 1.0
    mvhd.u16(0x0100);      // volume 1.0
    mvhd.zeros(10);
    matrixIdentity(mvhd);
    mvhd.zeros(24);  // preview, poster, selection, current time
    mvhd.u32(2);     // next track id
    Atom tkhd;
    tkhd.u32(0x0000000F);  // enabled, in movie, in preview, in poster
    tkhd.u32(0);
    tkhd.u32(0);
    tkhd.u32(1);  // track id
    tkhd.u32(0);
    tkhd.u32(duration);
    tkhd.zeros(8);
    tkhd.u16(0);  // layer
    tkhd.u16(0);  // alternate group
    tkhd.u16(0);  // volume (video)
    tkhd.u16(0);
    matrixIdentity(tkhd);
    tkhd.u32(static_cast<std::uint32_t>(W) << 16);
    tkhd.u32(static_cast<std::uint32_t>(H) << 16);
    Atom mdhd;
    mdhd.u32(0);
    mdhd.u32(0);
    mdhd.u32(0);
    mdhd.u32(timescale);
    mdhd.u32(duration);
    mdhd.u16(0x7FFF);  // language: unspecified
    mdhd.u16(0);       // quality
    auto hdlr = [](const char* kind, const char* sub, const char* name) {
        Atom h;
        h.u32(0);
        h.tag(kind);
        h.tag(sub);
        h.zeros(12);
        const std::size_t len = std::strlen(name);
        h.u8(static_cast<std::uint8_t>(len));  // Pascal string, as QuickTime writes it
        h.d.insert(h.d.end(), name, name + len);
        return h;
    };
    Atom vmhd;
    vmhd.u32(0x00000001);
    vmhd.u16(0x0040);  // graphics mode: dither copy
    vmhd.zeros(6);
    Atom dref;
    dref.u32(0);
    dref.u32(1);
    dref.add(Atom::box("alis", {}, {0, 0, 0, 1}));  // self-contained
    Atom entry;  // video sample description, 'jpeg' = Photo-JPEG
    entry.zeros(6);
    entry.u16(1);  // data reference index
    entry.u16(0);
    entry.u16(0);
    entry.tag("appl");
    entry.u32(0);    // temporal quality
    entry.u32(512);  // spatial quality
    entry.u16(static_cast<std::uint32_t>(W));
    entry.u16(static_cast<std::uint32_t>(H));
    entry.u32(0x00480000);  // 72 dpi
    entry.u32(0x00480000);
    entry.u32(0);  // data size
    entry.u16(1);  // frames per sample
    const char* compressor = "Photo - JPEG";
    entry.u8(static_cast<std::uint8_t>(std::strlen(compressor)));
    entry.d.insert(entry.d.end(), compressor, compressor + std::strlen(compressor));
    entry.zeros(31 - std::strlen(compressor));
    entry.u16(24);      // depth
    entry.u16(0xFFFF);  // no colour table
    Atom stsd;
    stsd.u32(0);
    stsd.u32(1);
    Atom sampleEntry = Atom::box("jpeg", {}, entry.d);
    stsd.add(sampleEntry);
    Atom stts;
    stts.u32(0);
    stts.u32(1);
    stts.u32(n);
    stts.u32(delta);
    Atom stsc;
    stsc.u32(0);
    stsc.u32(1);
    stsc.u32(1);
    stsc.u32(1);
    stsc.u32(1);
    Atom stsz;
    stsz.u32(0);
    stsz.u32(0);  // sizes differ
    stsz.u32(n);
    for (const auto& j : jpegs) stsz.u32(static_cast<std::uint32_t>(j.size()));
    Atom stco;
    stco.u32(0);
    stco.u32(n);
    for (std::uint32_t off : offsets) stco.u32(off);
    const Atom stbl = Atom::box("stbl", {Atom::box("stsd", {}, stsd.d), Atom::box("stts", {}, stts.d), Atom::box("stsc", {}, stsc.d),
                                         Atom::box("stsz", {}, stsz.d), Atom::box("stco", {}, stco.d)});
    const Atom minf = Atom::box("minf", {Atom::box("vmhd", {}, vmhd.d), Atom::box("hdlr", {}, hdlr("dhlr", "alis", "NeuraCraft data").d),
                                         Atom::box("dinf", {Atom::box("dref", {}, dref.d)}), stbl});
    const Atom mdia = Atom::box("mdia", {Atom::box("mdhd", {}, mdhd.d), Atom::box("hdlr", {}, hdlr("mhlr", "vide", "NeuraCraft video").d), minf});
    const Atom trak = Atom::box("trak", {Atom::box("tkhd", {}, tkhd.d), mdia});
    const Atom moov = Atom::box("moov", {Atom::box("mvhd", {}, mvhd.d), trak});

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("saveMjpegMov: cannot open " + path);
    f.write(reinterpret_cast<const char*>(ftyp.d.data()), static_cast<std::streamsize>(ftyp.d.size()));
    std::vector<std::uint8_t> mdatHeader;
    putBE32(mdatHeader, static_cast<std::uint32_t>(mdatSize));
    mdatHeader.insert(mdatHeader.end(), {'m', 'd', 'a', 't'});
    f.write(reinterpret_cast<const char*>(mdatHeader.data()), 8);
    for (const auto& j : jpegs) f.write(reinterpret_cast<const char*>(j.data()), static_cast<std::streamsize>(j.size()));
    f.write(reinterpret_cast<const char*>(moov.d.data()), static_cast<std::streamsize>(moov.d.size()));
    if (!f) throw std::runtime_error("saveMjpegMov: write failed for " + path);
}

} // namespace nc
