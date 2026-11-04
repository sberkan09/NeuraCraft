#include "Image.h"
#include "Backend.h"
#include "Parallel.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
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
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += x[i];
            const double mean = sum / static_cast<double>(n);
            double var = 0.0;
            for (std::size_t i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
            const float inv = static_cast<float>(1.0 / std::sqrt(var / static_cast<double>(n) + epsilon_));
            invStd_[t] = inv;
            const float m = static_cast<float>(mean);
            float* y = out.data() + b * out.cols() + g * n;
            float* xn = training ? normalized_.data() + b * out.cols() + g * n : nullptr;
            for (std::size_t c = 0; c < perGroup; ++c) {
                const std::size_t ch = g * perGroup + c;
                const float gm = gamma_(0, ch), bt = beta_(0, ch);
                for (std::size_t p = c * HW; p < (c + 1) * HW; ++p) {
                    const float v = (x[p] - m) * inv;
                    if (xn) xn[p] = v;
                    y[p] = v * gm + bt;
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

void savePng(const std::string& path, const Image& image) {
    checkImage(image, "savePng");
    std::uint8_t colorType = 0;
    switch (image.channels) {
    case 1: colorType = 0; break;
    case 3: colorType = 2; break;
    case 4: colorType = 6; break;
    default: throw std::invalid_argument("savePng: 1, 3 or 4 channels");
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("savePng: cannot open " + path);
    const std::uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    f.write(reinterpret_cast<const char*>(signature), 8);
    std::vector<std::uint8_t> ihdr;
    putBE32(ihdr, static_cast<std::uint32_t>(image.width));
    putBE32(ihdr, static_cast<std::uint32_t>(image.height));
    ihdr.insert(ihdr.end(), {8, colorType, 0, 0, 0});
    writeChunk(f, "IHDR", ihdr);

    // Raw scanlines (filter byte 0) in a zlib stream of stored deflate blocks.
    const std::size_t stride = image.width * image.channels;
    std::vector<std::uint8_t> raw;
    raw.reserve((stride + 1) * image.height);
    for (std::size_t y = 0; y < image.height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), image.pixels.begin() + static_cast<long>(y * stride),
                   image.pixels.begin() + static_cast<long>((y + 1) * stride));
    }
    std::vector<std::uint8_t> z{0x78, 0x01};
    for (std::size_t pos = 0; pos < raw.size() || pos == 0;) {
        const std::size_t len = std::min<std::size_t>(65535, raw.size() - pos);
        const bool last = pos + len == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<std::uint8_t>(len & 0xFF));
        z.push_back(static_cast<std::uint8_t>(len >> 8));
        z.push_back(static_cast<std::uint8_t>(~len & 0xFF));
        z.push_back(static_cast<std::uint8_t>((~len >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<long>(pos), raw.begin() + static_cast<long>(pos + len));
        pos += len;
        if (last) break;
    }
    std::uint32_t a = 1, b = 0;  // Adler-32
    for (std::uint8_t v : raw) {
        a = (a + v) % 65521;
        b = (b + a) % 65521;
    }
    putBE32(z, (b << 16) | a);
    writeChunk(f, "IDAT", z);
    writeChunk(f, "IEND", {});
    if (!f) throw std::runtime_error("savePng: write failed for " + path);
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
    Inflater(const std::uint8_t* data, std::size_t size) : d_(data), n_(size) {}

    std::vector<std::uint8_t> run() {
        std::vector<std::uint8_t> out;
        for (bool last = false; !last;) {
            last = bits(1);
            const unsigned type = bits(2);
            if (type == 0) {
                bitPos_ = (bitPos_ + 7) & ~std::size_t(7);
                const unsigned len = bits(16), nlen = bits(16);
                if ((len ^ 0xFFFF) != nlen) throw std::runtime_error("loadPng: corrupt stored block");
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
                const std::size_t from = out.size() - back;
                for (unsigned i = 0; i < len; ++i) out.push_back(out[from + i]);
            }
        }
    }

    const std::uint8_t* d_;
    std::size_t n_, bitPos_ = 0;
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
    if (interlace) throw std::runtime_error("loadPng: interlaced PNGs are not supported");
    const unsigned samples = colorType == 0 ? 1 : colorType == 2 ? 3 : colorType == 3 ? 1 : colorType == 4 ? 2 : colorType == 6 ? 4 : 0;
    if (!samples || (depth != 8 && depth != 16 && !((colorType == 0 || colorType == 3) && depth < 8)))
        throw std::runtime_error("loadPng: unsupported colour type / bit depth in " + path);
    if (idat.size() < 2 || (idat[0] & 0x0F) != 8) throw std::runtime_error("loadPng: bad zlib stream in " + path);
    const std::vector<std::uint8_t> raw = Inflater(idat.data() + 2, idat.size() - 2).run();
    const std::size_t bitsPerPixel = samples * depth, stride = (width * bitsPerPixel + 7) / 8, bpp = std::max<std::size_t>(1, bitsPerPixel / 8);
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

} // namespace nc
