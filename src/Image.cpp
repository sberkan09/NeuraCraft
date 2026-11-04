#include "Image.h"
#include "Backend.h"
#include "Parallel.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

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
