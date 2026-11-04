#include "Backend.h"
#include "Dense.h"
#include "Diffusion.h"
#include "Json.h"
#include "Parallel.h"
#include "Pretrained.h"
#include "Random.h"
#include "SafeTensors.h"

#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
bool exists(const std::string& path) { return static_cast<bool>(std::ifstream(path)); }

void siluInPlace(Matrix& m) {
    float* x = m.data();
    parallelFor(0, m.size(), 1 << 15, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
    });
}

void addInPlace(Matrix& a, const Matrix& b) {
    if (!a.sameShape(b)) throw std::logic_error("UNet: residual shape mismatch");
    float* x = a.data();
    const float* y = b.data();
    parallelFor(0, a.size(), 1 << 15, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) x[i] += y[i];
    });
}

// (C x HW) channel-major row <-> (HW x C) token rows.
Matrix toTokens(const Matrix& x, std::size_t C, std::size_t HW) { return Matrix(C, HW, std::vector<float>(x.data(), x.data() + C * HW)).transpose(); }
Matrix fromTokens(const Matrix& t) {
    const Matrix chw = t.transpose();
    return Matrix(1, chw.size(), std::vector<float>(chw.data(), chw.data() + chw.size()));
}
} // namespace

Matrix multiHeadAttention(const Matrix& q, const Matrix& k, const Matrix& v, std::size_t heads) {
    const std::size_t N = q.rows(), M = k.rows(), C = q.cols();
    if (heads == 0 || C % heads != 0 || k.cols() != C || v.cols() != C || v.rows() != M)
        throw std::invalid_argument("multiHeadAttention: shapes do not match");
    const std::size_t dh = C / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    Matrix out(N, C);
    const std::size_t chunk = std::max<std::size_t>(1, std::min(N, (std::size_t(1) << 22) / std::max<std::size_t>(1, M)));
    std::vector<float> scores(chunk * M);
    for (std::size_t h = 0; h < heads; ++h)
        for (std::size_t q0 = 0; q0 < N; q0 += chunk) {
            const std::size_t rows = std::min(chunk, N - q0);
            // scores = Q_h K_h^T (rows x M)
            gemm(false, true, rows, M, dh, q.data() + q0 * C + h * dh, C, k.data() + h * dh, C, scores.data(), M, false);
            parallelFor(0, rows, 4, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    float* s = scores.data() + r * M;
                    float m = -std::numeric_limits<float>::infinity();
                    for (std::size_t j = 0; j < M; ++j) m = std::max(m, s[j] *= scale);
                    float sum = 0.0f;
                    for (std::size_t j = 0; j < M; ++j) sum += s[j] = std::exp(s[j] - m);
                    const float inv = 1.0f / sum;
                    for (std::size_t j = 0; j < M; ++j) s[j] *= inv;
                }
            });
            // out[q0.., h] = scores V_h
            gemm(false, false, rows, dh, M, scores.data(), M, v.data() + h * dh, C, out.data() + q0 * C + h * dh, C, false);
        }
    return out;
}

Matrix timestepEmbedding(float timestep, std::size_t dim, bool flipSinToCos, float freqShift) {
    const std::size_t half = dim / 2;
    Matrix e(1, dim);
    for (std::size_t i = 0; i < half; ++i) {
        const double exponent = -std::log(10000.0) * static_cast<double>(i) / (static_cast<double>(half) - freqShift);
        const double a = static_cast<double>(timestep) * std::exp(exponent);
        const float s = static_cast<float>(std::sin(a)), c = static_cast<float>(std::cos(a));
        e(0, i) = flipSinToCos ? c : s;
        e(0, half + i) = flipSinToCos ? s : c;
    }
    return e;
}

// ---- Blocks -----------------------------------------------------------------------------------

namespace {
struct Weights {
    explicit Weights(const std::string& file) : file(file), st(file) {}
    Matrix take(const std::string& name, std::size_t rows, std::size_t cols) {
        Matrix m = st.load(name);
        if (lora) lora->apply(name, m);
        if (m.size() != rows * cols)
            throw std::runtime_error("loadUNet: " + name + " in " + file + " has " + std::to_string(m.size()) + " values, expected " +
                                     std::to_string(rows * cols));
        return Matrix(rows, cols, std::move(m.values()));
    }
    bool has(const std::string& name) const { return st.has(name); }
    void conv(Conv2D& c, const std::string& p) {
        const std::size_t out = c.outputShape().channels, patch = c.inputShape().channels * c.kernelSize() * c.kernelSize();
        c.weights() = take(p + ".weight", out, patch).transpose();
        c.bias() = take(p + ".bias", 1, out);
        if (int8) c.quantize();
    }
    void norm(GroupNorm& g, const std::string& p, std::size_t ch) {
        g.gamma() = take(p + ".weight", 1, ch);
        g.beta() = take(p + ".bias", 1, ch);
    }
    void layerNorm(LayerNorm& g, const std::string& p, std::size_t ch) {
        g.gamma() = take(p + ".weight", 1, ch);
        g.beta() = take(p + ".bias", 1, ch);
    }
    // PyTorch Linear (out x in, or out x in x 1 x 1) -> Dense (in x out); a missing bias is zero.
    void linear(Dense& d, const std::string& p) {
        d.weights() = take(p + ".weight", d.outputSize(), d.inputSize()).transpose();
        d.bias() = has(p + ".bias") ? take(p + ".bias", 1, d.outputSize()) : Matrix(1, d.outputSize());
        if (int8) d.quantize();
    }
    std::string file;
    SafeTensors st;
    bool int8 = false;
    LoraMerger* lora = nullptr;
};

struct TimeResnet {
    ImageShape in;
    std::size_t out;
    GroupNorm n1, n2;
    Conv2D c1, c2;
    Dense temb;
    std::unique_ptr<Conv2D> shortcut;

    TimeResnet(ImageShape input, std::size_t outChannels, std::size_t tembDim, std::size_t groups, float eps)
        : in(input), out(outChannels), n1(input, groups, eps), n2({outChannels, input.height, input.width}, groups, eps),
          c1(input, outChannels, 3, 1, 1), c2({outChannels, input.height, input.width}, outChannels, 3, 1, 1),
          temb(tembDim, outChannels) {
        if (input.channels != outChannels) shortcut = std::make_unique<Conv2D>(input, outChannels, 1);
    }
    void load(Weights& w, const std::string& p) {
        w.norm(n1, p + ".norm1", in.channels);
        w.conv(c1, p + ".conv1");
        w.linear(temb, p + ".time_emb_proj");
        w.norm(n2, p + ".norm2", out);
        w.conv(c2, p + ".conv2");
        if (shortcut) w.conv(*shortcut, p + ".conv_shortcut");
    }
    // tembAct: silu(time embedding), 1 x tembDim
    Matrix forward(const Matrix& x, const Matrix& tembAct) {
        Matrix h = n1.forward(x, false);
        siluInPlace(h);
        h = c1.forward(h, false);
        const Matrix t = temb.forward(tembAct, false);
        const std::size_t HW = in.height * in.width;
        parallelFor(0, out, 8, [&](std::size_t c0, std::size_t c1_) {
            for (std::size_t c = c0; c < c1_; ++c) {
                float* plane = h.data() + c * HW;
                const float add = t(0, c);
                for (std::size_t p = 0; p < HW; ++p) plane[p] += add;
            }
        });
        h = n2.forward(h, false);
        siluInPlace(h);
        h = c2.forward(h, false);
        if (shortcut) addInPlace(h, shortcut->forward(x, false));
        else addInPlace(h, x);
        return h;
    }
};

// diffusers Transformer2DModel with one BasicTransformerBlock: self-attention,
// cross-attention to the text, GEGLU feed-forward; linear or 1x1 projections.
struct CrossTransformer {
    ImageShape shape;
    std::size_t heads;
    GroupNorm norm;
    Dense projIn, projOut;
    LayerNorm ln1, ln2, ln3;
    Dense q1, k1, v1, o1, q2, k2, v2, o2, ff1, ff2;

    CrossTransformer(ImageShape s, std::size_t h, std::size_t contextDim, std::size_t groups)
        : shape(s), heads(h), norm(s, groups, 1e-6f), projIn(s.channels, s.channels), projOut(s.channels, s.channels),
          ln1(s.channels), ln2(s.channels), ln3(s.channels), q1(s.channels, s.channels), k1(s.channels, s.channels),
          v1(s.channels, s.channels), o1(s.channels, s.channels), q2(s.channels, s.channels), k2(contextDim, s.channels),
          v2(contextDim, s.channels), o2(s.channels, s.channels), ff1(s.channels, 8 * s.channels), ff2(4 * s.channels, s.channels) {}

    void load(Weights& w, const std::string& p) {
        const std::size_t C = shape.channels;
        w.norm(norm, p + ".norm", C);
        w.linear(projIn, p + ".proj_in");
        w.linear(projOut, p + ".proj_out");
        const std::string b = p + ".transformer_blocks.0";
        w.layerNorm(ln1, b + ".norm1", C);
        w.layerNorm(ln2, b + ".norm2", C);
        w.layerNorm(ln3, b + ".norm3", C);
        w.linear(q1, b + ".attn1.to_q");
        w.linear(k1, b + ".attn1.to_k");
        w.linear(v1, b + ".attn1.to_v");
        w.linear(o1, b + ".attn1.to_out.0");
        w.linear(q2, b + ".attn2.to_q");
        w.linear(k2, b + ".attn2.to_k");
        w.linear(v2, b + ".attn2.to_v");
        w.linear(o2, b + ".attn2.to_out.0");
        w.linear(ff1, b + ".ff.net.0.proj");
        w.linear(ff2, b + ".ff.net.2");
    }
    Matrix forward(const Matrix& x, const Matrix& context) {
        const std::size_t C = shape.channels, HW = shape.height * shape.width;
        Matrix h = projIn.forward(toTokens(norm.forward(x, false), C, HW), false);  // HW x C
        Matrix a = ln1.forward(h, false);
        addInPlace(h, o1.forward(multiHeadAttention(q1.forward(a, false), k1.forward(a, false), v1.forward(a, false), heads), false));
        a = ln2.forward(h, false);
        addInPlace(h, o2.forward(multiHeadAttention(q2.forward(a, false), k2.forward(context, false), v2.forward(context, false), heads),
                                 false));
        a = ln3.forward(h, false);
        const Matrix g = ff1.forward(a, false);  // HW x 8C: [hidden | gate]
        Matrix geglu(HW, 4 * C);
        parallelFor(0, HW, 16, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t r = r0; r < r1; ++r) {
                const float* row = g.data() + r * 8 * C;
                float* o = geglu.data() + r * 4 * C;
                for (std::size_t j = 0; j < 4 * C; ++j) {
                    const float gate = row[4 * C + j];
                    o[j] = row[j] * 0.5f * gate * (1.0f + std::erf(gate * 0.70710678118654752f));
                }
            }
        });
        addInPlace(h, ff2.forward(geglu, false));
        Matrix out = fromTokens(projOut.forward(h, false));
        addInPlace(out, x);
        return out;
    }
};

// AnimateDiff's motion module (diffusers AnimateDiffTransformer3D with one
// BasicTransformerBlock): for every pixel, the frames form a sequence of
// channel vectors that attend to each other.
struct TemporalTransformer {
    static constexpr std::size_t kMaxFrames = 32;
    ImageShape shape;
    std::size_t heads, groups;
    Matrix gamma, beta;  // GroupNorm over channels x frames x pixels
    Dense projIn, projOut;
    LayerNorm ln1, ln2, ln3;
    Dense q1, k1, v1, o1, q2, k2, v2, o2, ff1, ff2;
    std::vector<float> pe;  // kMaxFrames x C sinusoidal frame positions

    TemporalTransformer(ImageShape s, std::size_t h, std::size_t g)
        : shape(s), heads(h), groups(g), projIn(s.channels, s.channels), projOut(s.channels, s.channels), ln1(s.channels),
          ln2(s.channels), ln3(s.channels), q1(s.channels, s.channels), k1(s.channels, s.channels), v1(s.channels, s.channels),
          o1(s.channels, s.channels), q2(s.channels, s.channels), k2(s.channels, s.channels), v2(s.channels, s.channels),
          o2(s.channels, s.channels), ff1(s.channels, 8 * s.channels), ff2(4 * s.channels, s.channels) {
        const std::size_t C = s.channels;
        pe.assign(kMaxFrames * C, 0.0f);
        for (std::size_t f = 0; f < kMaxFrames; ++f)
            for (std::size_t i = 0; i < C; i += 2) {
                const double a = static_cast<double>(f) * std::exp(static_cast<double>(i) * (-std::log(10000.0) / static_cast<double>(C)));
                pe[f * C + i] = static_cast<float>(std::sin(a));
                if (i + 1 < C) pe[f * C + i + 1] = static_cast<float>(std::cos(a));
            }
    }

    void load(Weights& w, const std::string& p) {
        const std::size_t C = shape.channels;
        gamma = w.take(p + ".norm.weight", 1, C);
        beta = w.take(p + ".norm.bias", 1, C);
        w.linear(projIn, p + ".proj_in");
        w.linear(projOut, p + ".proj_out");
        const std::string b = p + ".transformer_blocks.0";
        w.layerNorm(ln1, b + ".norm1", C);
        w.layerNorm(ln2, b + ".norm2", C);
        w.layerNorm(ln3, b + ".norm3", C);
        w.linear(q1, b + ".attn1.to_q");
        w.linear(k1, b + ".attn1.to_k");
        w.linear(v1, b + ".attn1.to_v");
        w.linear(o1, b + ".attn1.to_out.0");
        w.linear(q2, b + ".attn2.to_q");
        w.linear(k2, b + ".attn2.to_k");
        w.linear(v2, b + ".attn2.to_v");
        w.linear(o2, b + ".attn2.to_out.0");
        w.linear(ff1, b + ".ff.net.0.proj");
        w.linear(ff2, b + ".ff.net.2");
        if (w.has(b + ".pos_embed.pe")) {  // the stored table, when the file has one
            const Matrix stored = w.take(b + ".pos_embed.pe", w.st.info(b + ".pos_embed.pe").elements() / C, C);
            if (stored.rows() < kMaxFrames) throw std::runtime_error("loadMotionModules: positional table shorter than 32 frames");
            std::copy(stored.data(), stored.data() + kMaxFrames * C, pe.begin());
        }
    }

    // Self-attention within each group of F consecutive rows (one pixel's frames).
    Matrix frameAttention(const Matrix& q, const Matrix& k, const Matrix& v, std::size_t F) const {
        const std::size_t rows = q.rows(), C = q.cols(), dh = C / heads, P = rows / F;
        const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
        Matrix out(rows, C);
        parallelFor(0, P, 8, [&](std::size_t p0, std::size_t p1) {
            std::vector<float> s(F);
            for (std::size_t p = p0; p < p1; ++p)
                for (std::size_t h = 0; h < heads; ++h)
                    for (std::size_t i = 0; i < F; ++i) {
                        const float* qi = q.data() + (p * F + i) * C + h * dh;
                        float m = -std::numeric_limits<float>::infinity();
                        for (std::size_t j = 0; j < F; ++j) {
                            const float* kj = k.data() + (p * F + j) * C + h * dh;
                            float d = 0.0f;
                            for (std::size_t e = 0; e < dh; ++e) d += qi[e] * kj[e];
                            m = std::max(m, s[j] = d * scale);
                        }
                        float sum = 0.0f;
                        for (std::size_t j = 0; j < F; ++j) sum += s[j] = std::exp(s[j] - m);
                        float* o = out.data() + (p * F + i) * C + h * dh;
                        for (std::size_t j = 0; j < F; ++j) {
                            const float wj = s[j] / sum;
                            const float* vj = v.data() + (p * F + j) * C + h * dh;
                            for (std::size_t e = 0; e < dh; ++e) o[e] += wj * vj[e];
                        }
                    }
        });
        return out;
    }

    // frames: F matrices of 1 x (C H W), updated in place (output + residual).
    void forward(std::vector<Matrix>& frames) {
        const std::size_t F = frames.size(), C = shape.channels, HW = shape.height * shape.width, cg = C / groups;
        if (F > kMaxFrames) throw std::invalid_argument("motion module: at most 32 frames");
        // Group statistics over the group's channels, all frames and all pixels.
        std::vector<float> mean(groups), inv(groups);
        parallelFor(0, groups, 1, [&](std::size_t g0, std::size_t g1) {
            for (std::size_t g = g0; g < g1; ++g) {
                double s = 0, q = 0;
                for (const Matrix& fr : frames)
                    for (std::size_t c = g * cg; c < (g + 1) * cg; ++c)
                        for (std::size_t p = 0; p < HW; ++p) {
                            const double v = fr.data()[c * HW + p];
                            s += v;
                            q += v * v;
                        }
                const double n = static_cast<double>(F * cg * HW), mu = s / n;
                mean[g] = static_cast<float>(mu);
                inv[g] = static_cast<float>(1.0 / std::sqrt(std::max(q / n - mu * mu, 0.0) + 1e-6));
            }
        });
        // Pixels in chunks (temporal attention never crosses pixels), so the
        // feed-forward's 8C-wide rows stay within a few tens of MB.
        const std::size_t P = std::max<std::size_t>(1, ((std::size_t(1) << 23) / (8 * C)) / F);
        for (std::size_t p0 = 0; p0 < HW; p0 += P) {
            const std::size_t np = std::min(P, HW - p0), rows = np * F;
            Matrix x(rows, C);
            parallelFor(0, C, 8, [&](std::size_t c0, std::size_t c1) {
                for (std::size_t c = c0; c < c1; ++c) {
                    const std::size_t g = c / cg;
                    const float a = inv[g] * gamma.data()[c], b = beta.data()[c] - mean[g] * a;
                    for (std::size_t f = 0; f < F; ++f) {
                        const float* src = frames[f].data() + c * HW + p0;
                        for (std::size_t p = 0; p < np; ++p) x.data()[(p * F + f) * C + c] = src[p] * a + b;
                    }
                }
            });
            Matrix h = projIn.forward(x, false);
            auto withPositions = [&](Matrix a) {
                for (std::size_t r = 0; r < rows; ++r) {
                    const float* e = pe.data() + (r % F) * C;
                    float* row = a.data() + r * C;
                    for (std::size_t c = 0; c < C; ++c) row[c] += e[c];
                }
                return a;
            };
            Matrix a = withPositions(ln1.forward(h, false));
            addInPlace(h, o1.forward(frameAttention(q1.forward(a, false), k1.forward(a, false), v1.forward(a, false), F), false));
            a = withPositions(ln2.forward(h, false));
            addInPlace(h, o2.forward(frameAttention(q2.forward(a, false), k2.forward(a, false), v2.forward(a, false), F), false));
            a = ln3.forward(h, false);
            const Matrix g = ff1.forward(a, false);
            Matrix geglu(rows, 4 * C);
            parallelFor(0, rows, 16, [&](std::size_t r0, std::size_t r1) {
                for (std::size_t r = r0; r < r1; ++r) {
                    const float* row = g.data() + r * 8 * C;
                    float* o = geglu.data() + r * 4 * C;
                    for (std::size_t j = 0; j < 4 * C; ++j) {
                        const float gate = row[4 * C + j];
                        o[j] = row[j] * 0.5f * gate * (1.0f + std::erf(gate * 0.70710678118654752f));
                    }
                }
            });
            addInPlace(h, ff2.forward(geglu, false));
            const Matrix y = projOut.forward(h, false);
            parallelFor(0, C, 8, [&](std::size_t c0, std::size_t c1) {
                for (std::size_t c = c0; c < c1; ++c)
                    for (std::size_t f = 0; f < F; ++f) {
                        float* dst = frames[f].data() + c * HW + p0;
                        for (std::size_t p = 0; p < np; ++p) dst[p] += y.data()[(p * F + f) * C + c];
                    }
            });
        }
    }
};

struct DownBlock {
    std::vector<std::unique_ptr<TimeResnet>> resnets;
    std::vector<std::unique_ptr<CrossTransformer>> attentions;
    std::vector<std::unique_ptr<TemporalTransformer>> motions;
    std::unique_ptr<Conv2D> downsample;
};
struct UpBlock {
    std::vector<std::unique_ptr<TimeResnet>> resnets;
    std::vector<std::unique_ptr<CrossTransformer>> attentions;
    std::vector<std::unique_ptr<TemporalTransformer>> motions;
    std::unique_ptr<Upsample2D> upsample;
    std::unique_ptr<Conv2D> upConv;
};

Matrix concatChannels(const Matrix& a, const Matrix& b) {
    Matrix out(1, a.size() + b.size());
    std::copy(a.data(), a.data() + a.size(), out.data());
    std::copy(b.data(), b.data() + b.size(), out.data() + a.size());
    return out;
}
} // namespace

struct UNet2DCondition::Impl {
    ImageShape latent;
    std::size_t contextDim = 0, timeDim = 0, params = 0;
    bool flipSinToCos = true;
    float freqShift = 0.0f;
    std::unique_ptr<Dense> t1, t2;
    std::unique_ptr<Conv2D> convIn, convOut;
    std::vector<DownBlock> downs;
    std::unique_ptr<TimeResnet> mid1, mid2;
    std::unique_ptr<CrossTransformer> midAttention;
    std::unique_ptr<TemporalTransformer> midMotion;
    std::vector<UpBlock> ups;
    std::unique_ptr<GroupNorm> normOut;
    std::size_t groups = 32;
    // Down / up block shapes after each resnet, for building motion modules.
    std::vector<std::vector<ImageShape>> downShapes, upShapes;
};

UNet2DCondition::UNet2DCondition() = default;
UNet2DCondition::~UNet2DCondition() = default;
UNet2DCondition::UNet2DCondition(UNet2DCondition&&) noexcept = default;
UNet2DCondition& UNet2DCondition::operator=(UNet2DCondition&&) noexcept = default;

ImageShape UNet2DCondition::latentShape() const { return impl_->latent; }
std::size_t UNet2DCondition::contextDim() const { return impl_->contextDim; }
std::size_t UNet2DCondition::parameterCount() const { return impl_->params; }
bool UNet2DCondition::hasMotionModules() const { return impl_->midMotion != nullptr; }

Matrix UNet2DCondition::forwardFrames(const Matrix& latents, float timestep, const Matrix& context) {
    return forwardFrames(latents, timestep, std::vector<Matrix>{context});
}

Matrix UNet2DCondition::forwardFrames(const Matrix& latents, float timestep, const std::vector<Matrix>& contexts) {
    Impl& m = *impl_;
    const std::size_t F = latents.rows();
    if (F == 0 || latents.cols() != m.latent.size())
        throw std::invalid_argument("UNet2DCondition::forwardFrames: expected F x " + std::to_string(m.latent.size()) + " latents");
    if (contexts.size() != 1 && contexts.size() != F)
        throw std::invalid_argument("UNet2DCondition::forwardFrames: give one context or one per frame");
    for (const Matrix& c : contexts)
        if (c.cols() != m.contextDim)
            throw std::invalid_argument("UNet2DCondition::forwardFrames: context must have " + std::to_string(m.contextDim) + " columns");
    auto ctx = [&](std::size_t f) -> const Matrix& { return contexts[contexts.size() == 1 ? 0 : f]; };
    Matrix out(F, m.latent.size());
    if (!hasMotionModules()) {
        for (std::size_t f = 0; f < F; ++f) {
            Matrix one(1, m.latent.size());
            std::copy(latents.data() + f * one.size(), latents.data() + (f + 1) * one.size(), one.data());
            const Matrix eps = forward(one, timestep, ctx(f));
            std::copy(eps.data(), eps.data() + eps.size(), out.data() + f * eps.size());
        }
        return out;
    }
    Matrix temb = m.t1->forward(timestepEmbedding(timestep, m.timeDim, m.flipSinToCos, m.freqShift), false);
    siluInPlace(temb);
    temb = m.t2->forward(temb, false);
    siluInPlace(temb);

    std::vector<Matrix> h(F);
    for (std::size_t f = 0; f < F; ++f) {
        Matrix one(1, m.latent.size());
        std::copy(latents.data() + f * one.size(), latents.data() + (f + 1) * one.size(), one.data());
        h[f] = m.convIn->forward(one, false);
    }
    std::vector<std::vector<Matrix>> skips{h};
    auto each = [&](auto&& fn) {
        for (Matrix& x : h) x = fn(x);
    };
    auto attend = [&](CrossTransformer& a) {
        for (std::size_t f = 0; f < F; ++f) h[f] = a.forward(h[f], ctx(f));
    };
    for (auto& d : m.downs) {
        for (std::size_t j = 0; j < d.resnets.size(); ++j) {
            each([&](const Matrix& x) { return d.resnets[j]->forward(x, temb); });
            if (!d.attentions.empty()) attend(*d.attentions[j]);
            d.motions[j]->forward(h);
            skips.push_back(h);
        }
        if (d.downsample) {
            each([&](const Matrix& x) { return d.downsample->forward(x, false); });
            skips.push_back(h);
        }
    }
    each([&](const Matrix& x) { return m.mid1->forward(x, temb); });
    attend(*m.midAttention);
    m.midMotion->forward(h);
    each([&](const Matrix& x) { return m.mid2->forward(x, temb); });
    for (auto& u : m.ups) {
        for (std::size_t j = 0; j < u.resnets.size(); ++j) {
            std::vector<Matrix>& s = skips.back();
            for (std::size_t f = 0; f < F; ++f) h[f] = u.resnets[j]->forward(concatChannels(h[f], s[f]), temb);
            skips.pop_back();
            if (!u.attentions.empty()) attend(*u.attentions[j]);
            u.motions[j]->forward(h);
        }
        if (u.upsample) each([&](const Matrix& x) { return u.upConv->forward(u.upsample->forward(x, false), false); });
    }
    for (std::size_t f = 0; f < F; ++f) {
        Matrix x = m.normOut->forward(h[f], false);
        siluInPlace(x);
        x = m.convOut->forward(x, false);
        std::copy(x.data(), x.data() + x.size(), out.data() + f * x.size());
    }
    return out;
}

void loadMotionModules(UNet2DCondition& unet, const std::string& file, bool int8) {
    UNet2DCondition::Impl& m = *unet.impl_;
    Weights w(file);
    w.int8 = int8;
    SkipParameterInit skip;
    // Heads: the adapter's config is not in the file; diffusers' default (and
    // every released AnimateDiff v1 / v2 / Lightning module) uses 8.
    const std::size_t heads = 8;
    for (std::size_t i = 0; i < m.downs.size(); ++i)
        for (std::size_t j = 0; j < m.downShapes[i].size(); ++j) {
            m.downs[i].motions.push_back(std::make_unique<TemporalTransformer>(m.downShapes[i][j], heads, m.groups));
            m.downs[i].motions.back()->load(w, "down_blocks." + std::to_string(i) + ".motion_modules." + std::to_string(j));
        }
    if (!w.has("mid_block.motion_modules.0.norm.weight"))
        throw std::runtime_error("loadMotionModules: " + file + " has no mid block motion module (AnimateDiff v1 is not supported)");
    m.midMotion = std::make_unique<TemporalTransformer>(m.downShapes.back().back(), heads, m.groups);
    m.midMotion->load(w, "mid_block.motion_modules.0");
    for (std::size_t i = 0; i < m.ups.size(); ++i)
        for (std::size_t j = 0; j < m.upShapes[i].size(); ++j) {
            m.ups[i].motions.push_back(std::make_unique<TemporalTransformer>(m.upShapes[i][j], heads, m.groups));
            m.ups[i].motions.back()->load(w, "up_blocks." + std::to_string(i) + ".motion_modules." + std::to_string(j));
        }
    for (const auto& name : w.st.names()) m.params += w.st.info(name).elements();
}

Matrix UNet2DCondition::forward(const Matrix& latent, float timestep, const Matrix& context) {
    Impl& m = *impl_;
    if (latent.rows() != 1 || latent.cols() != m.latent.size())
        throw std::invalid_argument("UNet2DCondition::forward: expected a 1 x " + std::to_string(m.latent.size()) + " latent");
    if (context.cols() != m.contextDim)
        throw std::invalid_argument("UNet2DCondition::forward: context must have " + std::to_string(m.contextDim) + " columns");
    Matrix temb = m.t1->forward(timestepEmbedding(timestep, m.timeDim, m.flipSinToCos, m.freqShift), false);
    siluInPlace(temb);
    temb = m.t2->forward(temb, false);
    siluInPlace(temb);  // every resnet applies SiLU before its projection

    Matrix h = m.convIn->forward(latent, false);
    std::vector<Matrix> skips{h};
    for (auto& d : m.downs) {
        for (std::size_t j = 0; j < d.resnets.size(); ++j) {
            h = d.resnets[j]->forward(h, temb);
            if (!d.attentions.empty()) h = d.attentions[j]->forward(h, context);
            skips.push_back(h);
        }
        if (d.downsample) {
            h = d.downsample->forward(h, false);
            skips.push_back(h);
        }
    }
    h = m.mid1->forward(h, temb);
    h = m.midAttention->forward(h, context);
    h = m.mid2->forward(h, temb);
    for (auto& u : m.ups) {
        for (std::size_t j = 0; j < u.resnets.size(); ++j) {
            h = u.resnets[j]->forward(concatChannels(h, skips.back()), temb);
            skips.pop_back();
            if (!u.attentions.empty()) h = u.attentions[j]->forward(h, context);
        }
        if (u.upsample) h = u.upConv->forward(u.upsample->forward(h, false), false);
    }
    h = m.normOut->forward(h, false);
    siluInPlace(h);
    return m.convOut->forward(h, false);
}

UNet2DCondition loadUNet(const std::string& directory, std::size_t latentWidth, std::size_t latentHeight, bool int8,
                         const std::vector<Lora>& loras) {
    const std::string dir = exists(directory + "/config.json") ? directory : directory + "/unet";
    const json::Value c = json::parse(readFile(dir + "/config.json"));
    std::string file = dir + "/diffusion_pytorch_model.safetensors";
    if (!exists(file)) file = dir + "/diffusion_pytorch_model.fp16.safetensors";
    if (!exists(file)) throw std::runtime_error("loadUNet: no diffusion_pytorch_model(.fp16).safetensors in " + dir);

    auto list = [&](const char* key) {
        std::vector<std::size_t> v;
        const json::Value& j = c.at(key);
        for (const auto& e : j.array) v.push_back(static_cast<std::size_t>(e.number));
        return v;
    };
    auto strings = [&](const char* key) {
        std::vector<std::string> v;
        for (const auto& e : c.at(key).array) v.push_back(e.string);
        return v;
    };
    auto unsupported = [&](const char* key, auto accept) {
        const json::Value* v = c.find(key);
        if (v && v->type != json::Value::Type::Null && !accept(*v))
            throw std::runtime_error(std::string("loadUNet: ") + key + " is not supported");
    };
    unsupported("class_embed_type", [](const json::Value&) { return false; });
    unsupported("addition_embed_type", [](const json::Value&) { return false; });
    unsupported("time_cond_proj_dim", [](const json::Value&) { return false; });
    unsupported("resnet_time_scale_shift", [](const json::Value& v) { return v.string == "default"; });
    unsupported("time_embedding_type", [](const json::Value& v) { return v.string == "positional"; });
    unsupported("transformer_layers_per_block", [](const json::Value& v) { return v.isNumber() && v.number == 1; });
    unsupported("act_fn", [](const json::Value& v) { return v.string == "silu"; });
    unsupported("mid_block_type", [](const json::Value& v) { return v.string == "UNetMidBlock2DCrossAttn"; });

    const std::vector<std::size_t> ch = list("block_out_channels");
    const std::size_t L = static_cast<std::size_t>(c.numberOr("layers_per_block", 2));
    const std::size_t groups = static_cast<std::size_t>(c.numberOr("norm_num_groups", 32));
    const float eps = static_cast<float>(c.numberOr("norm_eps", 1e-5));
    const std::size_t ctx = static_cast<std::size_t>(c.at("cross_attention_dim").number);
    // diffusers passes attention_head_dim on as the number of heads when num_attention_heads is unset.
    const json::Value* headsKey = c.find("num_attention_heads");
    if (!headsKey || headsKey->type == json::Value::Type::Null) headsKey = &c.at("attention_head_dim");
    std::vector<std::size_t> heads;
    if (headsKey->isNumber()) heads.assign(ch.size(), static_cast<std::size_t>(headsKey->number));
    else for (const auto& e : headsKey->array) heads.push_back(static_cast<std::size_t>(e.number));
    const std::vector<std::string> downTypes = strings("down_block_types"), upTypes = strings("up_block_types");
    if (heads.size() != ch.size() || downTypes.size() != ch.size() || upTypes.size() != ch.size())
        throw std::runtime_error("loadUNet: block lists disagree in length");
    const std::size_t factor = std::size_t(1) << (ch.size() - 1);
    if (latentWidth == 0 || latentHeight == 0 || latentWidth % factor || latentHeight % factor)
        throw std::invalid_argument("loadUNet: latent width and height must be multiples of " + std::to_string(factor));

    UNet2DCondition unet;
    unet.impl_ = std::make_unique<UNet2DCondition::Impl>();
    auto& m = *unet.impl_;
    m.latent = {static_cast<std::size_t>(c.numberOr("in_channels", 4)), latentHeight, latentWidth};
    m.contextDim = ctx;
    m.timeDim = ch[0];
    m.flipSinToCos = !c.find("flip_sin_to_cos") || c.at("flip_sin_to_cos").boolean;
    m.freqShift = static_cast<float>(c.numberOr("freq_shift", 0));
    m.groups = groups;
    const std::size_t tembDim = 4 * ch[0];
    Weights w(file);
    w.int8 = int8;
    LoraMerger merger(loras, "unet");
    if (!merger.empty()) w.lora = &merger;
    SkipParameterInit skip;

    m.t1 = std::make_unique<Dense>(m.timeDim, tembDim);
    w.linear(*m.t1, "time_embedding.linear_1");
    m.t2 = std::make_unique<Dense>(tembDim, tembDim);
    w.linear(*m.t2, "time_embedding.linear_2");
    m.convIn = std::make_unique<Conv2D>(m.latent, ch[0], 3, 1, 1);
    w.conv(*m.convIn, "conv_in");

    ImageShape s = m.convIn->outputShape();
    std::vector<std::size_t> skipChannels{s.channels};
    for (std::size_t i = 0; i < ch.size(); ++i) {
        DownBlock d;
        const std::string p = "down_blocks." + std::to_string(i);
        const bool cross = downTypes[i] == "CrossAttnDownBlock2D";
        if (!cross && downTypes[i] != "DownBlock2D") throw std::runtime_error("loadUNet: unsupported block " + downTypes[i]);
        m.downShapes.emplace_back();
        for (std::size_t j = 0; j < L; ++j) {
            d.resnets.push_back(std::make_unique<TimeResnet>(s, ch[i], tembDim, groups, eps));
            d.resnets.back()->load(w, p + ".resnets." + std::to_string(j));
            s.channels = ch[i];
            m.downShapes.back().push_back(s);
            if (cross) {
                d.attentions.push_back(std::make_unique<CrossTransformer>(s, heads[i], ctx, groups));
                d.attentions.back()->load(w, p + ".attentions." + std::to_string(j));
            }
            skipChannels.push_back(s.channels);
        }
        if (i + 1 < ch.size()) {
            d.downsample = std::make_unique<Conv2D>(s, s.channels, 3, 2, 1);
            w.conv(*d.downsample, p + ".downsamplers.0.conv");
            s = d.downsample->outputShape();
            skipChannels.push_back(s.channels);
        }
        m.downs.push_back(std::move(d));
    }
    m.mid1 = std::make_unique<TimeResnet>(s, s.channels, tembDim, groups, eps);
    m.mid1->load(w, "mid_block.resnets.0");
    m.midAttention = std::make_unique<CrossTransformer>(s, heads.back(), ctx, groups);
    m.midAttention->load(w, "mid_block.attentions.0");
    m.mid2 = std::make_unique<TimeResnet>(s, s.channels, tembDim, groups, eps);
    m.mid2->load(w, "mid_block.resnets.1");

    for (std::size_t i = 0; i < ch.size(); ++i) {
        UpBlock u;
        const std::string p = "up_blocks." + std::to_string(i);
        const std::size_t out = ch[ch.size() - 1 - i];
        const bool cross = upTypes[i] == "CrossAttnUpBlock2D";
        if (!cross && upTypes[i] != "UpBlock2D") throw std::runtime_error("loadUNet: unsupported block " + upTypes[i]);
        m.upShapes.emplace_back();
        for (std::size_t j = 0; j <= L; ++j) {
            const ImageShape in{s.channels + skipChannels.back(), s.height, s.width};
            skipChannels.pop_back();
            u.resnets.push_back(std::make_unique<TimeResnet>(in, out, tembDim, groups, eps));
            u.resnets.back()->load(w, p + ".resnets." + std::to_string(j));
            s.channels = out;
            m.upShapes.back().push_back(s);
            if (cross) {
                u.attentions.push_back(std::make_unique<CrossTransformer>(s, heads[ch.size() - 1 - i], ctx, groups));
                u.attentions.back()->load(w, p + ".attentions." + std::to_string(j));
            }
        }
        if (i + 1 < ch.size()) {
            u.upsample = std::make_unique<Upsample2D>(s, 2);
            s = u.upsample->outputShape();
            u.upConv = std::make_unique<Conv2D>(s, s.channels, 3, 1, 1);
            w.conv(*u.upConv, p + ".upsamplers.0.conv");
        }
        m.ups.push_back(std::move(u));
    }
    m.normOut = std::make_unique<GroupNorm>(s, groups, eps);
    w.norm(*m.normOut, "conv_norm_out", s.channels);
    m.convOut = std::make_unique<Conv2D>(s, static_cast<std::size_t>(c.numberOr("out_channels", 4)), 3, 1, 1);
    w.conv(*m.convOut, "conv_out");
    if (merger.layers() && !merger.merged())
        throw std::runtime_error("loadUNet: the LoRA matches no layer of this UNet (made for another model?)");
    for (const auto& name : w.st.names()) m.params += w.st.info(name).elements();
    return unet;
}

// ---- Euler scheduler -------------------------------------------------------------------------

namespace {
EulerScheduler makeEuler(std::size_t steps, std::size_t trainSteps, double betaStart, double betaEnd, const std::string& schedule,
                         const std::string& spacing, std::size_t offset) {
    if (steps == 0 || steps > trainSteps) throw std::invalid_argument("EulerScheduler: steps must be in 1..train steps");
    std::vector<double> sigmasAll(trainSteps);
    double cumprod = 1.0;
    for (std::size_t i = 0; i < trainSteps; ++i) {
        const double f = trainSteps > 1 ? static_cast<double>(i) / static_cast<double>(trainSteps - 1) : 0.0;
        double beta;
        if (schedule == "scaled_linear") {
            const double r = std::sqrt(betaStart) + f * (std::sqrt(betaEnd) - std::sqrt(betaStart));
            beta = r * r;
        } else if (schedule == "linear") {
            beta = betaStart + f * (betaEnd - betaStart);
        } else {
            throw std::runtime_error("EulerScheduler: beta_schedule '" + schedule + "' is not supported");
        }
        cumprod *= 1.0 - beta;
        sigmasAll[i] = std::sqrt((1.0 - cumprod) / cumprod);
    }
    EulerScheduler s;
    const double T = static_cast<double>(trainSteps);
    for (std::size_t i = 0; i < steps; ++i) {
        double t;
        if (spacing == "trailing") t = std::round(T - static_cast<double>(i) * T / static_cast<double>(steps)) - 1.0;
        else if (spacing == "leading")
            t = static_cast<double>((steps - 1 - i) * (trainSteps / steps) + offset);
        else if (spacing == "linspace")
            t = steps == 1 ? T - 1.0 : (T - 1.0) * static_cast<double>(steps - 1 - i) / static_cast<double>(steps - 1);
        else throw std::runtime_error("EulerScheduler: timestep_spacing '" + spacing + "' is not supported");
        s.timesteps.push_back(t);
        // Linear interpolation of the training sigmas at fractional timesteps.
        const double tc = std::min(std::max(t, 0.0), T - 1.0);
        const std::size_t lo = static_cast<std::size_t>(std::floor(tc)), hi = std::min(lo + 1, trainSteps - 1);
        const double frac = tc - static_cast<double>(lo);
        s.sigmas.push_back(sigmasAll[lo] * (1.0 - frac) + sigmasAll[hi] * frac);
    }
    s.sigmas.push_back(0.0);
    const double maxSigma = *std::max_element(s.sigmas.begin(), s.sigmas.end());
    s.initNoiseSigma = spacing == "linspace" || spacing == "trailing" ? maxSigma : std::sqrt(maxSigma * maxSigma + 1.0);
    return s;
}
} // namespace

EulerScheduler EulerScheduler::make(std::size_t steps, const std::string& schedule, const std::string& spacing, double betaStart,
                                    double betaEnd, std::size_t trainSteps, std::size_t offset) {
    return makeEuler(steps, trainSteps, betaStart, betaEnd, schedule, spacing, offset);
}

EulerScheduler EulerScheduler::standard(std::size_t steps) {
    return makeEuler(steps, 1000, 0.00085, 0.012, "scaled_linear", "trailing", 1);
}

EulerScheduler EulerScheduler::fromConfig(const std::string& path, std::size_t steps) {
    const json::Value c = json::parse(readFile(path));
    if (const json::Value* p = c.find("prediction_type"); p && p->string != "epsilon")
        throw std::runtime_error("EulerScheduler: prediction_type '" + p->string + "' is not supported");
    if (const json::Value* k = c.find("use_karras_sigmas"); k && k->boolean)
        throw std::runtime_error("EulerScheduler: Karras sigmas are not supported");
    return makeEuler(steps, static_cast<std::size_t>(c.numberOr("num_train_timesteps", 1000)), c.numberOr("beta_start", 0.00085),
                     c.numberOr("beta_end", 0.012), c.find("beta_schedule") ? c.at("beta_schedule").string : "scaled_linear",
                     c.find("timestep_spacing") ? c.at("timestep_spacing").string : "leading",
                     static_cast<std::size_t>(c.numberOr("steps_offset", 0)));
}

EulerScheduler EulerScheduler::lcm(std::size_t steps, const std::string& schedule, double betaStart, double betaEnd,
                                   std::size_t originalSteps, std::size_t trainSteps) {
    if (steps == 0 || steps > originalSteps || originalSteps == 0 || originalSteps > trainSteps)
        throw std::invalid_argument("EulerScheduler::lcm: need 1 <= steps <= original steps <= train steps");
    std::vector<double> sigmasAll(trainSteps);
    double cumprod = 1.0;
    for (std::size_t i = 0; i < trainSteps; ++i) {
        const double f = trainSteps > 1 ? static_cast<double>(i) / static_cast<double>(trainSteps - 1) : 0.0;
        double beta;
        if (schedule == "scaled_linear") {
            const double r = std::sqrt(betaStart) + f * (std::sqrt(betaEnd) - std::sqrt(betaStart));
            beta = r * r;
        } else if (schedule == "linear") beta = betaStart + f * (betaEnd - betaStart);
        else throw std::runtime_error("EulerScheduler: beta_schedule '" + schedule + "' is not supported");
        cumprod *= 1.0 - beta;
        sigmasAll[i] = std::sqrt((1.0 - cumprod) / cumprod);
    }
    // LCM's original timesteps (k, 2k, ..., trainSteps) - 1, reversed, every (originalSteps / steps)-th.
    const std::size_t k = trainSteps / originalSteps, skip = originalSteps / steps;
    EulerScheduler s;
    s.consistency = true;
    for (std::size_t i = 0; i < steps; ++i) {
        const std::size_t t = (originalSteps - i * skip) * k - 1;
        s.timesteps.push_back(static_cast<double>(t));
        s.sigmas.push_back(sigmasAll[t]);
    }
    s.sigmas.push_back(0.0);
    s.initNoiseSigma = std::sqrt(s.sigmas[0] * s.sigmas[0] + 1.0);  // unit noise in the model's (VP) space
    return s;
}

EulerScheduler EulerScheduler::lcmFromConfig(const std::string& path, std::size_t steps) {
    const json::Value c = json::parse(readFile(path));
    if (const json::Value* p = c.find("prediction_type"); p && p->string != "epsilon")
        throw std::runtime_error("EulerScheduler: prediction_type '" + p->string + "' is not supported");
    return lcm(steps, c.find("beta_schedule") ? c.at("beta_schedule").string : "scaled_linear", c.numberOr("beta_start", 0.00085),
               c.numberOr("beta_end", 0.012), static_cast<std::size_t>(c.numberOr("original_inference_steps", 50)),
               static_cast<std::size_t>(c.numberOr("num_train_timesteps", 1000)));
}

double EulerScheduler::inputScale(std::size_t i) const { return 1.0 / std::sqrt(sigmas.at(i) * sigmas.at(i) + 1.0); }

void EulerScheduler::step(std::size_t i, const Matrix& noise, Matrix& latent) const {
    if (!noise.sameShape(latent)) throw std::invalid_argument("EulerScheduler::step: shapes differ");
    if (consistency) {
        // The latent is kept in the sigma (VE) form x = x0 + sigma n, as for Euler; the model's
        // (VP) sample is x / sqrt(sigma^2 + 1). LCM: denoised = c_out x0 + c_skip sample.
        const double sigma = sigmas.at(i), next = sigmas.at(i + 1), ts = timesteps.at(i) * 10.0;
        const float cSkip = static_cast<float>(0.25 / (ts * ts + 0.25)), cOut = static_cast<float>(ts / std::sqrt(ts * ts + 0.25));
        const float s = static_cast<float>(sigma), vp = static_cast<float>(1.0 / std::sqrt(sigma * sigma + 1.0));
        const Matrix fresh = next > 0.0 ? Matrix::normal(latent.rows(), latent.cols(), 0.0f, 1.0f) : Matrix();
        for (std::size_t k = 0; k < latent.size(); ++k) {
            const float x = latent.data()[k];
            const float denoised = cOut * (x - s * noise.data()[k]) + cSkip * x * vp;
            latent.data()[k] = next > 0.0 ? denoised + static_cast<float>(next) * fresh.data()[k] : denoised;
        }
        return;
    }
    // x0 = x - sigma eps; d = (x - x0) / sigma = eps; x_next = x + d (sigma_next - sigma)
    const float dt = static_cast<float>(sigmas.at(i + 1) - sigmas.at(i));
    for (std::size_t k = 0; k < latent.size(); ++k) latent.data()[k] += noise.data()[k] * dt;
}

// ---- Text to image -------------------------------------------------------------------------

namespace {
using PipelineClock = std::chrono::steady_clock;

void reportStage(const TextToImageOptions& o, const std::string& stage, PipelineClock::time_point start) {
    if (o.progress) o.progress(stage, std::chrono::duration<double>(PipelineClock::now() - start).count());
}

void checkSize(const TextToImageOptions& o) {
    if (o.width % 64 || o.height % 64 || o.width == 0 || o.height == 0)
        throw std::invalid_argument("textToImage: width and height must be multiples of 64");
}

// CLIP hidden states of the prompt (and of the negative prompt when guided).
std::pair<Matrix, Matrix> encodePrompts(const std::string& dir, const std::string& prompt, const TextToImageOptions& o) {
    const auto t = PipelineClock::now();
    const ClipTokenizer tok = ClipTokenizer::fromDirectory(dir + "/tokenizer");
    ClipTextEncoder enc = loadClipTextEncoder(dir, o.int8, o.loras);
    // SD 2 / Turbo pad with "!" (id 0); SD 1.x pads with <|endoftext|>.
    int pad = tok.endOfText();
    if (std::ifstream(dir + "/tokenizer/tokenizer_config.json")) {
        const json::Value c = json::parse(readFile(dir + "/tokenizer/tokenizer_config.json"));
        if (const json::Value* p = c.find("pad_token"); p && p->isString() && p->string == "!") pad = 0;
    }
    std::pair<Matrix, Matrix> out;
    out.first = enc.encode(tok.encodeForModel(prompt, enc.length, pad));
    if (o.guidance > 1.0f) out.second = enc.encode(tok.encodeForModel(o.negativePrompt, enc.length, pad));
    reportStage(o, "text encoder", t);
    return out;
}

EulerScheduler schedulerFor(const std::string& dir, std::size_t steps, const std::string& sampler = "euler") {
    const std::string config = dir + "/scheduler/scheduler_config.json";
    if (sampler == "lcm") return exists(config) ? EulerScheduler::lcmFromConfig(config, steps) : EulerScheduler::lcm(steps);
    if (sampler != "euler") throw std::invalid_argument("unknown sampler '" + sampler + "' (euler, lcm)");
    return exists(config) ? EulerScheduler::fromConfig(config, steps) : EulerScheduler::standard(steps);
}

// Runs the scheduler's steps [first, end) on `latent` with the UNet.
void denoise(const std::string& dir, const std::pair<Matrix, Matrix>& context, const EulerScheduler& sched, std::size_t first,
             Matrix& latent, const TextToImageOptions& o) {
    auto t = PipelineClock::now();
    UNet2DCondition unet = loadUNet(dir, o.width / 8, o.height / 8, o.int8, o.loras);
    reportStage(o, "load unet", t);
    if (latent.empty()) {  // text to image: a Gaussian latent at the first sigma
        setSeed(o.seed);
        latent = Matrix::normal(1, unet.latentShape().size(), 0.0f, 1.0f);
        for (float& v : latent.values()) v *= static_cast<float>(sched.initNoiseSigma);
    }
    for (std::size_t i = first; i < sched.timesteps.size(); ++i) {
        t = PipelineClock::now();
        Matrix in = latent;
        const float scale = static_cast<float>(sched.inputScale(i));
        for (float& v : in.values()) v *= scale;
        Matrix eps = unet.forward(in, static_cast<float>(sched.timesteps[i]), context.first);
        if (o.guidance > 1.0f) {
            const Matrix uncond = unet.forward(in, static_cast<float>(sched.timesteps[i]), context.second);
            for (std::size_t k = 0; k < eps.size(); ++k) eps.data()[k] = uncond.data()[k] + o.guidance * (eps.data()[k] - uncond.data()[k]);
        }
        sched.step(i, eps, latent);
        reportStage(o, "step " + std::to_string(i - first + 1) + "/" + std::to_string(sched.timesteps.size() - first), t);
    }
}
} // namespace

Matrix textToLatent(const std::string& dir, const std::string& prompt, const TextToImageOptions& o) {
    checkSize(o);
    const auto context = encodePrompts(dir, prompt, o);
    Matrix latent;
    denoise(dir, context, schedulerFor(dir, o.steps, o.sampler), 0, latent, o);
    return latent;
}

Image imageToImage(const std::string& dir, const Image& input, const std::string& prompt, const TextToImageOptions& o) {
    checkSize(o);
    if (o.strength < 0.0f || o.strength > 1.0f) throw std::invalid_argument("imageToImage: strength must be in [0, 1]");
    const Image fitted = fitImage(toRgb(input), o.width, o.height);
    Matrix latent;
    {
        const auto t = PipelineClock::now();
        if (!o.tinyVae.empty()) latent = loadTinyVae(o.tinyVae, o.width, o.height, true, false).encode(imageToTensor(fitted));
        else latent = loadVae(dir, o.width, o.height, true, false).encode(imageToTensor(fitted));
        reportStage(o, o.tinyVae.empty() ? "vae encode" : "taesd encode", t);
    }
    // As diffusers: the last steps * strength steps of the schedule run, starting from the
    // encoded image with noise at that step's sigma.
    const EulerScheduler sched = schedulerFor(dir, o.steps, o.sampler);
    const std::size_t run = std::min(o.steps, static_cast<std::size_t>(static_cast<float>(o.steps) * o.strength + 1e-4f));
    if (run > 0) {
        const std::size_t first = o.steps - run;
        const auto context = encodePrompts(dir, prompt, o);
        setSeed(o.seed);
        const Matrix noise = Matrix::normal(1, latent.size(), 0.0f, 1.0f);
        const float sigma = static_cast<float>(sched.sigmas[first]);
        for (std::size_t k = 0; k < latent.size(); ++k) latent.data()[k] += sigma * noise.data()[k];
        denoise(dir, context, sched, first, latent, o);
    }
    return decodeLatent(dir, latent, o);
}

Image decodeLatent(const std::string& dir, const Matrix& latent, const TextToImageOptions& o) {
    const auto t = std::chrono::steady_clock::now();
    Matrix img;
    ImageShape shape;
    if (!o.tinyVae.empty()) {
        TinyVae vae = loadTinyVae(o.tinyVae, o.width, o.height, false, true);
        img = vae.decode(latent);
        shape = vae.image;
    } else {
        Vae vae = loadVae(dir, o.width, o.height, false, true);
        img = vae.decode(latent);
        shape = vae.image;
    }
    if (o.progress) o.progress(o.tinyVae.empty() ? "vae decode" : "taesd decode",
                               std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
    return tensorToImage(img.data(), shape);
}

Image textToImage(const std::string& dir, const std::string& prompt, const TextToImageOptions& o) {
    return decodeLatent(dir, textToLatent(dir, prompt, o), o);
}

// ---- Text to video (AnimateDiff) -------------------------------------------------------------------

std::vector<Image> textToVideo(const std::string& dir, const std::string& prompt, const TextToVideoOptions& v) {
    TextToImageOptions o;
    o.width = v.width;
    o.height = v.height;
    o.seed = v.seed;
    o.guidance = v.guidance;
    o.negativePrompt = v.negativePrompt;
    o.tinyVae = v.tinyVae;
    o.int8 = v.int8;
    o.progress = v.progress;
    o.loras = v.loras;
    checkSize(o);
    if (v.frames == 0) throw std::invalid_argument("textToVideo: frames must be > 0");
    if (v.context == 0 || v.context > 32 || v.overlap >= v.context)
        throw std::invalid_argument("textToVideo: context must be in 1..32 and overlap below it");
    if (!v.init.empty() && v.init.size() != 1 && v.init.size() != v.frames)
        throw std::invalid_argument("textToVideo: give one init image or one per frame");
    if (v.strength < 0.0f || v.strength > 1.0f) throw std::invalid_argument("textToVideo: strength must be in [0, 1]");
    std::string motion = v.motion;
    if (motion.empty()) {
        const std::string md = dir + "/motion";
        if (std::filesystem::is_directory(md))
            for (const auto& e : std::filesystem::directory_iterator(md))
                if (e.path().extension() == ".safetensors") {
                    if (!motion.empty()) throw std::runtime_error("textToVideo: several motion modules in " + md + "; choose one");
                    motion = e.path().string();
                }
        if (motion.empty()) throw std::runtime_error("textToVideo: no motion module (.safetensors) in " + md);
    }
    // Text conditions: one for all frames, or per frame between prompt-travel keyframes.
    std::vector<Matrix> conds;
    Matrix uncond;
    {
        auto te = PipelineClock::now();
        const ClipTokenizer tok = ClipTokenizer::fromDirectory(dir + "/tokenizer");
        ClipTextEncoder enc = loadClipTextEncoder(dir, o.int8, o.loras);
        int pad = tok.endOfText();
        if (std::ifstream(dir + "/tokenizer/tokenizer_config.json")) {
            const json::Value c = json::parse(readFile(dir + "/tokenizer/tokenizer_config.json"));
            if (const json::Value* p = c.find("pad_token"); p && p->isString() && p->string == "!") pad = 0;
        }
        auto encode = [&](const std::string& s) { return enc.encode(tok.encodeForModel(s, enc.length, pad)); };
        std::vector<std::pair<std::size_t, Matrix>> keys{{0, encode(prompt)}};
        std::vector<std::pair<std::size_t, std::string>> at = v.promptAt;
        std::sort(at.begin(), at.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [frame, text] : at) {
            if (frame == 0 || frame >= v.frames) throw std::invalid_argument("textToVideo: prompt keyframes must be in 1..frames-1");
            if (frame == keys.back().first) throw std::invalid_argument("textToVideo: two prompts at frame " + std::to_string(frame));
            keys.emplace_back(frame, encode(text));
        }
        if (keys.size() == 1) conds.push_back(keys[0].second);
        else
            for (std::size_t f = 0; f < v.frames; ++f) {
                std::size_t k = 0;
                while (k + 1 < keys.size() && keys[k + 1].first <= f) ++k;
                if (k + 1 == keys.size()) {
                    conds.push_back(keys[k].second);
                    continue;
                }
                const float w = static_cast<float>(f - keys[k].first) / static_cast<float>(keys[k + 1].first - keys[k].first);
                Matrix c = keys[k].second;
                const Matrix& b = keys[k + 1].second;
                for (std::size_t i = 0; i < c.size(); ++i) c.data()[i] += w * (b.data()[i] - c.data()[i]);
                conds.push_back(std::move(c));
            }
        if (o.guidance > 1.0f) uncond = encode(o.negativePrompt);
        reportStage(o, "text encoder", te);
    }
    auto t = PipelineClock::now();
    UNet2DCondition unet = loadUNet(dir, o.width / 8, o.height / 8, o.int8, o.loras);
    loadMotionModules(unet, motion, o.int8);
    reportStage(o, "load unet + motion", t);
    const EulerScheduler sched =
        v.sampler == "lcm"         ? schedulerFor(dir, v.steps, "lcm")
        : v.betaSchedule == "config" ? schedulerFor(dir, v.steps)
                                     : EulerScheduler::make(v.steps, v.betaSchedule, "trailing");
    const std::size_t F = v.frames, n = unet.latentShape().size();
    std::size_t first = 0;
    setSeed(v.seed);
    Matrix latents = Matrix::normal(F, n, 0.0f, 1.0f);
    if (v.init.empty()) {
        for (float& x : latents.values()) x *= static_cast<float>(sched.initNoiseSigma);
    } else {
        // Encode the init frame(s) and noise them to the starting sigma (as imageToImage).
        const std::size_t run = std::min(v.steps, static_cast<std::size_t>(static_cast<float>(v.steps) * v.strength + 1e-4f));
        first = v.steps - run;
        const float sigma = static_cast<float>(sched.sigmas[first]);
        const auto te = PipelineClock::now();
        std::unique_ptr<TinyVae> tiny;
        std::unique_ptr<Vae> vae;
        if (!o.tinyVae.empty()) tiny = std::make_unique<TinyVae>(loadTinyVae(o.tinyVae, o.width, o.height, true, false));
        else vae = std::make_unique<Vae>(loadVae(dir, o.width, o.height, true, false));
        Matrix z;
        for (std::size_t f = 0; f < F; ++f) {
            if (f < v.init.size()) {
                const Matrix x = imageToTensor(fitImage(toRgb(v.init[f]), o.width, o.height));
                z = tiny ? tiny->encode(x) : vae->encode(x);
            }
            float* row = latents.data() + f * n;
            for (std::size_t k = 0; k < n; ++k) row[k] = z.data()[k] + sigma * row[k];
        }
        reportStage(o, o.tinyVae.empty() ? "vae encode" : "taesd encode", te);
    }
    // Sliding windows over the frames (one window when they fit).
    std::vector<std::size_t> starts;
    const std::size_t ctxLen = std::min(v.context, F), stride = std::max<std::size_t>(1, ctxLen - std::min(v.overlap, ctxLen - 1));
    for (std::size_t s = 0;; s += stride) {
        if (s + ctxLen >= F) {
            starts.push_back(F - ctxLen);
            break;
        }
        starts.push_back(s);
    }
    auto predict = [&](const Matrix& in, float timestep, const std::vector<Matrix>& ctx) {
        if (starts.size() == 1) return unet.forwardFrames(in, timestep, ctx);
        Matrix sum(F, n);
        std::vector<float> weight(F, 0.0f);
        for (std::size_t s : starts) {
            Matrix win(ctxLen, n);
            std::copy(in.data() + s * n, in.data() + (s + ctxLen) * n, win.data());
            const Matrix e = unet.forwardFrames(
                win, timestep, ctx.size() == 1 ? ctx : std::vector<Matrix>(ctx.begin() + static_cast<long>(s), ctx.begin() + static_cast<long>(s + ctxLen)));
            for (std::size_t f = 0; f < ctxLen; ++f) {
                // Pyramid weights: frames near a window's middle count most.
                const float wgt = static_cast<float>(std::min(f + 1, ctxLen - f));
                weight[s + f] += wgt;
                const float* src = e.data() + f * n;
                float* dst = sum.data() + (s + f) * n;
                for (std::size_t k = 0; k < n; ++k) dst[k] += wgt * src[k];
            }
        }
        for (std::size_t f = 0; f < F; ++f) {
            float* row = sum.data() + f * n;
            for (std::size_t k = 0; k < n; ++k) row[k] /= weight[f];
        }
        return sum;
    };
    for (std::size_t i = first; i < sched.timesteps.size(); ++i) {
        t = PipelineClock::now();
        Matrix in = latents;
        const float scale = static_cast<float>(sched.inputScale(i));
        for (float& x : in.values()) x *= scale;
        Matrix eps = predict(in, static_cast<float>(sched.timesteps[i]), conds);
        if (o.guidance > 1.0f) {
            const Matrix un = predict(in, static_cast<float>(sched.timesteps[i]), std::vector<Matrix>{uncond});
            for (std::size_t k = 0; k < eps.size(); ++k) eps.data()[k] = un.data()[k] + o.guidance * (eps.data()[k] - un.data()[k]);
        }
        sched.step(i, eps, latents);
        reportStage(o, "step " + std::to_string(i + 1) + "/" + std::to_string(sched.timesteps.size()), t);
    }
    unet = UNet2DCondition();  // free the UNet before decoding
    t = PipelineClock::now();
    std::unique_ptr<TinyVae> tiny;
    std::unique_ptr<Vae> vae;
    if (!o.tinyVae.empty()) tiny = std::make_unique<TinyVae>(loadTinyVae(o.tinyVae, o.width, o.height, false, true));
    else vae = std::make_unique<Vae>(loadVae(dir, o.width, o.height, false, true));
    std::vector<Image> frames;
    for (std::size_t f = 0; f < F; ++f) {
        Matrix one(1, n);
        std::copy(latents.data() + f * n, latents.data() + (f + 1) * n, one.data());
        const Matrix img = tiny ? tiny->decode(one) : vae->decode(one);
        frames.push_back(tensorToImage(img.data(), {3, o.height, o.width}));
    }
    reportStage(o, o.tinyVae.empty() ? "vae decode" : "taesd decode", t);
    return frames;
}

// ---- Persistent model and animations --------------------------------------------------------------

struct StableDiffusion::Impl {
    std::string dir;
    TextToImageOptions options;
    std::unique_ptr<ClipTokenizer> tokenizer;
    ClipTextEncoder clip;
    int pad = 0;
    UNet2DCondition unet;
    std::unique_ptr<TinyVae> tiny;
    std::unique_ptr<Vae> vae;
    bool encoder = false;
};

StableDiffusion::StableDiffusion(const std::string& dir, const TextToImageOptions& o, bool withEncoder) : impl_(std::make_unique<Impl>()) {
    checkSize(o);
    Impl& m = *impl_;
    m.dir = dir;
    m.options = o;
    m.encoder = withEncoder;
    auto t = PipelineClock::now();
    m.tokenizer = std::make_unique<ClipTokenizer>(ClipTokenizer::fromDirectory(dir + "/tokenizer"));
    m.clip = loadClipTextEncoder(dir, o.int8, o.loras);
    m.pad = m.tokenizer->endOfText();
    if (std::ifstream(dir + "/tokenizer/tokenizer_config.json")) {
        const json::Value c = json::parse(readFile(dir + "/tokenizer/tokenizer_config.json"));
        if (const json::Value* p = c.find("pad_token"); p && p->isString() && p->string == "!") m.pad = 0;
    }
    reportStage(o, "load text encoder", t);
    t = PipelineClock::now();
    m.unet = loadUNet(dir, o.width / 8, o.height / 8, o.int8, o.loras);
    reportStage(o, "load unet", t);
    t = PipelineClock::now();
    if (!o.tinyVae.empty()) m.tiny = std::make_unique<TinyVae>(loadTinyVae(o.tinyVae, o.width, o.height, withEncoder, true));
    else m.vae = std::make_unique<Vae>(loadVae(dir, o.width, o.height, withEncoder, true));
    reportStage(o, "load autoencoder", t);
}

StableDiffusion::~StableDiffusion() = default;
StableDiffusion::StableDiffusion(StableDiffusion&&) noexcept = default;
StableDiffusion& StableDiffusion::operator=(StableDiffusion&&) noexcept = default;

std::size_t StableDiffusion::width() const { return impl_->options.width; }
const TextToImageOptions& StableDiffusion::impl_options() const { return impl_->options; }
std::size_t StableDiffusion::height() const { return impl_->options.height; }

Matrix StableDiffusion::encodePrompt(const std::string& prompt) {
    return impl_->clip.encode(impl_->tokenizer->encodeForModel(prompt, impl_->clip.length, impl_->pad));
}

Matrix StableDiffusion::noise(unsigned seed) const {
    setSeed(seed);
    return Matrix::normal(1, impl_->unet.latentShape().size(), 0.0f, 1.0f);
}

EulerScheduler StableDiffusion::scheduler(std::size_t steps) const { return schedulerFor(impl_->dir, steps, impl_->options.sampler); }

void StableDiffusion::denoise(Matrix& latent, const Matrix& context, const EulerScheduler& sched, std::size_t first, float guidance,
                              const Matrix& negative) {
    for (std::size_t i = first; i < sched.timesteps.size(); ++i) {
        Matrix in = latent;
        const float scale = static_cast<float>(sched.inputScale(i));
        for (float& v : in.values()) v *= scale;
        Matrix eps = impl_->unet.forward(in, static_cast<float>(sched.timesteps[i]), context);
        if (guidance > 1.0f) {
            const Matrix uncond = impl_->unet.forward(in, static_cast<float>(sched.timesteps[i]), negative);
            for (std::size_t k = 0; k < eps.size(); ++k) eps.data()[k] = uncond.data()[k] + guidance * (eps.data()[k] - uncond.data()[k]);
        }
        sched.step(i, eps, latent);
    }
}

Matrix StableDiffusion::encodeImage(const Image& image) {
    if (!impl_->encoder) throw std::logic_error("StableDiffusion: loaded without an image encoder");
    const Matrix x = imageToTensor(fitImage(toRgb(image), width(), height()));
    return impl_->tiny ? impl_->tiny->encode(x) : impl_->vae->encode(x);
}

Image StableDiffusion::decode(const Matrix& latent) {
    const Matrix img = impl_->tiny ? impl_->tiny->decode(latent) : impl_->vae->decode(latent);
    return tensorToImage(img.data(), {3, height(), width()});
}

Image StableDiffusion::generate(const std::string& prompt, unsigned seed, std::size_t steps) {
    const EulerScheduler sched = scheduler(steps);
    Matrix latent = noise(seed);
    for (float& v : latent.values()) v *= static_cast<float>(sched.initNoiseSigma);
    denoise(latent, encodePrompt(prompt), sched);
    return decode(latent);
}

Image StableDiffusion::transform(const Image& input, const std::string& prompt, unsigned seed, std::size_t steps, float strength) {
    if (strength < 0.0f || strength > 1.0f) throw std::invalid_argument("StableDiffusion::transform: strength must be in [0, 1]");
    Matrix latent = encodeImage(input);
    const EulerScheduler sched = scheduler(steps);
    const std::size_t run = std::min(steps, static_cast<std::size_t>(static_cast<float>(steps) * strength + 1e-4f));
    if (run > 0) {
        const std::size_t first = steps - run;
        const Matrix n = noise(seed);
        const float sigma = static_cast<float>(sched.sigmas[first]);
        for (std::size_t k = 0; k < latent.size(); ++k) latent.data()[k] += sigma * n.data()[k];
        denoise(latent, encodePrompt(prompt), sched, first);
    }
    return decode(latent);
}

namespace {
// Spherical interpolation of two noise latents (keeps the norm of a Gaussian sample).
Matrix slerp(const Matrix& a, const Matrix& b, double t) {
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a.data()[i]) * b.data()[i];
        na += static_cast<double>(a.data()[i]) * a.data()[i];
        nb += static_cast<double>(b.data()[i]) * b.data()[i];
    }
    const double cosTheta = std::max(-1.0, std::min(1.0, dot / std::sqrt(na * nb)));
    const double theta = std::acos(cosTheta);
    Matrix out(a.rows(), a.cols());
    if (theta < 1e-4) {  // nearly the same direction: linear is exact enough
        for (std::size_t i = 0; i < a.size(); ++i) out.data()[i] = static_cast<float>((1 - t) * a.data()[i] + t * b.data()[i]);
        return out;
    }
    const double s = std::sin(theta), wa = std::sin((1 - t) * theta) / s, wb = std::sin(t * theta) / s;
    for (std::size_t i = 0; i < a.size(); ++i) out.data()[i] = static_cast<float>(wa * a.data()[i] + wb * b.data()[i]);
    return out;
}

// Per-channel histogram matching to a reference picture: keeps feedback loops
// from drifting towards over-saturated or shifted colours.
void matchColors(Image& image, const Image& reference) {
    const std::size_t C = image.channels, n = image.width * image.height, m = reference.width * reference.height;
    for (std::size_t c = 0; c < C; ++c) {
        double hi[256] = {}, hr[256] = {};
        for (std::size_t i = 0; i < n; ++i) hi[image.pixels[i * C + c]] += 1.0 / static_cast<double>(n);
        for (std::size_t i = 0; i < m; ++i) hr[reference.pixels[i * C + c]] += 1.0 / static_cast<double>(m);
        // Map each level to the reference level at the same cumulative fraction (mid-bin).
        std::vector<std::uint8_t> map(256);
        double ci = 0, cr = hr[0];
        int r = 0;
        for (int v = 0; v < 256; ++v) {
            const double target = ci + hi[v] / 2;
            ci += hi[v];
            while (r < 255 && cr < target) cr += hr[++r];
            map[v] = static_cast<std::uint8_t>(r);
        }
        for (std::size_t i = 0; i < n; ++i) image.pixels[i * C + c] = map[image.pixels[i * C + c]];
    }
}

// Unsharp mask (3x3 binomial blur) plus Gaussian pixel noise, both in place.
void sharpenAndNoise(Image& image, float amount, float noise, unsigned seed) {
    const std::size_t W = image.width, H = image.height, C = image.channels;
    const std::vector<std::uint8_t> src = image.pixels;
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, noise * 255.0f);
    for (std::size_t y = 0; y < H; ++y)
        for (std::size_t x = 0; x < W; ++x)
            for (std::size_t c = 0; c < C; ++c) {
                float blur = 0.0f;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i) {
                        const std::size_t yy = static_cast<std::size_t>(std::min<long>(std::max<long>(static_cast<long>(y) + j, 0), static_cast<long>(H) - 1));
                        const std::size_t xx = static_cast<std::size_t>(std::min<long>(std::max<long>(static_cast<long>(x) + i, 0), static_cast<long>(W) - 1));
                        blur += static_cast<float>((j == 0 ? 2 : 1) * (i == 0 ? 2 : 1)) * src[(yy * W + xx) * C + c];
                    }
                const float v = src[(y * W + x) * C + c];
                float out = v + amount * (v - blur / 16.0f);
                if (noise > 0.0f) out += gauss(rng);
                image.pixels[(y * W + x) * C + c] = static_cast<std::uint8_t>(std::lround(std::min(255.0f, std::max(0.0f, out))));
            }
}

void frameDone(const TextToImageOptions& o, std::size_t i, std::size_t n, PipelineClock::time_point start) {
    reportStage(o, "frame " + std::to_string(i + 1) + "/" + std::to_string(n), start);
}
} // namespace

std::vector<Image> morphAnimation(StableDiffusion& sd, const std::vector<std::string>& prompts, const std::vector<unsigned>& seeds,
                                  const MorphOptions& o) {
    if (prompts.empty()) throw std::invalid_argument("morphAnimation: no prompts");
    if (seeds.size() != 1 && seeds.size() != prompts.size())
        throw std::invalid_argument("morphAnimation: give one seed, or one per prompt");
    if (o.framesPerSegment == 0) throw std::invalid_argument("morphAnimation: framesPerSegment must be > 0");
    const std::size_t keys = prompts.size();
    std::vector<Matrix> contexts, noises;
    for (std::size_t i = 0; i < keys; ++i) {
        contexts.push_back(sd.encodePrompt(prompts[i]));
        noises.push_back(sd.noise(seeds.size() == 1 ? seeds[0] : seeds[i]));
    }
    const EulerScheduler sched = sd.scheduler(o.steps);
    const std::size_t segments = keys == 1 ? 1 : (o.loop ? keys : keys - 1);
    const std::size_t total = segments * o.framesPerSegment + (o.loop || keys == 1 ? 0 : 1);
    const TextToImageOptions& opts = sd.impl_options();
    std::vector<Image> frames;
    for (std::size_t f = 0; f < total; ++f) {
        const auto t = PipelineClock::now();
        const std::size_t seg = std::min(f / o.framesPerSegment, segments - 1);
        const double u = static_cast<double>(f - seg * o.framesPerSegment) / static_cast<double>(o.framesPerSegment);
        const std::size_t a = seg % keys, b = (seg + 1) % keys;
        // Smoothstep easing, so each keyframe holds for a moment.
        const double w = u * u * (3 - 2 * u);
        Matrix context(contexts[a].rows(), contexts[a].cols());
        for (std::size_t k = 0; k < context.size(); ++k)
            context.data()[k] = static_cast<float>((1 - w) * contexts[a].data()[k] + w * contexts[b].data()[k]);
        Matrix latent = slerp(noises[a], noises[b], w);
        for (float& v : latent.values()) v *= static_cast<float>(sched.initNoiseSigma);
        sd.denoise(latent, context, sched);
        frames.push_back(sd.decode(latent));
        frameDone(opts, f, total, t);
    }
    return frames;
}

std::vector<Image> zoomAnimation(StableDiffusion& sd, const std::vector<std::string>& prompts, const ZoomOptions& o, const Image* start) {
    if (prompts.empty()) throw std::invalid_argument("zoomAnimation: no prompts");
    if (o.frames == 0) throw std::invalid_argument("zoomAnimation: frames must be > 0");
    const TextToImageOptions& opts = sd.impl_options();
    std::vector<Image> frames;
    auto t = PipelineClock::now();
    if (start) frames.push_back(fitImage(toRgb(*start), sd.width(), sd.height()));
    else frames.push_back(sd.transform(sd.generate(prompts[0], o.seed, 1), prompts[0], o.seed + 7919u, o.steps, o.strength));
    frameDone(opts, 0, o.frames, t);
    for (std::size_t f = 1; f < o.frames; ++f) {
        t = PipelineClock::now();
        const std::string& prompt = prompts[std::min(prompts.size() - 1, f * prompts.size() / o.frames)];
        Image moved = transformImage(frames.back(), o.zoom, o.dx, o.dy, o.rotate);
        if (o.sharpen > 0.0f || o.noise > 0.0f) sharpenAndNoise(moved, o.sharpen, o.noise, o.seed * 7919u + static_cast<unsigned>(f));
        Image next = sd.transform(moved, prompt, o.seed + static_cast<unsigned>(f), o.steps, o.strength);
        if (o.colorCoherence) matchColors(next, frames.front());
        frames.push_back(std::move(next));
        frameDone(opts, f, o.frames, t);
    }
    return frames;
}

} // namespace nc
