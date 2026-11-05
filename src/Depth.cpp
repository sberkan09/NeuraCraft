#include "Depth.h"
#include "Backend.h"
#include "Diffusion.h"
#include "Json.h"
#include "Parallel.h"
#include "SafeTensors.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
struct Map {
    std::size_t C = 0, H = 0, W = 0;
    std::vector<float> v;
    Map() = default;
    Map(std::size_t c, std::size_t h, std::size_t w) : C(c), H(h), W(w), v(c * h * w, 0.0f) {}
    float* plane(std::size_t c) { return v.data() + c * H * W; }
    const float* plane(std::size_t c) const { return v.data() + c * H * W; }
};

struct ConvW {
    Matrix weight;  // conv: out x (in k k); transposed: in x (out k k)
    std::vector<float> bias;
    std::size_t in = 0, out = 0, k = 1;
};

// k x k convolution with stride and padding (im2col + GEMM).
Map conv(const Map& x, const ConvW& c, std::size_t stride, std::size_t pad) {
    const std::size_t k = c.k, OH = (x.H + 2 * pad - k) / stride + 1, OW = (x.W + 2 * pad - k) / stride + 1, P = OH * OW;
    const std::size_t patch = x.C * k * k;
    Map y(c.out, OH, OW);
    if (k == 1 && stride == 1 && pad == 0) {
        gemm(false, false, c.out, P, x.C, c.weight.data(), x.C, x.v.data(), P, y.v.data(), P, false);
    } else {
        Matrix cols(patch, P);
        parallelFor(0, x.C, 1, [&](std::size_t c0, std::size_t c1) {
            for (std::size_t ch = c0; ch < c1; ++ch)
                for (std::size_t ky = 0; ky < k; ++ky)
                    for (std::size_t kx = 0; kx < k; ++kx) {
                        float* dst = cols.data() + ((ch * k + ky) * k + kx) * P;
                        const float* src = x.plane(ch);
                        for (std::size_t oy = 0; oy < OH; ++oy) {
                            const long sy = static_cast<long>(oy * stride + ky) - static_cast<long>(pad);
                            for (std::size_t ox = 0; ox < OW; ++ox) {
                                const long sx = static_cast<long>(ox * stride + kx) - static_cast<long>(pad);
                                dst[oy * OW + ox] = sy < 0 || sx < 0 || sy >= static_cast<long>(x.H) || sx >= static_cast<long>(x.W)
                                                        ? 0.0f
                                                        : src[static_cast<std::size_t>(sy) * x.W + static_cast<std::size_t>(sx)];
                            }
                        }
                    }
        });
        gemm(false, false, c.out, P, patch, c.weight.data(), patch, cols.data(), P, y.v.data(), P, false);
    }
    if (!c.bias.empty())
        for (std::size_t o = 0; o < c.out; ++o) {
            float* p = y.plane(o);
            for (std::size_t i = 0; i < P; ++i) p[i] += c.bias[o];
        }
    return y;
}

// Transposed convolution with kernel = stride = s, no padding (each input pixel -> an s x s block).
Map convTransposeBlocks(const Map& x, const ConvW& c) {
    const std::size_t s = c.k, P = x.H * x.W;
    Matrix cols(c.out * s * s, P);
    gemm(true, false, c.out * s * s, P, c.in, c.weight.data(), c.out * s * s, x.v.data(), P, cols.data(), P, false);
    Map y(c.out, x.H * s, x.W * s);
    parallelFor(0, c.out, 1, [&](std::size_t o0, std::size_t o1) {
        for (std::size_t o = o0; o < o1; ++o)
            for (std::size_t ky = 0; ky < s; ++ky)
                for (std::size_t kx = 0; kx < s; ++kx) {
                    const float* src = cols.data() + ((o * s + ky) * s + kx) * P;
                    for (std::size_t iy = 0; iy < x.H; ++iy)
                        for (std::size_t ix = 0; ix < x.W; ++ix)
                            y.plane(o)[(iy * s + ky) * y.W + ix * s + kx] = src[iy * x.W + ix] + c.bias[o];
                }
    });
    return y;
}

// Bilinear resize, PyTorch semantics for both align_corners settings.
Map resize(const Map& x, std::size_t H, std::size_t W, bool alignCorners) {
    Map y(x.C, H, W);
    auto src = [&](std::size_t i, std::size_t outN, std::size_t inN) {
        if (alignCorners) return outN > 1 ? static_cast<double>(i) * (inN - 1) / (outN - 1) : 0.0;
        return std::max(0.0, (i + 0.5) * static_cast<double>(inN) / outN - 0.5);
    };
    parallelFor(0, H, 8, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const double fy = src(r, H, x.H);
            const std::size_t y0 = std::min(static_cast<std::size_t>(fy), x.H - 1), y1 = std::min(y0 + 1, x.H - 1);
            const float ay = static_cast<float>(fy - y0);
            for (std::size_t q = 0; q < W; ++q) {
                const double fx = src(q, W, x.W);
                const std::size_t x0 = std::min(static_cast<std::size_t>(fx), x.W - 1), x1 = std::min(x0 + 1, x.W - 1);
                const float ax = static_cast<float>(fx - x0);
                for (std::size_t c = 0; c < x.C; ++c) {
                    const float* p = x.plane(c);
                    y.plane(c)[r * W + q] = (p[y0 * x.W + x0] * (1 - ax) + p[y0 * x.W + x1] * ax) * (1 - ay) +
                                            (p[y1 * x.W + x0] * (1 - ax) + p[y1 * x.W + x1] * ax) * ay;
                }
            }
        }
    });
    return y;
}

// Bicubic resize as PyTorch (A = -0.75, align_corners = false, clamped indices).
Map resizeBicubic(const Map& x, std::size_t H, std::size_t W) {
    auto cubic = [](double t) {
        const double A = -0.75;
        t = std::abs(t);
        if (t <= 1) return ((A + 2) * t - (A + 3)) * t * t + 1;
        if (t < 2) return ((A * t - 5 * A) * t + 8 * A) * t - 4 * A;
        return 0.0;
    };
    Map y(x.C, H, W);
    for (std::size_t r = 0; r < H; ++r) {
        const double fy = (r + 0.5) * static_cast<double>(x.H) / H - 0.5;
        const long iy = static_cast<long>(std::floor(fy));
        const double ty = fy - iy;
        for (std::size_t q = 0; q < W; ++q) {
            const double fx = (q + 0.5) * static_cast<double>(x.W) / W - 0.5;
            const long ix = static_cast<long>(std::floor(fx));
            const double tx = fx - ix;
            for (std::size_t c = 0; c < x.C; ++c) {
                double s = 0;
                for (int m = -1; m <= 2; ++m) {
                    const std::size_t yy = static_cast<std::size_t>(std::min<long>(std::max<long>(iy + m, 0), static_cast<long>(x.H) - 1));
                    const double wy = cubic(m - ty);
                    for (int n = -1; n <= 2; ++n) {
                        const std::size_t xx = static_cast<std::size_t>(std::min<long>(std::max<long>(ix + n, 0), static_cast<long>(x.W) - 1));
                        s += wy * cubic(n - tx) * x.plane(c)[yy * x.W + xx];
                    }
                }
                y.plane(c)[r * W + q] = static_cast<float>(s);
            }
        }
    }
    return y;
}

void relu(Map& m) {
    for (float& v : m.v) v = std::max(v, 0.0f);
}

void layerNormRows(Matrix& x, const std::vector<float>& g, const std::vector<float>& b, float eps) {
    const std::size_t D = x.cols();
    parallelFor(0, x.rows(), 16, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            float* row = x.data() + r * D;
            double mean = 0, var = 0;
            for (std::size_t i = 0; i < D; ++i) mean += row[i];
            mean /= static_cast<double>(D);
            for (std::size_t i = 0; i < D; ++i) var += (row[i] - mean) * (row[i] - mean);
            const float inv = static_cast<float>(1.0 / std::sqrt(var / static_cast<double>(D) + eps));
            for (std::size_t i = 0; i < D; ++i) row[i] = static_cast<float>(row[i] - mean) * inv * g[i] + b[i];
        }
    });
}

struct Linear {
    Matrix wt;  // in x out
    std::vector<float> bias;
    Matrix forward(const Matrix& x) const {
        Matrix y(x.rows(), wt.cols());
        gemm(false, false, x.rows(), wt.cols(), wt.rows(), x.data(), wt.rows(), wt.data(), wt.cols(), y.data(), wt.cols(), false);
        for (std::size_t r = 0; r < y.rows(); ++r)
            for (std::size_t c = 0; c < y.cols(); ++c) y.data()[r * y.cols() + c] += bias[c];
        return y;
    }
};
} // namespace

struct DepthEstimator::Impl {
    std::size_t dim = 384, heads = 6, patch = 14, layers = 12, fusion = 64;
    float eps = 1e-6f, maxDepth = 1.0f;
    bool metric = false;
    std::vector<std::size_t> outIndices;
    std::vector<float> reassembleFactors;
    ConvW patchEmbed;
    std::vector<float> cls;
    Map pos;  // dim x 37 x 37 (patch positions), plus its class token row below
    std::vector<float> clsPos;
    struct Layer {
        std::vector<float> n1g, n1b, n2g, n2b, ls1, ls2;
        Linear q, k, v, o, fc1, fc2;
    };
    std::vector<Layer> blocks;
    std::vector<float> lnG, lnB;
    std::vector<ConvW> reProj, reResize, neckConvs;
    struct Fusion {
        ConvW proj, r1c1, r1c2, r2c1, r2c2;
    };
    std::vector<Fusion> fusions;
    ConvW head1, head2, head3;
};

DepthEstimator::DepthEstimator(const std::string& dir) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    auto read = [](const std::string& p) {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("DepthEstimator: cannot open " + p);
        std::stringstream s;
        s << f.rdbuf();
        return s.str();
    };
    const json::Value c = json::parse(read(dir + "/config.json"));
    const json::Value& bc = c.at("backbone_config");
    m.dim = static_cast<std::size_t>(bc.numberOr("hidden_size", 384));
    m.heads = static_cast<std::size_t>(bc.numberOr("num_attention_heads", 6));
    m.patch = static_cast<std::size_t>(bc.numberOr("patch_size", 14));
    m.layers = static_cast<std::size_t>(bc.numberOr("num_hidden_layers", 12));
    m.eps = static_cast<float>(bc.numberOr("layer_norm_eps", 1e-6));
    for (const auto& e : bc.at("out_indices").array) m.outIndices.push_back(static_cast<std::size_t>(e.number));
    for (const auto& e : c.at("reassemble_factors").array) m.reassembleFactors.push_back(static_cast<float>(e.number));
    m.fusion = static_cast<std::size_t>(c.numberOr("fusion_hidden_size", 64));
    m.maxDepth = static_cast<float>(c.numberOr("max_depth", 1));
    if (const json::Value* t = c.find("depth_estimation_type"); t && t->isString()) m.metric = t->string == "metric";
    if (const json::Value* a = bc.find("hidden_act"); a && a->isString() && a->string != "gelu")
        throw std::runtime_error("DepthEstimator: activation " + a->string + " is not supported");
    if (const json::Value* s = bc.find("use_swiglu_ffn"); s && s->boolean) throw std::runtime_error("DepthEstimator: SwiGLU FFN is not supported");

    const SafeTensors st(dir + "/model.safetensors");
    auto vec = [&](const std::string& n) {
        const Matrix x = st.load(n);
        return std::vector<float>(x.data(), x.data() + x.size());
    };
    auto convW = [&](const std::string& n, bool transposed, bool bias = true) {
        ConvW w;
        const auto& info = st.info(n + ".weight");
        if (info.shape.size() != 4) throw std::runtime_error("DepthEstimator: " + n + " is not a convolution");
        const Matrix x = st.load(n + ".weight");
        w.k = info.shape[2];
        w.in = transposed ? info.shape[0] : info.shape[1];
        w.out = transposed ? info.shape[1] : info.shape[0];
        w.weight = transposed ? Matrix(w.in, w.out * w.k * w.k, std::vector<float>(x.data(), x.data() + x.size()))
                              : Matrix(w.out, w.in * w.k * w.k, std::vector<float>(x.data(), x.data() + x.size()));
        if (bias) w.bias = vec(n + ".bias");
        return w;
    };
    auto linear = [&](const std::string& n) {
        Linear l;
        const auto& info = st.info(n + ".weight");
        l.wt = st.load(n + ".weight");  // out x in
        l.wt = Matrix(info.shape[0], info.shape[1], std::vector<float>(l.wt.data(), l.wt.data() + l.wt.size())).transpose();
        l.bias = vec(n + ".bias");
        return l;
    };
    const std::string b = "backbone.";
    m.patchEmbed = convW(b + "embeddings.patch_embeddings.projection", false);
    m.cls = vec(b + "embeddings.cls_token");
    const std::vector<float> pe = vec(b + "embeddings.position_embeddings");
    const std::size_t positions = pe.size() / m.dim - 1, side = static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(positions))));
    if (side * side != positions) throw std::runtime_error("DepthEstimator: position embeddings are not square");
    m.clsPos.assign(pe.begin(), pe.begin() + static_cast<long>(m.dim));
    m.pos = Map(m.dim, side, side);
    for (std::size_t p = 0; p < positions; ++p)
        for (std::size_t d = 0; d < m.dim; ++d) m.pos.plane(d)[p] = pe[(p + 1) * m.dim + d];
    for (std::size_t i = 0; i < m.layers; ++i) {
        const std::string p = b + "encoder.layer." + std::to_string(i) + ".";
        Impl::Layer l;
        l.n1g = vec(p + "norm1.weight");
        l.n1b = vec(p + "norm1.bias");
        l.n2g = vec(p + "norm2.weight");
        l.n2b = vec(p + "norm2.bias");
        l.ls1 = st.has(p + "layer_scale1.lambda1") ? vec(p + "layer_scale1.lambda1") : std::vector<float>(m.dim, 1.0f);
        l.ls2 = st.has(p + "layer_scale2.lambda1") ? vec(p + "layer_scale2.lambda1") : std::vector<float>(m.dim, 1.0f);
        l.q = linear(p + "attention.attention.query");
        l.k = linear(p + "attention.attention.key");
        l.v = linear(p + "attention.attention.value");
        l.o = linear(p + "attention.output.dense");
        l.fc1 = linear(p + "mlp.fc1");
        l.fc2 = linear(p + "mlp.fc2");
        m.blocks.push_back(std::move(l));
    }
    m.lnG = vec(b + "layernorm.weight");
    m.lnB = vec(b + "layernorm.bias");
    for (std::size_t i = 0; i < m.reassembleFactors.size(); ++i) {
        const std::string p = "neck.reassemble_stage.layers." + std::to_string(i) + ".";
        m.reProj.push_back(convW(p + "projection", false));
        const float f = m.reassembleFactors[i];
        m.reResize.push_back(f > 1 ? convW(p + "resize", true) : f < 1 ? convW(p + "resize", false) : ConvW{});
        m.neckConvs.push_back(convW("neck.convs." + std::to_string(i), false, false));
        const std::string q = "neck.fusion_stage.layers." + std::to_string(i) + ".";
        m.fusions.push_back({convW(q + "projection", false), convW(q + "residual_layer1.convolution1", false),
                             convW(q + "residual_layer1.convolution2", false), convW(q + "residual_layer2.convolution1", false),
                             convW(q + "residual_layer2.convolution2", false)});
    }
    m.head1 = convW("head.conv1", false);
    m.head2 = convW("head.conv2", false);
    m.head3 = convW("head.conv3", false);
}

DepthEstimator::~DepthEstimator() = default;
DepthEstimator::DepthEstimator(DepthEstimator&&) noexcept = default;
DepthEstimator& DepthEstimator::operator=(DepthEstimator&&) noexcept = default;

std::vector<float> DepthEstimator::forward(const std::vector<float>& chw, std::size_t H, std::size_t W) {
    Impl& m = *impl_;
    if (H % m.patch || W % m.patch || chw.size() != 3 * H * W)
        throw std::invalid_argument("DepthEstimator::forward: a 3 x H x W input with H, W multiples of the patch size");
    const std::size_t ph = H / m.patch, pw = W / m.patch, N = ph * pw, D = m.dim;
    Map img(3, H, W);
    img.v = chw;
    const Map patches = conv(img, m.patchEmbed, m.patch, 0);  // D x ph x pw
    // Tokens: class token, then patches row by row; plus position embeddings (bicubic when the grid differs).
    const Map pos = (ph == m.pos.H && pw == m.pos.W) ? m.pos : resizeBicubic(m.pos, ph, pw);
    Matrix x(N + 1, D);
    for (std::size_t d = 0; d < D; ++d) x.data()[d] = m.cls[d] + m.clsPos[d];
    for (std::size_t p = 0; p < N; ++p)
        for (std::size_t d = 0; d < D; ++d) x.data()[(p + 1) * D + d] = patches.plane(d)[p] + pos.plane(d)[p];
    std::vector<Map> features;
    for (std::size_t i = 0; i < m.layers; ++i) {
        const Impl::Layer& l = m.blocks[i];
        Matrix a = x;
        layerNormRows(a, l.n1g, l.n1b, m.eps);
        const Matrix att = l.o.forward(multiHeadAttention(l.q.forward(a), l.k.forward(a), l.v.forward(a), m.heads));
        for (std::size_t r = 0; r <= N; ++r)
            for (std::size_t d = 0; d < D; ++d) x.data()[r * D + d] += l.ls1[d] * att.data()[r * D + d];
        a = x;
        layerNormRows(a, l.n2g, l.n2b, m.eps);
        Matrix hidden = l.fc1.forward(a);
        for (float& v : hidden.values()) v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
        const Matrix mlp = l.fc2.forward(hidden);
        for (std::size_t r = 0; r <= N; ++r)
            for (std::size_t d = 0; d < D; ++d) x.data()[r * D + d] += l.ls2[d] * mlp.data()[r * D + d];
        if (std::find(m.outIndices.begin(), m.outIndices.end(), i + 1) != m.outIndices.end()) {
            Matrix n = x;
            layerNormRows(n, m.lnG, m.lnB, m.eps);
            Map f(D, ph, pw);
            for (std::size_t p = 0; p < N; ++p)
                for (std::size_t d = 0; d < D; ++d) f.plane(d)[p] = n.data()[(p + 1) * D + d];
            features.push_back(std::move(f));
        }
    }
    if (features.size() != m.reassembleFactors.size()) throw std::runtime_error("DepthEstimator: out_indices do not match the neck");
    // Neck: reassemble (project, resize), 3x3 convs to the fusion width.
    for (std::size_t i = 0; i < features.size(); ++i) {
        Map f = conv(features[i], m.reProj[i], 1, 0);
        const float factor = m.reassembleFactors[i];
        if (factor > 1) f = convTransposeBlocks(f, m.reResize[i]);
        else if (factor < 1) f = conv(f, m.reResize[i], static_cast<std::size_t>(std::lround(1 / factor)), 1);
        features[i] = conv(f, m.neckConvs[i], 1, 1);
    }
    auto preAct = [&](const Map& in, const ConvW& c1, const ConvW& c2) {
        Map h = in;
        relu(h);
        h = conv(h, c1, 1, 1);
        relu(h);
        h = conv(h, c2, 1, 1);
        for (std::size_t i = 0; i < h.v.size(); ++i) h.v[i] += in.v[i];
        return h;
    };
    // Fusion, from the coarsest map up; each step ends at the next map's size (the last doubles).
    const std::size_t F = features.size();
    Map fused;
    for (std::size_t s = 0; s < F; ++s) {
        const Map& feat = features[F - 1 - s];
        const Impl::Fusion& L = m.fusions[s];
        Map h;
        if (s == 0) h = feat;
        else {
            Map res = feat;
            if (res.H != fused.H || res.W != fused.W) res = resize(res, fused.H, fused.W, false);
            h = fused;
            const Map r = preAct(res, L.r1c1, L.r1c2);
            for (std::size_t i = 0; i < h.v.size(); ++i) h.v[i] += r.v[i];
        }
        h = preAct(h, L.r2c1, L.r2c2);
        const std::size_t th = s + 1 < F ? features[F - 2 - s].H : h.H * 2, tw = s + 1 < F ? features[F - 2 - s].W : h.W * 2;
        h = resize(h, th, tw, true);
        fused = conv(h, L.proj, 1, 0);
    }
    // Head: halve channels, upsample to the input size, two more convs.
    Map d = conv(fused, m.head1, 1, 1);
    d = resize(d, ph * m.patch, pw * m.patch, true);
    d = conv(d, m.head2, 1, 1);
    relu(d);
    d = conv(d, m.head3, 1, 0);
    std::vector<float> out(d.v.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = (m.metric ? 1.0f / (1.0f + std::exp(-d.v[i])) : std::max(d.v[i], 0.0f)) * m.maxDepth;
    return out;
}

std::vector<float> DepthEstimator::estimate(const Image& input, std::size_t inputSize) {
    const Image im = toRgb(input);
    const std::size_t p = impl_->patch;
    // Shorter side about inputSize, both sides multiples of the patch size (aspect ratio kept).
    const double scale = static_cast<double>(inputSize) / static_cast<double>(std::min(im.width, im.height));
    auto snap = [&](double v) { return std::max<std::size_t>(p, static_cast<std::size_t>(std::lround(v / p)) * p); };
    const std::size_t W = snap(im.width * scale), H = snap(im.height * scale);
    const Image r = resizeImage(im, W, H);
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> chw(3 * W * H);
    for (std::size_t i = 0; i < W * H; ++i)
        for (std::size_t c = 0; c < 3; ++c) chw[c * W * H + i] = (r.pixels[i * 3 + c] / 255.0f - mean[c]) / stdv[c];
    Map d(1, H, W);
    d.v = forward(chw, H, W);
    return resize(d, im.height, im.width, false).v;
}

Image DepthEstimator::depthImage(const Image& image, std::size_t inputSize) {
    const std::vector<float> d = estimate(image, inputSize);
    const auto [lo, hi] = std::minmax_element(d.begin(), d.end());
    const float range = std::max(*hi - *lo, 1e-6f);
    Image out;
    out.width = image.width;
    out.height = image.height;
    out.pixels.resize(d.size() * 3);
    for (std::size_t i = 0; i < d.size(); ++i)
        std::fill_n(out.pixels.data() + i * 3, 3, static_cast<std::uint8_t>(std::lround((d[i] - *lo) / range * 255.0f)));
    return out;
}

std::vector<Image> parallaxAnimation(const Image& input, const std::vector<float>& depth, const ParallaxOptions& o) {
    const Image im = toRgb(input);
    const std::size_t W = im.width, H = im.height;
    if (depth.size() != W * H) throw std::invalid_argument("parallaxAnimation: depth must have one value per pixel");
    if (o.frames == 0) throw std::invalid_argument("parallaxAnimation: frames must be > 0");
    const auto [lo, hi] = std::minmax_element(depth.begin(), depth.end());
    const float range = std::max(*hi - *lo, 1e-6f);
    std::vector<float> near(depth.size());  // 0 = farthest, 1 = nearest
    for (std::size_t i = 0; i < depth.size(); ++i) near[i] = (depth[i] - *lo) / range;
    std::vector<Image> frames;
    const double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;
    for (std::size_t f = 0; f < o.frames; ++f) {
        const double u = static_cast<double>(f) / static_cast<double>(o.loop ? o.frames : std::max<std::size_t>(1, o.frames - 1));
        const double angle = 2 * 3.14159265358979323846 * u;
        const double dx = o.loop ? std::sin(angle) : (u * 2 - 1);
        const double dy = o.loop ? 0.3 * (1 - std::cos(angle)) : 0.0;  // starts (and ends) at rest
        const double zoom = o.loop ? 0.5 * (1 - std::cos(angle)) : u;
        Image out;
        out.width = W;
        out.height = H;
        out.pixels.assign(W * H * 3, 0);
        std::vector<float> zbuf(W * H, -1.0f);
        // Forward splat: each source pixel moves by its nearness (covering the 2x2 pixels around
        // where it lands, against cracks).
        for (std::size_t y = 0; y < H; ++y)
            for (std::size_t x = 0; x < W; ++x) {
                const float n = near[y * W + x];
                const double s = 1.0 + o.zoom * zoom * n;
                const double tx = cx + (x - cx) * s + dx * o.amplitude * W * n;
                const double ty = cy + (y - cy) * s + dy * o.amplitude * W * n;
                const long ix = static_cast<long>(std::floor(tx)), iy = static_cast<long>(std::floor(ty));
                const int ex = tx - ix > 1e-6 ? 1 : 0, ey = ty - iy > 1e-6 ? 1 : 0;  // a pixel that lands exactly stays one pixel
                for (int oy = 0; oy <= ey; ++oy)
                    for (int ox = 0; ox <= ex; ++ox) {
                        const long px = ix + ox, py = iy + oy;
                        if (px < 0 || py < 0 || px >= static_cast<long>(W) || py >= static_cast<long>(H)) continue;
                        const std::size_t j = static_cast<std::size_t>(py) * W + static_cast<std::size_t>(px);
                        if (n > zbuf[j]) {
                            zbuf[j] = n;
                            std::copy_n(im.pixels.data() + (y * W + x) * 3, 3, out.pixels.data() + j * 3);
                        }
                    }
            }
        // Holes: take the farther of the nearest filled pixels to the left and right.
        for (std::size_t y = 0; y < H; ++y)
            for (std::size_t x = 0; x < W; ++x) {
                const std::size_t j = y * W + x;
                if (zbuf[j] >= 0.0f) continue;
                long l = static_cast<long>(x) - 1, r = static_cast<long>(x) + 1;
                while (l >= 0 && zbuf[y * W + static_cast<std::size_t>(l)] < 0.0f) --l;
                while (r < static_cast<long>(W) && zbuf[y * W + static_cast<std::size_t>(r)] < 0.0f) ++r;
                long pick = -1;
                if (l >= 0 && r < static_cast<long>(W))
                    pick = zbuf[y * W + static_cast<std::size_t>(l)] <= zbuf[y * W + static_cast<std::size_t>(r)] ? l : r;
                else pick = l >= 0 ? l : r;
                if (pick >= 0 && pick < static_cast<long>(W))
                    std::copy_n(out.pixels.data() + (y * W + static_cast<std::size_t>(pick)) * 3, 3, out.pixels.data() + j * 3);
            }
        frames.push_back(std::move(out));
    }
    return frames;
}

} // namespace nc
