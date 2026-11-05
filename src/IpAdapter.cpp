#include "IpAdapter.h"
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
struct Linear {
    Matrix wt;  // in x out
    std::vector<float> bias;
    Matrix forward(const Matrix& x) const {
        Matrix y(x.rows(), wt.cols());
        gemm(false, false, x.rows(), wt.cols(), wt.rows(), x.data(), wt.rows(), wt.data(), wt.cols(), y.data(), wt.cols(), false);
        if (!bias.empty())
            for (std::size_t r = 0; r < y.rows(); ++r)
                for (std::size_t c = 0; c < y.cols(); ++c) y.data()[r * y.cols() + c] += bias[c];
        return y;
    }
};

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

std::vector<float> vec(const SafeTensors& st, const std::string& n) {
    const Matrix x = st.load(n);
    return std::vector<float>(x.data(), x.data() + x.size());
}

Linear linear(const SafeTensors& st, const std::string& n, bool bias = true) {
    Linear l;
    const auto& info = st.info(n + ".weight");
    const Matrix w = st.load(n + ".weight");
    l.wt = Matrix(info.shape[0], info.shape[1], std::vector<float>(w.data(), w.data() + w.size())).transpose();
    if (bias) l.bias = vec(st, n + ".bias");
    return l;
}
} // namespace

struct ClipImageEncoder::Impl {
    std::size_t dim = 1280, heads = 16, patch = 14, size = 224, layers = 32;
    float eps = 1e-5f;
    bool quickGelu = false;
    Matrix patchW;  // dim x (3 * patch * patch)
    std::vector<float> cls, pos, preG, preB, postG, postB;
    struct Layer {
        std::vector<float> n1g, n1b, n2g, n2b;
        Linear q, k, v, o, fc1, fc2;
    };
    std::vector<Layer> blocks;
    Linear proj;
};

ClipImageEncoder::ClipImageEncoder(const std::string& dir) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    std::ifstream f(dir + "/config.json");
    if (!f) throw std::runtime_error("ClipImageEncoder: cannot open " + dir + "/config.json");
    std::stringstream s;
    s << f.rdbuf();
    json::Value c = json::parse(s.str());
    if (const json::Value* v = c.find("vision_config")) c = *v;
    m.dim = static_cast<std::size_t>(c.numberOr("hidden_size", 1280));
    m.heads = static_cast<std::size_t>(c.numberOr("num_attention_heads", 16));
    m.patch = static_cast<std::size_t>(c.numberOr("patch_size", 14));
    m.size = static_cast<std::size_t>(c.numberOr("image_size", 224));
    m.layers = static_cast<std::size_t>(c.numberOr("num_hidden_layers", 32));
    m.eps = static_cast<float>(c.numberOr("layer_norm_eps", 1e-5));
    if (const json::Value* a = c.find("hidden_act"); a && a->isString()) {
        if (a->string == "quick_gelu") m.quickGelu = true;
        else if (a->string != "gelu") throw std::runtime_error("ClipImageEncoder: activation " + a->string + " is not supported");
    }
    const SafeTensors st(dir + "/model.safetensors");
    const std::string p = "vision_model.";
    const Matrix pw = st.load(p + "embeddings.patch_embedding.weight");
    m.patchW = Matrix(m.dim, pw.size() / m.dim, std::vector<float>(pw.data(), pw.data() + pw.size()));
    m.cls = vec(st, p + "embeddings.class_embedding");
    m.pos = vec(st, p + "embeddings.position_embedding.weight");
    m.preG = vec(st, p + "pre_layrnorm.weight");
    m.preB = vec(st, p + "pre_layrnorm.bias");
    m.postG = vec(st, p + "post_layernorm.weight");
    m.postB = vec(st, p + "post_layernorm.bias");
    const std::size_t tokens = (m.size / m.patch) * (m.size / m.patch) + 1;
    if (m.pos.size() != tokens * m.dim) throw std::runtime_error("ClipImageEncoder: position embeddings do not match the image size");
    for (std::size_t i = 0; i < m.layers; ++i) {
        const std::string l = p + "encoder.layers." + std::to_string(i) + ".";
        Impl::Layer L;
        L.n1g = vec(st, l + "layer_norm1.weight");
        L.n1b = vec(st, l + "layer_norm1.bias");
        L.n2g = vec(st, l + "layer_norm2.weight");
        L.n2b = vec(st, l + "layer_norm2.bias");
        L.q = linear(st, l + "self_attn.q_proj");
        L.k = linear(st, l + "self_attn.k_proj");
        L.v = linear(st, l + "self_attn.v_proj");
        L.o = linear(st, l + "self_attn.out_proj");
        L.fc1 = linear(st, l + "mlp.fc1");
        L.fc2 = linear(st, l + "mlp.fc2");
        m.blocks.push_back(std::move(L));
    }
    m.proj = linear(st, "visual_projection", false);
}

ClipImageEncoder::~ClipImageEncoder() = default;
ClipImageEncoder::ClipImageEncoder(ClipImageEncoder&&) noexcept = default;
ClipImageEncoder& ClipImageEncoder::operator=(ClipImageEncoder&&) noexcept = default;
std::size_t ClipImageEncoder::imageSize() const { return impl_->size; }

Matrix ClipImageEncoder::forward(const std::vector<float>& chw) {
    Impl& m = *impl_;
    const std::size_t S = m.size, g = S / m.patch, N = g * g, D = m.dim, P = m.patch;
    if (chw.size() != 3 * S * S) throw std::invalid_argument("ClipImageEncoder::forward: expected a 3 x " + std::to_string(S) + " x " + std::to_string(S) + " input");
    // Patches as rows (N x 3 P P, channel-major inside a patch), times the patch weights.
    Matrix patches(N, 3 * P * P);
    for (std::size_t py = 0; py < g; ++py)
        for (std::size_t px = 0; px < g; ++px)
            for (std::size_t c = 0; c < 3; ++c)
                for (std::size_t y = 0; y < P; ++y)
                    for (std::size_t x = 0; x < P; ++x)
                        patches.data()[(py * g + px) * 3 * P * P + (c * P + y) * P + x] = chw[c * S * S + (py * P + y) * S + px * P + x];
    Matrix emb(N, D);
    gemm(false, true, N, D, 3 * P * P, patches.data(), 3 * P * P, m.patchW.data(), 3 * P * P, emb.data(), D, false);
    Matrix x(N + 1, D);
    for (std::size_t d = 0; d < D; ++d) x.data()[d] = m.cls[d] + m.pos[d];
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t d = 0; d < D; ++d) x.data()[(n + 1) * D + d] = emb.data()[n * D + d] + m.pos[(n + 1) * D + d];
    layerNormRows(x, m.preG, m.preB, m.eps);
    for (const Impl::Layer& L : m.blocks) {
        Matrix a = x;
        layerNormRows(a, L.n1g, L.n1b, m.eps);
        const Matrix att = L.o.forward(multiHeadAttention(L.q.forward(a), L.k.forward(a), L.v.forward(a), m.heads));
        for (std::size_t i = 0; i < x.size(); ++i) x.data()[i] += att.data()[i];
        a = x;
        layerNormRows(a, L.n2g, L.n2b, m.eps);
        Matrix h = L.fc1.forward(a);
        for (float& v : h.values()) v = m.quickGelu ? v / (1.0f + std::exp(-1.702f * v)) : 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
        const Matrix mlp = L.fc2.forward(h);
        for (std::size_t i = 0; i < x.size(); ++i) x.data()[i] += mlp.data()[i];
    }
    Matrix pooled(1, D);
    std::copy(x.data(), x.data() + D, pooled.data());
    layerNormRows(pooled, m.postG, m.postB, m.eps);
    return m.proj.forward(pooled);
}

Matrix ClipImageEncoder::embed(const Image& image) {
    const std::size_t S = impl_->size;
    // Shorter side to S, centre crop S x S (fitImage), CLIP normalization.
    const Image im = fitImage(toRgb(image), S, S);
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f}, stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    std::vector<float> chw(3 * S * S);
    for (std::size_t i = 0; i < S * S; ++i)
        for (std::size_t c = 0; c < 3; ++c) chw[c * S * S + i] = (im.pixels[i * 3 + c] / 255.0f - mean[c]) / stdv[c];
    return forward(chw);
}

IpAdapter::IpAdapter(const std::string& file) : file_(file) {
    const SafeTensors st(file);
    if (!st.has("image_proj.proj.weight"))
        throw std::runtime_error("IpAdapter: " + file + " has no image_proj.proj (only the basic SD 1.5 IP-Adapter is supported)");
    const auto& info = st.info("image_proj.proj.weight");
    const Matrix w = st.load("image_proj.proj.weight");
    projT = Matrix(info.shape[0], info.shape[1], std::vector<float>(w.data(), w.data() + w.size())).transpose();
    projBias = vec(st, "image_proj.proj.bias");
    normG = vec(st, "image_proj.norm.weight");
    normB = vec(st, "image_proj.norm.bias");
    dim_ = normG.size();
    tokens_ = projBias.size() / dim_;
}

Matrix IpAdapter::tokens(const Matrix& e) const {
    if (e.rows() != 1 || e.cols() != projT.rows()) throw std::invalid_argument("IpAdapter::tokens: expected a 1 x " + std::to_string(projT.rows()) + " image embedding");
    Matrix y(1, projT.cols());
    gemm(false, false, 1, projT.cols(), projT.rows(), e.data(), projT.rows(), projT.data(), projT.cols(), y.data(), projT.cols(), false);
    Matrix t(tokens_, dim_);
    for (std::size_t i = 0; i < y.size(); ++i) t.data()[i] = y.data()[i] + projBias[i];
    layerNormRows(t, normG, normB, 1e-5f);
    return t;
}

} // namespace nc
