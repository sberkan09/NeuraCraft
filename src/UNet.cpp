#include "Backend.h"
#include "Dense.h"
#include "Diffusion.h"
#include "Json.h"
#include "Parallel.h"
#include "Pretrained.h"
#include "Random.h"
#include "SafeTensors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

struct DownBlock {
    std::vector<std::unique_ptr<TimeResnet>> resnets;
    std::vector<std::unique_ptr<CrossTransformer>> attentions;
    std::unique_ptr<Conv2D> downsample;
};
struct UpBlock {
    std::vector<std::unique_ptr<TimeResnet>> resnets;
    std::vector<std::unique_ptr<CrossTransformer>> attentions;
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
    std::vector<UpBlock> ups;
    std::unique_ptr<GroupNorm> normOut;
};

UNet2DCondition::UNet2DCondition() = default;
UNet2DCondition::~UNet2DCondition() = default;
UNet2DCondition::UNet2DCondition(UNet2DCondition&&) noexcept = default;
UNet2DCondition& UNet2DCondition::operator=(UNet2DCondition&&) noexcept = default;

ImageShape UNet2DCondition::latentShape() const { return impl_->latent; }
std::size_t UNet2DCondition::contextDim() const { return impl_->contextDim; }
std::size_t UNet2DCondition::parameterCount() const { return impl_->params; }

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

UNet2DCondition loadUNet(const std::string& directory, std::size_t latentWidth, std::size_t latentHeight, bool int8) {
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
    const std::size_t tembDim = 4 * ch[0];
    Weights w(file);
    w.int8 = int8;
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
        for (std::size_t j = 0; j < L; ++j) {
            d.resnets.push_back(std::make_unique<TimeResnet>(s, ch[i], tembDim, groups, eps));
            d.resnets.back()->load(w, p + ".resnets." + std::to_string(j));
            s.channels = ch[i];
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
        for (std::size_t j = 0; j <= L; ++j) {
            const ImageShape in{s.channels + skipChannels.back(), s.height, s.width};
            skipChannels.pop_back();
            u.resnets.push_back(std::make_unique<TimeResnet>(in, out, tembDim, groups, eps));
            u.resnets.back()->load(w, p + ".resnets." + std::to_string(j));
            s.channels = out;
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

double EulerScheduler::inputScale(std::size_t i) const { return 1.0 / std::sqrt(sigmas.at(i) * sigmas.at(i) + 1.0); }

void EulerScheduler::step(std::size_t i, const Matrix& noise, Matrix& latent) const {
    if (!noise.sameShape(latent)) throw std::invalid_argument("EulerScheduler::step: shapes differ");
    // x0 = x - sigma eps; d = (x - x0) / sigma = eps; x_next = x + d (sigma_next - sigma)
    const float dt = static_cast<float>(sigmas.at(i + 1) - sigmas.at(i));
    for (std::size_t k = 0; k < latent.size(); ++k) latent.data()[k] += noise.data()[k] * dt;
}

// ---- Text to image -------------------------------------------------------------------------

Matrix textToLatent(const std::string& dir, const std::string& prompt, const TextToImageOptions& o) {
    using Clock = std::chrono::steady_clock;
    auto report = [&](const std::string& stage, Clock::time_point start) {
        if (o.progress) o.progress(stage, std::chrono::duration<double>(Clock::now() - start).count());
    };
    if (o.width % 64 || o.height % 64 || o.width == 0 || o.height == 0)
        throw std::invalid_argument("textToImage: width and height must be multiples of 64");
    const bool guided = o.guidance > 1.0f;

    Matrix context, negative;
    {
        auto t = Clock::now();
        const ClipTokenizer tok = ClipTokenizer::fromDirectory(dir + "/tokenizer");
        ClipTextEncoder enc = loadClipTextEncoder(dir, o.int8);
        // SD 2 / Turbo pad with "!" (id 0); SD 1.x pads with <|endoftext|>.
        int pad = tok.endOfText();
        {
            std::ifstream cfg(dir + "/tokenizer/tokenizer_config.json");
            if (cfg) {
                const json::Value c = json::parse(readFile(dir + "/tokenizer/tokenizer_config.json"));
                if (const json::Value* p = c.find("pad_token"); p && p->isString() && p->string == "!") pad = 0;
            }
        }
        context = enc.encode(tok.encodeForModel(prompt, enc.length, pad));
        if (guided) negative = enc.encode(tok.encodeForModel(o.negativePrompt, enc.length, pad));
        report("text encoder", t);
    }

    const std::size_t lw = o.width / 8, lh = o.height / 8;
    const std::string schedulerConfig = dir + "/scheduler/scheduler_config.json";
    const EulerScheduler sched = exists(schedulerConfig) ? EulerScheduler::fromConfig(schedulerConfig, o.steps) : EulerScheduler::standard(o.steps);
    Matrix latent;
    {
        auto t = Clock::now();
        UNet2DCondition unet = loadUNet(dir, lw, lh, o.int8);
        report("load unet", t);
        setSeed(o.seed);
        latent = Matrix::normal(1, unet.latentShape().size(), 0.0f, 1.0f);
        for (float& v : latent.values()) v *= static_cast<float>(sched.initNoiseSigma);
        for (std::size_t i = 0; i < sched.timesteps.size(); ++i) {
            t = Clock::now();
            Matrix in = latent;
            const float scale = static_cast<float>(sched.inputScale(i));
            for (float& v : in.values()) v *= scale;
            Matrix eps = unet.forward(in, static_cast<float>(sched.timesteps[i]), context);
            if (guided) {
                const Matrix uncond = unet.forward(in, static_cast<float>(sched.timesteps[i]), negative);
                for (std::size_t k = 0; k < eps.size(); ++k)
                    eps.data()[k] = uncond.data()[k] + o.guidance * (eps.data()[k] - uncond.data()[k]);
            }
            sched.step(i, eps, latent);
            report("step " + std::to_string(i + 1) + "/" + std::to_string(sched.timesteps.size()), t);
        }
    }
    return latent;
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

} // namespace nc
