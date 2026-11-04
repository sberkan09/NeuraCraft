#include "Diffusion.h"
#include "Activations.h"
#include "Dense.h"
#include "Json.h"
#include "SafeTensors.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

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

// Sets parameters from a diffusers checkpoint while the network is being built.
class Weights {
public:
    explicit Weights(const std::string& file) : file_(file), st_(file) {}

    Matrix vector(const std::string& name, std::size_t n) {
        Matrix m = st_.load(name);
        if (m.size() != n) fail(name, m.size(), n);
        return Matrix(1, n, std::move(m.values()));
    }
    // PyTorch (out, in, k, k) -> Conv2D's (in*k*k) x out.
    void conv(Conv2D& c, const std::string& prefix) {
        const ImageShape in = c.inputShape();
        const std::size_t out = c.outputShape().channels, patch = in.channels * c.kernelSize() * c.kernelSize();
        Matrix w = st_.load(prefix + ".weight");
        if (w.size() != out * patch) fail(prefix + ".weight", w.size(), out * patch);
        c.weights() = Matrix(out, patch, std::move(w.values())).transpose();
        c.bias() = vector(prefix + ".bias", out);
    }
    void convNoBias(Conv2D& c, const std::string& prefix) {
        const ImageShape in = c.inputShape();
        const std::size_t out = c.outputShape().channels, patch = in.channels * c.kernelSize() * c.kernelSize();
        Matrix w = st_.load(prefix + ".weight");
        if (w.size() != out * patch) fail(prefix + ".weight", w.size(), out * patch);
        c.weights() = Matrix(out, patch, std::move(w.values())).transpose();
        c.bias() = Matrix(1, out);
    }
    void norm(GroupNorm& g, const std::string& prefix, std::size_t channels) {
        g.gamma() = vector(prefix + ".weight", channels);
        g.beta() = vector(prefix + ".bias", channels);
    }
    void linear(Matrix& w, Matrix& b, const std::string& prefix, std::size_t out, std::size_t in) {
        Matrix m = st_.load(prefix + ".weight");
        if (m.size() != out * in) fail(prefix + ".weight", m.size(), out * in);  // (out, in) or (out, in, 1, 1)
        w = Matrix(out, in, std::move(m.values()));
        b = vector(prefix + ".bias", out);
    }

private:
    [[noreturn]] void fail(const std::string& name, std::size_t got, std::size_t want) {
        throw std::runtime_error("loadVae: " + name + " in " + file_ + " has " + std::to_string(got) + " values, expected " +
                                 std::to_string(want));
    }
    std::string file_;
    SafeTensors st_;
};

template <typename T>
T& as(Layer& layer, const char* what) {
    auto* p = dynamic_cast<T*>(&layer);
    if (!p) throw std::logic_error(std::string("loadVae: unexpected layer layout (") + what + ")");
    return *p;
}

struct Builder {
    Network& net;
    Weights& w;
    ImageShape s;
    std::size_t groups;
    float eps;

    void conv(std::size_t out, std::size_t k, std::size_t stride, std::size_t pad, const std::string& name) {
        auto& c = net.add<Conv2D>(s, out, k, stride, pad);
        w.conv(c, name);
        s = c.outputShape();
    }
    void resnet(std::size_t out, const std::string& p) {
        auto& r = net.add<ResnetBlock2D>(s, out, groups, eps);
        w.norm(as<GroupNorm>(r.main().layer(0), "norm1"), p + ".norm1", s.channels);
        w.conv(as<Conv2D>(r.main().layer(2), "conv1"), p + ".conv1");
        w.norm(as<GroupNorm>(r.main().layer(3), "norm2"), p + ".norm2", out);
        w.conv(as<Conv2D>(r.main().layer(5), "conv2"), p + ".conv2");
        if (r.shortcut().layerCount()) w.conv(as<Conv2D>(r.shortcut().layer(0), "conv_shortcut"), p + ".conv_shortcut");
        s = r.outputShape();
    }
    void attention(const std::string& p) {
        auto& a = net.add<SpatialSelfAttention>(s, 1, groups, eps);
        const std::size_t C = s.channels;
        w.norm(a.norm(), p + ".group_norm", C);
        w.linear(a.query(), a.queryBias(), p + ".to_q", C, C);
        w.linear(a.key(), a.keyBias(), p + ".to_k", C, C);
        w.linear(a.value(), a.valueBias(), p + ".to_v", C, C);
        w.linear(a.out(), a.outBias(), p + ".to_out.0", C, C);
    }
    void normOut(const std::string& p) {
        auto& g = net.add<GroupNorm>(s, groups, eps);
        w.norm(g, p, s.channels);
        net.add<SiLU>();
    }
};
} // namespace

Matrix Vae::encode(const Matrix& img) {
    const Matrix moments = encoder.predict(img);
    const std::size_t n = latent.size();
    Matrix z(moments.rows(), n);
    for (std::size_t r = 0; r < moments.rows(); ++r)
        for (std::size_t i = 0; i < n; ++i) z(r, i) = moments(r, i) * scalingFactor;  // mean: the first half
    return z;
}

Matrix Vae::decode(const Matrix& z) {
    Matrix unscaled = z;
    for (float& v : unscaled.values()) v /= scalingFactor;
    return decoder.predict(unscaled);
}

Vae loadVae(const std::string& directory, std::size_t width, std::size_t height, bool withEncoder, bool withDecoder) {
    const std::string dir = exists(directory + "/config.json") ? directory : directory + "/vae";
    const json::Value c = json::parse(readFile(dir + "/config.json"));
    std::string file = dir + "/diffusion_pytorch_model.safetensors";
    if (!exists(file)) file = dir + "/diffusion_pytorch_model.fp16.safetensors";
    if (!exists(file)) throw std::runtime_error("loadVae: no diffusion_pytorch_model(.fp16).safetensors in " + dir);

    std::vector<std::size_t> channels;
    for (const auto& v : c.at("block_out_channels").array) channels.push_back(static_cast<std::size_t>(v.number));
    if (channels.empty()) throw std::runtime_error("loadVae: block_out_channels is empty");
    const std::size_t layers = static_cast<std::size_t>(c.numberOr("layers_per_block", 2));
    const std::size_t groups = static_cast<std::size_t>(c.numberOr("norm_num_groups", 32));
    const std::size_t latentC = static_cast<std::size_t>(c.numberOr("latent_channels", 4));
    const std::size_t inC = static_cast<std::size_t>(c.numberOr("in_channels", 3));
    const std::size_t outC = static_cast<std::size_t>(c.numberOr("out_channels", 3));
    if (const json::Value* act = c.find("act_fn"); act && act->string != "silu")
        throw std::runtime_error("loadVae: activation '" + act->string + "' is not supported (silu only)");
    const std::size_t factor = std::size_t(1) << (channels.size() - 1);
    if (width == 0 || height == 0 || width % factor || height % factor)
        throw std::invalid_argument("loadVae: width and height must be multiples of " + std::to_string(factor));
    const float eps = 1e-6f;

    Vae vae;
    vae.scalingFactor = static_cast<float>(c.numberOr("scaling_factor", 0.18215));
    vae.image = {inC, height, width};
    vae.latent = {latentC, height / factor, width / factor};
    Weights w(file);
    SkipParameterInit skip;

    if (withEncoder) {
        Builder b{vae.encoder, w, vae.image, groups, eps};
        b.conv(channels[0], 3, 1, 1, "encoder.conv_in");
        for (std::size_t i = 0; i < channels.size(); ++i) {
            const std::string p = "encoder.down_blocks." + std::to_string(i);
            for (std::size_t j = 0; j < layers; ++j) b.resnet(channels[i], p + ".resnets." + std::to_string(j));
            if (i + 1 < channels.size()) {
                auto& pad = vae.encoder.add<Pad2D>(b.s, 0, 1, 0, 1);  // diffusers pads (0, 1, 0, 1) before a stride-2 conv
                b.s = pad.outputShape();
                b.conv(channels[i], 3, 2, 0, p + ".downsamplers.0.conv");
            }
        }
        b.resnet(b.s.channels, "encoder.mid_block.resnets.0");
        b.attention("encoder.mid_block.attentions.0");
        b.resnet(b.s.channels, "encoder.mid_block.resnets.1");
        b.normOut("encoder.conv_norm_out");
        b.conv(2 * latentC, 3, 1, 1, "encoder.conv_out");
        b.conv(2 * latentC, 1, 1, 0, "quant_conv");
    }
    if (withDecoder) {
        Builder b{vae.decoder, w, vae.latent, groups, eps};
        b.conv(latentC, 1, 1, 0, "post_quant_conv");
        b.conv(channels.back(), 3, 1, 1, "decoder.conv_in");
        b.resnet(b.s.channels, "decoder.mid_block.resnets.0");
        b.attention("decoder.mid_block.attentions.0");
        b.resnet(b.s.channels, "decoder.mid_block.resnets.1");
        for (std::size_t i = 0; i < channels.size(); ++i) {
            const std::string p = "decoder.up_blocks." + std::to_string(i);
            const std::size_t out = channels[channels.size() - 1 - i];
            for (std::size_t j = 0; j <= layers; ++j) b.resnet(out, p + ".resnets." + std::to_string(j));
            if (i + 1 < channels.size()) {
                auto& up = vae.decoder.add<Upsample2D>(b.s, 2);
                b.s = up.outputShape();
                b.conv(out, 3, 1, 1, p + ".upsamplers.0.conv");
            }
        }
        b.normOut("decoder.conv_norm_out");
        b.conv(outC, 3, 1, 1, "decoder.conv_out");
    }
    return vae;
}

// ---- Tiny autoencoder (TAESD) -----------------------------------------------------------------

namespace {
// AutoencoderTinyBlock: relu(conv(relu(conv(relu(conv(x))))) + skip(x)), skip = identity or a 1x1 conv.
void tinyBlock(Network& net, Weights& w, ImageShape s, std::size_t out, const std::string& p) {
    auto& r = net.add<Residual>();
    auto& c0 = r.add<Conv2D>(s, out, 3, 1, 1);
    w.conv(c0, p + ".conv.0");
    r.add<ReLU>();
    auto& c2 = r.add<Conv2D>(ImageShape{out, s.height, s.width}, out, 3, 1, 1);
    w.conv(c2, p + ".conv.2");
    r.add<ReLU>();
    auto& c4 = r.add<Conv2D>(ImageShape{out, s.height, s.width}, out, 3, 1, 1);
    w.conv(c4, p + ".conv.4");
    if (s.channels != out) throw std::runtime_error("loadTinyVae: blocks that change the channel count are not supported");
    net.add<ReLU>();
}

void squashInPlace(Matrix& m, float scale, float offset) {
    for (float& v : m.values()) v = v * scale + offset;
}
} // namespace

Matrix TinyVae::encode(const Matrix& img) {
    Matrix x = img;
    squashInPlace(x, 0.5f, 0.5f);  // [-1, 1] -> [0, 1]
    return encoder.predict(x);
}

Matrix TinyVae::decode(const Matrix& z) {
    Matrix x = z;
    for (float& v : x.values()) v = std::tanh(v / 3.0f) * 3.0f;  // soft clamp to (-3, 3)
    Matrix y = decoder.predict(x);
    squashInPlace(y, 2.0f, -1.0f);  // [0, 1] -> [-1, 1]
    return y;
}

TinyVae loadTinyVae(const std::string& dir, std::size_t width, std::size_t height, bool withEncoder, bool withDecoder) {
    const json::Value c = json::parse(readFile(dir + "/config.json"));
    const std::string file = dir + "/diffusion_pytorch_model.safetensors";
    if (!exists(file)) throw std::runtime_error("loadTinyVae: no diffusion_pytorch_model.safetensors in " + dir);
    if (const json::Value* a = c.find("act_fn"); a && a->string != "relu")
        throw std::runtime_error("loadTinyVae: activation '" + a->string + "' is not supported (relu only)");
    if (const json::Value* sf = c.find("shift_factor"); sf && sf->isNumber() && sf->number != 0.0)
        throw std::runtime_error("loadTinyVae: shift_factor is not supported");
    auto list = [&](const char* key) {
        std::vector<std::size_t> v;
        for (const auto& e : c.at(key).array) v.push_back(static_cast<std::size_t>(e.number));
        return v;
    };
    const std::vector<std::size_t> encCh = list("encoder_block_out_channels"), encN = list("num_encoder_blocks");
    const std::vector<std::size_t> decCh = list("decoder_block_out_channels"), decN = list("num_decoder_blocks");
    const std::size_t latentC = static_cast<std::size_t>(c.numberOr("latent_channels", 4));
    const std::size_t inC = static_cast<std::size_t>(c.numberOr("in_channels", 3)), outC = static_cast<std::size_t>(c.numberOr("out_channels", 3));
    const std::size_t up = static_cast<std::size_t>(c.numberOr("upsampling_scaling_factor", 2));
    const std::size_t factor = std::size_t(1) << (decCh.size() - 1);
    if (width == 0 || height == 0 || width % factor || height % factor)
        throw std::invalid_argument("loadTinyVae: width and height must be multiples of " + std::to_string(factor));

    TinyVae vae;
    vae.image = {inC, height, width};
    vae.latent = {latentC, height / factor, width / factor};
    Weights w(file);
    SkipParameterInit skip;
    if (withEncoder) {
        Network& net = vae.encoder;
        ImageShape s = vae.image;
        std::size_t idx = 0;
        for (std::size_t i = 0; i < encCh.size(); ++i) {
            const std::string p = "encoder.layers." + std::to_string(idx++);
            auto& conv = i == 0 ? net.add<Conv2D>(s, encCh[0], 3, 1, 1) : net.add<Conv2D>(s, encCh[i], 3, 2, 1);
            if (i == 0) w.conv(conv, p);
            else w.convNoBias(conv, p);
            s = conv.outputShape();
            for (std::size_t b = 0; b < encN[i]; ++b) tinyBlock(net, w, s, encCh[i], "encoder.layers." + std::to_string(idx++));
        }
        auto& out = net.add<Conv2D>(s, latentC, 3, 1, 1);
        w.conv(out, "encoder.layers." + std::to_string(idx));
    }
    if (withDecoder) {
        Network& net = vae.decoder;
        ImageShape s = vae.latent;
        std::size_t idx = 0;
        auto& in = net.add<Conv2D>(s, decCh[0], 3, 1, 1);
        w.conv(in, "decoder.layers." + std::to_string(idx));
        idx += 2;  // conv, activation
        net.add<ReLU>();
        s = in.outputShape();
        for (std::size_t i = 0; i < decCh.size(); ++i) {
            const bool last = i + 1 == decCh.size();
            for (std::size_t b = 0; b < decN[i]; ++b) tinyBlock(net, w, s, decCh[i], "decoder.layers." + std::to_string(idx++));
            if (!last) {
                auto& u = net.add<Upsample2D>(s, up);
                s = u.outputShape();
                ++idx;
            }
            auto& conv = net.add<Conv2D>(s, last ? outC : decCh[i], 3, 1, 1);
            if (last) w.conv(conv, "decoder.layers." + std::to_string(idx++));
            else w.convNoBias(conv, "decoder.layers." + std::to_string(idx++));
            s = conv.outputShape();
        }
    }
    return vae;
}

// ---- CLIP text encoder ------------------------------------------------------------------------

Matrix ClipTextEncoder::encode(const std::vector<int>& ids) {
    if (ids.size() != length)
        throw std::invalid_argument("ClipTextEncoder::encode: expected " + std::to_string(length) + " ids, got " +
                                    std::to_string(ids.size()));
    Matrix x(1, ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) x(0, i) = static_cast<float>(ids[i]);
    return net.predict(x);
}

ClipTextEncoder loadClipTextEncoder(const std::string& directory, bool int8, const std::vector<Lora>& loras) {
    const std::string dir = exists(directory + "/config.json") ? directory : directory + "/text_encoder";
    const json::Value c = json::parse(readFile(dir + "/config.json"));
    std::string file = dir + "/model.safetensors";
    if (!exists(file)) file = dir + "/model.fp16.safetensors";
    if (!exists(file)) throw std::runtime_error("loadClipTextEncoder: no model(.fp16).safetensors in " + dir);
    const std::string act = c.find("hidden_act") ? c.at("hidden_act").string : "quick_gelu";
    GELU::Approximation gelu;
    if (act == "gelu") gelu = GELU::Approximation::Erf;
    else if (act == "quick_gelu") gelu = GELU::Approximation::Sigmoid;
    else throw std::runtime_error("loadClipTextEncoder: activation '" + act + "' is not supported");

    ClipTextEncoder enc;
    const std::size_t V = static_cast<std::size_t>(c.at("vocab_size").number);
    const std::size_t D = static_cast<std::size_t>(c.at("hidden_size").number);
    const std::size_t H = static_cast<std::size_t>(c.at("num_attention_heads").number);
    const std::size_t F = static_cast<std::size_t>(c.at("intermediate_size").number);
    enc.layers = static_cast<std::size_t>(c.at("num_hidden_layers").number);
    enc.length = static_cast<std::size_t>(c.numberOr("max_position_embeddings", 77));
    enc.dim = D;
    const float eps = static_cast<float>(c.numberOr("layer_norm_eps", 1e-5));
    SafeTensors st(file);
    LoraMerger lora(loras, "text_encoder");
    auto load = [&](const std::string& name, std::size_t rows, std::size_t cols) {
        Matrix m = st.load(name);
        lora.apply(name, m);
        if (m.size() != rows * cols)
            throw std::runtime_error("loadClipTextEncoder: " + name + " has " + std::to_string(m.size()) + " values, expected " +
                                     std::to_string(rows * cols));
        return Matrix(rows, cols, std::move(m.values()));
    };
    const std::string p = "text_model.";
    {
        SkipParameterInit skip;
        enc.context = std::make_shared<SequenceContext>();
        auto& emb = enc.net.add<TokenEmbedding>(V, D, enc.length, enc.context, true);
        emb.tokenTable() = load(p + "embeddings.token_embedding.weight", V, D);
        emb.positionTable() = load(p + "embeddings.position_embedding.weight", enc.length, D);
        for (std::size_t i = 0; i < enc.layers; ++i) {
            const std::string l = p + "encoder.layers." + std::to_string(i) + ".";
            auto& a = enc.net.add<Residual>();
            auto& ln1 = a.add<LayerNorm>(D, eps);
            ln1.gamma() = load(l + "layer_norm1.weight", 1, D);
            ln1.beta() = load(l + "layer_norm1.bias", 1, D);
            auto& att = a.add<CausalSelfAttention>(D, H, enc.context);
            // [Q | K | V] stacked as (3D x D) out x in, transposed to the layer's in x out.
            Matrix qkv(3 * D, D);
            const char* parts[3] = {"q_proj", "k_proj", "v_proj"};
            Matrix bias(1, 3 * D);
            for (std::size_t k = 0; k < 3; ++k) {
                const Matrix w = load(l + "self_attn." + parts[k] + ".weight", D, D);
                std::copy(w.data(), w.data() + w.size(), qkv.data() + k * D * D);
                const Matrix b = load(l + "self_attn." + parts[k] + ".bias", 1, D);
                std::copy(b.data(), b.data() + D, bias.data() + k * D);
            }
            att.setWeights(qkv.transpose(), load(l + "self_attn.out_proj.weight", D, D).transpose());
            att.qkvBias() = std::move(bias);
            att.outBias() = load(l + "self_attn.out_proj.bias", 1, D);

            auto& m = enc.net.add<Residual>();
            auto& ln2 = m.add<LayerNorm>(D, eps);
            ln2.gamma() = load(l + "layer_norm2.weight", 1, D);
            ln2.beta() = load(l + "layer_norm2.bias", 1, D);
            auto& fc1 = m.add<Dense>(D, F);
            fc1.weights() = load(l + "mlp.fc1.weight", F, D).transpose();
            fc1.bias() = load(l + "mlp.fc1.bias", 1, F);
            m.add<GELU>(gelu);
            auto& fc2 = m.add<Dense>(F, D);
            fc2.weights() = load(l + "mlp.fc2.weight", D, F).transpose();
            fc2.bias() = load(l + "mlp.fc2.bias", 1, D);
        }
        auto& lnf = enc.net.add<LayerNorm>(D, eps);
        lnf.gamma() = load(p + "final_layer_norm.weight", 1, D);
        lnf.beta() = load(p + "final_layer_norm.bias", 1, D);
    }
    enc.net.releaseGradients();
    if (int8) enc.net.quantize();
    return enc;
}

} // namespace nc
