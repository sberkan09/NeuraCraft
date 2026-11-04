#include "Diffusion.h"
#include "Json.h"
#include "SafeTensors.h"

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

} // namespace nc
