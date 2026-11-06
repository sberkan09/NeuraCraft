#include "GpuGraph.h"

#include "Backend.h"
#include "Conv.h"
#include "Image.h"
#include "Profile.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace nc::gpugraph {

#ifndef NEURACRAFT_HAS_CUDA

bool enabled() { return false; }
bool forward(Network&, const Matrix&, Matrix&) { return false; }

#else

bool enabled() {
    static const bool off = [] {
        const char* v = std::getenv("NEURACRAFT_GPU_GRAPH");
        return v && std::strcmp(v, "0") == 0;
    }();
    return !off && currentDevice() == Device::GPU && cuda::available();
}

namespace {
namespace dv = cuda::dev;

bool supported(Network& net);

bool supported(Layer& l) {
    if (auto* c = dynamic_cast<Conv2D*>(&l)) return !c->quantized();
    if (dynamic_cast<GroupNorm*>(&l) || dynamic_cast<SiLU*>(&l) || dynamic_cast<Upsample2D*>(&l) || dynamic_cast<Pad2D*>(&l)) return true;
    if (auto* r = dynamic_cast<ResnetBlock2D*>(&l)) return supported(r->main()) && supported(r->shortcut());
    if (auto* a = dynamic_cast<SpatialSelfAttention*>(&l)) return a->heads() > 0 && a->shape().channels % a->heads() == 0;
    return false;
}

bool supported(Network& net) {
    for (std::size_t i = 0; i < net.layerCount(); ++i)
        if (!supported(net.layer(i))) return false;
    return true;
}

// The activation between two layers: owned, or borrowed from the caller (never written then).
struct Value {
    DeviceTensor owned;
    const float* borrowed = nullptr;
    std::size_t channels = 0, height = 0, width = 0;
    const float* get() const { return borrowed ? borrowed : owned.get(); }
    std::size_t size() const { return channels * height * width; }
};

// Makes `v` writable (copies a borrowed input).
float* writable(Value& v, std::size_t B) {
    if (v.borrowed) {
        DeviceTensor t(B * v.size());
        dv::copy(t.get(), v.borrowed, B * v.size());
        v.owned = std::move(t);
        v.borrowed = nullptr;
    }
    return v.owned.get();
}

Value run(Network& net, const float* input, std::size_t B, ImageShape shape);

void attention(SpatialSelfAttention& a, Value& v, std::size_t B) {
    const std::size_t C = v.channels, HW = v.height * v.width;
    GroupNorm& n = a.norm();
    DeviceTensor normed(B * C * HW);
    dv::groupNorm(v.get(), normed.get(), B, C, HW, n.groups(), n.epsilon(), n.gamma().data(), n.beta().data(), false);
    DeviceTensor tokens(HW * C), q(HW * C), k(HW * C), val(HW * C), o(HW * C), y(HW * C);
    DeviceTensor out(B * C * HW);
    for (std::size_t b = 0; b < B; ++b) {
        dv::toTokens(normed.get() + b * C * HW, tokens.get(), C, HW);
        // The projections are PyTorch Linear (out x in), applied to every pixel's channel vector.
        dv::linear(tokens.get(), HW, C, a.query().data(), true, a.queryBias().data(), C, q.get());
        dv::linear(tokens.get(), HW, C, a.key().data(), true, a.keyBias().data(), C, k.get());
        dv::linear(tokens.get(), HW, C, a.value().data(), true, a.valueBias().data(), C, val.get());
        dv::attention(q.get(), k.get(), val.get(), o.get(), HW, HW, C, a.heads());
        dv::linear(o.get(), HW, C, a.out().data(), true, a.outBias().data(), C, y.get());
        dv::fromTokens(y.get(), out.get() + b * C * HW, HW, C);
    }
    dv::add(out.get(), v.get(), B * C * HW);
    v.owned = std::move(out);
    v.borrowed = nullptr;
}

Value run(Network& net, const float* input, std::size_t B, ImageShape shape) {
    Value v;
    v.borrowed = input;
    v.channels = shape.channels;
    v.height = shape.height;
    v.width = shape.width;
    for (std::size_t i = 0; i < net.layerCount(); ++i) {
        Layer& l = net.layer(i);
        if (auto* c = dynamic_cast<Conv2D*>(&l)) {
            const ImageShape in = c->inputShape(), out = c->outputShape();
            if (in.channels != v.channels || in.height != v.height || in.width != v.width)
                throw std::logic_error("gpugraph: Conv2D input shape mismatch");
            DeviceTensor y(B * out.size());
            dv::conv2d(v.get(), B, in.channels, in.height, in.width, c->weights().data(), false, c->bias().empty() ? nullptr : c->bias().data(),
                       out.channels, c->kernelSize(), c->stride(), c->padding(), y.get());
            v.owned = std::move(y);
            v.borrowed = nullptr;
            v.channels = out.channels;
            v.height = out.height;
            v.width = out.width;
        } else if (auto* g = dynamic_cast<GroupNorm*>(&l)) {
            // GroupNorm then SiLU (every norm of a VAE): one pass.
            const bool silu = i + 1 < net.layerCount() && dynamic_cast<SiLU*>(&net.layer(i + 1));
            DeviceTensor y(B * v.size());
            dv::groupNorm(v.get(), y.get(), B, v.channels, v.height * v.width, g->groups(), g->epsilon(), g->gamma().data(), g->beta().data(), silu);
            v.owned = std::move(y);
            v.borrowed = nullptr;
            if (silu) ++i;
        } else if (dynamic_cast<SiLU*>(&l)) {
            dv::silu(writable(v, B), B * v.size());
        } else if (auto* u = dynamic_cast<Upsample2D*>(&l)) {
            const std::size_t f = u->factor();
            DeviceTensor y(B * v.size() * f * f);
            dv::upsample(v.get(), y.get(), B, v.channels, v.height, v.width, f);
            v.owned = std::move(y);
            v.borrowed = nullptr;
            v.height *= f;
            v.width *= f;
        } else if (auto* p = dynamic_cast<Pad2D*>(&l)) {
            const std::size_t H = v.height + p->top() + p->bottom(), W = v.width + p->left() + p->right();
            DeviceTensor y(B * v.channels * H * W);
            dv::pad(v.get(), y.get(), B, v.channels, v.height, v.width, p->top(), p->bottom(), p->left(), p->right());
            v.owned = std::move(y);
            v.borrowed = nullptr;
            v.height = H;
            v.width = W;
        } else if (auto* r = dynamic_cast<ResnetBlock2D*>(&l)) {
            const ImageShape s{v.channels, v.height, v.width};
            Value h = run(r->main(), v.get(), B, s);
            float* out = writable(h, B);
            if (r->shortcut().layerCount()) {
                const Value sc = run(r->shortcut(), v.get(), B, s);
                dv::add(out, sc.get(), B * h.size());
            } else {
                dv::add(out, v.get(), B * h.size());
            }
            v = std::move(h);
        } else if (auto* a = dynamic_cast<SpatialSelfAttention*>(&l)) {
            attention(*a, v, B);
        } else {
            throw std::logic_error("gpugraph: unsupported layer " + l.name());
        }
    }
    return v;
}

// The shape of a network's input, from its first layer.
bool inputShape(Network& net, ImageShape& s) {
    if (net.layerCount() == 0) return false;
    Layer& l = net.layer(0);
    if (auto* c = dynamic_cast<Conv2D*>(&l)) s = c->inputShape();
    else if (auto* g = dynamic_cast<GroupNorm*>(&l)) s = g->shape();
    else if (auto* u = dynamic_cast<Upsample2D*>(&l)) s = u->inputShape();
    else if (auto* p = dynamic_cast<Pad2D*>(&l)) s = p->inputShape();
    else if (auto* r = dynamic_cast<ResnetBlock2D*>(&l)) s = r->inputShape();
    else if (auto* a = dynamic_cast<SpatialSelfAttention*>(&l)) s = a->shape();
    else return false;
    return true;
}
} // namespace

bool forward(Network& net, const Matrix& input, Matrix& output) {
    if (!enabled() || input.rows() == 0) return false;
    ImageShape shape;
    if (!inputShape(net, shape) || input.cols() != shape.size() || !supported(net)) return false;
    NC_PROFILE("gpu graph");
    dv::Session session;
    const DeviceTensor x = DeviceTensor::upload(input);
    const Value y = run(net, x.get(), input.rows(), shape);
    output = Matrix(input.rows(), y.size());
    dv::download(output.data(), y.get(), output.size());
    return true;
}

#endif

} // namespace nc::gpugraph
