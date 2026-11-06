// The ONNX operators: n-dimensional, row-major, broadcasting like numpy. Products and convolutions
// go to the library's GEMM (gemm() in Backend.h); the rest are plain loops, parallel where they
// are big.

#include "OnnxOps.h"

#include "Backend.h"
#include "Parallel.h"
#include "VectorMath.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nc::onnx {

namespace {

using Shape = std::vector<std::int64_t>;
using Ins = std::vector<const Tensor*>;
using Outs = std::vector<Tensor>;

std::size_t numel(const Shape& s) {
    std::size_t n = 1;
    for (std::int64_t d : s) {
        if (d < 0) throw std::runtime_error("negative dimension");
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

std::string shapeText(const Shape& s) {
    std::string t = "[";
    for (std::size_t k = 0; k < s.size(); ++k) t += (k ? "," : "") + std::to_string(s[k]);
    return t + "]";
}

std::vector<std::size_t> strides(const Shape& s) {
    std::vector<std::size_t> st(s.size(), 1);
    for (std::size_t k = s.size(); k-- > 1;) st[k - 1] = st[k] * static_cast<std::size_t>(s[k]);
    return st;
}

std::int64_t normAxis(std::int64_t axis, std::size_t rank) {
    const std::int64_t r = static_cast<std::int64_t>(rank);
    if (axis < -r || axis >= std::max<std::int64_t>(r, 1)) throw std::runtime_error("axis " + std::to_string(axis) + " out of range for rank " + std::to_string(rank));
    return axis < 0 ? axis + r : axis;
}

Tensor floatTensor(Shape s) {
    Tensor t;
    t.kind = Tensor::Kind::Float;
    t.elemType = 1;
    t.f.assign(numel(s), 0.0f);
    t.shape = std::move(s);
    return t;
}
Tensor intTensor(Shape s, int elemType = 7) {
    Tensor t;
    t.kind = Tensor::Kind::Int;
    t.elemType = elemType;
    t.i.assign(numel(s), 0);
    t.shape = std::move(s);
    return t;
}
// Same kind and type as `like`.
Tensor tensorLike(const Tensor& like, Shape s) {
    Tensor t = like.kind == Tensor::Kind::Float ? floatTensor(std::move(s)) : intTensor(std::move(s), like.elemType);
    t.elemType = like.elemType;
    return t;
}

const Tensor& need(const Ins& in, std::size_t k) {
    if (k >= in.size() || !in[k]) throw std::runtime_error("missing input " + std::to_string(k));
    return *in[k];
}
const Tensor* opt(const Ins& in, std::size_t k) { return k < in.size() ? in[k] : nullptr; }

std::vector<std::int64_t> asInts(const Tensor& t) {
    if (t.kind == Tensor::Kind::Int) return t.i;
    std::vector<std::int64_t> v(t.f.size());
    for (std::size_t k = 0; k < v.size(); ++k) v[k] = static_cast<std::int64_t>(t.f[k]);
    return v;
}
std::vector<float> asFloats(const Tensor& t) {
    if (t.kind == Tensor::Kind::Float) return t.f;
    return std::vector<float>(t.i.begin(), t.i.end());
}
float scalarFloat(const Tensor& t) {
    if (t.size() < 1) throw std::runtime_error("empty scalar");
    return t.kind == Tensor::Kind::Float ? t.f[0] : static_cast<float>(t.i[0]);
}

// Copies element `from` of src into element `to` of dst (same kind).
inline void copyElem(const Tensor& src, std::size_t from, Tensor& dst, std::size_t to) {
    if (src.kind == Tensor::Kind::Float) dst.f[to] = src.f[from];
    else dst.i[to] = src.i[from];
}

// ---- broadcasting ----

Shape broadcastShape(const Shape& a, const Shape& b) {
    const std::size_t r = std::max(a.size(), b.size());
    Shape out(r);
    for (std::size_t k = 0; k < r; ++k) {
        const std::int64_t x = k + a.size() >= r ? a[k + a.size() - r] : 1, y = k + b.size() >= r ? b[k + b.size() - r] : 1;
        if (x != y && x != 1 && y != 1) throw std::runtime_error("cannot broadcast " + shapeText(a) + " with " + shapeText(b));
        out[k] = x == 1 ? y : x;
    }
    return out;
}

// Strides of `in` read as `out` (0 along broadcast dimensions).
std::vector<std::size_t> broadcastStrides(const Shape& in, const Shape& out) {
    std::vector<std::size_t> st(out.size(), 0);
    const std::vector<std::size_t> s = strides(in);
    for (std::size_t k = 0; k < in.size(); ++k) {
        const std::size_t o = k + out.size() - in.size();
        st[o] = in[k] == 1 ? 0 : s[k];
    }
    return st;
}

// Calls f(outIndex, offsets...) over the output, offsets into each broadcast input.
template <std::size_t N, typename F>
void broadcastLoop(const Shape& out, const std::array<std::vector<std::size_t>, N>& st, F&& f) {
    const std::size_t total = numel(out);
    if (!total) return;
    const std::size_t r = out.size();
    if (r == 0) {
        std::array<std::size_t, N> off{};
        f(0, off);
        return;
    }
    const std::size_t inner = static_cast<std::size_t>(out[r - 1]);
    const std::size_t outer = total / inner;
    parallelFor(0, outer, std::max<std::size_t>(1, 16384 / std::max<std::size_t>(inner, 1)), [&](std::size_t o0, std::size_t o1) {
        std::vector<std::size_t> idx(r, 0);
        for (std::size_t o = o0; o < o1; ++o) {
            // index of row o over dims 0..r-2
            std::size_t rem = o;
            std::array<std::size_t, N> base{};
            for (std::size_t d = r - 1; d-- > 0;) {
                const std::size_t i = rem % static_cast<std::size_t>(out[d]);
                rem /= static_cast<std::size_t>(out[d]);
                for (std::size_t n = 0; n < N; ++n) base[n] += i * st[n][d];
            }
            std::array<std::size_t, N> off = base;
            for (std::size_t j = 0; j < inner; ++j) {
                f(o * inner + j, off);
                for (std::size_t n = 0; n < N; ++n) off[n] += st[n][r - 1];
            }
        }
    });
}

template <typename FF, typename FI>
Tensor binary(const Tensor& a, const Tensor& b, FF ff, FI fi, bool boolOut = false) {
    if (a.kind != b.kind) throw std::runtime_error("mixed float and integer operands");
    const Shape out = broadcastShape(a.shape, b.shape);
    Tensor r = boolOut ? intTensor(out, 9) : tensorLike(a, out);
    const std::array<std::vector<std::size_t>, 2> st{broadcastStrides(a.shape, out), broadcastStrides(b.shape, out)};
    const bool same = a.shape == b.shape;
    if (a.kind == Tensor::Kind::Float) {
        if (boolOut) {
            broadcastLoop<2>(out, st, [&](std::size_t o, const std::array<std::size_t, 2>& x) { r.i[o] = ff(a.f[x[0]], b.f[x[1]]) != 0.0f; });
        } else if (same) {
            const std::size_t n = r.f.size();
            parallelFor(0, n, 65536, [&](std::size_t i0, std::size_t i1) { for (std::size_t i = i0; i < i1; ++i) r.f[i] = ff(a.f[i], b.f[i]); });
        } else if (b.size() == 1) {
            const float y = b.f[0];
            const std::size_t n = r.f.size();
            parallelFor(0, n, 65536, [&](std::size_t i0, std::size_t i1) { for (std::size_t i = i0; i < i1; ++i) r.f[i] = ff(a.f[i], y); });
        } else {
            broadcastLoop<2>(out, st, [&](std::size_t o, const std::array<std::size_t, 2>& x) { r.f[o] = ff(a.f[x[0]], b.f[x[1]]); });
        }
    } else {
        broadcastLoop<2>(out, st, [&](std::size_t o, const std::array<std::size_t, 2>& x) {
            const std::int64_t v = fi(a.i[x[0]], b.i[x[1]]);
            r.i[o] = boolOut ? (v != 0) : v;
        });
    }
    return r;
}

template <typename F>
Tensor unaryFloat(const Tensor& a, F f) {
    if (a.kind != Tensor::Kind::Float) throw std::runtime_error("expects a float tensor");
    Tensor r = tensorLike(a, a.shape);
    const std::size_t n = a.f.size();
    parallelFor(0, n, 32768, [&](std::size_t i0, std::size_t i1) { for (std::size_t i = i0; i < i1; ++i) r.f[i] = f(a.f[i]); });
    return r;
}

// ---- elementwise operators ----

void addUnary(std::map<std::string, OpFn>& t, const std::string& name, std::function<float(float)> f) {
    t[name] = [f](const Node&, const Ins& in, int) -> Outs { return {unaryFloat(need(in, 0), f)}; };
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

Tensor erfTensor(const Tensor& a) {
    Tensor r = tensorLike(a, a.shape);
    const std::size_t n = a.f.size();
    parallelFor(0, n, 32768, [&](std::size_t i0, std::size_t i1) { erfArray(a.f.data() + i0, r.f.data() + i0, i1 - i0); });
    return r;
}

Tensor variadic(const Ins& in, const std::function<float(float, float)>& ff, const std::function<std::int64_t(std::int64_t, std::int64_t)>& fi) {
    Tensor acc = need(in, 0);
    for (std::size_t k = 1; k < in.size(); ++k)
        if (in[k]) acc = binary(acc, *in[k], ff, fi);
    return acc;
}

// ---- products ----

// C[M x N] = A[M x K] * B[K x N] (row-major, contiguous).
void matmul2d(const float* A, const float* B, float* C, std::size_t M, std::size_t N, std::size_t K, bool transA = false, bool transB = false,
              bool accumulate = false) {
    if (!M || !N) return;
    if (!K) {
        if (!accumulate) std::fill(C, C + M * N, 0.0f);
        return;
    }
    gemm(transA, transB, M, N, K, A, transA ? M : K, B, transB ? K : N, C, N, accumulate);
}

Tensor matMul(const Tensor& a0, const Tensor& b0) {
    if (a0.kind != Tensor::Kind::Float || b0.kind != Tensor::Kind::Float) throw std::runtime_error("MatMul on integers is not supported");
    Shape as = a0.shape, bs = b0.shape;
    const bool aVec = as.size() == 1, bVec = bs.size() == 1;
    if (aVec) as.insert(as.begin(), 1);
    if (bVec) bs.push_back(1);
    const std::size_t M = static_cast<std::size_t>(as[as.size() - 2]), K = static_cast<std::size_t>(as.back());
    const std::size_t N = static_cast<std::size_t>(bs.back());
    if (static_cast<std::size_t>(bs[bs.size() - 2]) != K) throw std::runtime_error("MatMul " + shapeText(a0.shape) + " x " + shapeText(b0.shape));
    const Shape aBatch(as.begin(), as.end() - 2), bBatch(bs.begin(), bs.end() - 2);
    const Shape batch = broadcastShape(aBatch, bBatch);
    Shape outShape = batch;
    if (!aVec) outShape.push_back(static_cast<std::int64_t>(M));
    if (!bVec) outShape.push_back(static_cast<std::int64_t>(N));
    Tensor r = floatTensor(outShape);
    const std::size_t nb = numel(batch);
    if (numel(bBatch) == 1 && numel(aBatch) == nb) {  // weights shared by every batch: one tall product
        matmul2d(a0.f.data(), b0.f.data(), r.f.data(), nb * M, N, K);
        return r;
    }
    const std::array<std::vector<std::size_t>, 2> st{broadcastStrides(aBatch, batch), broadcastStrides(bBatch, batch)};
    std::vector<std::array<std::size_t, 2>> offs(nb);
    if (batch.empty()) offs[0] = {0, 0};
    else broadcastLoop<2>(batch, st, [&](std::size_t o, const std::array<std::size_t, 2>& x) { offs[o] = x; });
    for (std::size_t k = 0; k < nb; ++k)
        matmul2d(a0.f.data() + offs[k][0] * M * K, b0.f.data() + offs[k][1] * K * N, r.f.data() + k * M * N, M, N, K);
    return r;
}

Outs gemmOp(const Node& n, const Ins& in, int) {
    const Tensor& A = need(in, 0);
    const Tensor& B = need(in, 1);
    const bool tA = n.i("transA", 0) != 0, tB = n.i("transB", 0) != 0;
    const float alpha = n.f("alpha", 1.0f), beta = n.f("beta", 1.0f);
    const std::size_t M = static_cast<std::size_t>(tA ? A.shape[1] : A.shape[0]), K = static_cast<std::size_t>(tA ? A.shape[0] : A.shape[1]);
    const std::size_t N = static_cast<std::size_t>(tB ? B.shape[0] : B.shape[1]);
    Tensor r = floatTensor({static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)});
    matmul2d(A.f.data(), B.f.data(), r.f.data(), M, N, K, tA, tB);
    if (alpha != 1.0f)
        for (float& v : r.f) v *= alpha;
    if (const Tensor* C = opt(in, 2); C && beta != 0.0f) {
        const std::array<std::vector<std::size_t>, 1> st{broadcastStrides(C->shape, r.shape)};
        broadcastLoop<1>(r.shape, st, [&](std::size_t o, const std::array<std::size_t, 1>& x) { r.f[o] += beta * C->f[x[0]]; });
    }
    return {r};
}

// ---- recurrent layers: RNN, GRU, LSTM ----

// An activation of the recurrent operators by its ONNX name.
std::function<float(float)> rnnActivation(const std::string& name, float alpha, float beta) {
    if (name == "Sigmoid") return [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    if (name == "Tanh") return [](float x) { return std::tanh(x); };
    if (name == "Relu") return [](float x) { return std::max(x, 0.0f); };
    if (name == "HardSigmoid") return [=](float x) { return std::clamp(alpha * x + beta, 0.0f, 1.0f); };
    if (name == "LeakyRelu") return [=](float x) { return x < 0 ? alpha * x : x; };
    if (name == "Affine") return [=](float x) { return alpha * x + beta; };
    if (name == "ScaledTanh") return [=](float x) { return alpha * std::tanh(beta * x); };
    if (name == "Elu") return [=](float x) { return x < 0 ? alpha * (std::exp(x) - 1.0f) : x; };
    if (name == "Softsign") return [](float x) { return x / (1.0f + std::fabs(x)); };
    if (name == "Softplus") return [](float x) { return std::log1p(std::exp(x)); };
    throw std::runtime_error("activation " + name + " is not supported");
}

// The shared part of RNN / GRU / LSTM: inputs X, W [dirs, G*H, I], R [dirs, G*H, H], B [dirs, 2*G*H],
// sequence_lens, initial_h (and for LSTM initial_c, P). `kind` is 'r', 'g' or 'l'.
Outs recurrent(const Node& n, const Ins& in, char kind) {
    const std::size_t G = kind == 'l' ? 4 : kind == 'g' ? 3 : 1;
    const bool batchFirst = n.i("layout", 0) != 0;
    const Tensor& X = need(in, 0);
    const Tensor& W = need(in, 1);
    const Tensor& R = need(in, 2);
    const Tensor* B = opt(in, 3);
    const Tensor* lens = opt(in, 4);
    const Tensor* h0 = opt(in, 5);
    const Tensor* c0 = kind == 'l' ? opt(in, 6) : nullptr;
    const Tensor* P = kind == 'l' ? opt(in, 7) : nullptr;
    if (X.shape.size() != 3) throw std::runtime_error("X must have rank 3");
    const std::size_t T = static_cast<std::size_t>(X.shape[batchFirst ? 1 : 0]), N = static_cast<std::size_t>(X.shape[batchFirst ? 0 : 1]),
                      I = static_cast<std::size_t>(X.shape[2]);
    const std::size_t D = static_cast<std::size_t>(W.shape.at(0)), H = static_cast<std::size_t>(n.i("hidden_size", R.shape.at(2)));
    const std::string direction = n.s("direction", "forward");
    if ((direction == "bidirectional") != (D == 2)) throw std::runtime_error("direction " + direction + " with " + std::to_string(D) + " weight sets");
    if (n.i("input_forget", 0)) throw std::runtime_error("input_forget is not supported");
    const float clip = n.attr("clip") ? n.f("clip", 0.0f) : 0.0f;
    const bool linearBeforeReset = n.i("linear_before_reset", 0) != 0;
    // activations: f, g (and h for LSTM) per direction
    std::vector<std::string> names = n.attr("activations") ? n.attr("activations")->strings : std::vector<std::string>{};
    const std::vector<float> alphas = n.attr("activation_alpha") ? n.attr("activation_alpha")->floats : std::vector<float>{},
                             betas = n.attr("activation_beta") ? n.attr("activation_beta")->floats : std::vector<float>{};
    const std::size_t perDir = kind == 'l' ? 3 : kind == 'g' ? 2 : 1;
    if (names.empty()) {
        for (std::size_t d = 0; d < D; ++d) {
            if (kind == 'l') names.insert(names.end(), {"Sigmoid", "Tanh", "Tanh"});
            else if (kind == 'g') names.insert(names.end(), {"Sigmoid", "Tanh"});
            else names.push_back("Tanh");
        }
    }
    if (names.size() != perDir * D) throw std::runtime_error("wrong number of activations");
    std::vector<std::function<float(float)>> act;
    for (std::size_t k = 0; k < names.size(); ++k) {
        // the defaults the ONNX spec gives the parameterised activations
        const bool hard = names[k] == "HardSigmoid";
        const float a = k < alphas.size() ? alphas[k] : hard ? 0.2f : names[k] == "LeakyRelu" ? 0.01f : 1.0f;
        const float b = k < betas.size() ? betas[k] : hard ? 0.5f : names[k] == "ScaledTanh" ? 1.0f : 0.0f;
        act.push_back(rnnActivation(names[k], a, b));
    }
    std::vector<std::size_t> len(N, T);
    if (lens && lens->size())
        for (std::size_t b = 0; b < N; ++b) len[b] = static_cast<std::size_t>(std::min<std::int64_t>(asInts(*lens).at(b), static_cast<std::int64_t>(T)));
    // X as [T, N, I]
    std::vector<float> x(T * N * I);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t b = 0; b < N; ++b)
            std::copy_n(X.f.data() + (batchFirst ? (b * T + t) : (t * N + b)) * I, I, x.data() + (t * N + b) * I);
    const std::size_t GH = G * H;
    std::vector<float> Y(T * D * N * H, 0.0f), Yh(D * N * H, 0.0f), Yc(D * N * H, 0.0f);
    for (std::size_t d = 0; d < D; ++d) {
        const bool reverse = direction == "reverse" || d == 1;
        const float* w = W.f.data() + d * GH * I;
        const float* r = R.f.data() + d * GH * H;
        const float* wb = B ? B->f.data() + d * 2 * GH : nullptr;
        const float* rb = B ? wb + GH : nullptr;
        const auto& f = act[d * perDir];
        const auto& g = perDir > 1 ? act[d * perDir + 1] : act[d * perDir];
        const auto& h = perDir > 2 ? act[d * perDir + 2] : act[d * perDir];
        // X W^T (+ both biases, except the recurrent bias of GRU's hidden gate when it applies after the reset)
        std::vector<float> xw(T * N * GH);
        gemm(false, true, T * N, GH, I, x.data(), I, w, I, xw.data(), GH, false);
        if (B)
            for (std::size_t row = 0; row < T * N; ++row)
                for (std::size_t c = 0; c < GH; ++c)
                    xw[row * GH + c] += wb[c] + (kind == 'g' && c >= 2 * H ? 0.0f : rb[c]);
        std::vector<float> hs(N * H, 0.0f), cs(N * H, 0.0f), hr(N * GH), rh(N * H), hh(N * H);
        if (h0)
            for (std::size_t b = 0; b < N; ++b)
                std::copy_n(h0->f.data() + (batchFirst ? (b * D + d) : (d * N + b)) * H, H, hs.data() + b * H);
        if (c0)
            for (std::size_t b = 0; b < N; ++b)
                std::copy_n(c0->f.data() + (batchFirst ? (b * D + d) : (d * N + b)) * H, H, cs.data() + b * H);
        const float* pi = P ? P->f.data() + d * 3 * H : nullptr;
        const float* po = pi ? pi + H : nullptr;
        const float* pf = pi ? pi + 2 * H : nullptr;
        auto clipped = [&](float v) { return clip > 0 ? std::clamp(v, -clip, clip) : v; };
        for (std::size_t s = 0; s < T; ++s) {
            // the recurrent product (GRU's hidden gate: below, when the reset comes first)
            const std::size_t cols = kind == 'g' && !linearBeforeReset ? 2 * H : GH;
            gemm(false, true, N, cols, H, hs.data(), H, r, H, hr.data(), GH, false);
            if (kind == 'g' && !linearBeforeReset) {
                for (std::size_t b = 0; b < N; ++b)
                    for (std::size_t k = 0; k < H; ++k) {
                        const std::size_t t = reverse ? len[b] - 1 - s : s;
                        if (s >= len[b]) continue;
                        const float rv = f(clipped(xw[(t * N + b) * GH + H + k] + hr[b * GH + H + k]));
                        rh[b * H + k] = rv * hs[b * H + k];
                    }
                gemm(false, true, N, H, H, rh.data(), H, r + 2 * H * H, H, hh.data(), H, false);
            }
            for (std::size_t b = 0; b < N; ++b) {
                if (s >= len[b]) continue;
                const std::size_t t = reverse ? len[b] - 1 - s : s;
                const float* xr = xw.data() + (t * N + b) * GH;
                const float* rr = hr.data() + b * GH;
                float* hb = hs.data() + b * H;
                float* cb = cs.data() + b * H;
                for (std::size_t k = 0; k < H; ++k) {
                    if (kind == 'r') {
                        hb[k] = f(clipped(xr[k] + rr[k]));
                    } else if (kind == 'g') {
                        const float z = f(clipped(xr[k] + rr[k]));
                        float cand;
                        if (linearBeforeReset) {
                            const float rv = f(clipped(xr[H + k] + rr[H + k]));
                            cand = g(clipped(xr[2 * H + k] + rv * (rr[2 * H + k] + (B ? rb[2 * H + k] : 0.0f))));
                        } else {
                            cand = g(clipped(xr[2 * H + k] + hh[b * H + k] + (B ? rb[2 * H + k] : 0.0f)));
                        }
                        hb[k] = (1.0f - z) * cand + z * hb[k];
                    } else {  // LSTM, gates i o f c
                        const float c = cb[k];
                        const float ig = f(clipped(xr[k] + rr[k] + (pi ? pi[k] * c : 0.0f)));
                        const float fg = f(clipped(xr[2 * H + k] + rr[2 * H + k] + (pf ? pf[k] * c : 0.0f)));
                        const float cg = g(clipped(xr[3 * H + k] + rr[3 * H + k]));
                        const float cn = fg * c + ig * cg;
                        const float og = f(clipped(xr[H + k] + rr[H + k] + (po ? po[k] * cn : 0.0f)));
                        cb[k] = cn;
                        hb[k] = og * h(cn);
                    }
                }
                std::copy_n(hb, H, Y.data() + ((t * D + d) * N + b) * H);
            }
        }
        for (std::size_t b = 0; b < N; ++b) {
            std::copy_n(hs.data() + b * H, H, Yh.data() + (batchFirst ? (b * D + d) : (d * N + b)) * H);
            std::copy_n(cs.data() + b * H, H, Yc.data() + (batchFirst ? (b * D + d) : (d * N + b)) * H);
        }
    }
    const auto T64 = static_cast<std::int64_t>(T), N64 = static_cast<std::int64_t>(N), D64 = static_cast<std::int64_t>(D), H64 = static_cast<std::int64_t>(H);
    Tensor y = floatTensor(batchFirst ? Shape{N64, T64, D64, H64} : Shape{T64, D64, N64, H64});
    if (batchFirst) {
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t d = 0; d < D; ++d)
                for (std::size_t b = 0; b < N; ++b)
                    std::copy_n(Y.data() + ((t * D + d) * N + b) * H, H, y.f.data() + ((b * T + t) * D + d) * H);
    } else {
        y.f = std::move(Y);
    }
    const Shape stateShape = batchFirst ? Shape{N64, D64, H64} : Shape{D64, N64, H64};
    Outs out{y, Tensor::floats(stateShape, std::move(Yh))};
    if (kind == 'l') out.push_back(Tensor::floats(stateShape, std::move(Yc)));
    return out;
}

// ---- convolution ----

struct ConvGeometry {
    std::size_t dims = 2;               // spatial dims (1 or 2; 1-D is run as 2-D with height 1)
    std::size_t kh = 1, kw = 1, sh = 1, sw = 1, dh = 1, dw = 1;
    std::int64_t padTop = 0, padLeft = 0, padBottom = 0, padRight = 0;
};

// Pads from the attributes (explicit or auto_pad) for input size `in` along one axis.
void autoPad(const std::string& mode, std::int64_t in, std::int64_t k, std::int64_t s, std::int64_t d, std::int64_t& b, std::int64_t& e) {
    if (mode == "SAME_UPPER" || mode == "SAME_LOWER") {
        const std::int64_t out = (in + s - 1) / s, total = std::max<std::int64_t>(0, (out - 1) * s + (k - 1) * d + 1 - in);
        b = mode == "SAME_UPPER" ? total / 2 : total - total / 2;
        e = total - b;
    } else if (mode == "VALID") {
        b = e = 0;
    }
}

ConvGeometry convGeometry(const Node& n, const Shape& x, const Shape& w, bool transposed = false) {
    ConvGeometry g;
    g.dims = x.size() - 2;
    if (g.dims != 1 && g.dims != 2) throw std::runtime_error("only 1-D and 2-D convolutions are supported");
    std::vector<std::int64_t> k = n.ints("kernel_shape"), s = n.ints("strides"), d = n.ints("dilations"), p = n.ints("pads");
    if (k.empty()) k.assign(w.begin() + 2, w.end());
    if (s.empty()) s.assign(g.dims, 1);
    if (d.empty()) d.assign(g.dims, 1);
    if (p.empty()) p.assign(2 * g.dims, 0);
    if (g.dims == 1) {
        g.kw = static_cast<std::size_t>(k[0]);
        g.sw = static_cast<std::size_t>(s[0]);
        g.dw = static_cast<std::size_t>(d[0]);
        g.padLeft = p[0];
        g.padRight = p[1];
    } else {
        g.kh = static_cast<std::size_t>(k[0]);
        g.kw = static_cast<std::size_t>(k[1]);
        g.sh = static_cast<std::size_t>(s[0]);
        g.sw = static_cast<std::size_t>(s[1]);
        g.dh = static_cast<std::size_t>(d[0]);
        g.dw = static_cast<std::size_t>(d[1]);
        g.padTop = p[0];
        g.padLeft = p[1];
        g.padBottom = p[2];
        g.padRight = p[3];
    }
    const std::string mode = n.s("auto_pad", "NOTSET");
    if (!transposed && mode != "NOTSET") {
        if (g.dims == 2) autoPad(mode, x[2], static_cast<std::int64_t>(g.kh), static_cast<std::int64_t>(g.sh), static_cast<std::int64_t>(g.dh), g.padTop, g.padBottom);
        autoPad(mode, x.back(), static_cast<std::int64_t>(g.kw), static_cast<std::int64_t>(g.sw), static_cast<std::int64_t>(g.dw), g.padLeft, g.padRight);
    }
    return g;
}

// The channel planes of a Conv input: one tensor, or the pieces of a Concat folded into the Conv
// (Model::load, "_concat"), each read where it is.
struct ConvInput {
    Shape shape;
    std::vector<std::pair<const Tensor*, std::size_t>> pieces;  // tensor, channels
    std::size_t plane = 0;
    // The plane of channel c of image b.
    const float* channel(std::size_t b, std::size_t c) const {
        for (const auto& [t, ch] : pieces) {
            if (c < ch) return t->f.data() + (b * ch + c) * plane;
            c -= ch;
        }
        return nullptr;
    }
};

ConvInput convInput(const Node& n, const Ins& in) {
    ConvInput ci;
    const Tensor& X0 = need(in, 0);
    ci.shape = X0.shape;
    ci.pieces.push_back({&X0, static_cast<std::size_t>(X0.shape.at(1))});
    const std::size_t extra = static_cast<std::size_t>(n.i("_concat", 1)) - 1;
    for (std::size_t k = 0; k < extra; ++k) {
        const Tensor& p = need(in, 3 + k);
        if (p.shape.size() != X0.shape.size() || p.kind != Tensor::Kind::Float) throw std::runtime_error("Concat: pieces of different ranks");
        for (std::size_t d = 0; d < p.shape.size(); ++d)
            if (d != 1 && p.shape[d] != X0.shape[d]) throw std::runtime_error("Concat " + shapeText(X0.shape) + " with " + shapeText(p.shape));
        ci.pieces.push_back({&p, static_cast<std::size_t>(p.shape[1])});
        ci.shape[1] += p.shape[1];
    }
    ci.plane = numel(Shape(X0.shape.begin() + 2, X0.shape.end()));
    return ci;
}

Outs convOp(const Node& n, const Ins& in, int) {
    const ConvInput xin = convInput(n, in);
    // The paths below other than the direct one read one tensor: concatenate there.
    Tensor joined;
    auto whole = [&]() -> const Tensor& {
        if (xin.pieces.size() == 1) return *xin.pieces[0].first;
        if (joined.f.empty()) {
            joined = floatTensor(xin.shape);
            const std::size_t Nb = static_cast<std::size_t>(xin.shape[0]), Ct = static_cast<std::size_t>(xin.shape[1]);
            for (std::size_t b = 0; b < Nb; ++b)
                for (std::size_t c = 0; c < Ct; ++c) std::copy_n(xin.channel(b, c), xin.plane, joined.f.data() + (b * Ct + c) * xin.plane);
        }
        return joined;
    };
    struct ShapeOnly { const Shape& shape; } X{xin.shape};
    const Tensor& W = need(in, 1);
    const Tensor* B = opt(in, 2);
    const ConvGeometry g = convGeometry(n, X.shape, W.shape);
    const std::size_t group = static_cast<std::size_t>(n.i("group", 1));
    const std::size_t N = static_cast<std::size_t>(X.shape[0]), C = static_cast<std::size_t>(X.shape[1]);
    const std::size_t H = g.dims == 2 ? static_cast<std::size_t>(X.shape[2]) : 1, Wd = static_cast<std::size_t>(X.shape.back());
    const std::size_t M = static_cast<std::size_t>(W.shape[0]), Cg = C / group, Mg = M / group;
    if (static_cast<std::size_t>(W.shape[1]) != Cg) throw std::runtime_error("Conv: weight " + shapeText(W.shape) + " does not fit input " + shapeText(X.shape));
    const std::int64_t oh64 = (static_cast<std::int64_t>(H) + g.padTop + g.padBottom - static_cast<std::int64_t>((g.kh - 1) * g.dh + 1)) / static_cast<std::int64_t>(g.sh) + 1;
    const std::int64_t ow64 = (static_cast<std::int64_t>(Wd) + g.padLeft + g.padRight - static_cast<std::int64_t>((g.kw - 1) * g.dw + 1)) / static_cast<std::int64_t>(g.sw) + 1;
    if (oh64 <= 0 || ow64 <= 0) throw std::runtime_error("Conv: output would be empty");
    const std::size_t OH = static_cast<std::size_t>(oh64), OW = static_cast<std::size_t>(ow64);
    Shape outShape = {static_cast<std::int64_t>(N), static_cast<std::int64_t>(M)};
    if (g.dims == 2) outShape.push_back(static_cast<std::int64_t>(OH));
    outShape.push_back(static_cast<std::int64_t>(OW));
    Tensor r = floatTensor(outShape);
    const std::size_t Kc = Cg * g.kh * g.kw;
    const bool pointwise = g.kh == 1 && g.kw == 1 && g.sh == 1 && g.sw == 1 && g.padTop == 0 && g.padLeft == 0 && g.padBottom == 0 && g.padRight == 0;
    // Output rows per pass so the column buffer (Kc x rows * OW) stays near 1 M floats: it is then
    // still in cache when the product reads it.
    const std::size_t rowsPerPass = std::max<std::size_t>(1, std::min(OH, (std::size_t(1) << 20) / std::max<std::size_t>(1, Kc * OW)));
    // Stride 1 with the blocked CPU kernel: no im2col at all. The input is copied once, padded and
    // channels-last; then every output pixel's taps are fixed offsets from its own position, so the
    // product reads them in place (gemmIndirectA). Pixels are computed over the padded width and the
    // few columns past the right edge dropped.
    if (Cg == 1 && !pointwise) {
        // Depthwise (MobileNet-style): each output channel reads one input channel; per tap, a scaled
        // input row segment is added to the output row (contiguous, so it vectorises at stride 1).
        const std::size_t KK = g.kh * g.kw;
        parallelFor(0, N * M, 1, [&](std::size_t p0, std::size_t p1) {
            for (std::size_t p = p0; p < p1; ++p) {
                const std::size_t b = p / M, m = p % M;
                const float* x = xin.channel(b, m / Mg);
                const float* w = W.f.data() + m * KK;
                float* y = r.f.data() + p * OH * OW;
                std::fill(y, y + OH * OW, 0.0f);
                for (std::size_t kj = 0; kj < g.kw; ++kj) {
                    const std::int64_t off = static_cast<std::int64_t>(kj * g.dw) - g.padLeft, sw = static_cast<std::int64_t>(g.sw);
                    const std::int64_t lo64 = off >= 0 ? 0 : (-off + sw - 1) / sw;
                    const std::int64_t hi64 = std::min<std::int64_t>(static_cast<std::int64_t>(OW), (static_cast<std::int64_t>(Wd) - off + sw - 1) / sw);
                    if (hi64 <= lo64) continue;
                    const std::size_t lo = static_cast<std::size_t>(lo64), hi = static_cast<std::size_t>(hi64);
                    for (std::size_t oy = 0; oy < OH; ++oy) {
                        float* d = y + oy * OW;
                        for (std::size_t ki = 0; ki < g.kh; ++ki) {
                            const std::int64_t iy = static_cast<std::int64_t>(oy * g.sh + ki * g.dh) - g.padTop;
                            if (iy < 0 || iy >= static_cast<std::int64_t>(H)) continue;
                            const float wv = w[ki * g.kw + kj];
                            const float* src = x + static_cast<std::size_t>(iy) * Wd;
                            if (g.sw == 1)
                                for (std::size_t ox = lo; ox < hi; ++ox) d[ox] += wv * src[static_cast<std::int64_t>(ox) + off];
                            else
                                for (std::size_t ox = lo; ox < hi; ++ox) d[ox] += wv * src[static_cast<std::int64_t>(ox) * sw + off];
                        }
                    }
                }
            }
        });
    } else
    if (!pointwise && gemmPackedAvailable() && g.sh == 1 && g.sw == 1 && Cg >= 16) {
        const std::size_t R = gemmPanelRows();
        const std::size_t Hp = H + static_cast<std::size_t>(g.padTop + g.padBottom), Wp = Wd + static_cast<std::size_t>(g.padLeft + g.padRight);
        const std::size_t rows = (OH - 1) * Wp + OW;
        std::vector<float> xp, tmp(rows * Mg);
        std::vector<std::int64_t> offs(Kc);
        {
            std::size_t k = 0;
            for (std::size_t c = 0; c < Cg; ++c)
                for (std::size_t ki = 0; ki < g.kh; ++ki)
                    for (std::size_t kj = 0; kj < g.kw; ++kj)
                        offs[k++] = static_cast<std::int64_t>(((ki * g.dh) * Wp + kj * g.dw) * Cg + c);
        }
        for (std::size_t b = 0; b < N; ++b)
            for (std::size_t gr = 0; gr < group; ++gr) {
                std::vector<const float*> planes(Cg);
                for (std::size_t c = 0; c < Cg; ++c) planes[c] = xin.channel(b, gr * Cg + c);
                const float* w = W.f.data() + gr * Mg * Kc;
                float* y = r.f.data() + (b * M + gr * Mg) * OH * OW;
                xp.assign((Hp * Wp + R + 1) * Cg, 0.0f);
                parallelFor(0, H, 1, [&](std::size_t y0, std::size_t y1) {
                    for (std::size_t iy = y0; iy < y1; ++iy) {
                        float* dst = xp.data() + ((iy + static_cast<std::size_t>(g.padTop)) * Wp + static_cast<std::size_t>(g.padLeft)) * Cg;
                        for (std::size_t c = 0; c < Cg; ++c) {
                            const float* src = planes[c] + iy * Wd;
                            for (std::size_t ix = 0; ix < Wd; ++ix) dst[ix * Cg + c] = src[ix];
                        }
                    }
                });
                gemmIndirectA(rows, Mg, Kc, xp.data(), Cg, offs.data(), true, w, Kc, tmp.data(), Mg);
                parallelFor(0, Mg, 1, [&](std::size_t m0, std::size_t m1) {
                    for (std::size_t m = m0; m < m1; ++m)
                        for (std::size_t oy = 0; oy < OH; ++oy) {
                            float* dst = y + (m * OH + oy) * OW;
                            const float* src = tmp.data() + (oy * Wp) * Mg + m;
                            for (std::size_t ox = 0; ox < OW; ++ox) dst[ox] = src[ox * Mg];
                        }
                });
            }
    } else
    // Otherwise (strided) with the blocked CPU kernel: pixels as the long side of the product, their
    // im2col rows written straight in the kernel's packed layout, the few output channels as columns.
    if (!pointwise && gemmPackedAvailable()) {
        const std::size_t R = gemmPanelRows(), P = OH * OW;
        // Up to 4 M floats of rows per product: bigger products split better over the threads.
        const std::size_t chunk = std::max(R, ((std::size_t(1) << 22) / std::max<std::size_t>(1, Kc)) / R * R);
        std::vector<float> packed, tmp;
        for (std::size_t b = 0; b < N; ++b)
            for (std::size_t gr = 0; gr < group; ++gr) {
                const float* x = whole().f.data() + (b * C + gr * Cg) * H * Wd;
                const float* w = W.f.data() + gr * Mg * Kc;
                float* y = r.f.data() + (b * M + gr * Mg) * P;
                for (std::size_t p0 = 0; p0 < P; p0 += chunk) {
                    const std::size_t pc = std::min(chunk, P - p0), panels = (pc + R - 1) / R;
                    packed.resize(panels * R * Kc);
                    parallelFor(0, panels, 1, [&](std::size_t a0, std::size_t a1) {
                        std::vector<std::int64_t> iy0(R), ix0(R);
                        std::vector<bool> live(R);
                        for (std::size_t pa = a0; pa < a1; ++pa) {
                            for (std::size_t i = 0; i < R; ++i) {
                                const std::size_t q = p0 + pa * R + i;
                                live[i] = q < p0 + pc;
                                iy0[i] = live[i] ? static_cast<std::int64_t>((q / OW) * g.sh) - g.padTop : 0;
                                ix0[i] = live[i] ? static_cast<std::int64_t>((q % OW) * g.sw) - g.padLeft : 0;
                            }
                            float* dst = packed.data() + pa * Kc * R;
                            const std::size_t q0 = p0 + pa * R;
                            if (live[R - 1] && q0 / OW == (q0 + R - 1) / OW) {
                                // The usual case: R neighbours in one output row - per (c, ki, kj) one input
                                // row segment (a straight copy when the stride is 1 and it is inside).
                                const std::int64_t iyBase = iy0[0], ixBase = ix0[0];
                                const std::int64_t sw = static_cast<std::int64_t>(g.sw), Wi = static_cast<std::int64_t>(Wd);
                                std::size_t k = 0;
                                for (std::size_t c = 0; c < Cg; ++c) {
                                    const float* plane = x + c * H * Wd;
                                    for (std::size_t ki = 0; ki < g.kh; ++ki) {
                                        const std::int64_t iy = iyBase + static_cast<std::int64_t>(ki * g.dh);
                                        if (iy < 0 || iy >= static_cast<std::int64_t>(H)) {
                                            std::fill(dst + k * R, dst + (k + g.kw) * R, 0.0f);
                                            k += g.kw;
                                            continue;
                                        }
                                        const float* row = plane + static_cast<std::size_t>(iy) * Wd;
                                        for (std::size_t kj = 0; kj < g.kw; ++kj, ++k) {
                                            float* d = dst + k * R;
                                            const std::int64_t ixs = ixBase + static_cast<std::int64_t>(kj * g.dw);
                                            if (sw == 1 && ixs >= 0 && ixs + static_cast<std::int64_t>(R) <= Wi) {
                                                if (R == 12) std::memcpy(d, row + ixs, 12 * sizeof(float));  // inlined for the AVX-512 / AVX tile heights
                                                else if (R == 6) std::memcpy(d, row + ixs, 6 * sizeof(float));
                                                else std::memcpy(d, row + ixs, R * sizeof(float));
                                            } else {
                                                for (std::size_t i = 0; i < R; ++i) {
                                                    const std::int64_t ix = ixs + static_cast<std::int64_t>(i) * sw;
                                                    d[i] = ix >= 0 && ix < Wi ? row[ix] : 0.0f;
                                                }
                                            }
                                        }
                                    }
                                }
                                continue;
                            }
                            std::size_t k = 0;
                            for (std::size_t c = 0; c < Cg; ++c) {
                                const float* plane = x + c * H * Wd;
                                for (std::size_t ki = 0; ki < g.kh; ++ki)
                                    for (std::size_t kj = 0; kj < g.kw; ++kj, ++k) {
                                        float* d = dst + k * R;
                                        for (std::size_t i = 0; i < R; ++i) {
                                            const std::int64_t iy = iy0[i] + static_cast<std::int64_t>(ki * g.dh), ix = ix0[i] + static_cast<std::int64_t>(kj * g.dw);
                                            d[i] = live[i] && iy >= 0 && iy < static_cast<std::int64_t>(H) && ix >= 0 && ix < static_cast<std::int64_t>(Wd)
                                                       ? plane[static_cast<std::size_t>(iy) * Wd + static_cast<std::size_t>(ix)]
                                                       : 0.0f;
                                        }
                                    }
                            }
                        }
                    });
                    tmp.resize(pc * Mg);
                    gemmPackedA(pc, Mg, Kc, packed.data(), true, w, Kc, tmp.data(), Mg);
                    parallelFor(0, Mg, 1, [&](std::size_t m0, std::size_t m1) {
                        for (std::size_t m = m0; m < m1; ++m) {
                            float* dst = y + m * P + p0;
                            for (std::size_t p = 0; p < pc; ++p) dst[p] = tmp[p * Mg + m];
                        }
                    });
                }
            }
    } else {
    std::vector<float> col;
    for (std::size_t b = 0; b < N; ++b) {
        for (std::size_t gr = 0; gr < group; ++gr) {
            const float* x = whole().f.data() + (b * C + gr * Cg) * H * Wd;
            const float* w = W.f.data() + gr * Mg * Kc;
            float* y = r.f.data() + (b * M + gr * Mg) * OH * OW;
            if (pointwise) {
                matmul2d(w, x, y, Mg, OH * OW, Cg);
                continue;
            }
            for (std::size_t r0 = 0; r0 < OH; r0 += rowsPerPass) {
                const std::size_t rows = std::min(rowsPerPass, OH - r0), cols = rows * OW;
                col.assign(Kc * cols, 0.0f);
                parallelFor(0, Kc, 1, [&](std::size_t k0, std::size_t k1) {
                    for (std::size_t k = k0; k < k1; ++k) {
                        const std::size_t c = k / (g.kh * g.kw), ki = (k / g.kw) % g.kh, kj = k % g.kw;
                        float* dst = col.data() + k * cols;
                        // Output columns whose input column is inside: ox in [lo, hi).
                        const std::int64_t off = static_cast<std::int64_t>(kj * g.dw) - g.padLeft, sw = static_cast<std::int64_t>(g.sw);
                        const std::int64_t lo64 = off >= 0 ? 0 : (-off + sw - 1) / sw;
                        const std::int64_t hi64 = std::min<std::int64_t>(static_cast<std::int64_t>(OW), (static_cast<std::int64_t>(Wd) - off + sw - 1) / sw);
                        const std::size_t lo = static_cast<std::size_t>(std::max<std::int64_t>(0, lo64)), hi = static_cast<std::size_t>(std::max<std::int64_t>(0, hi64));
                        for (std::size_t oy = 0; oy < rows; ++oy) {
                            const std::int64_t iy = static_cast<std::int64_t>((r0 + oy) * g.sh + ki * g.dh) - g.padTop;
                            if (iy < 0 || iy >= static_cast<std::int64_t>(H) || lo >= hi) continue;
                            const float* src = x + (c * H + static_cast<std::size_t>(iy)) * Wd;
                            float* d = dst + oy * OW;
                            if (g.sw == 1) {
                                std::memcpy(d + lo, src + static_cast<std::int64_t>(lo) + off, (hi - lo) * sizeof(float));
                            } else {
                                for (std::size_t ox = lo; ox < hi; ++ox) d[ox] = src[static_cast<std::int64_t>(ox) * sw + off];
                            }
                        }
                    }
                });
                // Straight into the output rows r0.. of every channel (row stride OH * OW).
                gemm(false, false, Mg, cols, Kc, w, Kc, col.data(), cols, y + r0 * OW, OH * OW, false);
            }
        }
    }
    }
    // Bias and a fused activation (Model::load folds a Relu / LeakyRelu / Clip that only reads this
    // output into the convolution) in one pass.
    const std::string act = n.s("_activation", "");
    if (B || !act.empty()) {
        const std::size_t plane = OH * OW;
        const float alpha = n.f("_alpha", 0.01f), lo = n.f("_min", -std::numeric_limits<float>::infinity()),
                    hi = n.f("_max", std::numeric_limits<float>::infinity());
        const int kind = act == "Relu" ? 1 : act == "LeakyRelu" ? 2 : act == "Clip" ? 3 : 0;
        parallelFor(0, N * M, 4, [&](std::size_t p0, std::size_t p1) {
            for (std::size_t p = p0; p < p1; ++p) {
                const float bias = B ? B->f[p % M] : 0.0f;
                float* y = r.f.data() + p * plane;
                switch (kind) {
                case 1: for (std::size_t k = 0; k < plane; ++k) y[k] = std::max(y[k] + bias, 0.0f); break;
                case 2: for (std::size_t k = 0; k < plane; ++k) { const float v = y[k] + bias; y[k] = v >= 0 ? v : alpha * v; } break;
                case 3: for (std::size_t k = 0; k < plane; ++k) y[k] = std::min(std::max(y[k] + bias, lo), hi); break;
                default: for (std::size_t k = 0; k < plane; ++k) y[k] += bias;
                }
            }
        });
    }
    return {r};
}

Outs convTransposeOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const Tensor& W = need(in, 1);
    const Tensor* B = opt(in, 2);
    ConvGeometry g = convGeometry(n, X.shape, W.shape, true);
    const std::size_t group = static_cast<std::size_t>(n.i("group", 1));
    const std::size_t N = static_cast<std::size_t>(X.shape[0]), C = static_cast<std::size_t>(X.shape[1]);
    const std::size_t H = g.dims == 2 ? static_cast<std::size_t>(X.shape[2]) : 1, Wd = static_cast<std::size_t>(X.shape.back());
    const std::size_t Mg = static_cast<std::size_t>(W.shape[1]), M = Mg * group, Cg = C / group;
    std::vector<std::int64_t> outPad = n.ints("output_padding"), outShapeAttr = n.ints("output_shape");
    if (outPad.empty()) outPad.assign(g.dims, 0);
    const std::int64_t opH = g.dims == 2 ? outPad[0] : 0, opW = outPad.back();
    const std::int64_t fullH = static_cast<std::int64_t>(g.sh * (H - 1) + (g.kh - 1) * g.dh + 1) + opH;
    const std::int64_t fullW = static_cast<std::int64_t>(g.sw * (Wd - 1) + (g.kw - 1) * g.dw + 1) + opW;
    const std::string mode = n.s("auto_pad", "NOTSET");
    if (!outShapeAttr.empty() || mode == "SAME_UPPER" || mode == "SAME_LOWER") {
        const std::int64_t wantH = !outShapeAttr.empty() ? (g.dims == 2 ? outShapeAttr[outShapeAttr.size() - 2] : 1) : static_cast<std::int64_t>(H * g.sh);
        const std::int64_t wantW = !outShapeAttr.empty() ? outShapeAttr.back() : static_cast<std::int64_t>(Wd * g.sw);
        auto split = [&](std::int64_t total, std::int64_t& b, std::int64_t& e) {
            total = std::max<std::int64_t>(0, total);
            if (mode == "SAME_UPPER") { b = total / 2; e = total - b; }
            else { e = total / 2; b = total - e; }
        };
        if (g.dims == 2) split(fullH - wantH, g.padTop, g.padBottom);
        split(fullW - wantW, g.padLeft, g.padRight);
    }
    const std::int64_t OH64 = fullH - g.padTop - g.padBottom, OW64 = fullW - g.padLeft - g.padRight;
    if (OH64 <= 0 || OW64 <= 0) throw std::runtime_error("ConvTranspose: output would be empty");
    const std::size_t OH = static_cast<std::size_t>(OH64), OW = static_cast<std::size_t>(OW64);
    Shape outShape = {static_cast<std::int64_t>(N), static_cast<std::int64_t>(M)};
    if (g.dims == 2) outShape.push_back(OH64);
    outShape.push_back(OW64);
    Tensor r = floatTensor(outShape);
    const std::size_t Kc = Mg * g.kh * g.kw, HW = H * Wd;
    std::vector<float> col(Kc * HW);
    for (std::size_t b = 0; b < N; ++b) {
        for (std::size_t gr = 0; gr < group; ++gr) {
            const float* x = X.f.data() + (b * C + gr * Cg) * HW;
            const float* w = W.f.data() + gr * Cg * Kc;  // Cg x Kc
            matmul2d(w, x, col.data(), Kc, HW, Cg, true, false);
            float* y = r.f.data() + (b * M + gr * Mg) * OH * OW;
            parallelFor(0, Mg, 1, [&](std::size_t m0, std::size_t m1) {
                for (std::size_t m = m0; m < m1; ++m)
                    for (std::size_t ki = 0; ki < g.kh; ++ki)
                        for (std::size_t kj = 0; kj < g.kw; ++kj) {
                            const float* src = col.data() + ((m * g.kh + ki) * g.kw + kj) * HW;
                            for (std::size_t iy = 0; iy < H; ++iy) {
                                const std::int64_t oy = static_cast<std::int64_t>(iy * g.sh + ki * g.dh) - g.padTop;
                                if (oy < 0 || oy >= OH64) continue;
                                float* dst = y + (m * OH + static_cast<std::size_t>(oy)) * OW;
                                for (std::size_t ix = 0; ix < Wd; ++ix) {
                                    const std::int64_t ox = static_cast<std::int64_t>(ix * g.sw + kj * g.dw) - g.padLeft;
                                    if (ox >= 0 && ox < OW64) dst[ox] += src[iy * Wd + ix];
                                }
                            }
                        }
            });
        }
    }
    if (B)
        for (std::size_t p = 0; p < N * M; ++p) {
            const float bias = B->f[p % M];
            for (std::size_t k = 0; k < OH * OW; ++k) r.f[p * OH * OW + k] += bias;
        }
    return {r};
}

// ---- pooling ----

Outs poolOp(const Node& n, const Ins& in, bool isMax) {
    const Tensor& X = need(in, 0);
    const std::size_t dims = X.shape.size() - 2;
    if (dims != 1 && dims != 2) throw std::runtime_error("only 1-D and 2-D pooling is supported");
    std::vector<std::int64_t> k = n.ints("kernel_shape"), s = n.ints("strides"), p = n.ints("pads"), d = n.ints("dilations");
    if (s.empty()) s.assign(dims, 1);
    if (p.empty()) p.assign(2 * dims, 0);
    if (d.empty()) d.assign(dims, 1);
    const bool ceil = n.i("ceil_mode", 0) != 0, countPad = n.i("count_include_pad", 0) != 0;
    const std::size_t N = static_cast<std::size_t>(X.shape[0]) * static_cast<std::size_t>(X.shape[1]);
    const std::int64_t H = dims == 2 ? X.shape[2] : 1, W = X.shape.back();
    const std::int64_t kh = dims == 2 ? k[0] : 1, kw = k.back(), sh = dims == 2 ? s[0] : 1, sw = s.back(), dh = dims == 2 ? d[0] : 1, dw = d.back();
    std::int64_t pt = dims == 2 ? p[0] : 0, pl = dims == 2 ? p[1] : p[0], pb = dims == 2 ? p[2] : 0, pr = dims == 2 ? p[3] : p[1];
    const std::string mode = n.s("auto_pad", "NOTSET");
    if (mode != "NOTSET") {
        if (dims == 2) autoPad(mode, H, kh, sh, dh, pt, pb);
        autoPad(mode, W, kw, sw, dw, pl, pr);
    }
    auto outSize = [&](std::int64_t in, std::int64_t kk, std::int64_t ss, std::int64_t dd, std::int64_t b, std::int64_t e) {
        const std::int64_t span = in + b + e - ((kk - 1) * dd + 1);
        std::int64_t o = (ceil ? (span + ss - 1) / ss : span / ss) + 1;
        if (ceil && (o - 1) * ss >= in + b) --o;  // the last window must start inside the input or left padding
        return o;
    };
    const std::int64_t OH = outSize(H, kh, sh, dh, pt, pb), OW = outSize(W, kw, sw, dw, pl, pr);
    Shape os = {X.shape[0], X.shape[1]};
    if (dims == 2) os.push_back(OH);
    os.push_back(OW);
    Tensor r = floatTensor(os);
    parallelFor(0, N, 1, [&](std::size_t c0, std::size_t c1) {
        for (std::size_t c = c0; c < c1; ++c) {
            const float* x = X.f.data() + c * static_cast<std::size_t>(H * W);
            float* y = r.f.data() + c * static_cast<std::size_t>(OH * OW);
            for (std::int64_t oy = 0; oy < OH; ++oy)
                for (std::int64_t ox = 0; ox < OW; ++ox) {
                    float acc = isMax ? -std::numeric_limits<float>::infinity() : 0.0f;
                    std::int64_t count = 0;
                    for (std::int64_t i = 0; i < kh; ++i) {
                        const std::int64_t iy = oy * sh + i * dh - pt;
                        for (std::int64_t j = 0; j < kw; ++j) {
                            const std::int64_t ix = ox * sw + j * dw - pl;
                            const bool inside = iy >= 0 && iy < H && ix >= 0 && ix < W;
                            if (inside) {
                                const float v = x[iy * W + ix];
                                acc = isMax ? std::max(acc, v) : acc + v;
                                ++count;
                            } else if (countPad && iy < H + pb && ix < W + pr) {
                                ++count;
                            }
                        }
                    }
                    y[oy * OW + ox] = isMax ? acc : (count ? acc / static_cast<float>(count) : 0.0f);
                }
        }
    });
    return {r};
}

Outs globalPool(const Ins& in, bool isMax) {
    const Tensor& X = need(in, 0);
    const std::size_t NC = static_cast<std::size_t>(X.shape[0] * X.shape[1]), plane = X.size() / std::max<std::size_t>(1, NC);
    Shape os(X.shape.size(), 1);
    os[0] = X.shape[0];
    os[1] = X.shape[1];
    Tensor r = floatTensor(os);
    for (std::size_t c = 0; c < NC; ++c) {
        const float* x = X.f.data() + c * plane;
        if (isMax) r.f[c] = *std::max_element(x, x + plane);
        else r.f[c] = static_cast<float>(std::accumulate(x, x + plane, 0.0) / static_cast<double>(plane));
    }
    return {r};
}

// ---- normalisations ----

// Normalises groups of `inner` consecutive values: (x - mean) / sqrt(var + eps).
void normalizeRuns(const float* x, float* y, std::size_t runs, std::size_t inner, float eps) {
    parallelFor(0, runs, 1, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* a = x + r * inner;
            float* b = y + r * inner;
            double mean = 0.0;
            for (std::size_t k = 0; k < inner; ++k) mean += a[k];
            mean /= static_cast<double>(inner);
            double var = 0.0;
            for (std::size_t k = 0; k < inner; ++k) {
                const double dlt = a[k] - mean;
                var += dlt * dlt;
            }
            var /= static_cast<double>(inner);
            const float inv = static_cast<float>(1.0 / std::sqrt(var + eps)), m = static_cast<float>(mean);
            for (std::size_t k = 0; k < inner; ++k) b[k] = (a[k] - m) * inv;
        }
    });
}

Outs layerNorm(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", -1), X.shape.size()));
    const float eps = n.f("epsilon", 1e-5f);
    std::size_t inner = 1;
    for (std::size_t k = axis; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    Tensor r = floatTensor(X.shape);
    normalizeRuns(X.f.data(), r.f.data(), X.size() / std::max<std::size_t>(1, inner), inner, eps);
    const Tensor& scale = need(in, 1);
    const Tensor* bias = opt(in, 2);
    Tensor y = binary(r, scale, [](float a, float b) { return a * b; }, [](std::int64_t a, std::int64_t b) { return a * b; });
    if (bias) y = binary(y, *bias, [](float a, float b) { return a + b; }, [](std::int64_t a, std::int64_t b) { return a + b; });
    return {y};
}

// Per-channel scale / bias for N x C x ... tensors.
void channelAffine(Tensor& y, const std::vector<float>& scale, const std::vector<float>& bias) {
    const std::size_t N = static_cast<std::size_t>(y.shape[0]), C = static_cast<std::size_t>(y.shape[1]), plane = y.size() / (N * C);
    parallelFor(0, N * C, 1, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const float s = scale.empty() ? 1.0f : scale[p % C], b = bias.empty() ? 0.0f : bias[p % C];
            float* v = y.f.data() + p * plane;
            for (std::size_t k = 0; k < plane; ++k) v[k] = v[k] * s + b;
        }
    });
}

Outs instanceNorm(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::size_t NC = static_cast<std::size_t>(X.shape[0] * X.shape[1]);
    Tensor r = floatTensor(X.shape);
    normalizeRuns(X.f.data(), r.f.data(), NC, X.size() / NC, n.f("epsilon", 1e-5f));
    channelAffine(r, need(in, 1).f, need(in, 2).f);
    return {r};
}

Outs groupNorm(const Node& n, const Ins& in, int opset) {
    const Tensor& X = need(in, 0);
    const std::size_t groups = static_cast<std::size_t>(n.i("num_groups", 1));
    const std::size_t N = static_cast<std::size_t>(X.shape[0]), C = static_cast<std::size_t>(X.shape[1]);
    Tensor r = floatTensor(X.shape);
    normalizeRuns(X.f.data(), r.f.data(), N * groups, X.size() / (N * groups), n.f("epsilon", 1e-5f));
    std::vector<float> scale = need(in, 1).f, bias = need(in, 2).f;
    if (opset < 21 && scale.size() == groups && groups != C) {  // opset 18: one scale / bias per group
        std::vector<float> s(C), b(C);
        for (std::size_t c = 0; c < C; ++c) {
            s[c] = scale[c / (C / groups)];
            b[c] = bias[c / (C / groups)];
        }
        scale = s;
        bias = b;
    }
    channelAffine(r, scale, bias);
    return {r};
}

Outs batchNorm(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::vector<float>&scale = need(in, 1).f, &bias = need(in, 2).f, &mean = need(in, 3).f, &var = need(in, 4).f;
    const float eps = n.f("epsilon", 1e-5f);
    const std::size_t C = static_cast<std::size_t>(X.shape.size() > 1 ? X.shape[1] : 1);
    std::vector<float> s(C), b(C);
    for (std::size_t c = 0; c < C; ++c) {
        s[c] = scale[c] / std::sqrt(var[c] + eps);
        b[c] = bias[c] - mean[c] * s[c];
    }
    Tensor r = X;
    if (X.shape.size() == 1) {
        for (std::size_t k = 0; k < r.f.size(); ++k) r.f[k] = r.f[k] * s[k] + b[k];
        return {r};
    }
    if (X.shape.size() == 2) {
        for (std::size_t k = 0; k < r.f.size(); ++k) r.f[k] = r.f[k] * s[k % C] + b[k % C];
        return {r};
    }
    channelAffine(r, s, b);
    return {r};
}

Outs lpNorm(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", -1), X.shape.size()));
    const std::int64_t p = n.i("p", 2);
    const std::size_t len = static_cast<std::size_t>(X.shape[axis]), inner = X.size() / numel(Shape(X.shape.begin(), X.shape.begin() + static_cast<std::ptrdiff_t>(axis) + 1));
    const std::size_t outer = X.size() / (len * inner);
    Tensor r = X;
    for (std::size_t o = 0; o < outer; ++o)
        for (std::size_t i = 0; i < inner; ++i) {
            double acc = 0.0;
            for (std::size_t k = 0; k < len; ++k) {
                const float v = X.f[(o * len + k) * inner + i];
                acc += p == 1 ? std::fabs(v) : static_cast<double>(v) * v;
            }
            const double norm = p == 1 ? acc : std::sqrt(acc);
            for (std::size_t k = 0; k < len; ++k) r.f[(o * len + k) * inner + i] = norm > 0 ? static_cast<float>(X.f[(o * len + k) * inner + i] / norm) : 0.0f;
        }
    return {r};
}

// ---- softmax ----

Outs softmaxOp(const Node& n, const Ins& in, int opset, bool log) {
    const Tensor& X = need(in, 0);
    const std::size_t rank = X.shape.size();
    const std::int64_t axisAttr = n.i("axis", opset >= 13 ? -1 : 1);
    const std::size_t axis = static_cast<std::size_t>(normAxis(axisAttr, rank));
    Tensor r = floatTensor(X.shape);
    if (opset < 13) {  // the input is seen as 2-D: [prod(dims < axis), prod(dims >= axis)]
        std::size_t inner = 1;
        for (std::size_t k = axis; k < rank; ++k) inner *= static_cast<std::size_t>(X.shape[k]);
        const std::size_t rows = X.size() / std::max<std::size_t>(1, inner);
        std::copy(X.f.begin(), X.f.end(), r.f.begin());
        for (std::size_t row = 0; row < rows; ++row) {
            float* v = r.f.data() + row * inner;
            const float m = *std::max_element(v, v + inner);
            double s = 0;
            for (std::size_t k = 0; k < inner; ++k) s += std::exp(v[k] - m);
            for (std::size_t k = 0; k < inner; ++k) v[k] = log ? static_cast<float>(v[k] - m - std::log(s)) : static_cast<float>(std::exp(v[k] - m) / s);
        }
        return {r};
    }
    const std::size_t len = static_cast<std::size_t>(X.shape[axis]);
    std::size_t inner = 1;
    for (std::size_t k = axis + 1; k < rank; ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    const std::size_t outer = X.size() / std::max<std::size_t>(1, len * inner);
    if (inner == 1 && len > 0) {  // the last axis: contiguous rows, vectorised
        std::copy(X.f.begin(), X.f.end(), r.f.begin());
        parallelFor(0, outer, std::max<std::size_t>(1, 8192 / len), [&](std::size_t o0, std::size_t o1) {
            for (std::size_t o = o0; o < o1; ++o) {
                float* row = r.f.data() + o * len;
                softmaxRow(row, len);
                if (log)
                    for (std::size_t k = 0; k < len; ++k) row[k] = std::log(row[k]);
            }
        });
        if (!log) return {r};
        // log-softmax as x - max - log(sum exp): more accurate for tiny probabilities
        parallelFor(0, outer, 64, [&](std::size_t o0, std::size_t o1) {
            for (std::size_t o = o0; o < o1; ++o) {
                const float* x = X.f.data() + o * len;
                const float m = *std::max_element(x, x + len);
                double sum = 0;
                for (std::size_t k = 0; k < len; ++k) sum += std::exp(x[k] - m);
                const float ls = static_cast<float>(std::log(sum));
                for (std::size_t k = 0; k < len; ++k) r.f[o * len + k] = x[k] - m - ls;
            }
        });
        return {r};
    }
    parallelFor(0, outer * inner, 64, [&](std::size_t q0, std::size_t q1) {
        std::vector<float> tmp(len);
        for (std::size_t q = q0; q < q1; ++q) {
            const std::size_t o = q / inner, i = q % inner;
            float m = -std::numeric_limits<float>::infinity();
            for (std::size_t k = 0; k < len; ++k) m = std::max(m, X.f[(o * len + k) * inner + i]);
            double s = 0;
            for (std::size_t k = 0; k < len; ++k) {
                tmp[k] = std::exp(X.f[(o * len + k) * inner + i] - m);
                s += tmp[k];
            }
            for (std::size_t k = 0; k < len; ++k)
                r.f[(o * len + k) * inner + i] = log ? static_cast<float>(X.f[(o * len + k) * inner + i] - m - std::log(s)) : static_cast<float>(tmp[k] / s);
        }
    });
    return {r};
}

// ---- shape operators ----

Outs reshapeOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    std::vector<std::int64_t> s = asInts(need(in, 1));
    const bool allowZero = n.i("allowzero", 0) != 0;
    std::int64_t known = 1;
    int unknown = -1;
    for (std::size_t k = 0; k < s.size(); ++k) {
        if (s[k] == 0 && !allowZero) s[k] = X.shape.at(k);
        if (s[k] == -1) unknown = static_cast<int>(k);
        else known *= s[k];
    }
    if (unknown >= 0) s[static_cast<std::size_t>(unknown)] = known ? static_cast<std::int64_t>(X.size()) / known : 0;
    if (numel(s) != X.size()) throw std::runtime_error("cannot reshape " + shapeText(X.shape) + " to " + shapeText(s));
    Tensor r = X;
    r.shape = s;
    return {r};
}

Outs flattenOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::int64_t axis = X.shape.empty() ? 0 : (n.i("axis", 1) < 0 ? n.i("axis", 1) + static_cast<std::int64_t>(X.shape.size()) : n.i("axis", 1));
    std::int64_t a = 1;
    for (std::int64_t k = 0; k < axis; ++k) a *= X.shape[static_cast<std::size_t>(k)];
    Tensor r = X;
    r.shape = {a, a ? static_cast<std::int64_t>(X.size()) / a : 0};
    return {r};
}

std::vector<std::int64_t> axesFrom(const Node& n, const Ins& in, std::size_t inputIndex, int opset, int inputSince) {
    if (opset >= inputSince) {
        const Tensor* a = opt(in, inputIndex);
        return a ? asInts(*a) : std::vector<std::int64_t>{};
    }
    return n.ints("axes");
}

Outs squeezeOp(const Node& n, const Ins& in, int opset) {
    const Tensor& X = need(in, 0);
    std::vector<std::int64_t> axes = axesFrom(n, in, 1, opset, 13);
    std::vector<bool> drop(X.shape.size(), false);
    if (axes.empty())
        for (std::size_t k = 0; k < X.shape.size(); ++k) drop[k] = X.shape[k] == 1;
    for (std::int64_t a : axes) drop[static_cast<std::size_t>(normAxis(a, X.shape.size()))] = true;
    Tensor r = X;
    r.shape.clear();
    for (std::size_t k = 0; k < X.shape.size(); ++k)
        if (!drop[k]) r.shape.push_back(X.shape[k]);
    return {r};
}

Outs unsqueezeOp(const Node& n, const Ins& in, int opset) {
    const Tensor& X = need(in, 0);
    std::vector<std::int64_t> axes = axesFrom(n, in, 1, opset, 13);
    const std::size_t rank = X.shape.size() + axes.size();
    std::vector<bool> isNew(rank, false);
    for (std::int64_t a : axes) isNew[static_cast<std::size_t>(normAxis(a, rank))] = true;
    Tensor r = X;
    r.shape.clear();
    std::size_t src = 0;
    for (std::size_t k = 0; k < rank; ++k) r.shape.push_back(isNew[k] ? 1 : X.shape[src++]);
    return {r};
}

Tensor transpose(const Tensor& X, const std::vector<std::int64_t>& permIn) {
    const std::size_t rank = X.shape.size();
    std::vector<std::int64_t> perm = permIn;
    if (perm.empty())
        for (std::size_t k = 0; k < rank; ++k) perm.push_back(static_cast<std::int64_t>(rank - 1 - k));
    Shape os(rank);
    for (std::size_t k = 0; k < rank; ++k) os[k] = X.shape[static_cast<std::size_t>(perm[k])];
    Tensor r = tensorLike(X, os);
    const std::vector<std::size_t> is = strides(X.shape);
    std::vector<std::size_t> st(rank);
    for (std::size_t k = 0; k < rank; ++k) st[k] = is[static_cast<std::size_t>(perm[k])];
    const std::array<std::vector<std::size_t>, 1> sts{st};
    if (X.kind == Tensor::Kind::Float)
        broadcastLoop<1>(os, sts, [&](std::size_t o, const std::array<std::size_t, 1>& x) { r.f[o] = X.f[x[0]]; });
    else
        broadcastLoop<1>(os, sts, [&](std::size_t o, const std::array<std::size_t, 1>& x) { r.i[o] = X.i[x[0]]; });
    return r;
}

Outs concatOp(const Node& n, const Ins& in, int) {
    std::vector<const Tensor*> parts;
    for (const Tensor* t : in)
        if (t) parts.push_back(t);
    const Tensor& first = *parts.at(0);
    const std::size_t rank = first.shape.size();
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", 0), rank));
    Shape os = first.shape;
    os[axis] = 0;
    for (const Tensor* t : parts) os[axis] += t->shape[axis];
    std::size_t outer = 1, inner = 1;
    for (std::size_t k = 0; k < axis; ++k) outer *= static_cast<std::size_t>(os[k]);
    for (std::size_t k = axis + 1; k < rank; ++k) inner *= static_cast<std::size_t>(os[k]);
    if (outer == 1) {  // the parts one after another: appended, no zero fill first
        Tensor r;
        r.kind = first.kind;
        r.elemType = first.elemType;
        r.shape = os;
        if (r.kind == Tensor::Kind::Float) {
            r.f.reserve(numel(os));
            for (const Tensor* t : parts) r.f.insert(r.f.end(), t->f.begin(), t->f.end());
        } else {
            r.i.reserve(numel(os));
            for (const Tensor* t : parts) r.i.insert(r.i.end(), t->i.begin(), t->i.end());
        }
        return {r};
    }
    Tensor r = tensorLike(first, os);
    const std::size_t outRow = static_cast<std::size_t>(os[axis]) * inner;
    std::vector<std::size_t> offsets;
    std::size_t offset = 0;
    for (const Tensor* t : parts) {
        offsets.push_back(offset);
        offset += static_cast<std::size_t>(t->shape[axis]) * inner;
    }
    // Every (outer index, part) is one contiguous copy; big ones are split over the threads.
    const std::size_t jobs = outer * parts.size();
    parallelFor(0, jobs, std::max<std::size_t>(1, jobs / 64), [&](std::size_t j0, std::size_t j1) {
        for (std::size_t j = j0; j < j1; ++j) {
            const std::size_t o = j / parts.size(), q = j % parts.size();
            const Tensor* t = parts[q];
            const std::size_t row = static_cast<std::size_t>(t->shape[axis]) * inner;
            if (r.kind == Tensor::Kind::Float) std::copy(t->f.begin() + o * row, t->f.begin() + (o + 1) * row, r.f.begin() + o * outRow + offsets[q]);
            else std::copy(t->i.begin() + o * row, t->i.begin() + (o + 1) * row, r.i.begin() + o * outRow + offsets[q]);
        }
    });
    return {r};
}

// Copies the sub-block [start, start + count) along `axis`.
Tensor sliceAxis(const Tensor& X, std::size_t axis, std::size_t start, std::size_t count) {
    Shape os = X.shape;
    os[axis] = static_cast<std::int64_t>(count);
    Tensor r = tensorLike(X, os);
    std::size_t outer = 1, inner = 1;
    for (std::size_t k = 0; k < axis; ++k) outer *= static_cast<std::size_t>(X.shape[k]);
    for (std::size_t k = axis + 1; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    const std::size_t len = static_cast<std::size_t>(X.shape[axis]);
    for (std::size_t o = 0; o < outer; ++o) {
        const std::size_t from = (o * len + start) * inner, to = o * count * inner;
        if (X.kind == Tensor::Kind::Float) std::copy(X.f.begin() + from, X.f.begin() + from + count * inner, r.f.begin() + to);
        else std::copy(X.i.begin() + from, X.i.begin() + from + count * inner, r.i.begin() + to);
    }
    return r;
}

Outs splitOp(const Node& n, const Ins& in, int opset) {
    const Tensor& X = need(in, 0);
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", 0), X.shape.size()));
    std::vector<std::int64_t> sizes = opset >= 13 ? (opt(in, 1) ? asInts(*in[1]) : std::vector<std::int64_t>{}) : n.ints("split");
    const std::size_t parts = n.outputs.size();
    const std::int64_t len = X.shape[axis];
    if (sizes.empty()) {
        const std::int64_t k = static_cast<std::int64_t>(n.i("num_outputs", static_cast<std::int64_t>(parts)));
        const std::int64_t each = (len + k - 1) / k;
        for (std::int64_t q = 0; q < k; ++q) sizes.push_back(std::min(each, len - q * each));
    }
    Outs out;
    std::size_t start = 0;
    for (std::int64_t s : sizes) {
        out.push_back(sliceAxis(X, axis, start, static_cast<std::size_t>(s)));
        start += static_cast<std::size_t>(s);
    }
    return out;
}

Outs sliceOp(const Node& n, const Ins& in, int opset) {
    const Tensor& X = need(in, 0);
    std::vector<std::int64_t> starts, ends, axes, steps;
    if (opset >= 10) {
        starts = asInts(need(in, 1));
        ends = asInts(need(in, 2));
        if (opt(in, 3)) axes = asInts(*in[3]);
        if (opt(in, 4)) steps = asInts(*in[4]);
    } else {
        starts = n.ints("starts");
        ends = n.ints("ends");
        axes = n.ints("axes");
    }
    const std::size_t rank = X.shape.size();
    if (axes.empty())
        for (std::size_t k = 0; k < starts.size(); ++k) axes.push_back(static_cast<std::int64_t>(k));
    if (steps.empty()) steps.assign(starts.size(), 1);
    std::vector<std::int64_t> b(rank, 0), st(rank, 1);
    Shape os = X.shape;
    for (std::size_t q = 0; q < axes.size(); ++q) {
        const std::size_t a = static_cast<std::size_t>(normAxis(axes[q], rank));
        const std::int64_t dim = X.shape[a], step = steps[q];
        if (step == 0) throw std::runtime_error("Slice step 0");
        std::int64_t s = starts[q], e = ends[q];
        if (s < 0) s += dim;
        if (e < 0) e += dim;
        if (step > 0) {
            s = std::clamp<std::int64_t>(s, 0, dim);
            e = std::clamp<std::int64_t>(e, 0, dim);
            os[a] = e > s ? (e - s + step - 1) / step : 0;
        } else {
            s = std::clamp<std::int64_t>(s, -1, dim - 1);
            e = std::clamp<std::int64_t>(e, -1, dim - 1);
            os[a] = s > e ? (s - e - step - 1) / (-step) : 0;
        }
        b[a] = s;
        st[a] = step;
    }
    Tensor r = tensorLike(X, os);
    const std::vector<std::size_t> is = strides(X.shape);
    const std::size_t total = r.size();
    for (std::size_t o = 0; o < total; ++o) {
        std::size_t rem = o, src = 0;
        for (std::size_t d = rank; d-- > 0;) {
            const std::size_t i = rem % static_cast<std::size_t>(os[d]);
            rem /= static_cast<std::size_t>(os[d]);
            src += static_cast<std::size_t>(b[d] + static_cast<std::int64_t>(i) * st[d]) * is[d];
        }
        copyElem(X, src, r, o);
    }
    return {r};
}

Outs gatherOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::vector<std::int64_t> idx = asInts(need(in, 1));
    const Shape& idxShape = in[1]->shape;
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", 0), X.shape.size()));
    Shape os(X.shape.begin(), X.shape.begin() + static_cast<std::ptrdiff_t>(axis));
    os.insert(os.end(), idxShape.begin(), idxShape.end());
    os.insert(os.end(), X.shape.begin() + static_cast<std::ptrdiff_t>(axis) + 1, X.shape.end());
    Tensor r = tensorLike(X, os);
    std::size_t outer = 1, inner = 1;
    for (std::size_t k = 0; k < axis; ++k) outer *= static_cast<std::size_t>(X.shape[k]);
    for (std::size_t k = axis + 1; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    const std::int64_t len = X.shape[axis];
    for (std::size_t o = 0; o < outer; ++o)
        for (std::size_t q = 0; q < idx.size(); ++q) {
            std::int64_t j = idx[q] < 0 ? idx[q] + len : idx[q];
            if (j < 0 || j >= len) throw std::runtime_error("Gather index " + std::to_string(idx[q]) + " out of range");
            const std::size_t from = (o * static_cast<std::size_t>(len) + static_cast<std::size_t>(j)) * inner, to = (o * idx.size() + q) * inner;
            if (X.kind == Tensor::Kind::Float) std::copy(X.f.begin() + from, X.f.begin() + from + inner, r.f.begin() + to);
            else std::copy(X.i.begin() + from, X.i.begin() + from + inner, r.i.begin() + to);
        }
    return {r};
}

Outs gatherElementsOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const Tensor& I = need(in, 1);
    const std::vector<std::int64_t> idx = asInts(I);
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", 0), X.shape.size()));
    Tensor r = tensorLike(X, I.shape);
    const std::vector<std::size_t> xs = strides(X.shape);
    for (std::size_t o = 0; o < idx.size(); ++o) {
        std::size_t rem = o, src = 0;
        for (std::size_t d = I.shape.size(); d-- > 0;) {
            std::size_t i = rem % static_cast<std::size_t>(I.shape[d]);
            rem /= static_cast<std::size_t>(I.shape[d]);
            if (d == axis) {
                const std::int64_t j = idx[o] < 0 ? idx[o] + X.shape[axis] : idx[o];
                i = static_cast<std::size_t>(j);
            }
            src += i * xs[d];
        }
        copyElem(X, src, r, o);
    }
    return {r};
}

Outs scatterNdOp(const Node& n, const Ins& in, int) {
    Tensor r = need(in, 0);
    const Tensor& I = need(in, 1);
    const Tensor& U = need(in, 2);
    const std::vector<std::int64_t> idx = asInts(I);
    const std::size_t q = static_cast<std::size_t>(I.shape.back()), updates = idx.size() / q;
    const std::vector<std::size_t> xs = strides(r.shape);
    std::size_t slice = 1;
    for (std::size_t k = q; k < r.shape.size(); ++k) slice *= static_cast<std::size_t>(r.shape[k]);
    const std::string reduction = n.s("reduction", "none");
    for (std::size_t u = 0; u < updates; ++u) {
        std::size_t base = 0;
        for (std::size_t k = 0; k < q; ++k) {
            std::int64_t j = idx[u * q + k];
            if (j < 0) j += r.shape[k];
            base += static_cast<std::size_t>(j) * xs[k];
        }
        for (std::size_t e = 0; e < slice; ++e) {
            if (r.kind == Tensor::Kind::Float) {
                float& dst = r.f[base + e];
                const float v = U.f[u * slice + e];
                dst = reduction == "add" ? dst + v : reduction == "mul" ? dst * v : v;
            } else {
                std::int64_t& dst = r.i[base + e];
                const std::int64_t v = U.i[u * slice + e];
                dst = reduction == "add" ? dst + v : reduction == "mul" ? dst * v : v;
            }
        }
    }
    return {r};
}

Outs expandOp(const Node&, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const Shape target = asInts(need(in, 1));
    const Shape os = broadcastShape(X.shape, target);
    Tensor r = tensorLike(X, os);
    const std::array<std::vector<std::size_t>, 1> st{broadcastStrides(X.shape, os)};
    if (X.kind == Tensor::Kind::Float) broadcastLoop<1>(os, st, [&](std::size_t o, const std::array<std::size_t, 1>& x) { r.f[o] = X.f[x[0]]; });
    else broadcastLoop<1>(os, st, [&](std::size_t o, const std::array<std::size_t, 1>& x) { r.i[o] = X.i[x[0]]; });
    return {r};
}

Outs tileOp(const Node&, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::vector<std::int64_t> reps = asInts(need(in, 1));
    Shape os = X.shape;
    for (std::size_t k = 0; k < os.size(); ++k) os[k] *= reps[k];
    Tensor r = tensorLike(X, os);
    const std::vector<std::size_t> xs = strides(X.shape);
    for (std::size_t o = 0; o < r.size(); ++o) {
        std::size_t rem = o, src = 0;
        for (std::size_t d = os.size(); d-- > 0;) {
            const std::size_t i = rem % static_cast<std::size_t>(os[d]);
            rem /= static_cast<std::size_t>(os[d]);
            src += (i % static_cast<std::size_t>(X.shape[d])) * xs[d];
        }
        copyElem(X, src, r, o);
    }
    return {r};
}

Outs padOp(const Node& n, const Ins& in, int opset) {
    const Tensor& X = need(in, 0);
    const std::size_t rank = X.shape.size();
    std::vector<std::int64_t> pads = opset >= 11 ? asInts(need(in, 1)) : n.ints("pads");
    float value = opset >= 11 ? (opt(in, 2) && in[2]->size() ? scalarFloat(*in[2]) : 0.0f) : n.f("value", 0.0f);
    if (opset >= 18 && opt(in, 3)) {  // pads for the listed axes only
        const std::vector<std::int64_t> axes = asInts(*in[3]);
        std::vector<std::int64_t> full(2 * rank, 0);
        for (std::size_t q = 0; q < axes.size(); ++q) {
            const std::size_t a = static_cast<std::size_t>(normAxis(axes[q], rank));
            full[a] = pads[q];
            full[a + rank] = pads[q + axes.size()];
        }
        pads = full;
    }
    const std::string mode = n.s("mode", "constant");
    Shape os = X.shape;
    for (std::size_t k = 0; k < rank; ++k) os[k] += pads[k] + pads[k + rank];
    Tensor r = tensorLike(X, os);
    const std::vector<std::size_t> xs = strides(X.shape);
    for (std::size_t o = 0; o < r.size(); ++o) {
        std::size_t rem = o, src = 0;
        bool outside = false;
        for (std::size_t d = rank; d-- > 0;) {
            const std::int64_t i = static_cast<std::int64_t>(rem % static_cast<std::size_t>(os[d])) - pads[d];
            rem /= static_cast<std::size_t>(os[d]);
            std::int64_t j = i;
            const std::int64_t len = X.shape[d];
            if (i < 0 || i >= len) {
                if (mode == "constant") outside = true;
                else if (mode == "edge") j = std::clamp<std::int64_t>(i, 0, len - 1);
                else if (mode == "reflect") {
                    if (len == 1) j = 0;
                    else {
                        const std::int64_t period = 2 * (len - 1);
                        j = ((i % period) + period) % period;
                        if (j >= len) j = period - j;
                    }
                } else if (mode == "wrap") j = ((i % len) + len) % len;
                else throw std::runtime_error("Pad mode " + mode);
            }
            if (!outside) src += static_cast<std::size_t>(j) * xs[d];
        }
        if (outside) {
            if (r.kind == Tensor::Kind::Float) r.f[o] = value;
            else r.i[o] = static_cast<std::int64_t>(value);
        } else {
            copyElem(X, src, r, o);
        }
    }
    return {r};
}

Outs depthToSpace(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::int64_t b = n.i("blocksize", 2), N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
    const bool crd = n.s("mode", "DCR") == "CRD";
    const std::int64_t oc = C / (b * b);
    Tensor r = floatTensor({N, oc, H * b, W * b});
    for (std::int64_t nn = 0; nn < N; ++nn)
        for (std::int64_t c = 0; c < oc; ++c)
            for (std::int64_t y = 0; y < H * b; ++y)
                for (std::int64_t x = 0; x < W * b; ++x) {
                    const std::int64_t by = y % b, bx = x % b;
                    const std::int64_t ic = crd ? c * b * b + by * b + bx : (by * b + bx) * oc + c;
                    r.f[static_cast<std::size_t>(((nn * oc + c) * H * b + y) * W * b + x)] = X.f[static_cast<std::size_t>(((nn * C + ic) * H + y / b) * W + x / b)];
                }
    return {r};
}

Outs spaceToDepth(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::int64_t b = n.i("blocksize", 2), N = X.shape[0], C = X.shape[1], H = X.shape[2], W = X.shape[3];
    Tensor r = floatTensor({N, C * b * b, H / b, W / b});
    for (std::int64_t nn = 0; nn < N; ++nn)
        for (std::int64_t c = 0; c < C; ++c)
            for (std::int64_t y = 0; y < H; ++y)
                for (std::int64_t x = 0; x < W; ++x) {
                    const std::int64_t oc = ((y % b) * b + (x % b)) * C + c;
                    r.f[static_cast<std::size_t>(((nn * C * b * b + oc) * (H / b) + y / b) * (W / b) + x / b)] = X.f[static_cast<std::size_t>(((nn * C + c) * H + y) * W + x)];
                }
    return {r};
}

// ---- reductions ----

Outs reduceOp(const Node& n, const Ins& in, int opset, const std::string& kind) {
    const Tensor& X = need(in, 0);
    const std::size_t rank = X.shape.size();
    // Axes moved from an attribute to an input: ReduceSum at opset 13, the others at 18.
    const int since = kind == "sum" ? 13 : 18;
    std::vector<std::int64_t> axes = axesFrom(n, in, 1, opset, since);
    const bool keep = n.i("keepdims", 1) != 0;
    if (axes.empty() && n.i("noop_with_empty_axes", 0) != 0) return {X};
    std::vector<bool> red(rank, axes.empty());
    for (std::int64_t a : axes) red[static_cast<std::size_t>(normAxis(a, rank))] = true;
    Shape os, kept;
    for (std::size_t k = 0; k < rank; ++k) {
        if (red[k]) {
            if (keep) os.push_back(1);
        } else {
            os.push_back(X.shape[k]);
        }
        kept.push_back(red[k] ? 1 : X.shape[k]);
    }
    const std::size_t outN = numel(kept);
    // Reduced axes forming one contiguous block (the usual case: the last axis for a layer norm):
    // outer x len x inner, summed straight from the data.
    {
        std::size_t first = rank, last = 0;
        for (std::size_t k = 0; k < rank; ++k)
            if (red[k] && X.shape[k] != 1) {
                first = std::min(first, k);
                last = k;
            }
        bool block = first < rank;
        for (std::size_t k = first; block && k <= last; ++k) block = red[k] || X.shape[k] == 1;
        if (block && X.kind == Tensor::Kind::Float && (kind == "sum" || kind == "mean" || kind == "max" || kind == "min" || kind == "sumsquare" || kind == "l2")) {
            std::size_t outer = 1, len = 1, inner = 1;
            for (std::size_t k = 0; k < first; ++k) outer *= static_cast<std::size_t>(X.shape[k]);
            for (std::size_t k = first; k <= last; ++k) len *= static_cast<std::size_t>(X.shape[k]);
            for (std::size_t k = last + 1; k < rank; ++k) inner *= static_cast<std::size_t>(X.shape[k]);
            Tensor r = floatTensor(os);
            parallelFor(0, outer, 1, [&](std::size_t o0, std::size_t o1) {
                std::vector<double> a(inner);
                for (std::size_t o = o0; o < o1; ++o) {
                    const float* x = X.f.data() + o * len * inner;
                    const bool mx = kind == "max", mn = kind == "min", sq = kind == "sumsquare" || kind == "l2";
                    for (std::size_t i = 0; i < inner; ++i) a[i] = mx ? -std::numeric_limits<double>::infinity() : mn ? std::numeric_limits<double>::infinity() : 0.0;
                    for (std::size_t l = 0; l < len; ++l) {
                        const float* row = x + l * inner;
                        if (mx) for (std::size_t i = 0; i < inner; ++i) a[i] = std::max(a[i], static_cast<double>(row[i]));
                        else if (mn) for (std::size_t i = 0; i < inner; ++i) a[i] = std::min(a[i], static_cast<double>(row[i]));
                        else if (sq) for (std::size_t i = 0; i < inner; ++i) a[i] += static_cast<double>(row[i]) * row[i];
                        else for (std::size_t i = 0; i < inner; ++i) a[i] += row[i];
                    }
                    for (std::size_t i = 0; i < inner; ++i) {
                        double v = a[i];
                        if (kind == "mean") v /= static_cast<double>(len);
                        else if (kind == "l2") v = std::sqrt(v);
                        r.f[o * inner + i] = static_cast<float>(v);
                    }
                }
            });
            return {r};
        }
    }
    std::vector<double> acc(outN, kind == "max" ? -std::numeric_limits<double>::infinity() : kind == "min" ? std::numeric_limits<double>::infinity() : kind == "prod" ? 1.0 : 0.0);
    std::vector<double> mx;
    if (kind == "logsumexp") mx.assign(outN, -std::numeric_limits<double>::infinity());
    const std::vector<std::size_t> ks = strides(kept);
    const std::vector<double> vals = [&] {
        std::vector<double> v(X.size());
        for (std::size_t k = 0; k < v.size(); ++k) v[k] = X.kind == Tensor::Kind::Float ? X.f[k] : static_cast<double>(X.i[k]);
        return v;
    }();
    auto target = [&](std::size_t idx) {
        std::size_t rem = idx, o = 0;
        for (std::size_t d = rank; d-- > 0;) {
            const std::size_t i = rem % static_cast<std::size_t>(X.shape[d]);
            rem /= static_cast<std::size_t>(X.shape[d]);
            if (!red[d]) o += i * ks[d];
        }
        return o;
    };
    if (kind == "logsumexp")
        for (std::size_t k = 0; k < vals.size(); ++k) {
            const std::size_t o = target(k);
            mx[o] = std::max(mx[o], vals[k]);
        }
    for (std::size_t k = 0; k < vals.size(); ++k) {
        const std::size_t o = target(k);
        const double v = vals[k];
        if (kind == "sum" || kind == "mean") acc[o] += v;
        else if (kind == "max") acc[o] = std::max(acc[o], v);
        else if (kind == "min") acc[o] = std::min(acc[o], v);
        else if (kind == "prod") acc[o] *= v;
        else if (kind == "sumsquare" || kind == "l2") acc[o] += v * v;
        else if (kind == "l1") acc[o] += std::fabs(v);
        else if (kind == "logsumexp") acc[o] += std::exp(v - mx[o]);
    }
    const double count = static_cast<double>(X.size()) / static_cast<double>(std::max<std::size_t>(1, outN));
    for (std::size_t o = 0; o < outN; ++o) {
        if (kind == "mean") acc[o] /= count;
        else if (kind == "l2") acc[o] = std::sqrt(acc[o]);
        else if (kind == "logsumexp") acc[o] = std::log(acc[o]) + mx[o];
    }
    Tensor r = tensorLike(X, os);
    for (std::size_t o = 0; o < outN; ++o) {
        if (r.kind == Tensor::Kind::Float) r.f[o] = static_cast<float>(acc[o]);
        else r.i[o] = static_cast<std::int64_t>(acc[o]);
    }
    return {r};
}

Outs argOp(const Node& n, const Ins& in, bool isMax) {
    const Tensor& X = need(in, 0);
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", 0), X.shape.size()));
    const bool keep = n.i("keepdims", 1) != 0, last = n.i("select_last_index", 0) != 0;
    const std::size_t len = static_cast<std::size_t>(X.shape[axis]);
    std::size_t inner = 1;
    for (std::size_t k = axis + 1; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    const std::size_t outer = X.size() / std::max<std::size_t>(1, len * inner);
    Shape os = X.shape;
    if (keep) os[axis] = 1;
    else os.erase(os.begin() + static_cast<std::ptrdiff_t>(axis));
    Tensor r = intTensor(os);
    const std::vector<float> v = asFloats(X);
    for (std::size_t o = 0; o < outer; ++o)
        for (std::size_t i = 0; i < inner; ++i) {
            std::size_t best = 0;
            for (std::size_t k = 1; k < len; ++k) {
                const float a = v[(o * len + k) * inner + i], b = v[(o * len + best) * inner + i];
                if (isMax ? (a > b || (last && a == b)) : (a < b || (last && a == b))) best = k;
            }
            r.i[o * inner + i] = static_cast<std::int64_t>(best);
        }
    return {r};
}

Outs cumSumOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::size_t axis = static_cast<std::size_t>(normAxis(asInts(need(in, 1)).at(0), X.shape.size()));
    const bool exclusive = n.i("exclusive", 0) != 0, reverse = n.i("reverse", 0) != 0;
    const std::size_t len = static_cast<std::size_t>(X.shape[axis]);
    std::size_t inner = 1;
    for (std::size_t k = axis + 1; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    const std::size_t outer = X.size() / std::max<std::size_t>(1, len * inner);
    Tensor r = tensorLike(X, X.shape);
    for (std::size_t o = 0; o < outer; ++o)
        for (std::size_t i = 0; i < inner; ++i) {
            double acc = 0.0;
            for (std::size_t q = 0; q < len; ++q) {
                const std::size_t k = reverse ? len - 1 - q : q, at = (o * len + k) * inner + i;
                const double v = X.kind == Tensor::Kind::Float ? X.f[at] : static_cast<double>(X.i[at]);
                if (exclusive) {
                    if (r.kind == Tensor::Kind::Float) r.f[at] = static_cast<float>(acc);
                    else r.i[at] = static_cast<std::int64_t>(acc);
                    acc += v;
                } else {
                    acc += v;
                    if (r.kind == Tensor::Kind::Float) r.f[at] = static_cast<float>(acc);
                    else r.i[at] = static_cast<std::int64_t>(acc);
                }
            }
        }
    return {r};
}

// ---- Resize ----

// Resamples one axis. mode: nearest / linear / cubic.
Tensor resizeAxis(const Tensor& X, std::size_t axis, std::size_t outLen, float scale, const std::string& mode, const std::string& coord,
                  const std::string& nearestMode, float cubicA, bool excludeOutside) {
    const std::size_t inLen = static_cast<std::size_t>(X.shape[axis]);
    Shape os = X.shape;
    os[axis] = static_cast<std::int64_t>(outLen);
    Tensor r = floatTensor(os);
    std::size_t outer = 1, inner = 1;
    for (std::size_t k = 0; k < axis; ++k) outer *= static_cast<std::size_t>(X.shape[k]);
    for (std::size_t k = axis + 1; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    // Source taps and weights per output position.
    std::vector<std::vector<std::pair<std::size_t, float>>> taps(outLen);
    for (std::size_t x = 0; x < outLen; ++x) {
        double src;
        const double xd = static_cast<double>(x);
        if (coord == "align_corners") src = outLen > 1 ? xd * (static_cast<double>(inLen) - 1) / (static_cast<double>(outLen) - 1) : 0.0;
        else if (coord == "asymmetric") src = xd / scale;
        else if (coord == "pytorch_half_pixel") src = outLen > 1 ? (xd + 0.5) / scale - 0.5 : 0.0;
        else if (coord == "tf_half_pixel_for_nn") src = (xd + 0.5) / scale;
        else if (coord == "half_pixel_symmetric") {
            const double adj = static_cast<double>(outLen) / (scale * static_cast<double>(inLen)), center = static_cast<double>(inLen) / 2.0;
            src = center * (1.0 - adj) + (xd + 0.5) / scale - 0.5;
        } else src = (xd + 0.5) / scale - 0.5;  // half_pixel
        const std::int64_t last = static_cast<std::int64_t>(inLen) - 1;
        auto clampIdx = [&](std::int64_t i) { return static_cast<std::size_t>(std::clamp<std::int64_t>(i, 0, last)); };
        if (mode == "nearest") {
            std::int64_t i;
            if (nearestMode == "floor") i = static_cast<std::int64_t>(std::floor(src));
            else if (nearestMode == "ceil") i = static_cast<std::int64_t>(std::ceil(src));
            else if (nearestMode == "round_prefer_ceil") i = static_cast<std::int64_t>(std::floor(src + 0.5));
            else i = static_cast<std::int64_t>(std::ceil(src - 0.5));  // round_prefer_floor
            taps[x] = {{clampIdx(i), 1.0f}};
        } else if (mode == "linear") {
            const double s = std::clamp(src, 0.0, static_cast<double>(last));
            const std::int64_t i0 = static_cast<std::int64_t>(std::floor(s));
            const double t = s - static_cast<double>(i0);
            taps[x] = {{clampIdx(i0), static_cast<float>(1.0 - t)}, {clampIdx(i0 + 1), static_cast<float>(t)}};
        } else {  // cubic
            const std::int64_t i0 = static_cast<std::int64_t>(std::floor(src));
            const double t = src - static_cast<double>(i0);
            auto w = [&](double d) {
                d = std::fabs(d);
                const double A = cubicA;
                if (d <= 1) return ((A + 2) * d - (A + 3)) * d * d + 1;
                if (d < 2) return ((A * d - 5 * A) * d + 8 * A) * d - 4 * A;
                return 0.0;
            };
            double sum = 0;
            std::vector<std::pair<std::size_t, float>> tp;
            for (int k = -1; k <= 2; ++k) {
                const std::int64_t i = i0 + k;
                double wt = w(t - k);
                if (excludeOutside && (i < 0 || i > last)) wt = 0;
                sum += wt;
                tp.push_back({clampIdx(i), static_cast<float>(wt)});
            }
            if (excludeOutside && sum != 0)
                for (auto& p : tp) p.second = static_cast<float>(p.second / sum);
            taps[x] = tp;
        }
    }
    parallelFor(0, outer, 1, [&](std::size_t o0, std::size_t o1) {
        for (std::size_t o = o0; o < o1; ++o)
            for (std::size_t x = 0; x < outLen; ++x) {
                float* dst = r.f.data() + (o * outLen + x) * inner;
                for (const auto& [i, w] : taps[x]) {
                    const float* src = X.f.data() + (o * inLen + i) * inner;
                    for (std::size_t k = 0; k < inner; ++k) dst[k] += w * src[k];
                }
            }
    });
    return r;
}

Outs resizeOp(const Node& n, const Ins& in, int opset, bool upsample) {
    const Tensor& X = need(in, 0);
    const std::size_t rank = X.shape.size();
    std::vector<float> scales;
    std::vector<std::int64_t> sizes;
    if (upsample) {
        scales = opset >= 9 ? asFloats(need(in, 1)) : n.attr("scales") ? n.attr("scales")->floats : std::vector<float>{};
    } else if (opset == 10) {
        scales = asFloats(need(in, 1));
    } else {
        if (const Tensor* s = opt(in, 2); s && s->size()) scales = asFloats(*s);
        if (const Tensor* z = opt(in, 3); z && z->size()) sizes = asInts(*z);
    }
    std::vector<std::int64_t> axes = n.ints("axes");
    if (axes.empty())
        for (std::size_t k = 0; k < rank; ++k) axes.push_back(static_cast<std::int64_t>(k));
    std::vector<float> fullScale(rank, 1.0f);
    std::vector<std::int64_t> outShape = X.shape;
    for (std::size_t q = 0; q < axes.size(); ++q) {
        const std::size_t a = static_cast<std::size_t>(normAxis(axes[q], rank));
        if (!sizes.empty()) {
            outShape[a] = sizes[q];
            fullScale[a] = static_cast<float>(sizes[q]) / static_cast<float>(X.shape[a]);
        } else if (!scales.empty()) {
            fullScale[a] = scales[q];
            outShape[a] = static_cast<std::int64_t>(std::floor(static_cast<double>(X.shape[a]) * scales[q]));
        }
    }
    const std::string policy = n.s("keep_aspect_ratio_policy", "stretch");
    if (!sizes.empty() && policy != "stretch") {
        float s = policy == "not_larger" ? std::numeric_limits<float>::max() : 0.0f;
        for (std::int64_t ax : axes) {
            const std::size_t a = static_cast<std::size_t>(normAxis(ax, rank));
            const float r = static_cast<float>(outShape[a]) / static_cast<float>(X.shape[a]);
            s = policy == "not_larger" ? std::min(s, r) : std::max(s, r);
        }
        for (std::int64_t ax : axes) {
            const std::size_t a = static_cast<std::size_t>(normAxis(ax, rank));
            fullScale[a] = s;
            outShape[a] = static_cast<std::int64_t>(std::round(static_cast<double>(X.shape[a]) * s));
        }
    }
    std::string mode = n.s("mode", "nearest");
    if (mode == "bilinear" || mode == "trilinear") mode = "linear";
    const std::string coord = (upsample || opset == 10) ? "asymmetric" : n.s("coordinate_transformation_mode", "half_pixel");
    const std::string nearest = (upsample || opset == 10) ? "floor" : n.s("nearest_mode", "round_prefer_floor");
    Tensor cur = X;
    if (cur.kind != Tensor::Kind::Float) throw std::runtime_error("Resize of integers is not supported");
    for (std::size_t a = 0; a < rank; ++a)
        if (outShape[a] != X.shape[a])
            cur = resizeAxis(cur, a, static_cast<std::size_t>(outShape[a]), fullScale[a], mode, coord, nearest, n.f("cubic_coeff_a", -0.75f),
                                 n.i("exclude_outside", 0) != 0);
    return {cur};
}

// ---- Einsum ----

Outs einsumOp(const Node& n, const Ins& in, int) {
    std::string eq = n.s("equation", "");
    eq.erase(std::remove(eq.begin(), eq.end(), ' '), eq.end());
    const std::size_t arrow = eq.find("->");
    std::string lhs = eq.substr(0, arrow), rhs = arrow == std::string::npos ? "" : eq.substr(arrow + 2);
    std::vector<std::string> terms;
    for (std::size_t p = 0, q; ; p = q + 1) {
        q = lhs.find(',', p);
        terms.push_back(lhs.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (q == std::string::npos) break;
    }
    std::vector<const Tensor*> ts;
    for (const Tensor* t : in)
        if (t) ts.push_back(t);
    if (ts.size() != terms.size() || ts.size() > 2) throw std::runtime_error("Einsum '" + eq + "': one or two operands are supported");
    // Expand "..." to letters not used elsewhere.
    std::string spare;
    for (char c = 'A'; c <= 'Z'; ++c)
        if (eq.find(c) == std::string::npos) spare += c;
    std::size_t ellRank = 0;
    for (std::size_t k = 0; k < terms.size(); ++k) {
        const std::size_t e = terms[k].find("...");
        if (e != std::string::npos) ellRank = std::max(ellRank, ts[k]->shape.size() - (terms[k].size() - 3));
    }
    const std::string ell = spare.substr(0, ellRank);
    for (std::size_t k = 0; k < terms.size(); ++k) {
        const std::size_t e = terms[k].find("...");
        if (e != std::string::npos) {
            const std::size_t r = ts[k]->shape.size() - (terms[k].size() - 3);
            terms[k] = terms[k].substr(0, e) + ell.substr(ellRank - r) + terms[k].substr(e + 3);
        }
    }
    if (arrow == std::string::npos) {  // implicit output: letters seen once, sorted
        std::map<char, int> cnt;
        for (const std::string& t : terms)
            for (char c : t) ++cnt[c];
        rhs = ell;
        for (auto& [c, k] : cnt)
            if (k == 1 && ell.find(c) == std::string::npos) rhs += c;
    } else if (rhs.find("...") != std::string::npos) {
        const std::size_t e = rhs.find("...");
        rhs = rhs.substr(0, e) + ell + rhs.substr(e + 3);
    }
    std::map<char, std::int64_t> dim;
    for (std::size_t k = 0; k < terms.size(); ++k) {
        if (terms[k].size() != ts[k]->shape.size()) throw std::runtime_error("Einsum '" + eq + "': term and rank disagree");
        for (std::size_t j = 0; j < terms[k].size(); ++j) {
            auto it = dim.find(terms[k][j]);
            if (it != dim.end() && it->second != ts[k]->shape[j] && it->second != 1 && ts[k]->shape[j] != 1)
                throw std::runtime_error("Einsum '" + eq + "': sizes of '" + std::string(1, terms[k][j]) + "' disagree");
            dim[terms[k][j]] = std::max(it == dim.end() ? 1 : it->second, ts[k]->shape[j]);
        }
    }
    // Sum out letters that appear in one operand only and not in the output; then a batched product.
    auto reduceTo = [&](const Tensor& t, const std::string& term, const std::string& keepLetters, std::string& outTerm) {
        std::vector<std::int64_t> axes;
        outTerm.clear();
        for (std::size_t j = 0; j < term.size(); ++j) {
            if (keepLetters.find(term[j]) == std::string::npos) axes.push_back(static_cast<std::int64_t>(j));
            else outTerm += term[j];
        }
        if (axes.empty()) return t;
        Node rn;
        rn.attrs["keepdims"].i = 0;
        rn.attrs["keepdims"].has = true;
        rn.attrs["axes"].ints = axes;
        rn.attrs["axes"].has = true;
        return reduceOp(rn, {&t}, 11, "sum")[0];
    };
    auto permute = [&](const Tensor& t, const std::string& term, const std::string& order) {
        std::vector<std::int64_t> perm;
        for (char c : order) perm.push_back(static_cast<std::int64_t>(term.find(c)));
        return transpose(t, perm);
    };
    if (ts.size() == 1) {
        std::string t1;
        const Tensor r = reduceTo(*ts[0], terms[0], rhs, t1);
        return {permute(r, t1, rhs)};
    }
    const std::string& A = terms[0];
    const std::string& B = terms[1];
    std::string a1, b1;
    const Tensor ta = reduceTo(*ts[0], A, B + rhs, a1), tb = reduceTo(*ts[1], B, A + rhs, b1);
    std::string batch, aOnly, bOnly, contract;
    for (char c : a1) {
        const bool inB = b1.find(c) != std::string::npos, inOut = rhs.find(c) != std::string::npos;
        if (inB && inOut) batch += c;
        else if (inB) contract += c;
        else aOnly += c;
    }
    for (char c : b1)
        if (a1.find(c) == std::string::npos) bOnly += c;
    auto prod = [&](const std::string& s) {
        std::int64_t p = 1;
        for (char c : s) p *= dim[c];
        return p;
    };
    const std::int64_t nb = prod(batch), m = prod(aOnly), kk = prod(contract), nn = prod(bOnly);
    // Broadcast size-1 batch dims by expanding first.
    auto expandTo = [&](const Tensor& t, const std::string& term) {
        Shape target;
        for (char c : term) target.push_back(dim[c]);
        if (target == t.shape) return t;
        Tensor shapeT = Tensor::ints({static_cast<std::int64_t>(target.size())}, target);
        return expandOp(Node{}, {&t, &shapeT}, 13)[0];
    };
    Tensor pa = permute(expandTo(ta, a1), a1, batch + aOnly + contract), pb = permute(expandTo(tb, b1), b1, batch + contract + bOnly);
    pa.shape = {nb, m, kk};
    pb.shape = {nb, kk, nn};
    Tensor r = matMul(pa, pb);
    Shape os;
    for (char c : batch + aOnly + bOnly) os.push_back(dim[c]);
    r.shape = os;
    return {permute(r, batch + aOnly + bOnly, rhs)};
}

// ---- misc ----

Outs castOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const int to = static_cast<int>(n.i("to", 1));
    Tensor r;
    r.shape = X.shape;
    r.elemType = to;
    if (isFloatType(to)) {
        r.kind = Tensor::Kind::Float;
        r.f = asFloats(X);
    } else {
        r.kind = Tensor::Kind::Int;
        if (X.kind == Tensor::Kind::Float) {
            r.i.resize(X.f.size());
            for (std::size_t k = 0; k < X.f.size(); ++k) r.i[k] = to == 9 ? (X.f[k] != 0.0f) : static_cast<std::int64_t>(X.f[k]);
        } else {
            r.i = X.i;
            if (to == 9)
                for (auto& v : r.i) v = v != 0;
        }
    }
    return {r};
}

Outs constantOp(const Node& n, const Ins&, int) {
    if (const Attribute* a = n.attr("value"); a && a->t) return {*a->t};
    if (const Attribute* a = n.attr("value_float")) return {Tensor::floats({}, {a->f})};
    if (const Attribute* a = n.attr("value_floats")) return {Tensor::floats({static_cast<std::int64_t>(a->floats.size())}, a->floats)};
    if (const Attribute* a = n.attr("value_int")) return {Tensor::ints({}, {a->i})};
    if (const Attribute* a = n.attr("value_ints")) return {Tensor::ints({static_cast<std::int64_t>(a->ints.size())}, a->ints)};
    throw std::runtime_error("Constant without a supported value");
}

Outs constantOfShapeOp(const Node& n, const Ins& in, int) {
    const Shape s = asInts(need(in, 0));
    const Attribute* a = n.attr("value");
    if (!a || !a->t) return {floatTensor(s)};
    Tensor r = tensorLike(*a->t, s);
    if (r.kind == Tensor::Kind::Float) std::fill(r.f.begin(), r.f.end(), a->t->f.at(0));
    else std::fill(r.i.begin(), r.i.end(), a->t->i.at(0));
    return {r};
}

Outs rangeOp(const Node&, const Ins& in, int) {
    const Tensor& s = need(in, 0);
    if (s.kind == Tensor::Kind::Float) {
        const float a = scalarFloat(s), b = scalarFloat(need(in, 1)), d = scalarFloat(need(in, 2));
        const std::int64_t count = std::max<std::int64_t>(0, static_cast<std::int64_t>(std::ceil((b - a) / d)));
        Tensor r = floatTensor({count});
        for (std::int64_t k = 0; k < count; ++k) r.f[static_cast<std::size_t>(k)] = a + d * static_cast<float>(k);
        return {r};
    }
    const std::int64_t a = s.i.at(0), b = need(in, 1).i.at(0), d = need(in, 2).i.at(0);
    const std::int64_t count = std::max<std::int64_t>(0, (b - a + d + (d > 0 ? -1 : 1)) / d);
    Tensor r = intTensor({count}, s.elemType);
    for (std::int64_t k = 0; k < count; ++k) r.i[static_cast<std::size_t>(k)] = a + d * k;
    return {r};
}

Outs whereOp(const Node&, const Ins& in, int) {
    const Tensor& C = need(in, 0);
    const Tensor& X = need(in, 1);
    const Tensor& Y = need(in, 2);
    const Shape os = broadcastShape(broadcastShape(C.shape, X.shape), Y.shape);
    Tensor r = tensorLike(X, os);
    const std::array<std::vector<std::size_t>, 3> st{broadcastStrides(C.shape, os), broadcastStrides(X.shape, os), broadcastStrides(Y.shape, os)};
    const std::vector<std::int64_t> cond = asInts(C);
    broadcastLoop<3>(os, st, [&](std::size_t o, const std::array<std::size_t, 3>& x) {
        if (r.kind == Tensor::Kind::Float) r.f[o] = cond[x[0]] ? X.f[x[1]] : Y.f[x[2]];
        else r.i[o] = cond[x[0]] ? X.i[x[1]] : Y.i[x[2]];
    });
    return {r};
}

Outs clipOp(const Node& n, const Ins& in, int opset) {
    const Tensor& X = need(in, 0);
    float lo = -std::numeric_limits<float>::infinity(), hi = std::numeric_limits<float>::infinity();
    if (opset >= 11) {
        if (opt(in, 1) && in[1]->size()) lo = scalarFloat(*in[1]);
        if (opt(in, 2) && in[2]->size()) hi = scalarFloat(*in[2]);
    } else {
        lo = n.f("min", lo);
        hi = n.f("max", hi);
    }
    if (X.kind == Tensor::Kind::Int) {
        Tensor r = X;
        for (auto& v : r.i) v = std::clamp<std::int64_t>(v, static_cast<std::int64_t>(std::max(lo, -9.2e18f)), static_cast<std::int64_t>(std::min(hi, 9.2e18f)));
        return {r};
    }
    return {unaryFloat(X, [lo, hi](float v) { return std::min(std::max(v, lo), hi); })};
}

Outs triluOp(const Node& n, const Ins& in, int) {
    Tensor r = need(in, 0);
    const std::int64_t k = opt(in, 1) ? asInts(*in[1]).at(0) : 0;
    const bool upper = n.i("upper", 1) != 0;
    const std::int64_t R = r.shape[r.shape.size() - 2], C = r.shape.back();
    const std::size_t mats = r.size() / static_cast<std::size_t>(R * C);
    for (std::size_t m = 0; m < mats; ++m)
        for (std::int64_t i = 0; i < R; ++i)
            for (std::int64_t j = 0; j < C; ++j) {
                const bool keep = upper ? j - i >= k : j - i <= k;
                if (!keep) {
                    const std::size_t at = m * static_cast<std::size_t>(R * C) + static_cast<std::size_t>(i * C + j);
                    if (r.kind == Tensor::Kind::Float) r.f[at] = 0.0f;
                    else r.i[at] = 0;
                }
            }
    return {r};
}

Outs topKOp(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::size_t K = static_cast<std::size_t>(asInts(need(in, 1)).at(0));
    const std::size_t axis = static_cast<std::size_t>(normAxis(n.i("axis", -1), X.shape.size()));
    const bool largest = n.i("largest", 1) != 0;
    const std::size_t len = static_cast<std::size_t>(X.shape[axis]);
    std::size_t inner = 1;
    for (std::size_t k = axis + 1; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    const std::size_t outer = X.size() / std::max<std::size_t>(1, len * inner);
    Shape os = X.shape;
    os[axis] = static_cast<std::int64_t>(K);
    Tensor vals = tensorLike(X, os), idx = intTensor(os);
    const std::vector<float> v = asFloats(X);
    std::vector<std::size_t> order(len);
    for (std::size_t o = 0; o < outer; ++o)
        for (std::size_t i = 0; i < inner; ++i) {
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                const float x = v[(o * len + a) * inner + i], y = v[(o * len + b) * inner + i];
                return largest ? x > y : x < y;
            });
            for (std::size_t q = 0; q < K; ++q) {
                const std::size_t to = (o * K + q) * inner + i;
                copyElem(X, (o * len + order[q]) * inner + i, vals, to);
                idx.i[to] = static_cast<std::int64_t>(order[q]);
            }
        }
    return {vals, idx};
}

Outs nonZeroOp(const Node&, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::vector<float> v = asFloats(X);
    std::vector<std::size_t> hits;
    for (std::size_t k = 0; k < v.size(); ++k)
        if (v[k] != 0.0f) hits.push_back(k);
    const std::size_t rank = X.shape.size();
    Tensor r = intTensor({static_cast<std::int64_t>(rank), static_cast<std::int64_t>(hits.size())});
    for (std::size_t h = 0; h < hits.size(); ++h) {
        std::size_t rem = hits[h];
        for (std::size_t d = rank; d-- > 0;) {
            r.i[d * hits.size() + h] = static_cast<std::int64_t>(rem % static_cast<std::size_t>(X.shape[d]));
            rem /= static_cast<std::size_t>(X.shape[d]);
        }
    }
    return {r};
}


// ---- quantised operators ----
// Integer products run on the float GEMM: exact while the sums stay below 2^24 in magnitude, and
// within float rounding beyond (the results are scaled back to floats anyway).

// Values of an integer tensor minus a zero point (scalar, or one per index along `axis`).
std::vector<float> minusZeroPoint(const Tensor& x, const Tensor* zp, std::size_t axis = 0) {
    std::vector<float> v(x.size());
    std::vector<std::int64_t> z = zp ? asInts(*zp) : std::vector<std::int64_t>{0};
    if (z.empty()) z = {0};
    std::size_t inner = 1;
    for (std::size_t k = axis + 1; k < x.shape.size(); ++k) inner *= static_cast<std::size_t>(x.shape[k]);
    const std::size_t len = x.shape.empty() ? 1 : static_cast<std::size_t>(x.shape[axis]);
    for (std::size_t k = 0; k < v.size(); ++k) {
        const std::int64_t zk = z.size() == 1 ? z[0] : z[(k / inner) % len];
        v[k] = static_cast<float>((x.kind == Tensor::Kind::Int ? x.i[k] : static_cast<std::int64_t>(x.f[k])) - zk);
    }
    return v;
}

// Saturating round-half-to-even into the zero point's type (uint8 by default).
Tensor quantize(const Shape& shape, const std::vector<float>& x, const std::vector<float>& scale, const std::vector<std::int64_t>& zp, int type,
                std::size_t axis, std::size_t blockSize = 0) {
    Tensor r = intTensor(shape, type);
    const std::int64_t lo = type == 3 ? -128 : type == 5 ? -32768 : type == 6 ? std::numeric_limits<std::int32_t>::min() : 0;
    const std::int64_t hi = type == 3 ? 127 : type == 4 ? 65535 : type == 5 ? 32767 : type == 6 ? std::numeric_limits<std::int32_t>::max() : 255;
    std::size_t inner = 1;
    for (std::size_t k = axis + 1; k < shape.size(); ++k) inner *= static_cast<std::size_t>(shape[k]);
    const std::size_t len = shape.empty() ? 1 : static_cast<std::size_t>(shape[axis]);
    for (std::size_t k = 0; k < x.size(); ++k) {
        std::size_t q = 0;
        if (scale.size() > 1) {
            const std::size_t a = (k / inner) % len;
            q = blockSize ? ((k / (inner * len)) * ((len + blockSize - 1) / blockSize) + a / blockSize) * inner + k % inner : a;
        }
        const float s = scale[std::min(q, scale.size() - 1)];
        const std::int64_t z = zp.empty() ? 0 : zp[std::min(q, zp.size() - 1)];
        // In float, as onnxruntime does: x / scale rounded half to even, then the zero point.
        const float v = std::nearbyint(x[k] / s) + static_cast<float>(z);
        r.i[k] = static_cast<std::int64_t>(std::clamp(static_cast<double>(v), static_cast<double>(lo), static_cast<double>(hi)));
    }
    return r;
}

Outs dequantizeLinear(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const std::vector<float> scale = asFloats(need(in, 1));
    const Tensor* zp = opt(in, 2);
    const std::size_t axis = X.shape.empty() ? 0 : static_cast<std::size_t>(normAxis(n.i("axis", 1), X.shape.size()));
    const std::size_t blockSize = static_cast<std::size_t>(n.i("block_size", 0));
    const std::vector<std::int64_t> z = zp ? asInts(*zp) : std::vector<std::int64_t>{};
    Tensor r = floatTensor(X.shape);
    std::size_t inner = 1;
    for (std::size_t k = axis + 1; k < X.shape.size(); ++k) inner *= static_cast<std::size_t>(X.shape[k]);
    const std::size_t len = X.shape.empty() ? 1 : static_cast<std::size_t>(X.shape[axis]);
    const std::vector<float> xv = asFloats(X);
    for (std::size_t k = 0; k < xv.size(); ++k) {
        std::size_t q = 0;
        if (scale.size() > 1) {
            const std::size_t a = (k / inner) % len;
            q = blockSize ? ((k / (inner * len)) * ((len + blockSize - 1) / blockSize) + a / blockSize) * inner + k % inner : a;
        }
        const float zk = z.empty() ? 0.0f : static_cast<float>(z[std::min(q, z.size() - 1)]);
        r.f[k] = (xv[k] - zk) * scale[std::min(q, scale.size() - 1)];
    }
    return {r};
}

Outs quantizeLinear(const Node& n, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    const Tensor* zp = opt(in, 2);
    const int type = zp ? zp->elemType : static_cast<int>(n.i("output_dtype", 2));
    const std::size_t axis = X.shape.empty() ? 0 : static_cast<std::size_t>(normAxis(n.i("axis", 1), X.shape.size()));
    return {quantize(X.shape, asFloats(X), asFloats(need(in, 1)), zp ? asInts(*zp) : std::vector<std::int64_t>{}, type, axis,
                     static_cast<std::size_t>(n.i("block_size", 0)))};
}

Outs dynamicQuantizeLinear(const Node&, const Ins& in, int) {
    const Tensor& X = need(in, 0);
    float lo = 0.0f, hi = 0.0f;
    for (float v : X.f) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const float scale = (hi - lo) / 255.0f;
    const float s = scale == 0.0f ? 1.0f : scale;
    const std::int64_t zp = static_cast<std::int64_t>(std::clamp(std::nearbyint(0.0f - lo / s), 0.0f, 255.0f));
    Tensor y = quantize(X.shape, X.f, {s}, {zp}, 2, 0);
    return {y, Tensor::floats({}, {scale}), Tensor::ints({}, {zp}, 2)};
}

// A float tensor of the same shape holding x - zero point (per column of B for MatMulInteger).
Tensor shifted(const Tensor& x, const Tensor* zp, bool perLastAxis) {
    Tensor t = floatTensor(x.shape);
    std::vector<std::int64_t> z = zp ? asInts(*zp) : std::vector<std::int64_t>{0};
    if (z.empty()) z = {0};
    const std::size_t last = x.shape.empty() ? 1 : static_cast<std::size_t>(x.shape.back());
    for (std::size_t k = 0; k < x.size(); ++k) {
        const std::int64_t zk = z.size() == 1 ? z[0] : (perLastAxis ? z[k % last] : z[0]);
        t.f[k] = static_cast<float>(x.i[k] - zk);
    }
    return t;
}

Tensor toInt32(const Tensor& f) {
    Tensor r = intTensor(f.shape, 6);
    for (std::size_t k = 0; k < f.f.size(); ++k) r.i[k] = static_cast<std::int64_t>(std::llround(f.f[k]));
    return r;
}

Outs matMulInteger(const Node&, const Ins& in, int) {
    const Tensor a = shifted(need(in, 0), opt(in, 2), false), b = shifted(need(in, 1), opt(in, 3), true);
    return {toInt32(matMul(a, b))};
}

Outs convInteger(const Node& n, const Ins& in, int o) {
    const Tensor& W = need(in, 1);
    Tensor x = shifted(need(in, 0), opt(in, 2), false);
    // w zero point: scalar or one per output channel
    Tensor w = floatTensor(W.shape);
    std::vector<std::int64_t> z = opt(in, 3) ? asInts(*in[3]) : std::vector<std::int64_t>{0};
    if (z.empty()) z = {0};
    const std::size_t per = W.size() / static_cast<std::size_t>(W.shape[0]);
    for (std::size_t k = 0; k < W.size(); ++k) w.f[k] = static_cast<float>(W.i[k] - (z.size() == 1 ? z[0] : z[k / per]));
    return {toInt32(convOp(n, {&x, &w}, o)[0])};
}

Outs qlinearMatMul(const Node&, const Ins& in, int) {
    const Tensor& A = need(in, 0);
    const Tensor& B = need(in, 3);
    Tensor a = floatTensor(A.shape), b = floatTensor(B.shape);
    a.f = minusZeroPoint(A, opt(in, 2), 0);
    const float as = scalarFloat(need(in, 1));
    for (float& v : a.f) v *= as;
    // b: scale / zero point per tensor or per output column (the last axis)
    const std::vector<float> bs = asFloats(need(in, 4));
    std::vector<std::int64_t> bz = opt(in, 5) ? asInts(*in[5]) : std::vector<std::int64_t>{0};
    const std::size_t Ncol = static_cast<std::size_t>(B.shape.back());
    for (std::size_t k = 0; k < B.size(); ++k) {
        const std::size_t c = k % Ncol;
        b.f[k] = static_cast<float>(B.i[k] - bz[bz.size() == 1 ? 0 : c]) * bs[bs.size() == 1 ? 0 : c];
    }
    const Tensor y = matMul(a, b);
    const Tensor* yz = opt(in, 7);
    return {quantize(y.shape, y.f, asFloats(need(in, 6)), yz ? asInts(*yz) : std::vector<std::int64_t>{0}, yz ? yz->elemType : 2, 0)};
}

Outs qlinearConv(const Node& n, const Ins& in, int o) {
    const Tensor& X = need(in, 0);
    const Tensor& W = need(in, 3);
    Tensor x = floatTensor(X.shape), w = floatTensor(W.shape);
    x.f = minusZeroPoint(X, opt(in, 2), 0);
    const float xs = scalarFloat(need(in, 1));
    for (float& v : x.f) v *= xs;
    const std::vector<float> ws = asFloats(need(in, 4));
    std::vector<std::int64_t> wz = opt(in, 5) ? asInts(*in[5]) : std::vector<std::int64_t>{0};
    const std::size_t per = W.size() / static_cast<std::size_t>(W.shape[0]);
    for (std::size_t k = 0; k < W.size(); ++k) {
        const std::size_t m = k / per;
        w.f[k] = static_cast<float>(W.i[k] - wz[wz.size() == 1 ? 0 : m]) * ws[ws.size() == 1 ? 0 : m];
    }
    Tensor bias;
    Ins cin{&x, &w};
    if (const Tensor* B = opt(in, 8)) {  // int32 bias in units of xs * ws
        bias = floatTensor(B->shape);
        for (std::size_t m = 0; m < B->size(); ++m) bias.f[m] = static_cast<float>(B->i[m]) * xs * ws[ws.size() == 1 ? 0 : m];
        cin.push_back(&bias);
    }
    const Tensor y = convOp(n, cin, o)[0];
    const Tensor* yz = opt(in, 7);
    return {quantize(y.shape, y.f, asFloats(need(in, 6)), yz ? asInts(*yz) : std::vector<std::int64_t>{0}, yz ? yz->elemType : 2, 0)};
}

// com.microsoft MatMulNBits: B packed [N][blocks][block_size * bits / 8] (low bits first), one scale
// per (column, block), optional packed or float zero points (default 2^(bits-1)), optional g_idx
// and bias. Y = A x W with W (K x N) dequantised.
Outs matMulNBits(const Node& n, const Ins& in, int) {
    const Tensor& A = need(in, 0);
    const Tensor& B = need(in, 1);
    const std::size_t K = static_cast<std::size_t>(n.i("K", 0)), N = static_cast<std::size_t>(n.i("N", 0));
    const std::size_t bits = static_cast<std::size_t>(n.i("bits", 4)), bs = static_cast<std::size_t>(n.i("block_size", 32));
    if (bits != 4 && bits != 8 && bits != 2) throw std::runtime_error("MatMulNBits: " + std::to_string(bits) + " bits");
    const std::size_t blocks = (K + bs - 1) / bs, blob = bs * bits / 8;
    const std::vector<float> scales = asFloats(need(in, 2));
    const Tensor* zp = opt(in, 3);
    const Tensor* gidx = opt(in, 4);
    const Tensor* bias = opt(in, 5);
    const std::vector<std::int64_t> packed = asInts(B);
    std::vector<std::int64_t> zpPacked;
    std::vector<float> zpFloat;
    if (zp) {
        if (zp->kind == Tensor::Kind::Int) zpPacked = zp->i;
        else zpFloat = zp->f;
    }
    const std::vector<std::int64_t> g = gidx ? asInts(*gidx) : std::vector<std::int64_t>{};
    const std::size_t zpBytesPerCol = (blocks * bits + 7) / 8, perByte = 8 / bits, mask = (1u << bits) - 1;
    const float defZp = static_cast<float>(1u << (bits - 1));
    Tensor W = floatTensor({static_cast<std::int64_t>(K), static_cast<std::int64_t>(N)});
    parallelFor(0, N, 16, [&](std::size_t n0, std::size_t n1) {
        for (std::size_t col = n0; col < n1; ++col)
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t blk = g.empty() ? k / bs : static_cast<std::size_t>(g[k]);
                const std::size_t j = k % bs;
                const std::size_t byte = (col * blocks + k / bs) * blob + j / perByte;
                const std::uint32_t q = (static_cast<std::uint32_t>(packed[byte]) >> ((j % perByte) * bits)) & mask;
                float z = defZp;
                if (!zpPacked.empty()) z = static_cast<float>((static_cast<std::uint32_t>(zpPacked[col * zpBytesPerCol + blk / perByte]) >> ((blk % perByte) * bits)) & mask);
                else if (!zpFloat.empty()) z = zpFloat[col * blocks + blk];
                W.f[k * N + col] = (static_cast<float>(q) - z) * scales[col * blocks + blk];
            }
    });
    Tensor y = matMul(A, W);
    if (bias)
        for (std::size_t k = 0; k < y.f.size(); ++k) y.f[k] += bias->f[k % N];
    return {y};
}

// com.microsoft MatMulBnb4: bitsandbytes 4-bit (FP4 or NF4) blocks of W^T (N x K, flattened), two
// codes per byte (high nibble first), one absmax per block.
Outs matMulBnb4(const Node& n, const Ins& in, int) {
    static const float nf4[16] = {-1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f, -0.28444138169288635f,
                                  -0.18477343022823334f, -0.09105003625154495f, 0.0f, 0.07958029955625534f, 0.16093020141124725f,
                                  0.24611230194568634f, 0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f};
    static const float fp4[16] = {0.0f, 5.208333333e-03f, 0.66666667f, 1.0f, 0.33333333f, 0.5f, 0.16666667f, 0.25f,
                                  -0.0f, -5.208333333e-03f, -0.66666667f, -1.0f, -0.33333333f, -0.5f, -0.16666667f, -0.25f};
    const Tensor& A = need(in, 0);
    const std::vector<std::int64_t> packed = asInts(need(in, 1));
    const std::vector<float> absmax = asFloats(need(in, 2));
    const std::size_t K = static_cast<std::size_t>(n.i("K", 0)), N = static_cast<std::size_t>(n.i("N", 0)), bs = static_cast<std::size_t>(n.i("block_size", 64));
    const float* lut = n.i("quant_type", 1) == 0 ? fp4 : nf4;
    Tensor W = floatTensor({static_cast<std::int64_t>(K), static_cast<std::int64_t>(N)});
    for (std::size_t e = 0; e < N * K; ++e) {
        const std::uint32_t b = static_cast<std::uint32_t>(packed[e / 2]);
        const std::uint32_t code = e % 2 == 0 ? (b >> 4) & 15 : b & 15;
        const std::size_t col = e / K, k = e % K;
        W.f[k * N + col] = lut[code] * absmax[e / bs];
    }
    return {matMul(A, W)};
}

std::map<std::string, OpFn> buildTable() {
    std::map<std::string, OpFn> t;
    auto add = [](float a, float b) { return a + b; };
    auto addI = [](std::int64_t a, std::int64_t b) { return a + b; };
    t["Add"] = [=](const Node&, const Ins& in, int) -> Outs { return {binary(need(in, 0), need(in, 1), add, addI)}; };
    t["Sub"] = [](const Node&, const Ins& in, int) -> Outs { return {binary(need(in, 0), need(in, 1), [](float a, float b) { return a - b; }, [](std::int64_t a, std::int64_t b) { return a - b; })}; };
    t["Mul"] = [](const Node&, const Ins& in, int) -> Outs { return {binary(need(in, 0), need(in, 1), [](float a, float b) { return a * b; }, [](std::int64_t a, std::int64_t b) { return a * b; })}; };
    t["Div"] = [](const Node&, const Ins& in, int) -> Outs {
        return {binary(need(in, 0), need(in, 1), [](float a, float b) { return a / b; }, [](std::int64_t a, std::int64_t b) {
            if (!b) throw std::runtime_error("integer division by zero");
            return a / b;
        })};
    };
    t["Pow"] = [](const Node&, const Ins& in, int) -> Outs {
        const Tensor& a = need(in, 0);
        Tensor b = need(in, 1);
        if (b.kind != a.kind) b = Tensor::floats(b.shape, asFloats(b));
        return {binary(a, b, [](float x, float y) { return y == 2.0f ? x * x : y == 0.5f ? std::sqrt(x) : std::pow(x, y); },
                       [](std::int64_t x, std::int64_t y) { return static_cast<std::int64_t>(std::pow(static_cast<double>(x), static_cast<double>(y))); })};
    };
    t["Mod"] = [](const Node& n, const Ins& in, int) -> Outs {
        const bool fm = n.i("fmod", 0) != 0;
        return {binary(need(in, 0), need(in, 1), [fm](float a, float b) {
            float r = std::fmod(a, b);
            if (!fm && r != 0 && ((r < 0) != (b < 0))) r += b;
            return r;
        }, [](std::int64_t a, std::int64_t b) {
            std::int64_t r = a % b;
            if (r != 0 && ((r < 0) != (b < 0))) r += b;
            return r;
        })};
    };
    t["Max"] = [](const Node&, const Ins& in, int) -> Outs { return {variadic(in, [](float a, float b) { return std::max(a, b); }, [](std::int64_t a, std::int64_t b) { return std::max(a, b); })}; };
    t["Min"] = [](const Node&, const Ins& in, int) -> Outs { return {variadic(in, [](float a, float b) { return std::min(a, b); }, [](std::int64_t a, std::int64_t b) { return std::min(a, b); })}; };
    t["Sum"] = [=](const Node&, const Ins& in, int) -> Outs { return {variadic(in, add, addI)}; };
    t["Mean"] = [=](const Node&, const Ins& in, int) -> Outs {
        Tensor s = variadic(in, add, addI);
        std::size_t count = 0;
        for (const Tensor* x : in) count += x != nullptr;
        for (float& v : s.f) v /= static_cast<float>(count);
        return {s};
    };
    auto cmp = [](std::function<bool(float, float)> f, std::function<bool(std::int64_t, std::int64_t)> g) -> OpFn {
        return [f, g](const Node&, const Ins& in, int) -> Outs {
            return {binary(need(in, 0), need(in, 1), [f](float a, float b) { return f(a, b) ? 1.0f : 0.0f; },
                           [g](std::int64_t a, std::int64_t b) { return static_cast<std::int64_t>(g(a, b)); }, true)};
        };
    };
    t["Equal"] = cmp([](float a, float b) { return a == b; }, [](std::int64_t a, std::int64_t b) { return a == b; });
    t["Less"] = cmp([](float a, float b) { return a < b; }, [](std::int64_t a, std::int64_t b) { return a < b; });
    t["Greater"] = cmp([](float a, float b) { return a > b; }, [](std::int64_t a, std::int64_t b) { return a > b; });
    t["LessOrEqual"] = cmp([](float a, float b) { return a <= b; }, [](std::int64_t a, std::int64_t b) { return a <= b; });
    t["GreaterOrEqual"] = cmp([](float a, float b) { return a >= b; }, [](std::int64_t a, std::int64_t b) { return a >= b; });
    t["And"] = cmp([](float a, float b) { return a && b; }, [](std::int64_t a, std::int64_t b) { return a && b; });
    t["Or"] = cmp([](float a, float b) { return a || b; }, [](std::int64_t a, std::int64_t b) { return a || b; });
    t["Xor"] = cmp([](float a, float b) { return (a != 0) != (b != 0); }, [](std::int64_t a, std::int64_t b) { return (a != 0) != (b != 0); });
    t["Not"] = [](const Node&, const Ins& in, int) -> Outs {
        Tensor r = need(in, 0);
        for (auto& v : r.i) v = !v;
        return {r};
    };

    addUnary(t, "Abs", [](float x) { return std::fabs(x); });
    addUnary(t, "Neg", [](float x) { return -x; });
    addUnary(t, "Sqrt", [](float x) { return std::sqrt(x); });
    addUnary(t, "Exp", [](float x) { return std::exp(x); });
    addUnary(t, "Log", [](float x) { return std::log(x); });
    addUnary(t, "Tanh", [](float x) { return std::tanh(x); });
    addUnary(t, "Sigmoid", sigmoid);
    addUnary(t, "Relu", [](float x) { return x > 0 ? x : 0.0f; });
    addUnary(t, "Reciprocal", [](float x) { return 1.0f / x; });
    addUnary(t, "Floor", [](float x) { return std::floor(x); });
    addUnary(t, "Ceil", [](float x) { return std::ceil(x); });
    addUnary(t, "Round", [](float x) { return std::nearbyint(x); });
    addUnary(t, "Sin", [](float x) { return std::sin(x); });
    addUnary(t, "Cos", [](float x) { return std::cos(x); });
    addUnary(t, "Sign", [](float x) { return static_cast<float>((x > 0) - (x < 0)); });
    addUnary(t, "Softplus", [](float x) { return x > 20 ? x : std::log1p(std::exp(x)); });
    addUnary(t, "Softsign", [](float x) { return x / (1 + std::fabs(x)); });
    addUnary(t, "Mish", [](float x) { return x * std::tanh(x > 20 ? x : std::log1p(std::exp(x))); });
    addUnary(t, "HardSwish", [](float x) { return x * std::min(std::max(x / 6.0f + 0.5f, 0.0f), 1.0f); });
    addUnary(t, "IsNaN", [](float x) { return std::isnan(x) ? 1.0f : 0.0f; });
    t["IsNaN"] = [](const Node&, const Ins& in, int) -> Outs {
        const Tensor& x = need(in, 0);
        Tensor r = intTensor(x.shape, 9);
        for (std::size_t k = 0; k < x.f.size(); ++k) r.i[k] = std::isnan(x.f[k]);
        return {r};
    };
    t["IsInf"] = [](const Node& n, const Ins& in, int) -> Outs {
        const Tensor& x = need(in, 0);
        const bool pos = n.i("detect_positive", 1) != 0, neg = n.i("detect_negative", 1) != 0;
        Tensor r = intTensor(x.shape, 9);
        for (std::size_t k = 0; k < x.f.size(); ++k) r.i[k] = std::isinf(x.f[k]) && ((x.f[k] > 0 && pos) || (x.f[k] < 0 && neg));
        return {r};
    };
    t["Erf"] = [](const Node&, const Ins& in, int) -> Outs { return {erfTensor(need(in, 0))}; };
    t["Gelu"] = [](const Node& n, const Ins& in, int) -> Outs {
        const Tensor& x = need(in, 0);
        if (n.s("approximate", "none") == "tanh")
            return {unaryFloat(x, [](float v) { return 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v))); })};
        Tensor s = unaryFloat(x, [](float v) { return v * 0.70710678118654752f; });
        Tensor e = erfTensor(s);
        for (std::size_t k = 0; k < e.f.size(); ++k) e.f[k] = 0.5f * x.f[k] * (1.0f + e.f[k]);
        return {e};
    };
    t["LeakyRelu"] = [](const Node& n, const Ins& in, int) -> Outs {
        const float a = n.f("alpha", 0.01f);
        return {unaryFloat(need(in, 0), [a](float x) { return x >= 0 ? x : a * x; })};
    };
    t["Elu"] = [](const Node& n, const Ins& in, int) -> Outs {
        const float a = n.f("alpha", 1.0f);
        return {unaryFloat(need(in, 0), [a](float x) { return x >= 0 ? x : a * (std::exp(x) - 1); })};
    };
    t["Celu"] = [](const Node& n, const Ins& in, int) -> Outs {
        const float a = n.f("alpha", 1.0f);
        return {unaryFloat(need(in, 0), [a](float x) { return std::max(0.0f, x) + std::min(0.0f, a * (std::exp(x / a) - 1)); })};
    };
    t["Selu"] = [](const Node& n, const Ins& in, int) -> Outs {
        const float a = n.f("alpha", 1.67326319217681884765625f), g = n.f("gamma", 1.05070102214813232421875f);
        return {unaryFloat(need(in, 0), [a, g](float x) { return x > 0 ? g * x : g * (a * std::exp(x) - a); })};
    };
    t["HardSigmoid"] = [](const Node& n, const Ins& in, int) -> Outs {
        const float a = n.f("alpha", 0.2f), b = n.f("beta", 0.5f);
        return {unaryFloat(need(in, 0), [a, b](float x) { return std::min(std::max(a * x + b, 0.0f), 1.0f); })};
    };
    t["ThresholdedRelu"] = [](const Node& n, const Ins& in, int) -> Outs {
        const float a = n.f("alpha", 1.0f);
        return {unaryFloat(need(in, 0), [a](float x) { return x > a ? x : 0.0f; })};
    };
    t["PRelu"] = [](const Node&, const Ins& in, int) -> Outs {
        return {binary(need(in, 0), need(in, 1), [](float x, float s) { return x >= 0 ? x : x * s; }, [](std::int64_t x, std::int64_t s) { return x >= 0 ? x : x * s; })};
    };
    t["Identity"] = [](const Node&, const Ins& in, int) -> Outs { return {need(in, 0)}; };
    t["Dropout"] = [](const Node& n, const Ins& in, int) -> Outs {
        Outs o{need(in, 0)};
        if (n.outputs.size() > 1) {
            Tensor mask = intTensor(in[0]->shape, 9);
            std::fill(mask.i.begin(), mask.i.end(), 1);
            o.push_back(mask);
        }
        return o;
    };
    t["Clip"] = clipOp;

    t["MatMul"] = [](const Node&, const Ins& in, int) -> Outs { return {matMul(need(in, 0), need(in, 1))}; };
    t["Gemm"] = gemmOp;
    t["RNN"] = [](const Node& n, const Ins& in, int) -> Outs { return recurrent(n, in, 'r'); };
    t["GRU"] = [](const Node& n, const Ins& in, int) -> Outs { return recurrent(n, in, 'g'); };
    t["LSTM"] = [](const Node& n, const Ins& in, int) -> Outs { return recurrent(n, in, 'l'); };
    t["Conv"] = convOp;
    t["ConvTranspose"] = convTransposeOp;
    t["MaxPool"] = [](const Node& n, const Ins& in, int) -> Outs { return poolOp(n, in, true); };
    t["AveragePool"] = [](const Node& n, const Ins& in, int) -> Outs { return poolOp(n, in, false); };
    t["GlobalAveragePool"] = [](const Node&, const Ins& in, int) -> Outs { return globalPool(in, false); };
    t["GlobalMaxPool"] = [](const Node&, const Ins& in, int) -> Outs { return globalPool(in, true); };
    t["BatchNormalization"] = batchNorm;
    t["InstanceNormalization"] = instanceNorm;
    t["LayerNormalization"] = layerNorm;
    t["GroupNormalization"] = groupNorm;
    t["LpNormalization"] = lpNorm;
    t["Softmax"] = [](const Node& n, const Ins& in, int o) -> Outs { return softmaxOp(n, in, o, false); };
    t["LogSoftmax"] = [](const Node& n, const Ins& in, int o) -> Outs { return softmaxOp(n, in, o, true); };

    t["Reshape"] = reshapeOp;
    t["Flatten"] = flattenOp;
    t["Squeeze"] = squeezeOp;
    t["Unsqueeze"] = unsqueezeOp;
    t["Transpose"] = [](const Node& n, const Ins& in, int) -> Outs { return {transpose(need(in, 0), n.ints("perm"))}; };
    t["Concat"] = concatOp;
    t["Split"] = splitOp;
    t["Slice"] = sliceOp;
    t["Gather"] = gatherOp;
    t["GatherElements"] = gatherElementsOp;
    t["ScatterND"] = scatterNdOp;
    t["Expand"] = expandOp;
    t["Tile"] = tileOp;
    t["Pad"] = padOp;
    t["DepthToSpace"] = depthToSpace;
    t["SpaceToDepth"] = spaceToDepth;
    t["Shape"] = [](const Node& n, const Ins& in, int) -> Outs {
        const Shape& s = need(in, 0).shape;
        const std::int64_t r = static_cast<std::int64_t>(s.size());
        std::int64_t b = n.i("start", 0), e = n.i("end", r);
        if (b < 0) b += r;
        if (e < 0) e += r;
        b = std::clamp<std::int64_t>(b, 0, r);
        e = std::clamp<std::int64_t>(e, 0, r);
        std::vector<std::int64_t> v(s.begin() + b, s.begin() + std::max(b, e));
        return {Tensor::ints({static_cast<std::int64_t>(v.size())}, v)};
    };
    t["Size"] = [](const Node&, const Ins& in, int) -> Outs { return {Tensor::ints({}, {static_cast<std::int64_t>(need(in, 0).size())})}; };
    t["Constant"] = constantOp;
    t["ConstantOfShape"] = constantOfShapeOp;
    t["Cast"] = castOp;
    t["CastLike"] = [](const Node&, const Ins& in, int o) -> Outs {
        Node c;
        c.attrs["to"].i = need(in, 1).elemType;
        c.attrs["to"].has = true;
        return castOp(c, {in[0]}, o);
    };
    t["Range"] = rangeOp;
    t["Where"] = whereOp;
    t["Trilu"] = triluOp;
    t["TopK"] = topKOp;
    t["NonZero"] = nonZeroOp;
    t["CumSum"] = cumSumOp;
    t["Einsum"] = einsumOp;
    t["Resize"] = [](const Node& n, const Ins& in, int o) -> Outs { return resizeOp(n, in, o, false); };
    t["Upsample"] = [](const Node& n, const Ins& in, int o) -> Outs { return resizeOp(n, in, o, true); };
    for (const auto& [name, kind] : std::vector<std::pair<std::string, std::string>>{
             {"ReduceMean", "mean"}, {"ReduceSum", "sum"}, {"ReduceMax", "max"}, {"ReduceMin", "min"}, {"ReduceProd", "prod"},
             {"ReduceL1", "l1"}, {"ReduceL2", "l2"}, {"ReduceSumSquare", "sumsquare"}, {"ReduceLogSumExp", "logsumexp"}}) {
        const std::string k = kind;
        t[name] = [k](const Node& n, const Ins& in, int o) -> Outs { return reduceOp(n, in, o, k); };
    }
    t["ArgMax"] = [](const Node& n, const Ins& in, int) -> Outs { return argOp(n, in, true); };
    t["ArgMin"] = [](const Node& n, const Ins& in, int) -> Outs { return argOp(n, in, false); };
    t["DequantizeLinear"] = dequantizeLinear;
    t["QuantizeLinear"] = quantizeLinear;
    t["DynamicQuantizeLinear"] = dynamicQuantizeLinear;
    t["MatMulInteger"] = matMulInteger;
    t["ConvInteger"] = convInteger;
    t["QLinearMatMul"] = qlinearMatMul;
    t["QLinearConv"] = qlinearConv;
    // Operators of other domains are keyed "domain::Op".
    t["com.microsoft::MatMulNBits"] = matMulNBits;
    t["com.microsoft::MatMulBnb4"] = matMulBnb4;
    t["com.microsoft::DequantizeLinear"] = dequantizeLinear;
    t["com.microsoft::QuantizeLinear"] = quantizeLinear;
    return t;
}

} // namespace

const std::map<std::string, OpFn>& operatorTable() {
    static const std::map<std::string, OpFn> table = buildTable();
    return table;
}

} // namespace nc::onnx
