#include "Interpolate.h"
#include "Backend.h"
#include "Parallel.h"
#include "SafeTensors.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nc {

namespace {
// A feature map: C planes of H x W floats.
struct Map {
    std::size_t C = 0, H = 0, W = 0;
    std::vector<float> v;
    Map() = default;
    Map(std::size_t c, std::size_t h, std::size_t w) : C(c), H(h), W(w), v(c * h * w, 0.0f) {}
    float* plane(std::size_t c) { return v.data() + c * H * W; }
    const float* plane(std::size_t c) const { return v.data() + c * H * W; }
};

Map concat(std::initializer_list<const Map*> parts) {
    std::size_t C = 0;
    for (const Map* p : parts) C += p->C;
    Map out(C, (*parts.begin())->H, (*parts.begin())->W);
    std::size_t at = 0;
    for (const Map* p : parts) {
        std::copy(p->v.begin(), p->v.end(), out.v.begin() + static_cast<long>(at));
        at += p->v.size();
    }
    return out;
}

void prelu(Map& m, const Matrix& a) {
    if (a.empty()) return;
    parallelFor(0, m.C, 1, [&](std::size_t c0, std::size_t c1) {
        for (std::size_t c = c0; c < c1; ++c) {
            const float s = a.data()[a.size() == 1 ? 0 : c];
            float* p = m.plane(c);
            for (std::size_t i = 0; i < m.H * m.W; ++i) p[i] = p[i] >= 0.0f ? p[i] : s * p[i];
        }
    });
}

// 3x3 convolution, padding 1, stride 1 or 2 (im2col + GEMM), then PReLU.
Map conv(const Map& x, const FrameInterpolator::Conv& c, std::size_t stride) {
    const std::size_t OH = (x.H + 2 - 3) / stride + 1, OW = (x.W + 2 - 3) / stride + 1, P = OH * OW, patch = x.C * 9;
    Matrix cols(patch, P);
    parallelFor(0, x.C, 1, [&](std::size_t c0, std::size_t c1) {
        for (std::size_t ch = c0; ch < c1; ++ch)
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    float* dst = cols.data() + (ch * 9 + static_cast<std::size_t>(ky * 3 + kx)) * P;
                    const float* src = x.plane(ch);
                    for (std::size_t oy = 0; oy < OH; ++oy) {
                        const long sy = static_cast<long>(oy * stride) + ky - 1;
                        for (std::size_t ox = 0; ox < OW; ++ox) {
                            const long sx = static_cast<long>(ox * stride) + kx - 1;
                            dst[oy * OW + ox] = sy < 0 || sx < 0 || sy >= static_cast<long>(x.H) || sx >= static_cast<long>(x.W)
                                                    ? 0.0f
                                                    : src[static_cast<std::size_t>(sy) * x.W + static_cast<std::size_t>(sx)];
                        }
                    }
                }
    });
    Map y(c.out, OH, OW);
    gemm(false, false, c.out, P, patch, c.weight.data(), patch, cols.data(), P, y.v.data(), P, false);
    for (std::size_t o = 0; o < c.out; ++o) {
        float* p = y.plane(o);
        const float b = c.bias.data()[o];
        for (std::size_t i = 0; i < P; ++i) p[i] += b;
    }
    prelu(y, c.prelu);
    return y;
}

// 4x4 transposed convolution, stride 2, padding 1 (doubles the size), then PReLU when it has one.
Map convTranspose(const Map& x, const FrameInterpolator::Conv& c) {
    const std::size_t P = x.H * x.W, OH = 2 * x.H, OW = 2 * x.W, kk = 16;
    // cols (out*16 x P) = W^T (out*16 x in) * X (in x P); W stored in x (out*16).
    Matrix cols(c.out * kk, P);
    gemm(true, false, c.out * kk, P, c.in, c.weight.data(), c.out * kk, x.v.data(), P, cols.data(), P, false);
    Map y(c.out, OH, OW);
    parallelFor(0, c.out, 1, [&](std::size_t o0, std::size_t o1) {
        for (std::size_t o = o0; o < o1; ++o) {
            float* dst = y.plane(o);
            for (std::size_t ky = 0; ky < 4; ++ky)
                for (std::size_t kx = 0; kx < 4; ++kx) {
                    const float* src = cols.data() + (o * kk + ky * 4 + kx) * P;
                    for (std::size_t iy = 0; iy < x.H; ++iy) {
                        const long oy = static_cast<long>(2 * iy + ky) - 1;
                        if (oy < 0 || oy >= static_cast<long>(OH)) continue;
                        for (std::size_t ix = 0; ix < x.W; ++ix) {
                            const long ox = static_cast<long>(2 * ix + kx) - 1;
                            if (ox < 0 || ox >= static_cast<long>(OW)) continue;
                            dst[static_cast<std::size_t>(oy) * OW + static_cast<std::size_t>(ox)] += src[iy * x.W + ix];
                        }
                    }
                }
            const float b = c.bias.data()[o];
            for (std::size_t i = 0; i < OH * OW; ++i) dst[i] += b;
        }
    });
    prelu(y, c.prelu);
    return y;
}

// Bilinear resize (PyTorch interpolate, align_corners = false, no antialiasing), values times `mul`.
Map resize(const Map& x, std::size_t H, std::size_t W, float mul = 1.0f) {
    if (H == x.H && W == x.W && mul == 1.0f) return x;
    Map y(x.C, H, W);
    const double sy = static_cast<double>(x.H) / H, sx = static_cast<double>(x.W) / W;
    parallelFor(0, H, 8, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const double fy = std::max(0.0, (r + 0.5) * sy - 0.5);
            const std::size_t y0 = std::min(static_cast<std::size_t>(fy), x.H - 1), y1 = std::min(y0 + 1, x.H - 1);
            const float ay = static_cast<float>(fy - y0);
            for (std::size_t q = 0; q < W; ++q) {
                const double fx = std::max(0.0, (q + 0.5) * sx - 0.5);
                const std::size_t x0 = std::min(static_cast<std::size_t>(fx), x.W - 1), x1 = std::min(x0 + 1, x.W - 1);
                const float ax = static_cast<float>(fx - x0);
                for (std::size_t c = 0; c < x.C; ++c) {
                    const float* p = x.plane(c);
                    const float v = (p[y0 * x.W + x0] * (1 - ax) + p[y0 * x.W + x1] * ax) * (1 - ay) +
                                    (p[y1 * x.W + x0] * (1 - ax) + p[y1 * x.W + x1] * ax) * ay;
                    y.plane(c)[r * W + q] = v * mul;
                }
            }
        }
    });
    return y;
}

// Backward warp: out(x, y) = img(x + fx, y + fy), bilinear, border clamped (grid_sample, align_corners = true).
Map warp(const Map& img, const Map& flow, std::size_t fc) {
    Map y(img.C, img.H, img.W);
    const float* fxp = flow.plane(fc);
    const float* fyp = flow.plane(fc + 1);
    parallelFor(0, img.H, 8, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r)
            for (std::size_t q = 0; q < img.W; ++q) {
                const std::size_t i = r * img.W + q;
                const float sx = std::min(std::max(static_cast<float>(q) + fxp[i], 0.0f), static_cast<float>(img.W - 1));
                const float sy = std::min(std::max(static_cast<float>(r) + fyp[i], 0.0f), static_cast<float>(img.H - 1));
                const std::size_t x0 = static_cast<std::size_t>(sx), y0 = static_cast<std::size_t>(sy);
                const std::size_t x1 = std::min(x0 + 1, img.W - 1), y1 = std::min(y0 + 1, img.H - 1);
                const float ax = sx - x0, ay = sy - y0;
                for (std::size_t c = 0; c < img.C; ++c) {
                    const float* p = img.plane(c);
                    y.plane(c)[i] = (p[y0 * img.W + x0] * (1 - ax) + p[y0 * img.W + x1] * ax) * (1 - ay) +
                                    (p[y1 * img.W + x0] * (1 - ax) + p[y1 * img.W + x1] * ax) * ay;
                }
            }
    });
    return y;
}

// IFBlock: flow (4 channels) and mask (1) corrections at 1/scale of the size.
void ifBlock(const FrameInterpolator::Block& b, const Map& x, const Map& flow, std::size_t scale, Map& flowOut, Map& maskOut) {
    const std::size_t H = x.H / scale, W = x.W / scale;
    const Map xs = resize(x, H, W), fs = resize(flow, H, W, 1.0f / static_cast<float>(scale));
    Map feat = conv(conv(concat({&xs, &fs}), b.c0a, 2), b.c0b, 2);
    for (const auto& pair : b.blocks) {
        Map r = conv(conv(feat, pair[0], 1), pair[1], 1);
        for (std::size_t i = 0; i < r.v.size(); ++i) r.v[i] += feat.v[i];
        feat = std::move(r);
    }
    const Map f = convTranspose(convTranspose(feat, b.up1a), b.up1b);
    const Map m = convTranspose(convTranspose(feat, b.up2a), b.up2b);
    flowOut = resize(f, x.H, x.W, static_cast<float>(scale));
    maskOut = resize(m, x.H, x.W);
}

Map toMap(const Image& im, std::size_t H, std::size_t W) {  // [0, 1], replicate-padded to H x W
    Map m(3, H, W);
    for (std::size_t y = 0; y < H; ++y)
        for (std::size_t x = 0; x < W; ++x) {
            const std::uint8_t* p = im.pixels.data() + (std::min(y, im.height - 1) * im.width + std::min(x, im.width - 1)) * 3;
            for (std::size_t c = 0; c < 3; ++c) m.plane(c)[y * W + x] = p[c] / 255.0f;
        }
    return m;
}
} // namespace

FrameInterpolator::FrameInterpolator(const std::string& file) {
    const SafeTensors st(file);
    auto load = [&](const std::string& p, bool transposed, bool prelu, Conv& c) {
        const auto& info = st.info(p + ".weight");
        c.bias = st.load(p + ".bias");
        const Matrix w = st.load(p + ".weight");
        if (info.shape.size() != 4) throw std::runtime_error("FrameInterpolator: " + p + " is not a convolution");
        c.k = info.shape[2];
        c.in = transposed ? info.shape[0] : info.shape[1];
        c.out = transposed ? info.shape[1] : info.shape[0];
        c.weight = transposed ? Matrix(c.in, c.out * c.k * c.k, std::vector<float>(w.data(), w.data() + w.size()))
                              : Matrix(c.out, c.in * c.k * c.k, std::vector<float>(w.data(), w.data() + w.size()));
        if (prelu) c.prelu = st.load(p.substr(0, p.size() - 1) + "1.weight");
    };
    for (const char* name : {"block0", "block1", "block2"}) {
        const std::string b = name;
        if (!st.has(b + ".conv0.0.0.weight")) throw std::runtime_error("FrameInterpolator: " + file + " has no " + b + " (RIFE IFNet expected)");
        Block k;
        load(b + ".conv0.0.0", false, true, k.c0a);
        load(b + ".conv0.1.0", false, true, k.c0b);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j) load(b + ".convblock" + std::to_string(i) + "." + std::to_string(j) + ".0", false, true, k.blocks[i][j]);
        load(b + ".conv1.0", true, false, k.up1a);
        k.up1a.prelu = st.load(b + ".conv1.1.weight");
        load(b + ".conv1.2", true, false, k.up1b);
        load(b + ".conv2.0", true, false, k.up2a);
        k.up2a.prelu = st.load(b + ".conv2.1.weight");
        load(b + ".conv2.2", true, false, k.up2b);
        if (k.c0a.in != 11 || k.up1b.out != 4 || k.up2b.out != 1 || k.up1a.k != 4)
            throw std::runtime_error("FrameInterpolator: unexpected layer shapes in " + file);
        blocks_.push_back(std::move(k));
    }
}

Image FrameInterpolator::middle(const Image& a0, const Image& b0) const {
    const Image a = toRgb(a0), b = toRgb(b0);
    if (a.width != b.width || a.height != b.height) throw std::invalid_argument("FrameInterpolator: frames differ in size");
    const std::size_t H = (a.height + 31) / 32 * 32, W = (a.width + 31) / 32 * 32;
    const Map img0 = toMap(a, H, W), img1 = toMap(b, H, W);
    Map flow(4, H, W), mask(1, H, W), w0 = img0, w1 = img1;
    const std::size_t scales[3] = {4, 2, 1};
    for (std::size_t i = 0; i < 3; ++i) {
        Map negMask = mask;
        for (float& v : negMask.v) v = -v;
        Map swapped(4, H, W);
        std::copy(flow.plane(2), flow.plane(2) + 2 * H * W, swapped.plane(0));
        std::copy(flow.plane(0), flow.plane(0) + 2 * H * W, swapped.plane(2));
        Map f0, m0, f1, m1;
        ifBlock(blocks_[i], concat({&w0, &w1, &mask}), flow, scales[i], f0, m0);
        ifBlock(blocks_[i], concat({&w1, &w0, &negMask}), swapped, scales[i], f1, m1);
        for (std::size_t p = 0; p < H * W; ++p) {
            flow.plane(0)[p] += (f0.plane(0)[p] + f1.plane(2)[p]) / 2;
            flow.plane(1)[p] += (f0.plane(1)[p] + f1.plane(3)[p]) / 2;
            flow.plane(2)[p] += (f0.plane(2)[p] + f1.plane(0)[p]) / 2;
            flow.plane(3)[p] += (f0.plane(3)[p] + f1.plane(1)[p]) / 2;
            mask.v[p] += (m0.v[p] - m1.v[p]) / 2;
        }
        w0 = warp(img0, flow, 0);
        w1 = warp(img1, flow, 2);
    }
    Image out;
    out.width = a.width;
    out.height = a.height;
    out.pixels.resize(a.width * a.height * 3);
    for (std::size_t y = 0; y < a.height; ++y)
        for (std::size_t x = 0; x < a.width; ++x) {
            const std::size_t i = y * W + x;
            const float m = 1.0f / (1.0f + std::exp(-mask.v[i]));
            for (std::size_t c = 0; c < 3; ++c) {
                const float v = w0.plane(c)[i] * m + w1.plane(c)[i] * (1 - m);
                out.pixels[(y * a.width + x) * 3 + c] = static_cast<std::uint8_t>(std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f));
            }
        }
    return out;
}

std::vector<Image> FrameInterpolator::interpolate(const std::vector<Image>& frames, std::size_t passes, bool loop) const {
    std::vector<Image> cur = frames;
    for (std::size_t pass = 0; pass < passes && cur.size() >= 2; ++pass) {
        std::vector<Image> next;
        for (std::size_t i = 0; i + 1 < cur.size(); ++i) {
            next.push_back(cur[i]);
            next.push_back(middle(cur[i], cur[i + 1]));
        }
        next.push_back(cur.back());
        if (loop) next.push_back(middle(cur.back(), cur.front()));
        cur = std::move(next);
    }
    return cur;
}

} // namespace nc
