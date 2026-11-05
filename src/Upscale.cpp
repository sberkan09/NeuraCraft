#include "Upscale.h"
#include "Backend.h"
#include "Parallel.h"
#include "SafeTensors.h"

#include <cmath>
#include <stdexcept>

namespace nc {

Upscaler::Upscaler(const std::string& file) {
    const SafeTensors st(file);
    for (std::size_t i = 0;; i += 2) {
        const std::string w = "body." + std::to_string(i) + ".weight";
        if (!st.has(w)) break;
        const auto& info = st.info(w);
        const std::string b = "body." + std::to_string(i) + ".bias";
        if (!st.has(b)) throw std::runtime_error("Upscaler: " + w + " has no bias (SRVGGNetCompact expected)");
        Layer l;
        l.bias = st.load(b);
        const std::size_t out = l.bias.size();
        if ((info.shape.size() == 4 && (info.shape[2] != 3 || info.shape[3] != 3)) || info.elements() % (out * 9))
            throw std::runtime_error("Upscaler: " + w + " is not a 3x3 convolution (SRVGGNetCompact expected)");
        const Matrix wm = st.load(w);
        l.weight = Matrix(out, info.elements() / out, std::vector<float>(wm.data(), wm.data() + wm.size()));
        const std::string p = "body." + std::to_string(i + 1) + ".weight";
        if (st.has(p)) {
            l.prelu = st.load(p);
            if (l.prelu.size() != out && l.prelu.size() != 1) throw std::runtime_error("Upscaler: bad PReLU " + p);
        }
        layers_.push_back(std::move(l));
    }
    if (layers_.size() < 2 || layers_.front().weight.cols() != 27 || !layers_.back().prelu.empty())
        throw std::runtime_error("Upscaler: " + file + " does not look like SRVGGNetCompact weights");
    for (std::size_t i = 1; i < layers_.size(); ++i)
        if (layers_[i].weight.cols() != layers_[i - 1].weight.rows() * 9)
            throw std::runtime_error("Upscaler: layer " + std::to_string(i) + " does not take the previous layer's channels");
    features_ = layers_.front().weight.rows();
    const std::size_t outCh = layers_.back().weight.rows();
    scale_ = static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(outCh / 3))));
    if (scale_ * scale_ * 3 != outCh) throw std::runtime_error("Upscaler: last layer has " + std::to_string(outCh) + " channels, not 3 * scale^2");
}

Image Upscaler::upscaleTile(const Image& im) const {
    const std::size_t W = im.width, H = im.height, P = W * H;
    // Channel-major float input in [0, 1].
    Matrix x(1, 3 * P);
    for (std::size_t i = 0; i < P; ++i)
        for (std::size_t c = 0; c < 3; ++c) x.data()[c * P + i] = im.pixels[i * 3 + c] / 255.0f;
    Matrix h = x;
    std::size_t channels = 3;
    for (const Layer& l : layers_) {
        // 3x3 "same" convolution: im2col per row of output channels via gemm on patch rows.
        const std::size_t out = l.weight.rows(), patch = channels * 9;
        Matrix cols(patch, P);
        parallelFor(0, channels, 1, [&](std::size_t c0, std::size_t c1) {
            for (std::size_t c = c0; c < c1; ++c)
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 3; ++kx) {
                        float* dst = cols.data() + ((c * 9) + static_cast<std::size_t>(ky * 3 + kx)) * P;
                        const float* src = h.data() + c * P;
                        for (std::size_t y = 0; y < H; ++y) {
                            const long sy = static_cast<long>(y) + ky - 1;
                            for (std::size_t xx = 0; xx < W; ++xx) {
                                const long sx = static_cast<long>(xx) + kx - 1;
                                dst[y * W + xx] = sy < 0 || sx < 0 || sy >= static_cast<long>(H) || sx >= static_cast<long>(W)
                                                      ? 0.0f
                                                      : src[static_cast<std::size_t>(sy) * W + static_cast<std::size_t>(sx)];
                            }
                        }
                    }
        });
        Matrix y(1, out * P);
        gemm(false, false, out, P, patch, l.weight.data(), patch, cols.data(), P, y.data(), P, false);
        parallelFor(0, out, 4, [&](std::size_t o0, std::size_t o1) {
            for (std::size_t o = o0; o < o1; ++o) {
                float* row = y.data() + o * P;
                const float b = l.bias.data()[o];
                const float a = l.prelu.empty() ? 1.0f : l.prelu.data()[l.prelu.size() == 1 ? 0 : o];
                for (std::size_t i = 0; i < P; ++i) {
                    const float v = row[i] + b;
                    row[i] = l.prelu.empty() || v >= 0.0f ? v : a * v;
                }
            }
        });
        h = std::move(y);
        channels = out;
    }
    // Pixel shuffle (channel c * s^2 + dy * s + dx -> pixel (y s + dy, x s + dx)) plus the nearest-upscaled input.
    const std::size_t s = scale_, OW = W * s, OH = H * s;
    Image out;
    out.width = OW;
    out.height = OH;
    out.pixels.resize(OW * OH * 3);
    parallelFor(0, H, 8, [&](std::size_t y0, std::size_t y1) {
        for (std::size_t y = y0; y < y1; ++y)
            for (std::size_t xx = 0; xx < W; ++xx)
                for (std::size_t c = 0; c < 3; ++c) {
                    const float base = x.data()[c * P + y * W + xx];
                    for (std::size_t dy = 0; dy < s; ++dy)
                        for (std::size_t dx = 0; dx < s; ++dx) {
                            const float v = h.data()[(c * s * s + dy * s + dx) * P + y * W + xx] + base;
                            out.pixels[((y * s + dy) * OW + xx * s + dx) * 3 + c] =
                                static_cast<std::uint8_t>(std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f));
                        }
                }
    });
    return out;
}

Image Upscaler::upscale(const Image& input, std::size_t tile) const {
    const Image im = toRgb(input);
    if (im.width == 0 || im.height == 0) throw std::invalid_argument("Upscaler::upscale: empty image");
    if (tile == 0 || (im.width <= tile && im.height <= tile)) return upscaleTile(im);
    // Overlapping tiles: each output pixel is taken from the tile in which it lies furthest from an edge.
    const std::size_t pad = layers_.size() + 2, s = scale_;  // the receptive field: tiles join without seams
    Image out;
    out.width = im.width * s;
    out.height = im.height * s;
    out.pixels.resize(out.width * out.height * 3);
    for (std::size_t ty = 0; ty < im.height; ty += tile)
        for (std::size_t tx = 0; tx < im.width; tx += tile) {
            const std::size_t x0 = tx >= pad ? tx - pad : 0, y0 = ty >= pad ? ty - pad : 0;
            const std::size_t x1 = std::min(im.width, tx + tile + pad), y1 = std::min(im.height, ty + tile + pad);
            Image part;
            part.width = x1 - x0;
            part.height = y1 - y0;
            part.pixels.resize(part.width * part.height * 3);
            for (std::size_t y = y0; y < y1; ++y)
                std::copy_n(im.pixels.data() + (y * im.width + x0) * 3, part.width * 3, part.pixels.data() + ((y - y0) * part.width) * 3);
            const Image up = upscaleTile(part);
            const std::size_t cx1 = std::min(im.width, tx + tile), cy1 = std::min(im.height, ty + tile);
            for (std::size_t y = ty * s; y < cy1 * s; ++y)
                std::copy_n(up.pixels.data() + ((y - y0 * s) * up.width + (tx - x0) * s) * 3, (cx1 - tx) * s * 3,
                            out.pixels.data() + (y * out.width + tx * s) * 3);
        }
    return out;
}

} // namespace nc
