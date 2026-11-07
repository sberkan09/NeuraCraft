#include "Stereo.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace nc {

StereoLayout stereoLayout(const std::string& name) {
    if (name == "sbs") return StereoLayout::SideBySide;
    if (name == "half_sbs") return StereoLayout::HalfSideBySide;
    if (name == "tb") return StereoLayout::TopBottom;
    if (name == "anaglyph") return StereoLayout::Anaglyph;
    throw std::invalid_argument("stereo: unknown layout \"" + name + "\" (sbs, half_sbs, tb, anaglyph)");
}

std::string stereoLayoutName(StereoLayout layout) {
    switch (layout) {
        case StereoLayout::SideBySide: return "sbs";
        case StereoLayout::HalfSideBySide: return "half_sbs";
        case StereoLayout::TopBottom: return "tb";
        case StereoLayout::Anaglyph: return "anaglyph";
    }
    return "sbs";
}

namespace {

// One eye: pixels move along their row by `shift` * (nearness - focus).
Image eyeView(const Image& im, const std::vector<float>& near, double shift, float focus) {
    const std::size_t W = im.width, H = im.height;
    Image out;
    out.width = W;
    out.height = H;
    out.pixels.assign(W * H * 3, 0);
    std::vector<float> z(W);
    for (std::size_t y = 0; y < H; ++y) {
        std::fill(z.begin(), z.end(), -1.0f);
        const std::uint8_t* src = im.pixels.data() + y * W * 3;
        std::uint8_t* dst = out.pixels.data() + y * W * 3;
        for (std::size_t x = 0; x < W; ++x) {
            const float n = near[y * W + x];
            const double tx = static_cast<double>(x) + shift * (n - focus);
            const long ix = static_cast<long>(std::floor(tx));
            const int extra = tx - static_cast<double>(ix) > 1e-6 ? 1 : 0;  // cover both neighbours against cracks
            for (int k = 0; k <= extra; ++k) {
                const long px = ix + k;
                if (px < 0 || px >= static_cast<long>(W)) continue;
                const auto j = static_cast<std::size_t>(px);
                if (n > z[j]) {
                    z[j] = n;
                    std::copy_n(src + x * 3, 3, dst + j * 3);
                }
            }
        }
        // Uncovered pixels (behind a near edge): the farther of the nearest filled pixels on each side.
        for (std::size_t x = 0; x < W; ++x) {
            if (z[x] >= 0.0f) continue;
            long l = static_cast<long>(x) - 1, r = static_cast<long>(x) + 1;
            while (l >= 0 && z[static_cast<std::size_t>(l)] < 0.0f) --l;
            while (r < static_cast<long>(W) && z[static_cast<std::size_t>(r)] < 0.0f) ++r;
            long pick;
            if (l >= 0 && r < static_cast<long>(W)) pick = z[static_cast<std::size_t>(l)] <= z[static_cast<std::size_t>(r)] ? l : r;
            else pick = l >= 0 ? l : r;
            if (pick >= 0 && pick < static_cast<long>(W)) std::copy_n(dst + static_cast<std::size_t>(pick) * 3, 3, dst + x * 3);
        }
    }
    return out;
}

// Halves the width by averaging pixel pairs (the last column alone when the width is odd).
Image halfWidth(const Image& im) {
    Image out;
    out.width = (im.width + 1) / 2;
    out.height = im.height;
    out.pixels.resize(out.width * out.height * 3);
    for (std::size_t y = 0; y < im.height; ++y)
        for (std::size_t x = 0; x < out.width; ++x) {
            const std::size_t a = (y * im.width + 2 * x) * 3, b = 2 * x + 1 < im.width ? a + 3 : a;
            for (int c = 0; c < 3; ++c)
                out.pixels[(y * out.width + x) * 3 + c] = static_cast<std::uint8_t>((im.pixels[a + c] + im.pixels[b + c] + 1) / 2);
        }
    return out;
}

const std::array<float, 256>& linearTable() {
    static const std::array<float, 256> t = [] {
        std::array<float, 256> v{};
        for (int i = 0; i < 256; ++i) {
            const double c = i / 255.0;
            v[i] = static_cast<float>(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4));
        }
        return v;
    }();
    return t;
}

std::uint8_t toSrgb(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    const double c = v <= 0.0031308 ? v * 12.92 : 1.055 * std::pow(v, 1 / 2.4) - 0.055;
    return static_cast<std::uint8_t>(std::lround(c * 255.0));
}

} // namespace

std::pair<Image, Image> stereoViews(const Image& picture, const std::vector<float>& nearness, const StereoOptions& o) {
    const Image im = toRgb(picture);
    if (nearness.size() != im.width * im.height) throw std::invalid_argument("stereoViews: nearness must have one value per pixel");
    if (!(o.strength >= 0.0f && o.strength <= 1.0f)) throw std::invalid_argument("stereoViews: strength must be in 0..1");
    std::vector<float> near(nearness.size());
    for (std::size_t i = 0; i < near.size(); ++i) near[i] = std::isfinite(nearness[i]) ? std::clamp(nearness[i], 0.0f, 1.0f) : 0.0f;
    const double half = 0.5 * o.strength * static_cast<double>(im.width);
    return {eyeView(im, near, half, o.focus), eyeView(im, near, -half, o.focus)};
}

Image stereoFrame(const Image& leftIn, const Image& rightIn, StereoLayout layout) {
    const Image left = toRgb(leftIn), right = toRgb(rightIn);
    if (left.width != right.width || left.height != right.height) throw std::invalid_argument("stereoFrame: the views differ in size");
    const std::size_t W = left.width, H = left.height;
    Image out;
    if (layout == StereoLayout::Anaglyph) {
        // Dubois' least-squares red-cyan matrices, on linear light.
        static const float L[9] = {0.456f, 0.500f, 0.176f, -0.040f, -0.038f, -0.016f, -0.015f, -0.021f, -0.005f};
        static const float R[9] = {-0.043f, -0.088f, -0.002f, 0.378f, 0.734f, -0.018f, -0.072f, -0.113f, 1.226f};
        const auto& lin = linearTable();
        out.width = W;
        out.height = H;
        out.pixels.resize(W * H * 3);
        for (std::size_t i = 0; i < W * H; ++i) {
            const float l[3] = {lin[left.pixels[i * 3]], lin[left.pixels[i * 3 + 1]], lin[left.pixels[i * 3 + 2]]};
            const float r[3] = {lin[right.pixels[i * 3]], lin[right.pixels[i * 3 + 1]], lin[right.pixels[i * 3 + 2]]};
            for (int c = 0; c < 3; ++c)
                out.pixels[i * 3 + c] = toSrgb(L[c * 3] * l[0] + L[c * 3 + 1] * l[1] + L[c * 3 + 2] * l[2] +
                                               R[c * 3] * r[0] + R[c * 3 + 1] * r[1] + R[c * 3 + 2] * r[2]);
        }
        return out;
    }
    if (layout == StereoLayout::TopBottom) {
        out.width = W;
        out.height = 2 * H;
        out.pixels = left.pixels;
        out.pixels.insert(out.pixels.end(), right.pixels.begin(), right.pixels.end());
        return out;
    }
    const Image a = layout == StereoLayout::HalfSideBySide ? halfWidth(left) : left;
    const Image b = layout == StereoLayout::HalfSideBySide ? halfWidth(right) : right;
    out.width = a.width + b.width;
    out.height = H;
    out.pixels.resize(out.width * H * 3);
    for (std::size_t y = 0; y < H; ++y) {
        std::copy_n(a.pixels.data() + y * a.width * 3, a.width * 3, out.pixels.data() + y * out.width * 3);
        std::copy_n(b.pixels.data() + y * b.width * 3, b.width * 3, out.pixels.data() + (y * out.width + a.width) * 3);
    }
    return out;
}

std::vector<float> StereoDepthRange::nearness(const std::vector<float>& depth) {
    std::vector<float> sample;
    const std::size_t step = std::max<std::size_t>(1, depth.size() / 65536);
    sample.reserve(depth.size() / step + 1);
    for (std::size_t i = 0; i < depth.size(); i += step)
        if (std::isfinite(depth[i])) sample.push_back(depth[i]);
    if (sample.empty()) return std::vector<float>(depth.size(), 0.0f);
    const auto at = [&](double q) {
        const auto k = static_cast<std::size_t>(q * static_cast<double>(sample.size() - 1));
        std::nth_element(sample.begin(), sample.begin() + static_cast<std::ptrdiff_t>(k), sample.end());
        return sample[k];
    };
    const float lo = at(0.02), hi = at(0.98);
    if (!started_) {
        lo_ = lo;
        hi_ = hi;
        started_ = true;
    } else {
        lo_ = smooth_ * lo_ + (1 - smooth_) * lo;
        hi_ = smooth_ * hi_ + (1 - smooth_) * hi;
    }
    const float range = std::max(hi_ - lo_, 1e-6f);
    std::vector<float> out(depth.size());
    for (std::size_t i = 0; i < depth.size(); ++i)
        out[i] = std::isfinite(depth[i]) ? std::clamp((depth[i] - lo_) / range, 0.0f, 1.0f) : 0.0f;
    return out;
}

} // namespace nc
