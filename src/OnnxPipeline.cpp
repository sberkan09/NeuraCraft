// What a workflow's "onnx" step does around Model::run: picture -> input tensor (size, layout,
// channel order, value range, mean / std), output tensor -> picture (image, mask or depth), and
// tiles for fixed-size super-resolution models. The "classify" step uses the same input side and
// reads the output as scores, one per label.

#include "Onnx.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nc::onnx {

namespace {

// Planar float picture: C x H x W.
struct Planes {
    std::size_t c = 0, h = 0, w = 0;
    std::vector<float> v;
};

std::string upper(std::string s) {
    for (char& ch : s) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    return s;
}

std::vector<float> floatList(const json::Value& v) {
    std::vector<float> out;
    if (v.isNumber()) out.push_back(static_cast<float>(v.number));
    else
        for (const auto& e : v.array) out.push_back(static_cast<float>(e.number));
    return out;
}

// An RGB / grey picture with replicated edges out to (w, h) from (x0, y0).
Image cropPadded(const Image& im, std::int64_t x0, std::int64_t y0, std::size_t w, std::size_t h) {
    Image out;
    out.width = w;
    out.height = h;
    out.channels = im.channels;
    out.pixels.resize(w * h * im.channels);
    for (std::size_t y = 0; y < h; ++y) {
        const std::size_t sy = static_cast<std::size_t>(std::clamp<std::int64_t>(y0 + static_cast<std::int64_t>(y), 0, static_cast<std::int64_t>(im.height) - 1));
        for (std::size_t x = 0; x < w; ++x) {
            const std::size_t sx = static_cast<std::size_t>(std::clamp<std::int64_t>(x0 + static_cast<std::int64_t>(x), 0, static_cast<std::int64_t>(im.width) - 1));
            for (std::size_t c = 0; c < im.channels; ++c) out.pixels[(y * w + x) * im.channels + c] = im.pixels[(sy * im.width + sx) * im.channels + c];
        }
    }
    return out;
}

Planes resizePlanes(const Planes& p, std::size_t w, std::size_t h) {
    if (p.w == w && p.h == h) return p;
    Planes out{p.c, h, w, std::vector<float>(p.c * h * w)};
    const float sx = static_cast<float>(p.w) / static_cast<float>(w), sy = static_cast<float>(p.h) / static_cast<float>(h);
    for (std::size_t c = 0; c < p.c; ++c)
        for (std::size_t y = 0; y < h; ++y) {
            const float fy = std::clamp((static_cast<float>(y) + 0.5f) * sy - 0.5f, 0.0f, static_cast<float>(p.h - 1));
            const std::size_t y0 = static_cast<std::size_t>(fy), y1 = std::min(y0 + 1, p.h - 1);
            const float ty = fy - static_cast<float>(y0);
            for (std::size_t x = 0; x < w; ++x) {
                const float fx = std::clamp((static_cast<float>(x) + 0.5f) * sx - 0.5f, 0.0f, static_cast<float>(p.w - 1));
                const std::size_t x0 = static_cast<std::size_t>(fx), x1 = std::min(x0 + 1, p.w - 1);
                const float tx = fx - static_cast<float>(x0);
                const float* pl = p.v.data() + c * p.h * p.w;
                const float a = pl[y0 * p.w + x0] * (1 - tx) + pl[y0 * p.w + x1] * tx, b = pl[y1 * p.w + x0] * (1 - tx) + pl[y1 * p.w + x1] * tx;
                out.v[(c * h + y) * w + x] = a * (1 - ty) + b * ty;
            }
        }
    return out;
}

} // namespace

json::Value onnxManifest(const std::string& modelPath) {
    const std::filesystem::path p = std::filesystem::path(modelPath).replace_extension(".json");
    if (!std::filesystem::exists(p)) return json::makeObject({});
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    json::Value v = json::parse(ss.str());
    if (!v.isObject()) throw std::runtime_error(p.string() + ": expected a JSON object");
    return v;
}

OnnxPipeline OnnxPipeline::fromModel(const Model& model, const json::Value& manifest) {
    OnnxPipeline p;
    if (model.inputs().empty() || model.outputs().empty()) throw std::runtime_error("onnx: the model has no input or output");
    const json::Value* in = manifest.find("input");
    const json::Value* out = manifest.find("output");
    p.inputName = in && in->find("name") ? in->at("name").string : model.inputs()[0].name;
    p.outputName = out && out->find("name") ? out->at("name").string : model.outputs()[0].name;
    const ValueInfo* info = nullptr;
    for (const ValueInfo& v : model.inputs())
        if (v.name == p.inputName) info = &v;
    if (!info) throw std::runtime_error("onnx: the model has no input '" + p.inputName + "'");
    if (in && in->find("layout")) p.nhwcIn = upper(in->at("layout").string) == "NHWC";
    else if (info->shape.size() == 4 && (info->shape[3] == 3 || info->shape[3] == 1) && info->shape[1] != 3 && info->shape[1] != 1) p.nhwcIn = true;
    if (in && in->find("channels")) {
        const std::string c = upper(in->at("channels").string);
        p.bgrIn = c == "BGR";
        p.grayIn = c == "GRAY" || c == "GREY";
    } else if (info->shape.size() == 4) {
        p.grayIn = (p.nhwcIn ? info->shape[3] : info->shape[1]) == 1;
    }
    // Size: the manifest's, else the model's own fixed one.
    if (in && in->find("size")) {
        const std::vector<float> s = floatList(in->at("size"));
        p.height = static_cast<int>(s.at(0));
        p.width = static_cast<int>(s.size() > 1 ? s[1] : s[0]);
    } else if (info->shape.size() == 4) {
        const std::int64_t h = p.nhwcIn ? info->shape[1] : info->shape[2], w = p.nhwcIn ? info->shape[2] : info->shape[3];
        if (h > 0 && w > 0) {
            p.height = static_cast<int>(h);
            p.width = static_cast<int>(w);
        }
    }
    if (in && in->find("multiple_of")) p.multipleOf = std::max(1, static_cast<int>(in->at("multiple_of").number));
    if (in && in->find("range")) {
        const std::vector<float> r = floatList(in->at("range"));
        p.inLow = r.at(0);
        p.inHigh = r.at(1);
    }
    if (in && in->find("mean")) p.mean = floatList(in->at("mean"));
    if (in && in->find("std")) p.stdev = floatList(in->at("std"));
    if (in && in->find("tile")) p.tileOverlap = static_cast<int>(in->at("tile").number);
    if (out) {
        if (out->find("kind")) p.kind = out->at("kind").string;
        if (out->find("layout")) p.nhwcOut = upper(out->at("layout").string) == "NHWC";
        if (out->find("channels")) p.bgrOut = upper(out->at("channels").string) == "BGR";
        if (out->find("invert")) p.invert = out->at("invert").boolean;
        if (out->find("apply")) p.applyMask = out->at("apply").boolean;
        if (const json::Value* l = out->find("labels"))
            for (const json::Value& x : l->array) p.labels.push_back(x.isString() ? x.string : std::to_string(p.labels.size()));
        if (out->find("activation")) {
            p.activation = out->at("activation").string;
            if (p.activation != "softmax" && p.activation != "sigmoid" && p.activation != "none")
                throw std::runtime_error("onnx: output.activation is softmax, sigmoid or none, not \"" + p.activation + "\"");
        }
        if (const json::Value* r = out->find("range")) {
            if (r->isString()) p.autoRange = r->string == "auto";
            else {
                const std::vector<float> v = floatList(*r);
                p.autoRange = false;
                p.outLow = v.at(0);
                p.outHigh = v.at(1);
            }
            p.rangeSet = true;
        }
    }
    p.fixedByModel = !(in && in->find("size")) && p.width > 0;
    return p;
}

namespace {

// The picture as the model takes it: grey or RGB.
Image sourcePicture(const OnnxPipeline& p, const Image& input) {
    if (!p.grayIn) return toRgb(input);
    const Image rgb = toRgb(input);
    Image g;
    g.width = rgb.width;
    g.height = rgb.height;
    g.channels = 1;
    g.pixels.resize(rgb.width * rgb.height);
    for (std::size_t k = 0; k < g.pixels.size(); ++k)
        g.pixels[k] = static_cast<std::uint8_t>(std::lround(0.299 * rgb.pixels[3 * k] + 0.587 * rgb.pixels[3 * k + 1] + 0.114 * rgb.pixels[3 * k + 2]));
    return g;
}

// A picture of the model's size -> its input tensor (layout, channel order, range, mean / std).
Tensor pictureTensor(const OnnxPipeline& p, const Image& im) {
    const std::size_t H = im.height, W = im.width, C = im.channels;
    Tensor t;
    t.kind = Tensor::Kind::Float;
    t.elemType = 1;
    t.shape = p.nhwcIn ? std::vector<std::int64_t>{1, static_cast<std::int64_t>(H), static_cast<std::int64_t>(W), static_cast<std::int64_t>(C)}
                       : std::vector<std::int64_t>{1, static_cast<std::int64_t>(C), static_cast<std::int64_t>(H), static_cast<std::int64_t>(W)};
    t.f.resize(C * H * W);
    for (std::size_t y = 0; y < H; ++y)
        for (std::size_t x = 0; x < W; ++x)
            for (std::size_t c = 0; c < C; ++c) {
                const std::size_t sc = p.bgrIn && C == 3 ? 2 - c : c;
                float v = im.pixels[(y * W + x) * C + sc] / 255.0f * (p.inHigh - p.inLow) + p.inLow;
                if (!p.mean.empty()) v -= p.mean[std::min(c, p.mean.size() - 1)];
                if (!p.stdev.empty()) v /= p.stdev[std::min(c, p.stdev.size() - 1)];
                t.f[p.nhwcIn ? (y * W + x) * C + c : (c * H + y) * W + x] = v;
            }
    return t;
}

} // namespace

std::vector<float> OnnxPipeline::classify(const Model& model, const Image& input) const {
    if (width <= 0 || height <= 0) throw std::runtime_error("classify: the model needs a fixed input size (give input.size in its manifest)");
    const Image src = sourcePicture(*this, input);
    const std::size_t tw = static_cast<std::size_t>(width), th = static_cast<std::size_t>(height);
    const Image sized = (src.width == tw && src.height == th) ? src : resizeImage(src, tw, th);
    const auto res = model.run({{inputName, pictureTensor(*this, sized)}}, {outputName});
    const Tensor& o = res.at(outputName);
    if (o.kind != Tensor::Kind::Float) throw std::runtime_error("classify: output '" + outputName + "' is not a float tensor");
    std::vector<float> v = o.f;
    if (!labels.empty() && v.size() != labels.size())
        throw std::runtime_error("classify: the model gives " + std::to_string(v.size()) + " values for " + std::to_string(labels.size()) + " labels");
    if (activation == "softmax" && !v.empty()) {
        const float top = *std::max_element(v.begin(), v.end());
        double sum = 0;
        for (float& x : v) sum += (x = std::exp(x - top));
        for (float& x : v) x = static_cast<float>(x / sum);
    } else if (activation == "sigmoid") {
        for (float& x : v) x = 1.0f / (1.0f + std::exp(-x));
    }
    return v;
}

Image OnnxPipeline::run(const Model& model, const Image& input) const {
    const Image src = sourcePicture(*this, input);

    // One forward pass over a picture of the size the model takes -> planes.
    auto forward = [&](const Image& im) {
        Tensor t = pictureTensor(*this, im);
        const auto res = model.run({{inputName, std::move(t)}}, {outputName});
        const Tensor& o = res.at(outputName);
        if (o.kind != Tensor::Kind::Float) throw std::runtime_error("onnx: output '" + outputName + "' is not a float tensor");
        Planes p;
        const auto& s = o.shape;
        bool nhwc = nhwcOut;
        if (s.size() == 4) {
            if (!nhwcOut && (s[3] == 1 || s[3] == 3) && s[1] != 1 && s[1] != 3) nhwc = true;
            p.c = static_cast<std::size_t>(nhwc ? s[3] : s[1]);
            p.h = static_cast<std::size_t>(nhwc ? s[1] : s[2]);
            p.w = static_cast<std::size_t>(nhwc ? s[2] : s[3]);
        } else if (s.size() == 3) {
            const bool chw = s[0] == 3 && s[1] > 4;
            p.c = chw ? 3 : 1;
            p.h = static_cast<std::size_t>(chw ? s[1] : s[s.size() - 2]);
            p.w = static_cast<std::size_t>(s.back());
            nhwc = false;
        } else if (s.size() == 2) {
            p.c = 1;
            p.h = static_cast<std::size_t>(s[0]);
            p.w = static_cast<std::size_t>(s[1]);
        } else {
            throw std::runtime_error("onnx: output '" + outputName + "' is not a picture (shape rank " + std::to_string(s.size()) + ")");
        }
        if (p.c != 1 && p.c != 3) throw std::runtime_error("onnx: output '" + outputName + "' has " + std::to_string(p.c) + " channels (1 or 3 make a picture)");
        p.v.resize(p.c * p.h * p.w);
        for (std::size_t c = 0; c < p.c; ++c)
            for (std::size_t k = 0; k < p.h * p.w; ++k) {
                const std::size_t sc = bgrOut && p.c == 3 ? 2 - c : c;
                p.v[c * p.h * p.w + k] = nhwc ? o.f[k * p.c + sc] : o.f[sc * p.h * p.w + k];
            }
        return p;
    };

    Planes result;
    std::size_t outW = 0, outH = 0;  // the final size
    if (width > 0 && height > 0) {
        const std::size_t tw = static_cast<std::size_t>(width), th = static_cast<std::size_t>(height);
        // A fixed-size model: in tiles when it enlarges (super-resolution) and the picture is bigger.
        Planes probe;
        bool tiled = false;
        if (fixedByModel && tileOverlap != 0 && (src.width > tw || src.height > th)) {
            probe = forward(cropPadded(src, 0, 0, tw, th));
            tiled = probe.w > tw && probe.c == 3 && kind != "mask" && kind != "depth";
        }
        if (tiled) {
            const std::size_t scale = probe.w / tw;
            const std::size_t overlap = static_cast<std::size_t>(tileOverlap > 0 ? tileOverlap : static_cast<int>(std::min(tw, th) / 8));
            const std::size_t stepX = tw > overlap ? tw - overlap : tw, stepY = th > overlap ? th - overlap : th;
            const std::size_t W = src.width * scale, H = src.height * scale;
            std::vector<float> acc(3 * W * H, 0.0f), wsum(W * H, 0.0f);
            for (std::size_t y0 = 0;; y0 += stepY) {
                const std::size_t ty = std::min(y0, src.height > th ? src.height - th : 0);
                for (std::size_t x0 = 0;; x0 += stepX) {
                    const std::size_t tx = std::min(x0, src.width > tw ? src.width - tw : 0);
                    const Planes p = (tx == 0 && ty == 0) ? probe : forward(cropPadded(src, static_cast<std::int64_t>(tx), static_cast<std::int64_t>(ty), tw, th));
                    const float ramp = static_cast<float>(std::max<std::size_t>(1, overlap * scale));
                    for (std::size_t y = 0; y < p.h; ++y) {
                        const std::size_t gy = ty * scale + y;
                        if (gy >= H) continue;
                        const float wy = std::min({1.0f, (static_cast<float>(y) + 1) / ramp, (static_cast<float>(p.h - y)) / ramp});
                        for (std::size_t x = 0; x < p.w; ++x) {
                            const std::size_t gx = tx * scale + x;
                            if (gx >= W) continue;
                            const float wx = std::min({1.0f, (static_cast<float>(x) + 1) / ramp, (static_cast<float>(p.w - x)) / ramp});
                            const float wgt = wx * wy;
                            for (std::size_t c = 0; c < 3; ++c) acc[(c * H + gy) * W + gx] += wgt * p.v[(c * p.h + y) * p.w + x];
                            wsum[gy * W + gx] += wgt;
                        }
                    }
                    if (tx + tw >= src.width) break;
                }
                if (ty + th >= src.height) break;
            }
            result = {3, H, W, std::move(acc)};
            for (std::size_t c = 0; c < 3; ++c)
                for (std::size_t k = 0; k < W * H; ++k) result.v[c * W * H + k] /= std::max(wsum[k], 1e-6f);
            outW = W;
            outH = H;
        } else {
            const Image resized = (src.width == tw && src.height == th) ? src : resizeImage(src, tw, th);
            result = forward(resized);
            // Back to the input size times the model's own scale (1 for masks, depth and same-size models).
            const double scale = result.c == 3 && kind != "mask" && kind != "depth" ? static_cast<double>(result.w) / static_cast<double>(tw) : 1.0;
            outW = static_cast<std::size_t>(std::lround(static_cast<double>(src.width) * scale));
            outH = static_cast<std::size_t>(std::lround(static_cast<double>(src.height) * scale));
        }
    } else {
        // Any size: pad (edges replicated) to a multiple of multipleOf, then crop the padding off.
        const std::size_t m = static_cast<std::size_t>(multipleOf);
        const std::size_t pw = (src.width + m - 1) / m * m, ph = (src.height + m - 1) / m * m;
        result = forward(pw == src.width && ph == src.height ? src : cropPadded(src, 0, 0, pw, ph));
        const double scale = static_cast<double>(result.w) / static_cast<double>(pw);
        const std::size_t cw = std::min(result.w, static_cast<std::size_t>(std::lround(src.width * scale))),
                          chh = std::min(result.h, static_cast<std::size_t>(std::lround(src.height * scale)));
        if (cw != result.w || chh != result.h) {
            Planes c{result.c, chh, cw, std::vector<float>(result.c * cw * chh)};
            for (std::size_t ch = 0; ch < result.c; ++ch)
                for (std::size_t y = 0; y < chh; ++y)
                    std::copy(result.v.begin() + static_cast<std::ptrdiff_t>((ch * result.h + y) * result.w),
                              result.v.begin() + static_cast<std::ptrdiff_t>((ch * result.h + y) * result.w + cw), c.v.begin() + static_cast<std::ptrdiff_t>((ch * chh + y) * cw));
            result = std::move(c);
        }
        const double s = result.c == 3 && kind != "mask" && kind != "depth" ? scale : 1.0;
        outW = static_cast<std::size_t>(std::lround(src.width * s));
        outH = static_cast<std::size_t>(std::lround(src.height * s));
    }
    result = resizePlanes(result, outW, outH);

    // Values -> 0..255: a fixed range, or (masks, depth) stretched from min to max.
    const bool single = result.c == 1;
    const bool stretch = rangeSet ? autoRange : single;
    float lo = outLow, hi = outHigh;
    if (stretch) {
        lo = std::numeric_limits<float>::max();
        hi = std::numeric_limits<float>::lowest();
        for (float v : result.v) {
            if (!std::isfinite(v)) continue;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    const float span = hi > lo ? hi - lo : 1.0f;
    auto byte = [&](float v) {
        float t = (v - lo) / span;
        if (invert) t = 1.0f - t;
        return static_cast<std::uint8_t>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
    };
    Image out;
    out.width = outW;
    out.height = outH;
    const std::size_t plane = outW * outH;
    if (single && applyMask) {  // cut the input out: RGBA, alpha from the mask
        const Image rgb = toRgb(input);
        const Image base = (rgb.width == outW && rgb.height == outH) ? rgb : resizeImage(rgb, outW, outH);
        out.channels = 4;
        out.pixels.resize(plane * 4);
        for (std::size_t k = 0; k < plane; ++k) {
            for (std::size_t c = 0; c < 3; ++c) out.pixels[4 * k + c] = base.pixels[3 * k + c];
            out.pixels[4 * k + 3] = byte(result.v[k]);
        }
        return out;
    }
    out.channels = 3;  // grey outputs as RGB, which every later step reads
    out.pixels.resize(plane * 3);
    for (std::size_t k = 0; k < plane; ++k)
        for (std::size_t c = 0; c < 3; ++c) out.pixels[3 * k + c] = byte(result.v[(single ? 0 : c) * plane + k]);
    return out;
}

} // namespace nc::onnx
