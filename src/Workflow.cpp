#include "Workflow.h"
#include "Depth.h"
#include "Diffusion.h"
#include "Interpolate.h"
#include "Json.h"
#include "Upscale.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
struct Step {
    const json::Value& v;
    std::string op;
    std::string base;
    std::map<std::string, Clip>& clips;

    bool has(const char* k) const { return v.find(k) != nullptr; }
    std::string str(const char* k, const std::string& fallback = "") const {
        const json::Value* x = v.find(k);
        if (!x) return fallback;
        if (!x->isString()) throw std::invalid_argument(op + ": \"" + k + "\" must be a string");
        return x->string;
    }
    std::string path(const char* k) const {
        const std::string s = str(k);
        if (s.empty()) return s;
        const std::filesystem::path p(s);
        return p.is_absolute() ? s : (std::filesystem::path(base) / p).string();
    }
    std::string need(const char* k) const {
        const std::string s = str(k);
        if (s.empty()) throw std::invalid_argument(op + ": needs \"" + k + "\"");
        return s;
    }
    double num(const char* k, double fallback) const { return v.numberOr(k, fallback); }
    std::size_t size(const char* k, std::size_t fallback) const { return static_cast<std::size_t>(num(k, static_cast<double>(fallback))); }
    bool flag(const char* k, bool fallback) const {
        const json::Value* x = v.find(k);
        return x ? x->boolean : fallback;
    }
    const Clip& clip(const char* k) const {
        const std::string name = need(k);
        auto it = clips.find(name);
        if (it == clips.end() || it->second.empty()) throw std::invalid_argument(op + ": no clip named \"" + name + "\"");
        return it->second;
    }
    const Image& image(const char* k) const { return clip(k).front(); }
    std::vector<std::string> strings(const char* k) const {
        std::vector<std::string> out;
        const json::Value* x = v.find(k);
        if (!x) return out;
        if (x->isString()) return {x->string};
        for (const auto& e : x->array) out.push_back(e.string);
        return out;
    }
    std::vector<Lora> loras() const {
        std::vector<Lora> out;
        for (std::string s : strings("lora")) {
            float scale = 1.0f;
            const std::size_t colon = s.rfind(':');
            if (colon != std::string::npos && colon + 1 < s.size() && (std::isdigit(static_cast<unsigned char>(s[colon + 1])) || s[colon + 1] == '.')) {
                scale = std::stof(s.substr(colon + 1));
                s = s.substr(0, colon);
            }
            const std::filesystem::path p(s);
            out.push_back({p.is_absolute() ? s : (std::filesystem::path(base) / p).string(), scale});
        }
        return out;
    }
};

TextToImageOptions imageOptions(const Step& s) {
    TextToImageOptions o;
    o.width = s.size("width", 512);
    o.height = s.size("height", 512);
    o.steps = s.size("steps", s.op == "img2img" ? 2 : s.op == "inpaint" ? 25 : 1);
    o.seed = static_cast<unsigned>(s.num("seed", 0));
    o.guidance = static_cast<float>(s.num("guidance", 0));
    o.negativePrompt = s.str("negative");
    o.strength = static_cast<float>(s.num("strength", s.op == "inpaint" ? 1.0 : 0.5));
    o.sampler = s.str("sampler", "euler");
    o.int8 = s.flag("int8", true);
    o.loras = s.loras();
    o.tinyVae = s.path("tiny_vae");
    const std::string model = s.path("model");
    if (o.tinyVae.empty() && std::filesystem::exists(model + "/taesd/config.json")) o.tinyVae = model + "/taesd";
    if (s.has("controlnet")) {
        o.controlNet = s.path("controlnet");
        o.controlImage = s.image("control");
        o.controlScale = static_cast<float>(s.num("control_scale", 1.0));
    }
    if (s.has("ip_adapter")) {
        o.ipAdapter = s.path("ip_adapter");
        o.imageEncoder = s.path("image_encoder");
        o.imagePrompt = s.image("image_prompt");
        o.ipScale = static_cast<float>(s.num("ip_scale", 0.6));
    }
    return o;
}

void save(const Step& s, const Clip& c) {
    const std::string path = s.path("path");
    if (path.empty()) throw std::invalid_argument("save: needs \"path\"");
    const float fps = static_cast<float>(s.num("fps", 8));
    auto ends = [&](const char* e) {
        const std::size_t n = std::strlen(e);
        return path.size() >= n && path.compare(path.size() - n, n, e) == 0;
    };
    if (path.back() == '/') {
        std::filesystem::create_directories(path);
        for (std::size_t i = 0; i < c.size(); ++i) {
            char name[32];
            std::snprintf(name, sizeof name, "frame_%04zu.png", i);
            savePng(path + name, c[i]);
        }
    } else if (ends(".jpg") || ends(".jpeg")) saveJpeg(path, c.front(), static_cast<int>(s.num("quality", 92)));
    else if (ends(".png")) c.size() == 1 ? savePng(path, c.front()) : saveApng(path, c, fps);
    else if (ends(".mov")) saveMjpegMov(path, c, fps, static_cast<int>(s.num("quality", 90)));
    else if (ends(".gif")) saveGif(path, c, fps);
    else throw std::invalid_argument("save: unknown file type for " + path);
}
} // namespace

std::map<std::string, Clip> runWorkflow(const std::string& text, const std::string& base, const std::function<void(const std::string&)>& log) {
    const json::Value doc = json::parse(text);
    const json::Value* steps = doc.find("steps");
    if (!steps || !steps->isArray()) throw std::invalid_argument("workflow: needs a \"steps\" array");
    std::map<std::string, Clip> clips;
    std::size_t index = 0;
    for (const json::Value& v : steps->array) {
        ++index;
        Step s{v, v.at("op").string, base, clips};
        const auto start = std::chrono::steady_clock::now();
        Clip out;
        const std::string& op = s.op;
        if (op == "load") {
            if (s.has("dir")) {
                std::vector<std::string> files;
                for (const auto& e : std::filesystem::directory_iterator(s.path("dir"))) {
                    const std::string ext = e.path().extension().string();
                    if (ext == ".png" || ext == ".ppm") files.push_back(e.path().string());
                }
                std::sort(files.begin(), files.end());
                for (const auto& f : files) out.push_back(toRgb(loadImage(f)));
            } else out.push_back(toRgb(loadImage(s.path("path"))));
        } else if (op == "save") {
            save(s, s.clip("in"));
        } else if (op == "resize" || op == "crop") {
            for (const Image& im : s.clip("in")) {
                const std::size_t w = s.size("width", im.width), h = s.size("height", im.height);
                out.push_back(op == "resize" ? resizeImage(im, w, h) : fitImage(im, w, h));
            }
        } else if (op == "concat") {
            for (const std::string& name : s.strings("in")) {
                auto it = clips.find(name);
                if (it == clips.end()) throw std::invalid_argument("concat: no clip named \"" + name + "\"");
                out.insert(out.end(), it->second.begin(), it->second.end());
            }
        } else if (op == "txt2img") {
            out.push_back(textToImage(s.path("model"), s.need("prompt"), imageOptions(s)));
        } else if (op == "img2img") {
            out.push_back(imageToImage(s.path("model"), s.image("in"), s.need("prompt"), imageOptions(s)));
        } else if (op == "inpaint") {
            const TextToImageOptions o = imageOptions(s);
            StableDiffusion sd(s.path("model"), o, true);
            out.push_back(sd.inpaint(s.image("in"), s.image("mask"), s.need("prompt"), o.seed, o.steps, o.strength,
                                     static_cast<float>(s.num("feather", 4))));
        } else if (op == "t2v") {
            const TextToImageOptions io = imageOptions(s);
            TextToVideoOptions o;
            o.width = io.width;
            o.height = io.height;
            o.frames = s.size("frames", 16);
            o.steps = s.size("steps", 4);
            o.seed = io.seed;
            o.guidance = static_cast<float>(s.num("guidance", 1));
            o.negativePrompt = io.negativePrompt;
            o.motion = s.path("motion");
            o.tinyVae = io.tinyVae;
            o.int8 = io.int8;
            o.loras = io.loras;
            o.sampler = io.sampler;
            o.betaSchedule = s.str("beta_schedule", "linear");
            o.context = s.size("context", 16);
            o.overlap = s.size("overlap", 4);
            if (s.has("init")) {
                o.init = s.clip("init");
                o.strength = static_cast<float>(s.num("strength", 0.75));
                if (o.init.size() > 1 && !s.has("frames")) o.frames = o.init.size();
            }
            if (s.has("controlnet")) {
                o.controlNet = io.controlNet;
                o.controlImages = s.clip("control");
                o.controlScale = io.controlScale;
            }
            o.ipAdapter = io.ipAdapter;
            o.imageEncoder = io.imageEncoder;
            o.imagePrompt = io.imagePrompt;
            o.ipScale = io.ipScale;
            if (const json::Value* p = v.find("prompt_at"))
                for (const auto& e : p->array) o.promptAt.emplace_back(static_cast<std::size_t>(e.array.at(0).number), e.array.at(1).string);
            out = textToVideo(s.path("model"), s.need("prompt"), o);
        } else if (op == "morph" || op == "zoom" || op == "outpaint") {
            TextToImageOptions io = imageOptions(s);
            StableDiffusion sd(s.path("model"), io, op != "morph");
            const std::vector<std::string> prompts = s.strings("prompts");
            if (prompts.empty()) throw std::invalid_argument(op + ": needs \"prompts\"");
            if (op == "morph") {
                MorphOptions mo;
                mo.framesPerSegment = s.size("frames", 12);
                mo.steps = s.size("steps", 1);
                mo.loop = s.flag("loop", true);
                std::vector<unsigned> seeds;
                if (const json::Value* x = v.find("seeds"))
                    for (const auto& e : x->array) seeds.push_back(static_cast<unsigned>(e.number));
                if (seeds.empty()) seeds = {io.seed};
                out = morphAnimation(sd, prompts, seeds, mo);
            } else if (op == "zoom") {
                ZoomOptions zo;
                zo.frames = s.size("frames", 24);
                zo.zoom = static_cast<float>(s.num("zoom", zo.zoom));
                zo.strength = static_cast<float>(s.num("strength", zo.strength));
                zo.steps = s.size("steps", zo.steps);
                zo.seed = io.seed;
                out = zoomAnimation(sd, prompts, zo, s.has("in") ? &s.image("in") : nullptr);
            } else {
                OutpaintZoomOptions oz;
                oz.keyframes = s.size("keyframes", 6);
                oz.shrink = static_cast<float>(s.num("shrink", 0.75));
                oz.framesPerKey = s.size("frames", 12);
                oz.steps = s.size("steps", 25);
                oz.seed = io.seed;
                oz.zoomIn = s.flag("zoom_in", false);
                oz.strength = static_cast<float>(s.num("strength", 0.9));
                out = outpaintZoom(sd, prompts, oz, s.has("in") ? &s.image("in") : nullptr);
            }
        } else if (op == "canny") {
            for (const Image& im : s.clip("in"))
                out.push_back(cannyEdges(im, static_cast<float>(s.num("low", 100)), static_cast<float>(s.num("high", 200))));
        } else if (op == "depth" || op == "parallax") {
            DepthEstimator de(s.path("model"));
            const std::size_t inputSize = s.size("size", 518);
            if (op == "depth")
                for (const Image& im : s.clip("in")) out.push_back(de.depthImage(im, inputSize));
            else {
                ParallaxOptions po;
                po.frames = s.size("frames", 48);
                po.amplitude = static_cast<float>(s.num("amplitude", po.amplitude));
                po.zoom = static_cast<float>(s.num("zoom", po.zoom));
                po.loop = s.flag("loop", true);
                out = parallaxAnimation(s.image("in"), de.estimate(s.image("in"), inputSize), po);
            }
        } else if (op == "upscale") {
            const Upscaler up(s.path("model"));
            for (const Image& im : s.clip("in")) out.push_back(up.upscale(im));
        } else if (op == "interpolate") {
            out = FrameInterpolator(s.path("model")).interpolate(s.clip("in"), s.size("passes", 1), s.flag("loop", false));
        } else {
            throw std::invalid_argument("workflow: unknown op \"" + op + "\" in step " + std::to_string(index));
        }
        const std::string outName = s.str("out");
        if (!out.empty()) clips[outName.empty() ? op : outName] = std::move(out);
        if (log) {
            std::ostringstream line;
            line << "step " << index << " " << op << ": " << std::fixed;
            line.precision(1);
            line << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s";
            log(line.str());
        }
    }
    return clips;
}

} // namespace nc
