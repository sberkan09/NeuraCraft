#include "Workflow.h"
#include "Depth.h"
#include "Diffusion.h"
#include "Interpolate.h"
#include "Json.h"
#include "ModelCache.h"
#include "Upscale.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
// Progress of the running step, with cancellation checks.
struct StepProgress {
    const WorkflowEvents* events = nullptr;
    std::size_t step = 0, steps = 0;
    std::string op;

    void check() const {
        if (events && events->cancelled && events->cancelled()) throw WorkflowCancelled();
    }
    void stage(const std::string& s) const {
        check();
        if (!events || !events->progress) return;
        double fraction = -1.0;
        const std::size_t slash = s.rfind('/');
        if (slash != std::string::npos && slash > 0) {
            std::size_t b = slash;
            while (b > 0 && std::isdigit(static_cast<unsigned char>(s[b - 1]))) --b;
            const std::string num = s.substr(b, slash - b), den = s.substr(slash + 1);
            if (!num.empty() && !den.empty() && den.find_first_not_of("0123456789") == std::string::npos && std::stod(den) > 0)
                fraction = std::stod(num) / std::stod(den);
        }
        events->progress(step, steps, op, s, fraction);
    }
    void frame(std::size_t i, std::size_t n) const { stage("frame " + std::to_string(i + 1) + "/" + std::to_string(n)); }
    std::function<void(const std::string&, double)> stageCallback() const {
        if (!events || (!events->progress && !events->cancelled)) return {};
        const StepProgress* self = this;
        return [self](const std::string& s, double) { self->stage(s); };
    }
    std::function<void(const Image&)> previewCallback() const {
        if (!events || !events->preview) return {};
        const StepProgress* self = this;
        return [self](const Image& im) {
            self->check();
            self->events->preview(self->step, im);
        };
    }
};

struct Step {
    const json::Value& v;
    std::string op;
    std::string base;
    std::map<std::string, Clip>& clips;
    const StepProgress* progress = nullptr;

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
    if (s.progress) {
        o.progress = s.progress->stageCallback();
        o.preview = s.progress->previewCallback();
    }
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

// Cache tags (ModelCache.h) of the models a step loads, so each one is released after its last use.
std::set<std::string> modelsUsed(const Step& s) {
    std::set<std::string> out;
    const std::string& op = s.op;
    const bool diffusion = op == "txt2img" || op == "img2img" || op == "inpaint" || op == "t2v" || op == "morph" ||
                           op == "zoom" || op == "outpaint";
    if (diffusion) {
        const std::size_t w = s.size("width", 512), h = s.size("height", 512);
        const std::string model = s.path("model");
        out.insert(modelTag(model));        // text encoder
        out.insert(modelTag(model, w, h));  // UNet (with motion modules / IP-Adapter), VAE
        std::string tiny = s.path("tiny_vae");
        if (tiny.empty() && std::filesystem::exists(model + "/taesd/config.json")) tiny = model + "/taesd";
        if (!tiny.empty()) out.insert(modelTag(tiny, w, h));
        if (s.has("controlnet")) out.insert(modelTag(s.path("controlnet"), w, h));
        if (s.has("ip_adapter")) {
            const std::string ip = s.path("ip_adapter");
            std::string enc = s.path("image_encoder");
            if (enc.empty()) enc = std::filesystem::path(ip).parent_path().append("image_encoder").string();
            out.insert(modelTag(ip));
            out.insert(modelTag(enc));
        }
    } else if (op == "depth" || op == "parallax" || op == "upscale" || op == "interpolate") {
        out.insert(modelTag(s.path("model")));
    }
    return out;
}

// blend: frame by frame, `b` (resized to a's size) over `a`. Clips of different lengths: a one-frame
// clip is used for every frame, otherwise the shorter one sets the length.
Clip blendClips(const Clip& a, const Clip& b, const Clip* mask, const std::string& mode, float weight) {
    if (mode != "mix" && mode != "add" && mode != "multiply" && mode != "difference")
        throw std::invalid_argument("blend: unknown mode \"" + mode + "\" (mix, add, multiply, difference)");
    const std::size_t n = a.size() == 1 ? b.size() : b.size() == 1 ? a.size() : std::min(a.size(), b.size());
    Clip out;
    for (std::size_t f = 0; f < n; ++f) {
        const Image base = toRgb(a[a.size() == 1 ? 0 : f]);
        Image over = toRgb(b[b.size() == 1 ? 0 : f]);
        if (over.width != base.width || over.height != base.height) over = resizeImage(over, base.width, base.height);
        Image m;
        if (mask) {
            m = toRgb((*mask)[mask->size() == 1 ? 0 : std::min(f, mask->size() - 1)]);
            if (m.width != base.width || m.height != base.height) m = resizeImage(m, base.width, base.height);
        }
        Image r = base;
        for (std::size_t i = 0; i < base.pixels.size(); ++i) {
            const float x = base.pixels[i] / 255.0f, y = over.pixels[i] / 255.0f;
            const float w = weight * (mask ? (m.pixels[i - i % 3] * 0.299f + m.pixels[i - i % 3 + 1] * 0.587f +
                                              m.pixels[i - i % 3 + 2] * 0.114f) / 255.0f
                                           : 1.0f);
            float v;
            if (mode == "mix") v = x + w * (y - x);
            else if (mode == "add") v = x + w * y;
            else if (mode == "multiply") v = x + w * (x * y - x);
            else v = x + w * (std::fabs(x - y) - x);
            r.pixels[i] = static_cast<std::uint8_t>(std::lround(255.0f * std::min(1.0f, std::max(0.0f, v))));
        }
        out.push_back(std::move(r));
    }
    return out;
}

template <typename T>
std::shared_ptr<T> smallModel(const char* kind, const std::string& path) {
    return cachedModel<T>(std::string(kind) + "|" + modelTag(path), modelTag(path), [&] { return std::make_shared<T>(path); });
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
    WorkflowEvents events;
    events.log = log;
    return runWorkflow(text, base, events);
}

std::map<std::string, Clip> runWorkflow(const std::string& text, const std::string& base, const WorkflowEvents& events) {
    const auto& log = events.log;
    const std::vector<std::string> problems = checkWorkflow(text);
    if (!problems.empty()) {
        std::string message = "workflow:";
        for (const std::string& p : problems) message += "\n  " + p;
        throw std::invalid_argument(message);
    }
    const json::Value doc = json::parse(text);
    const json::Value* steps = doc.find("steps");
    std::map<std::string, Clip> clips;

    // Models stay loaded from the step that first needs them to the last one that does.
    const bool cache = doc.find("cache") ? doc.at("cache").boolean : true;
    std::unique_ptr<ModelCacheScope> cacheScope;
    if (cache) cacheScope = std::make_unique<ModelCacheScope>();
    std::vector<std::set<std::string>> needed;  // needed[i]: models used from step i on
    for (const json::Value& v : steps->array) {
        std::map<std::string, Clip> none;
        needed.push_back(modelsUsed(Step{v, v.at("op").string, base, none}));
    }
    for (std::size_t i = needed.size(); i-- > 1;) needed[i - 1].insert(needed[i].begin(), needed[i].end());

    std::size_t index = 0;
    for (const json::Value& v : steps->array) {
        ++index;
        StepProgress progress{&events, index, steps->array.size(), v.at("op").string};
        progress.stage("start");
        Step s{v, v.at("op").string, base, clips, &progress};
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
            const Clip& in = s.clip("in");
            for (const Image& im : in) {
                progress.frame(static_cast<std::size_t>(&im - in.data()), in.size());
                const std::size_t w = s.size("width", im.width), h = s.size("height", im.height);
                out.push_back(op == "resize" ? resizeImage(im, w, h) : fitImage(im, w, h));
            }
        } else if (op == "concat") {
            for (const std::string& name : s.strings("in")) {
                auto it = clips.find(name);
                if (it == clips.end()) throw std::invalid_argument("concat: no clip named \"" + name + "\"");
                out.insert(out.end(), it->second.begin(), it->second.end());
            }
        } else if (op == "blend") {
            const std::vector<std::string> names = s.strings("in");
            if (names.size() != 2) throw std::invalid_argument("blend: \"in\" needs two clips");
            auto find = [&](const std::string& name) -> const Clip& {
                auto it = clips.find(name);
                if (it == clips.end() || it->second.empty()) throw std::invalid_argument("blend: no clip named \"" + name + "\"");
                return it->second;
            };
            out = blendClips(find(names[0]), find(names[1]), s.has("mask") ? &s.clip("mask") : nullptr, s.str("mode", "mix"),
                             static_cast<float>(s.num("weight", 0.5)));
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
            o.progress = io.progress;
            o.preview = io.preview;
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
            const Clip& in = s.clip("in");
            for (const Image& im : in) {
                progress.frame(static_cast<std::size_t>(&im - in.data()), in.size());
                out.push_back(cannyEdges(im, static_cast<float>(s.num("low", 100)), static_cast<float>(s.num("high", 200))));
            }
        } else if (op == "depth" || op == "parallax") {
            const auto depthHold = smallModel<DepthEstimator>("depth", s.path("model"));
            DepthEstimator& de = *depthHold;
            const std::size_t inputSize = s.size("size", 518);
            if (op == "depth") {
                const Clip& in = s.clip("in");
                for (const Image& im : in) {
                    progress.frame(static_cast<std::size_t>(&im - in.data()), in.size());
                    out.push_back(de.depthImage(im, inputSize));
                }
            }
            else {
                ParallaxOptions po;
                po.frames = s.size("frames", 48);
                po.amplitude = static_cast<float>(s.num("amplitude", po.amplitude));
                po.zoom = static_cast<float>(s.num("zoom", po.zoom));
                po.loop = s.flag("loop", true);
                out = parallaxAnimation(s.image("in"), de.estimate(s.image("in"), inputSize), po);
            }
        } else if (op == "upscale") {
            const auto up = smallModel<Upscaler>("upscaler", s.path("model"));
            const Clip& in = s.clip("in");
            for (const Image& im : in) {
                progress.frame(static_cast<std::size_t>(&im - in.data()), in.size());
                out.push_back(up->upscale(im));
            }
        } else if (op == "interpolate") {
            out = smallModel<FrameInterpolator>("rife", s.path("model"))->interpolate(s.clip("in"), s.size("passes", 1), s.flag("loop", false));
        } else {
            throw std::invalid_argument("workflow: unknown op \"" + op + "\" in step " + std::to_string(index));
        }
        if (cache) releaseModelsExcept(index < needed.size() ? needed[index] : std::set<std::string>{});
        const std::string outName = s.str("out");
        const std::string clipName = out.empty() ? std::string() : (outName.empty() ? op : outName);
        if (!out.empty()) clips[clipName] = std::move(out);
        if (events.output) {
            static const Clip none;
            events.output(index, op, clipName, clipName.empty() ? none : clips[clipName],
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
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
