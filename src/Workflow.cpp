#include "Workflow.h"
#include "Backend.h"
#include "Onnx.h"
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
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    double maxSeconds = 0;

    void check() const {
        if (events && events->cancelled && events->cancelled()) throw WorkflowCancelled();
        if (std::chrono::steady_clock::now() > deadline)
            throw WorkflowLimitExceeded("the workflow ran longer than " + std::to_string(static_cast<long long>(maxSeconds)) + " s");
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
        if (events && events->progress) events->progress(step, steps, op, s, fraction);
    }
    void frame(std::size_t i, std::size_t n) const { stage("frame " + std::to_string(i + 1) + "/" + std::to_string(n)); }
    std::function<void(const std::string&, double)> stageCallback() const {
        if ((!events || (!events->progress && !events->cancelled)) && deadline == std::chrono::steady_clock::time_point::max()) return {};
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

// Whether `path` is inside one of `roots` once ".." and symlinks are resolved.
bool insideRoots(const std::string& path, const std::vector<std::string>& roots) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p = fs::weakly_canonical(fs::absolute(path, ec), ec);
    if (ec) return false;
    for (const std::string& r : roots) {
        const fs::path root = fs::weakly_canonical(fs::absolute(r, ec), ec);
        if (ec) continue;
        auto pi = p.begin();
        bool inside = true;
        for (auto ri = root.begin(); ri != root.end(); ++ri, ++pi) {
            if (ri->empty()) continue;  // a trailing separator
            if (pi == p.end() || *pi != *ri) {
                inside = false;
                break;
            }
        }
        if (inside) return true;
    }
    return false;
}

void allowRead(const WorkflowLimits* limits, const std::string& path) {
    if (!limits || !limits->sandboxed() || path.empty()) return;
    if (!insideRoots(path, limits->readRoots)) throw WorkflowLimitExceeded("reading " + path + " is not allowed");
}
void allowWrite(const WorkflowLimits* limits, const std::string& path) {
    if (!limits || !limits->sandboxed()) return;
    if (limits->writeRoot.empty() || !insideRoots(path, {limits->writeRoot})) throw WorkflowLimitExceeded("writing " + path + " is not allowed");
}

struct Step {
    const json::Value& v;
    std::string op;
    std::string base;
    std::map<std::string, Clip>& clips;
    const StepProgress* progress = nullptr;
    const WorkflowLimits* limits = nullptr;

    bool has(const char* k) const { return v.find(k) != nullptr; }
    std::string str(const char* k, const std::string& fallback = "") const {
        const json::Value* x = v.find(k);
        if (!x) return fallback;
        if (!x->isString()) throw std::invalid_argument(op + ": \"" + k + "\" must be a string");
        return x->string;
    }
    // A file field as a path (relative to the base folder), checked against the sandbox.
    std::string path(const char* k) const {
        const std::string s = str(k);
        if (s.empty()) return s;
        const std::filesystem::path p(s);
        const std::string full = p.is_absolute() ? s : (std::filesystem::path(base) / p).string();
        if (std::string(k) == "path" && op == "save") allowWrite(limits, full);
        else allowRead(limits, full);
        return full;
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
            allowRead(limits, out.back().file);
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
    o.vae = s.path("vae");
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
        if (o.imageEncoder.empty() && s.limits && s.limits->sandboxed()) {
            o.imageEncoder = std::filesystem::path(o.ipAdapter).parent_path().append("image_encoder").string();
            allowRead(s.limits, o.imageEncoder);
        }
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
        if (s.has("vae")) out.insert(modelTag(s.path("vae"), w, h));
        if (s.has("controlnet")) out.insert(modelTag(s.path("controlnet"), w, h));
        if (s.has("ip_adapter")) {
            const std::string ip = s.path("ip_adapter");
            std::string enc = s.path("image_encoder");
            if (enc.empty()) enc = std::filesystem::path(ip).parent_path().append("image_encoder").string();
            out.insert(modelTag(ip));
            out.insert(modelTag(enc));
        }
    } else if (op == "depth" || op == "parallax" || op == "upscale" || op == "interpolate" || op == "onnx") {
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
    // Missing folders are created (outputs/<run>/image.png).
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
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

std::shared_ptr<const Clip> WorkflowResultCache::find(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
        if (it->key == key) {
            entries_.splice(entries_.begin(), entries_, it);
            ++hits_;
            return it->clip;
        }
    return nullptr;
}

void WorkflowResultCache::put(const std::string& key, const Clip& clip) {
    std::size_t bytes = 0;
    for (const Image& im : clip) bytes += im.pixels.size();
    if (bytes > maxBytes_) return;
    auto copy = std::make_shared<const Clip>(clip);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it)
        if (it->key == key) {
            bytes_ -= it->bytes;
            entries_.erase(it);
            break;
        }
    entries_.push_front({key, std::move(copy), bytes});
    bytes_ += bytes;
    while (bytes_ > maxBytes_ && !entries_.empty()) {
        bytes_ -= entries_.back().bytes;
        entries_.pop_back();
    }
}

void WorkflowResultCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    bytes_ = 0;
}

std::size_t WorkflowResultCache::bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_;
}
std::size_t WorkflowResultCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}
std::size_t WorkflowResultCache::hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hits_;
}

namespace {
std::string fnv64(const std::string& s) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// Size and modification time of a file (or of every file directly in a folder), so edited inputs and
// replaced models change the key.
std::string fileStamp(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto stamp = [&](const fs::path& p) {
        const auto size = fs::file_size(p, ec);
        if (ec) return std::string("?");
        const auto t = fs::last_write_time(p, ec).time_since_epoch().count();
        return std::to_string(size) + "@" + std::to_string(static_cast<long long>(t));
    };
    if (fs::is_regular_file(path, ec)) return stamp(path);
    if (!fs::is_directory(path, ec)) return "-";
    std::vector<std::string> parts;
    for (const auto& e : fs::directory_iterator(path, ec))
        if (e.is_regular_file(ec)) parts.push_back(e.path().filename().string() + "=" + stamp(e.path()));
    std::sort(parts.begin(), parts.end());
    std::string out;
    for (const auto& p : parts) out += p + ";";
    return fnv64(out);
}

// Key of a step's result: its fields, the keys of the clips it reads and stamps of the files it names.
std::string stepKey(const json::Value& v, const std::string& base, const std::map<std::string, std::string>& clipKeys) {
    const WorkflowOp* op = findWorkflowOp(v.at("op").string);
    std::string sig = v.at("op").string + "\n" + base + "\n";
    std::vector<std::pair<std::string, const json::Value*>> fields;
    for (const auto& [k, x] : v.object)
        if (k != "op" && k != "out" && k != "ui" && k != "label" && k != "comment" && (k.empty() || k[0] != '_')) fields.emplace_back(k, &x);
    std::sort(fields.begin(), fields.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [k, x] : fields) {
        const WorkflowField* f = nullptr;
        if (op)
            for (const WorkflowField& g : op->fields)
                if (g.name == k) f = &g;
        // Clips count by what made them, not by name, so renaming a clip keeps the results.
        sig += k + "=" + (f && (f->type == "clip" || f->type == "clips") ? std::string() : json::dump(*x)) + "\n";
        if (!f) continue;
        auto names = [&] {
            std::vector<std::string> out;
            if (x->isString()) out.push_back(x->string);
            else
                for (const auto& e : x->array)
                    if (e.isString()) out.push_back(e.string);
            return out;
        };
        if (f->type == "clip" || f->type == "clips") {
            for (const std::string& n : names()) {
                auto it = clipKeys.find(n);
                sig += "  <" + (it == clipKeys.end() ? "?" + n : it->second) + "\n";
            }
        } else if (f->type == "file" || f->type == "dir" || f->type == "model" || f->type == "loras") {
            for (std::string n : names()) {
                if (f->type == "loras") {
                    const std::size_t colon = n.rfind(':');
                    if (colon != std::string::npos && colon > 1) n = n.substr(0, colon);
                }
                const std::filesystem::path p(n);
                sig += "  @" + fileStamp(p.is_absolute() ? n : (std::filesystem::path(base) / p).string()) + "\n";
            }
        }
    }
    return fnv64(sig) + fnv64(sig + "#");
}
} // namespace

std::map<std::string, Clip> runWorkflow(const std::string& text, const std::string& base, const WorkflowEvents& events) {
    return runWorkflow(text, base, events, nullptr);
}

std::map<std::string, Clip> runWorkflow(const std::string& text, const std::string& base, const WorkflowEvents& events,
                                        WorkflowResultCache* results) {
    return runWorkflow(text, base, events, results, WorkflowLimits{});
}

namespace {
const std::set<std::string> generationOps{"txt2img", "img2img", "inpaint", "t2v", "morph", "zoom", "outpaint"};

// The files a step names, as written (LoRA scales removed).
void stepFiles(const json::Value& v, const std::function<void(const WorkflowField&, const std::string&)>& f) {
    const json::Value* opv = v.find("op");
    const WorkflowOp* op = opv && opv->isString() ? findWorkflowOp(opv->string) : nullptr;
    if (!op) return;
    for (const WorkflowField& field : op->fields) {
        if (field.type != "model" && field.type != "file" && field.type != "dir" && field.type != "loras") continue;
        const json::Value* x = v.find(field.name);
        if (!x) continue;
        auto one = [&](const std::string& value) {
            std::string name = value;
            if (field.type == "loras") {
                const std::size_t colon = name.rfind(':');
                if (colon != std::string::npos && colon + 1 < name.size() && (std::isdigit(static_cast<unsigned char>(name[colon + 1])) || name[colon + 1] == '.'))
                    name = name.substr(0, colon);
            }
            if (!name.empty()) f(field, name);
        };
        if (x->isString()) one(x->string);
        else if (x->isArray())
            for (const auto& e : x->array)
                if (e.isString()) one(e.string);
    }
}

void checkClip(const WorkflowLimits& limits, const Clip& clip, const std::string& what) {
    if (limits.maxFrames && clip.size() > limits.maxFrames)
        throw WorkflowLimitExceeded(what + " has " + std::to_string(clip.size()) + " frames, more than the limit of " + std::to_string(limits.maxFrames));
    if (limits.maxPixels)
        for (const Image& im : clip)
            if (im.width * im.height > limits.maxPixels)
                throw WorkflowLimitExceeded(what + " is " + std::to_string(im.width) + " x " + std::to_string(im.height) + ", more than the limit of " +
                                            std::to_string(limits.maxPixels) + " pixels");
}

// Sets the picture readers' bound for the run.
struct DecodeLimit {
    std::size_t before;
    explicit DecodeLimit(std::size_t pixels) : before(maxDecodePixels()) {
        if (pixels) setMaxDecodePixels(pixels);
    }
    ~DecodeLimit() { setMaxDecodePixels(before); }
};
} // namespace

std::vector<std::string> checkWorkflowLimits(const std::string& text, const std::string& base, const WorkflowLimits& limits) {
    std::vector<std::string> out;
    json::Value doc;
    try {
        doc = json::parse(text);
    } catch (const std::exception& e) {
        return {e.what()};
    }
    const json::Value* steps = doc.find("steps");
    if (!steps || !steps->isArray()) return out;
    if (limits.maxSteps && steps->array.size() > limits.maxSteps)
        out.push_back(std::to_string(steps->array.size()) + " steps, more than the limit of " + std::to_string(limits.maxSteps));
    std::size_t index = 0;
    for (const json::Value& v : steps->array) {
        ++index;
        const std::string where = "step " + std::to_string(index) + ": ";
        const json::Value* opv = v.find("op");
        const std::string op = opv && opv->isString() ? opv->string : "";
        const double w = v.numberOr("width", 512), h = v.numberOr("height", 512);
        if (limits.maxPixels && (v.find("width") || v.find("height") || generationOps.count(op)) && w * h > static_cast<double>(limits.maxPixels))
            out.push_back(where + "width x height is more than the limit of " + std::to_string(limits.maxPixels) + " pixels");
        if (limits.maxPixels && v.find("size") && op == "depth") {
            const double sz = v.numberOr("size", 518);
            if (sz * sz > static_cast<double>(limits.maxPixels)) out.push_back(where + "size is more than the pixel limit");
        }
        if (limits.maxDenoiseSteps && generationOps.count(op) && v.numberOr("steps", 0) > static_cast<double>(limits.maxDenoiseSteps))
            out.push_back(where + "more than " + std::to_string(limits.maxDenoiseSteps) + " denoising steps");
        if (limits.maxFrames) {
            double frames = v.numberOr("frames", 0);
            if (op == "morph") frames *= std::max<double>(1, static_cast<double>(v.find("prompts") ? v.at("prompts").array.size() : 1));
            if (op == "outpaint") frames *= v.numberOr("keyframes", 6);
            if (frames > static_cast<double>(limits.maxFrames)) out.push_back(where + "more frames than the limit of " + std::to_string(limits.maxFrames));
        }
        if (limits.sandboxed()) {
            stepFiles(v, [&](const WorkflowField& field, const std::string& value) {
                const std::filesystem::path p(value);
                const std::string full = p.is_absolute() ? value : (std::filesystem::path(base) / p).string();
                try {
                    if (op == "save" && field.name == "path") allowWrite(&limits, full);
                    else allowRead(&limits, full);
                } catch (const std::exception& e) {
                    out.push_back(where + e.what());
                }
            });
        }
    }
    return out;
}

std::string mapWorkflowFiles(const std::string& text, const std::function<std::string(const WorkflowField&, const std::string&)>& map) {
    json::Value doc = json::parse(text);
    json::Value* steps = nullptr;
    for (auto& [k, x] : doc.object)
        if (k == "steps") steps = &x;
    if (!steps || !steps->isArray()) return text;
    for (json::Value& v : steps->array) {
        const json::Value* opv = v.find("op");
        const WorkflowOp* op = opv && opv->isString() ? findWorkflowOp(opv->string) : nullptr;
        if (!op) continue;
        for (auto& [k, x] : v.object) {
            const WorkflowField* field = nullptr;
            for (const WorkflowField& f : op->fields)
                if (f.name == k) field = &f;
            if (!field || (field->type != "model" && field->type != "file" && field->type != "dir" && field->type != "loras")) continue;
            auto one = [&](json::Value& e) {
                if (!e.isString() || e.string.empty()) return;
                std::string name = e.string, suffix;
                if (field->type == "loras") {
                    const std::size_t colon = name.rfind(':');
                    if (colon != std::string::npos && colon + 1 < name.size() && (std::isdigit(static_cast<unsigned char>(name[colon + 1])) || name[colon + 1] == '.')) {
                        suffix = name.substr(colon);
                        name = name.substr(0, colon);
                    }
                }
                e.string = map(*field, name) + suffix;
            };
            if (x.isString()) one(x);
            else if (x.isArray())
                for (json::Value& e : x.array) one(e);
        }
    }
    return json::dump(doc);
}

std::map<std::string, Clip> runWorkflow(const std::string& text, const std::string& base, const WorkflowEvents& events,
                                        WorkflowResultCache* results, const WorkflowLimits& limits) {
    const auto& log = events.log;
    const std::vector<std::string> problems = checkWorkflow(text);
    if (!problems.empty()) {
        std::string message = "workflow:";
        for (const std::string& p : problems) message += "\n  " + p;
        throw std::invalid_argument(message);
    }
    if (const std::vector<std::string> over = checkWorkflowLimits(text, base, limits); !over.empty()) {
        std::string message = "workflow limits:";
        for (const std::string& p : over) message += "\n  " + p;
        throw WorkflowLimitExceeded(message);
    }
    const DecodeLimit decodeLimit(limits.maxPixels);
    const GpuWeightScope weightScope;  // layer weights stay on the GPU for the run
    const auto deadline = limits.maxSeconds > 0
                              ? std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(limits.maxSeconds))
                              : std::chrono::steady_clock::time_point::max();
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

    std::map<std::string, std::string> clipKeys;  // clip name -> key of the step that made it
    std::size_t index = 0;
    for (const json::Value& v : steps->array) {
        ++index;
        StepProgress progress{&events, index, steps->array.size(), v.at("op").string, deadline, limits.maxSeconds};
        const std::string& stepOp = v.at("op").string;
        const std::string key = results ? stepKey(v, base, clipKeys) : std::string();
        const std::string outField = v.find("out") && v.at("out").isString() ? v.at("out").string : std::string();
        if (results && stepOp != "save") {
            if (std::shared_ptr<const Clip> hit = results->find(key); hit && !hit->empty()) {
                progress.stage("cached");
                const std::string clipName = outField.empty() ? stepOp : outField;
                clips[clipName] = *hit;
                clipKeys[clipName] = key;
                if (cache) releaseModelsExcept(index < needed.size() ? needed[index] : std::set<std::string>{});
                if (events.output) events.output(index, stepOp, clipName, clips[clipName], 0.0);
                if (log) log("step " + std::to_string(index) + " " + stepOp + ": cached");
                continue;
            }
        }
        progress.stage("start");
        Step s{v, v.at("op").string, base, clips, &progress, &limits};
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
                if (limits.maxFrames && files.size() > limits.maxFrames) throw WorkflowLimitExceeded("load: more frames than the limit");
                for (const auto& f : files) {
                    allowRead(&limits, f);
                    out.push_back(toRgb(loadImage(f)));
                }
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
            o.vae = io.vae;
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
            if (limits.maxPixels)
                for (const Image& im : in)
                    if (im.width * im.height * up->scale() * up->scale() > limits.maxPixels) throw WorkflowLimitExceeded("upscale: the result would be bigger than the pixel limit");
            for (const Image& im : in) {
                progress.frame(static_cast<std::size_t>(&im - in.data()), in.size());
                out.push_back(up->upscale(im));
            }
        } else if (op == "onnx") {
            const std::string path = s.path("model");
            const auto model = cachedModel<onnx::Model>("onnx|" + modelTag(path), modelTag(path),
                                                         [&] { return std::make_shared<onnx::Model>(onnx::Model::load(path)); });
            onnx::OnnxPipeline pipe = onnx::OnnxPipeline::fromModel(*model, onnx::onnxManifest(path));
            if (s.has("kind") && s.str("kind", "auto") != "auto") pipe.kind = s.str("kind", "auto");
            if (s.has("apply")) pipe.applyMask = s.flag("apply", false);
            if (s.has("invert")) pipe.invert = s.flag("invert", false);
            if (s.has("tile")) pipe.tileOverlap = static_cast<int>(s.num("tile", -1));
            const Clip& in = s.clip("in");
            for (const Image& im : in) {
                progress.frame(static_cast<std::size_t>(&im - in.data()), in.size());
                out.push_back(pipe.run(*model, im));
            }
        } else if (op == "interpolate") {
            const std::size_t passes = s.size("passes", 1), n = s.clip("in").size();
            const std::size_t made = passes > 16 ? std::numeric_limits<std::size_t>::max()
                                     : s.flag("loop", false) ? n << passes : n ? ((n - 1) << passes) + 1 : 0;
            if (limits.maxFrames && made > limits.maxFrames)
                throw WorkflowLimitExceeded("interpolate: the result would have more frames than the limit");
            out = smallModel<FrameInterpolator>("rife", s.path("model"))->interpolate(s.clip("in"), passes, s.flag("loop", false));
        } else {
            throw std::invalid_argument("workflow: unknown op \"" + op + "\" in step " + std::to_string(index));
        }
        if (cache) releaseModelsExcept(index < needed.size() ? needed[index] : std::set<std::string>{});
        checkClip(limits, out, "step " + std::to_string(index) + " (" + op + ")");
        const std::string outName = s.str("out");
        const std::string clipName = out.empty() ? std::string() : (outName.empty() ? op : outName);
        if (!out.empty()) {
            if (results) results->put(key, out);
            clipKeys[clipName] = key;
            clips[clipName] = std::move(out);
        }
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
