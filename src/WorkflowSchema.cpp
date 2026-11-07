#include "Json.h"
#include "SafeTensors.h"
#include "Workflow.h"
#include "Checkpoint.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {

// Field helpers: name, type, default (JSON text, "" = none), doc; then the model kind
// for "model" fields, the options for "enum" fields.
WorkflowField field(const char* name, const char* type, const char* def, const char* doc, bool required = false) {
    WorkflowField f;
    f.name = name;
    f.type = type;
    f.defaultJson = def;
    f.doc = doc;
    f.required = required;
    return f;
}
WorkflowField model(const char* name, const char* kind, const char* doc, bool required = false) {
    WorkflowField f = field(name, "model", "", doc, required);
    f.model = kind;
    return f;
}
WorkflowField choice(const char* name, std::vector<std::string> options, const char* def, const char* doc) {
    WorkflowField f = field(name, "enum", def, doc);
    f.options = std::move(options);
    return f;
}
WorkflowField number(const char* name, const char* type, const char* def, const char* doc, double lo, double hi) {
    WorkflowField f = field(name, type, def, doc);
    f.min = lo;
    f.max = hi;
    return f;
}

// Fields every Stable Diffusion step shares.
std::vector<WorkflowField> diffusionFields(const char* steps, const char* strength, bool prompt = true) {
    std::vector<WorkflowField> f;
    f.push_back(model("model", "sd", "Stable Diffusion model: a diffusers folder or a single-file .safetensors checkpoint", true));
    if (prompt) f.push_back(field("prompt", "text", "", "What to draw", true));
    f.push_back(field("negative", "string", "", "What to avoid (used with guidance > 1)"));
    f.push_back(number("steps", "int", steps, "Denoising steps", 1, 200));
    f.push_back(number("seed", "int", "0", "Noise seed", 0, 4294967295.0));
    f.push_back(number("guidance", "float", "0", "Classifier-free guidance scale (0 or 1 = off)", 0, 30));
    f.push_back(number("width", "int", "512", "Width in pixels (multiple of 8)", 64, 2048));
    f.push_back(number("height", "int", "512", "Height in pixels (multiple of 8)", 64, 2048));
    if (strength[0]) f.push_back(number("strength", "float", strength, "How much of the input is redrawn (0..1)", 0, 1));
    f.push_back(choice("sampler", {"euler", "lcm"}, "\"euler\"", "Sampler (lcm with an LCM model or LoRA)"));
    f.push_back(field("int8", "bool", "true", "int8 weights (half the memory, slightly less exact)"));
    f.push_back(field("lora", "loras", "[]", "LoRA files, \"file\" or \"file:scale\""));
    f.push_back(model("tiny_vae", "taesd", "TAESD decoder (default: the model's taesd folder if present)"));
    f.push_back(model("vae", "vae", "Another VAE: a diffusers VAE folder or a VAE .safetensors (default: the model's own)"));
    f.push_back(model("controlnet", "controlnet", "ControlNet model folder"));
    f.push_back(field("control", "clip", "", "Control picture(s) for the ControlNet"));
    f.push_back(number("control_scale", "float", "1", "ControlNet strength", 0, 2));
    f.push_back(model("ip_adapter", "ip-adapter", "IP-Adapter weights (image prompt)"));
    f.push_back(model("image_encoder", "image-encoder", "CLIP vision encoder (default: beside the IP-Adapter)"));
    f.push_back(field("image_prompt", "clip", "", "Picture used as a prompt by the IP-Adapter"));
    f.push_back(number("ip_scale", "float", "0.6", "IP-Adapter strength", 0, 2));
    return f;
}

std::vector<WorkflowField> plus(std::vector<WorkflowField> a, std::vector<WorkflowField> b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::vector<WorkflowOp> buildOps() {
    std::vector<WorkflowOp> ops;
    auto add = [&](const char* name, const char* title, const char* group, const char* doc, bool output,
                   std::vector<WorkflowField> fields) {
        WorkflowOp op;
        op.name = name;
        op.title = title;
        op.group = group;
        op.doc = doc;
        op.output = output;
        op.fields = std::move(fields);
        if (output) op.fields.push_back(field("out", "string", "", "Name of what this step makes (default: the op's name)"));
        ops.push_back(std::move(op));
    };

    add("load", "Load", "input", "Pictures from a file or a folder of frames, or a 3D model (.glb, .obj, .ply)", true,
        {field("path", "file", "", "Picture (.png, .ppm) or 3D model (.glb, .obj, .ply) file"), field("dir", "dir", "", "Folder of frames, in name order")});
    ops.back().makes = "any";
    add("save", "Save", "input",
        "Writes a clip: .png / .jpg for one frame, .gif / .png / .mov for a clip, a folder ending in / for frames; a 3D model: .glb, .obj, .ply",
        false,
        {field("in", "clip", "", "Clip or 3D model to write", true), field("path", "file", "", "Output file", true),
         number("fps", "float", "8", "Frames per second", 1, 120), number("quality", "int", "90", "JPEG / MJPEG quality", 1, 100)});
    add("resize", "Resize", "edit", "Scales every frame", true,
        {field("in", "clip", "", "Input", true), number("width", "int", "", "Width (default: unchanged)", 1, 16384),
         number("height", "int", "", "Height (default: unchanged)", 1, 16384)});
    add("crop", "Crop", "edit", "Centre crop (scaled to cover first)", true,
        {field("in", "clip", "", "Input", true), number("width", "int", "", "Width", 1, 16384), number("height", "int", "", "Height", 1, 16384)});
    add("concat", "Concat", "edit", "Clips one after another", true, {field("in", "clips", "", "Clips in order", true)});
    {
        WorkflowField mode = choice("mode", {"mix", "add", "multiply", "difference"}, "\"mix\"", "How b is combined with a");
        add("blend", "Blend", "edit", "b over a frame by frame (b resized to a)", true,
            {field("in", "clips", "", "Two clips: [a, b]", true), mode, number("weight", "float", "0.5", "Amount of b", 0, 1),
             field("mask", "clip", "", "Where to blend (white = full weight)")});
    }
    add("txt2img", "Text to image", "image", "One picture from a prompt", true, diffusionFields("1", ""));
    add("img2img", "Image to image", "image", "Redraws a picture", true,
        plus({field("in", "clip", "", "Picture to redraw", true)}, diffusionFields("2", "0.5")));
    add("inpaint", "Inpaint", "image", "Redraws the masked part of a picture (needs an inpainting model)", true,
        plus({field("in", "clip", "", "Picture", true), field("mask", "clip", "", "Mask (white = redraw)", true),
              number("feather", "float", "4", "Soft edge of the paste-back, in pixels", 0, 64)},
             diffusionFields("25", "1")));
    {
        std::vector<WorkflowField> f = diffusionFields("4", "0.75");
        for (WorkflowField& x : f)
            if (x.name == "guidance") x.defaultJson = "1";
        f.push_back(model("motion", "motion", "Motion module (default: the model's motion folder)"));
        f.push_back(number("frames", "int", "16", "Number of frames", 1, 1024));
        f.push_back(field("init", "clip", "", "Picture or clip to start from (with strength)"));
        f.push_back(field("prompt_at", "keyframes", "[]", "Prompt travel: [[frame, \"prompt\"], ...]"));
        f.push_back(number("context", "int", "16", "Frames per sampling window", 1, 64));
        f.push_back(number("overlap", "int", "4", "Frames shared by neighbouring windows", 0, 32));
        f.push_back(choice("beta_schedule", {"linear", "scaled_linear"}, "\"linear\"", "Noise schedule of the motion model"));
        add("t2v", "Text to video", "video", "AnimateDiff clip from a prompt", true, f);
    }
    add("morph", "Prompt morph", "video", "Smooth morph through prompts", true,
        plus({field("prompts", "strings", "", "Prompts in order", true), field("seeds", "ints", "[]", "Seeds (one per prompt, or one)"),
              number("frames", "int", "12", "Frames per segment", 1, 256), field("loop", "bool", "true", "Morph back to the first prompt")},
             diffusionFields("1", "", false)));
    add("zoom", "Zoom", "video", "Endless zoom by redrawing each frame", true,
        plus({field("prompts", "strings", "", "Prompts (spread over the frames)", true), field("in", "clip", "", "First frame (default: drawn)"),
              number("frames", "int", "24", "Frames", 1, 1024), number("zoom", "float", "1.04", "Zoom per frame (> 1 zooms in)", 0.5, 2)},
             diffusionFields("4", "0.5", false)));
    add("outpaint", "Outpaint zoom", "video", "Zoom out by painting the border of each keyframe", true,
        plus({field("prompts", "strings", "", "Prompts", true), field("in", "clip", "", "Innermost picture (default: drawn)"),
              number("keyframes", "int", "6", "Keyframes", 1, 64), number("shrink", "float", "0.75", "Keyframe scale step", 0.3, 0.95),
              number("frames", "int", "12", "Frames between keyframes", 1, 256), field("zoom_in", "bool", "false", "Play inward")},
             diffusionFields("25", "0.9", false)));
    add("scribble", "Scribble", "analysis",
        "A drawing as the scribble ControlNet's input: its strokes white on black, thickened (dark strokes on light paper by default)", true,
        {field("in", "clip", "", "Drawing", true), field("dark_strokes", "bool", "true", "The strokes are darker than the paper"),
         number("threshold", "int", "128", "Brightness that separates strokes from paper", 1, 254),
         number("thickness", "int", "2", "Pixels added on every side of a stroke", 0, 32)});
    add("canny", "Canny edges", "analysis", "Edge map for a Canny ControlNet", true,
        {field("in", "clip", "", "Input", true), number("low", "float", "100", "Low threshold", 0, 1000),
         number("high", "float", "200", "High threshold", 0, 1000)});
    add("depth", "Depth", "analysis", "Depth map (Depth Anything V2)", true,
        {model("model", "depth", "Depth Anything V2 folder", true), field("in", "clip", "", "Input", true),
         number("size", "int", "518", "Network input size (multiple of 14)", 14, 2044)});
    add("parallax", "3D photo", "video", "A moving-camera clip from one picture and its depth", true,
        {model("model", "depth", "Depth Anything V2 folder", true), field("in", "clip", "", "Picture", true),
         number("frames", "int", "48", "Frames", 2, 1024), number("amplitude", "float", "0.03", "Camera sway (fraction of width)", 0, 0.2),
         number("zoom", "float", "0.06", "Dolly-in of near pixels", 0, 0.5), field("loop", "bool", "true", "Return to the start"),
         number("size", "int", "518", "Depth network input size", 14, 2044)});
    add("upscale", "Upscale", "enhance", "Real-ESRGAN (compact) super-resolution", true,
        {model("model", "esrgan", "Real-ESRGAN weights", true), field("in", "clip", "", "Input", true)});
    add("onnx", "ONNX model", "enhance",
        "Any image model exported to ONNX (background removal, depth, upscaling, restoration, style): pre- and post-processing from "
        "the manifest next to it (<model>.json), or the fields here",
        true,
        {model("model", "onnx", "The .onnx file", true), field("in", "clip", "", "Input", true),
         choice("kind", {"auto", "image", "mask", "depth"}, "\"auto\"", "What the output is (auto: by channel count)"),
         field("apply", "bool", "false", "Mask models: cut the input out with the mask (transparent background)"),
         field("invert", "bool", "false", "Invert the output"),
         number("tile", "int", "-1", "Fixed-size upscalers: tile overlap in pixels (-1 automatic, 0 never tile)", -1, 512)});
    add("interpolate", "Interpolate", "enhance", "RIFE in-between frames (each pass doubles the frame rate)", true,
        {model("model", "rife", "RIFE weights", true), field("in", "clip", "", "Clip", true),
         number("passes", "int", "1", "Doublings", 1, 4), field("loop", "bool", "false", "Also between the last and first frame")});
    add("depth_mesh", "Depth to 3D", "3d", "A 3D relief of a picture, raised by its depth and textured with it", true,
        {model("model", "depth", "Depth Anything V2 folder", true), field("in", "clip", "", "Picture", true),
         number("grid", "int", "256", "Vertices along the longer side", 16, 1024), number("depth", "float", "0.4", "Relief height (fraction of the height)", 0, 2),
         number("edges", "float", "0.08", "Largest depth step inside a triangle (0..1; 0 keeps all, joining foreground and background)", 0, 1),
         number("size", "int", "518", "Depth network input size", 14, 2044)});
    ops.back().makes = "mesh";
    add("image_to_3d", "Image to 3D", "3d",
        "A whole 3D model from one picture, made by a 3D service (TRELLIS.2 on a GPU); a transparent background is the object's outline", true,
        {field("in", "clip", "", "Picture (with a transparent background: its alpha is the object)", true),
         choice("quality", {"fast", "standard", "high"}, "\"standard\"", "fast: 512^3 voxels, standard: 1024^3, high: 1536^3"),
         number("seed", "int", "42", "Seed", 0, 2147483647), number("texture", "int", "2048", "Texture size", 512, 4096),
         number("faces", "int", "500000", "Triangle budget", 10000, 2000000)});
    ops.back().makes = "mesh";
    add("rig", "Auto-rig", "3d",
        "A skeleton and skin weights for a 3D model, ready to pose and animate (SkinTokens on a GPU, by a rigging service); "
        "saved as .glb",
        true,
        {field("in", "mesh", "", "3D model", true), number("seed", "int", "42", "Seed (another seed, another skeleton)", 0, 2147483647),
         field("postprocess", "bool", "true", "Smooth the skin weights over the model's volume")});
    ops.back().makes = "mesh";
    add("animate", "Animate", "3d",
        "Moves a rigged figure with motion capture (a .bvh file, as CMU's free library has them): legs, arms, spine and head found "
        "on both skeletons, the hips carried along; saved as an animated .glb",
        true,
        {field("in", "mesh", "", "Rigged 3D model (from rig)", true), model("capture", "mocap", "Motion capture (.bvh)", true),
         number("fps", "float", "30", "Frames a second", 1, 120), number("start", "float", "0", "Seconds into the motion", 0, 36000),
         number("seconds", "float", "0", "How much of the motion (0: all of it)", 0, 600), number("speed", "float", "1", "Playback speed", 0.1, 10),
         field("in_place", "bool", "false", "Keep the figure in place (no walking away)")});
    ops.back().makes = "mesh";
    add("render", "Render 3D", "3d", "Pictures of a 3D model: a still, or a turntable clip", true,
        {field("in", "mesh", "", "3D model", true), number("width", "int", "512", "Width", 16, 4096), number("height", "int", "512", "Height", 16, 4096),
         number("frames", "int", "1", "Frames (more than 1: a turntable)", 1, 720), number("turn", "float", "360", "Degrees turned over the frames", -1080, 1080),
         number("yaw", "float", "0", "Starting angle around the model (degrees)", -360, 360), number("pitch", "float", "15", "Camera height (degrees)", -89, 89),
         number("fov", "float", "35", "Field of view (degrees)", 5, 120), choice("background", {"white", "grey", "black"}, "\"white\"", "Background"),
         field("lighting", "bool", "true", "Shade by a light (off: flat colours)"),
         choice("stereo", {"none", "sbs", "half_sbs", "tb", "anaglyph"}, "\"none\"",
                "Stereo 3D: both eyes' views side by side (half_sbs: in the same width), top and bottom, or red-cyan"),
         number("eye", "float", "3", "Stereo: degrees between the two eyes' cameras", 0, 20),
         field("play", "bool", "true", "Animated models: play the animation over the frames (frames default to all of it, turn to 0)")});
    add("stereo", "Stereo 3D", "3d",
        "Stereoscopic 3D from pictures or video and their depth: left and right eye views for 3D TVs, VR players and "
        "red-cyan glasses",
        true,
        {model("model", "depth", "Depth Anything V2 folder", true), field("in", "clip", "", "Pictures or video", true),
         choice("layout", {"half_sbs", "sbs", "tb", "anaglyph"}, "\"half_sbs\"",
                "half_sbs: both views in the original width (3D TVs, most VR players); sbs: twice as wide; tb: top and bottom; "
                "anaglyph: red-cyan glasses"),
         number("strength", "float", "0.03", "Depth effect: largest disparity as a fraction of the width", 0, 0.1),
         number("focus", "float", "0.5", "Screen plane (0: everything comes out of the screen, 1: everything behind it)", 0, 1),
         number("smooth", "float", "0.8", "Video: depth range smoothing between frames (0: none)", 0, 0.99),
         number("size", "int", "518", "Depth network input size", 14, 2044)});
    return ops;
}

json::Value fieldJson(const WorkflowField& f) {
    std::vector<std::pair<std::string, json::Value>> m{{"name", json::makeString(f.name)},
                                                         {"type", json::makeString(f.type)},
                                                         {"doc", json::makeString(f.doc)},
                                                         {"required", json::makeBool(f.required)}};
    if (!f.defaultJson.empty()) m.emplace_back("default", json::parse(f.defaultJson));
    if (!f.model.empty()) m.emplace_back("model", json::makeString(f.model));
    if (!f.options.empty()) {
        std::vector<json::Value> o;
        for (const auto& x : f.options) o.push_back(json::makeString(x));
        m.emplace_back("options", json::makeArray(o));
    }
    if (f.min == f.min) m.emplace_back("min", json::makeNumber(f.min));
    if (f.max == f.max) m.emplace_back("max", json::makeNumber(f.max));
    return json::makeObject(m);
}

// Keys any step may carry besides its fields: the op, editor data and comments.
bool freeKey(const std::string& k) { return k == "op" || k == "ui" || k == "label" || k == "comment" || (!k.empty() && k[0] == '_'); }

std::size_t editDistance(const std::string& a, const std::string& b) {
    std::vector<std::size_t> row(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i) {
        std::size_t prev = row[0];
        row[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const std::size_t cur = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] == b[j - 1] ? 0 : 1)});
            prev = cur;
        }
    }
    return row[b.size()];
}

std::string readText(const std::filesystem::path& p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Kind of a model folder from its config files, "" if not a model.
std::string folderKind(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(dir / "model_index.json", ec) || fs::exists(dir / "unet" / "config.json", ec)) {
        if (fs::exists(dir / "text_encoder_2" / "config.json", ec)) return "sdxl";
        try {
            const json::Value c = json::parse(readText(dir / "unet" / "config.json"));
            if (c.numberOr("in_channels", 4) == 9) return "sd-inpaint";
        } catch (const std::exception&) {
        }
        return "sd";
    }
    if (!fs::exists(dir / "config.json", ec)) return "";
    try {
        const json::Value c = json::parse(readText(dir / "config.json"));
        const json::Value* cls = c.find("_class_name");
        const std::string name = cls && cls->isString() ? cls->string : "";
        if (name == "ControlNetModel") return "controlnet";
        if (name == "AutoencoderTiny") return "taesd";
        if (name == "AutoencoderKL") return "vae";
        if (const json::Value* arch = c.find("architectures"))
            for (const auto& a : arch->array) {
                if (a.string.find("DepthAnything") != std::string::npos) return "depth";
                if (a.string.find("CLIPVision") != std::string::npos) return "image-encoder";
            }
        if (const json::Value* type = c.find("model_type")) {
            if (type->string == "depth_anything") return "depth";
            if (type->string == "clip_vision_model") return "image-encoder";
        }
    } catch (const std::exception&) {
    }
    return "";
}

// Kind of a .safetensors file from its tensor names, "" if unknown.
std::string fileKind(const std::filesystem::path& file) {
    try {
        const SafeTensors st(file.string());
        if (st.has("model.diffusion_model.input_blocks.0.0.weight")) {  // a single-file Stable Diffusion checkpoint
            const CheckpointInfo info = inspectCheckpoint(file.string());
            if (info.family == "sdxl") return info.inChannels == 9 ? "" : "sdxl";
            if (info.family != "sd1" && info.family != "sd2") return "";
            return info.inChannels == 9 ? "sd-inpaint" : "sd";
        }
        if (st.has("control_model.input_blocks.0.0.weight")) {  // a ControlNet in the original layout
            const std::string family = inspectCheckpoint(file.string()).family;
            return family == "sd1" || family == "sd2" ? "controlnet" : "";
        }
        if (st.has("encoder.conv_in.weight") && st.has("decoder.conv_out.weight") && st.has("post_quant_conv.weight")) return "vae";
        bool motion = false, lora = false, ip = false, esrgan = false, rife = false;
        for (const std::string& n : st.names()) {
            if (n.find("motion_modules") != std::string::npos) motion = true;
            if (n.rfind("lora_", 0) == 0 || n.find(".lora_A") != std::string::npos || n.find("lora_down") != std::string::npos) lora = true;
            if (n.rfind("image_proj", 0) == 0 || n.find("ip_adapter") != std::string::npos) ip = true;
            if (n.rfind("body.", 0) == 0) esrgan = true;
            if (n.rfind("block0.", 0) == 0 || n.find("encode.") != std::string::npos) rife = true;
        }
        if (motion) return "motion";
        if (ip) return "ip-adapter";
        if (lora) return "lora";
        if (esrgan) return "esrgan";
        if (rife) return "rife";
    } catch (const std::exception&) {
    }
    return "";
}

} // namespace

const std::vector<WorkflowOp>& workflowOps() {
    static const std::vector<WorkflowOp> ops = buildOps();
    return ops;
}

const WorkflowOp* findWorkflowOp(const std::string& name) {
    for (const WorkflowOp& op : workflowOps())
        if (op.name == name) return &op;
    return nullptr;
}

std::string workflowSchemaJson() {
    std::vector<json::Value> ops;
    for (const WorkflowOp& op : workflowOps()) {
        std::vector<json::Value> fields;
        for (const WorkflowField& f : op.fields) fields.push_back(fieldJson(f));
        ops.push_back(json::makeObject({{"name", json::makeString(op.name)},
                                        {"title", json::makeString(op.title)},
                                        {"group", json::makeString(op.group)},
                                        {"doc", json::makeString(op.doc)},
                                        {"output", json::makeBool(op.output)},
                                        {"makes", json::makeString(op.output ? op.makes : std::string())},
                                        {"fields", json::makeArray(fields)}}));
    }
    return json::dump(json::makeObject({{"version", json::makeNumber(1)}, {"ops", json::makeArray(ops)}}));
}

std::vector<std::string> checkWorkflow(const std::string& text) {
    std::vector<std::string> problems;
    json::Value doc;
    try {
        doc = json::parse(text);
    } catch (const std::exception& e) {
        return {std::string("not valid JSON: ") + e.what()};
    }
    const json::Value* steps = doc.find("steps");
    if (!steps || !steps->isArray()) return {"needs a \"steps\" array"};
    std::size_t index = 0;
    for (const json::Value& v : steps->array) {
        ++index;
        const std::string where = "step " + std::to_string(index);
        if (!v.isObject()) {
            problems.push_back(where + ": not an object");
            continue;
        }
        const json::Value* opv = v.find("op");
        if (!opv || !opv->isString()) {
            problems.push_back(where + ": needs \"op\"");
            continue;
        }
        const WorkflowOp* op = findWorkflowOp(opv->string);
        if (!op) {
            std::string best;
            std::size_t bestD = 3;
            for (const WorkflowOp& o : workflowOps())
                if (editDistance(o.name, opv->string) < bestD) {
                    bestD = editDistance(o.name, opv->string);
                    best = o.name;
                }
            problems.push_back(where + ": unknown op \"" + opv->string + "\"" + (best.empty() ? "" : " (did you mean \"" + best + "\"?)"));
            continue;
        }
        const std::string at = where + " (" + op->name + ")";
        for (const auto& kv : v.object) {
            if (freeKey(kv.first)) continue;
            const WorkflowField* f = nullptr;
            for (const WorkflowField& x : op->fields)
                if (x.name == kv.first) f = &x;
            if (!f) {
                std::string best;
                std::size_t bestD = 3;
                for (const WorkflowField& x : op->fields)
                    if (editDistance(x.name, kv.first) < bestD) {
                        bestD = editDistance(x.name, kv.first);
                        best = x.name;
                    }
                problems.push_back(at + ": unknown field \"" + kv.first + "\"" + (best.empty() ? "" : " (did you mean \"" + best + "\"?)"));
                continue;
            }
            const json::Value& x = kv.second;
            const std::string& t = f->type;
            bool ok = true;
            if (t == "int" || t == "float") ok = x.isNumber();
            else if (t == "bool") ok = x.type == json::Value::Type::Bool;
            else if (t == "enum") ok = x.isString() && std::find(f->options.begin(), f->options.end(), x.string) != f->options.end();
            else if (t == "clips" || t == "strings" || t == "loras") ok = x.isString() || x.isArray();
            else if (t == "ints" || t == "keyframes") ok = x.isArray();
            else ok = x.isString();
            if (!ok) {
                std::string want = t == "enum" ? "one of" : "a " + t;
                if (t == "enum")
                    for (const auto& o : f->options) want += " " + o;
                problems.push_back(at + ": \"" + f->name + "\" must be " + want);
            } else if ((t == "int" || t == "float") && f->min == f->min && (x.number < f->min || x.number > f->max)) {
                std::ostringstream r;
                r << at << ": \"" << f->name << "\" = " << x.number << " is outside " << f->min << " .. " << f->max;
                problems.push_back(r.str());
            }
        }
        for (const WorkflowField& f : op->fields)
            if (f.required && !v.find(f.name)) problems.push_back(at + ": needs \"" + f.name + "\"");
        if (op->name == "load" && !v.find("path") && !v.find("dir")) problems.push_back(at + ": needs \"path\" or \"dir\"");
    }
    return problems;
}

std::vector<ModelInfo> findModels(const std::string& root, std::size_t maxDepth) {
    namespace fs = std::filesystem;
    std::vector<ModelInfo> out;
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return out;
    std::vector<std::pair<fs::path, std::size_t>> stack{{fs::path(root), 0}};
    while (!stack.empty()) {
        const auto [dir, depth] = stack.back();
        stack.pop_back();
        for (const auto& e : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
            const fs::path p = e.path();
            if (p.filename().string().rfind('.', 0) == 0) continue;
            if (e.is_directory(ec)) {
                const std::string kind = folderKind(p);
                if (!kind.empty()) {
                    std::uintmax_t bytes = 0;
                    for (const auto& f : fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied, ec))
                        if (f.is_regular_file(ec)) bytes += f.file_size(ec);
                    out.push_back({p.string(), kind, p.filename().string(), static_cast<std::size_t>(bytes)});
                    // Folders inside a model can be models of their own (taesd, image_encoder, motion files).
                }
                if (depth + 1 < maxDepth) stack.push_back({p, depth + 1});
            } else if (e.is_regular_file(ec) && p.extension() == ".onnx") {
                out.push_back({p.string(), "onnx", p.filename().string(), static_cast<std::size_t>(e.file_size(ec))});
            } else if (e.is_regular_file(ec) && p.extension() == ".bvh") {
                out.push_back({p.string(), "mocap", p.filename().string(), static_cast<std::size_t>(e.file_size(ec))});
            } else if (e.is_regular_file(ec) && p.extension() == ".safetensors") {
                if (p.parent_path().filename() == "unet" || p.parent_path().filename() == "vae" ||
                    p.parent_path().filename() == "text_encoder" || folderKind(p.parent_path()) != "")
                    continue;  // a part of a model folder
                const std::string kind = fileKind(p);
                if (!kind.empty()) out.push_back({p.string(), kind, p.filename().string(), static_cast<std::size_t>(e.file_size(ec))});
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const ModelInfo& a, const ModelInfo& b) { return a.path < b.path; });
    return out;
}

std::string modelsJson(const std::vector<ModelInfo>& models) {
    std::vector<json::Value> items;
    for (const ModelInfo& m : models)
        items.push_back(json::makeObject({{"path", json::makeString(m.path)},
                                          {"kind", json::makeString(m.kind)},
                                          {"name", json::makeString(m.name)},
                                          {"bytes", json::makeNumber(static_cast<double>(m.bytes))}}));
    return json::dump(json::makeArray(items));
}

} // namespace nc
