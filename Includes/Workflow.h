#ifndef NEURACRAFT_WORKFLOW_H
#define NEURACRAFT_WORKFLOW_H

// Image / video workflows described in JSON and run step by step, each step
// reading and writing named clips (a picture is a one-frame clip):
//
//   {"steps": [
//     {"op": "txt2img", "model": "data/animatediff", "prompt": "a red fox", "steps": 4,
//      "sampler": "lcm", "lora": ["data/loras/lcm-lora-sdv1-5.safetensors"], "out": "fox"},
//     {"op": "t2v", "model": "data/animatediff", "prompt": "a red fox walking", "init": "fox", "out": "clip"},
//     {"op": "interpolate", "model": "data/interpolate/rife-v4-flownet.safetensors", "in": "clip", "out": "clip"},
//     {"op": "save", "in": "clip", "path": "fox.mov", "fps": 16}]}
//
// See runWorkflow for the operations and their fields.

#include "Image.h"

#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace nc {

using Clip = std::vector<Image>;

// Runs the workflow (JSON text). Operations ("op"), with "in" / "out" clip names:
//   load {path | dir} - save {in, path, fps, quality} (.png / .jpg for one frame; .gif / .png / .mov for clips; a
//   directory path ending in '/' writes PNG frames) - resize {in, width, height} - crop {in, width, height} (centred)
//   txt2img / img2img / inpaint {model, prompt, negative, steps, seed, guidance, width, height, strength, sampler,
//   tiny_vae, lora: [file or "file:scale"], controlnet, control (clip), ip_adapter, image_prompt (clip), ip_scale,
//   in (img2img / inpaint), mask (clip)} - t2v {model, prompt, frames, steps, seed, width, height, init (clip),
//   strength, motion, sampler, lora, controlnet, control, ip_adapter, image_prompt, prompt_at: [[frame, "prompt"]],
//   context, overlap} - morph {model, prompts, seeds, frames, steps} - zoom {model, prompts, frames, zoom, strength,
//   steps, seed, in} - outpaint {model, prompts, keyframes, shrink, frames, steps, seed, in, sampler, lora, guidance}
//   - canny {in, low, high} - depth {model, in, size} - parallax {model, in, frames, amplitude, zoom}
//   - upscale {model, in} - interpolate {model, in, passes, loop} - concat {in: [clips]}
//   - blend {in: [a, b], mode: mix | add | multiply | difference, weight (0.5), mask (clip)}: b over a frame by
//   frame (b resized to a; a one-frame clip is used for every frame), e.g. a skip connection from an earlier step.
// Any step can read any clip made before it, so a workflow is a graph run in the listed order.
// Models are cached between steps ("cache": false at the top level turns it off): each one is loaded by the
// first step that needs it and released after the last one (ModelCache.h).
// Relative paths are taken from `baseDirectory`. `log` receives one line per step.
// Returns the named clips after the last step.
std::map<std::string, Clip> runWorkflow(const std::string& json, const std::string& baseDirectory = ".",
                                        const std::function<void(const std::string&)>& log = {});

// ---- Schema: what each operation takes (drives validation, docs and the Studio UI) ----

struct WorkflowField {
    std::string name;
    // int, float, bool, string, text (a prompt), enum (see options), clip (a clip name), clips (names),
    // model (see model), loras ("file" / "file:scale" list), file, dir, strings, ints, keyframes ([[frame, "prompt"]])
    std::string type;
    std::string defaultJson;            // JSON text of the default, "" = none
    std::string doc;
    std::string model;                  // for model fields: sd, controlnet, motion, lora, taesd, ip-adapter,
                                        // image-encoder, depth, esrgan, rife
    std::vector<std::string> options;   // for enum fields
    double min = std::numeric_limits<double>::quiet_NaN(), max = std::numeric_limits<double>::quiet_NaN();
    bool required = false;
};

struct WorkflowOp {
    std::string name, title, group, doc;  // group: input, edit, image, video, analysis, enhance
    bool output = true;                   // makes a clip (named by "out")
    std::vector<WorkflowField> fields;
};

const std::vector<WorkflowOp>& workflowOps();
const WorkflowOp* findWorkflowOp(const std::string& name);  // nullptr if unknown
// {"version": 1, "ops": [{name, title, group, doc, output, fields: [{name, type, doc, required, default?,
// model?, options?, min?, max?}]}]}
std::string workflowSchemaJson();

// Problems with a workflow before running it: unknown ops or fields (with a "did you mean"), wrong
// types, numbers out of range, missing required fields. Empty when it looks runnable. Besides its
// fields a step may carry "op", "ui" (editor data such as a node position), "label", "comment" and
// keys starting with "_"; they are ignored.
std::vector<std::string> checkWorkflow(const std::string& json);

// Models under a folder, recognised by their config files or tensor names.
struct ModelInfo {
    std::string path, kind, name;  // kind: as WorkflowField::model, plus sd-inpaint
    std::size_t bytes = 0;
};
std::vector<ModelInfo> findModels(const std::string& root, std::size_t maxDepth = 4);
std::string modelsJson(const std::vector<ModelInfo>& models);  // [{path, kind, name, bytes}]

} // namespace nc

#endif // NEURACRAFT_WORKFLOW_H
