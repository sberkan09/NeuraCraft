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

#include <functional>
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
//   - upscale {model, in} - interpolate {model, in, passes, loop} - concat {in: [clips]}.
// Relative paths are taken from `baseDirectory`. `log` receives one line per step.
// Returns the named clips after the last step.
std::map<std::string, Clip> runWorkflow(const std::string& json, const std::string& baseDirectory = ".",
                                        const std::function<void(const std::string&)>& log = {});

} // namespace nc

#endif // NEURACRAFT_WORKFLOW_H
