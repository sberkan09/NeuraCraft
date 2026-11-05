#ifndef NEURACRAFT_CHECKPOINT_H
#define NEURACRAFT_CHECKPOINT_H

// Stable Diffusion models in two layouts:
//
//  - a diffusers folder: unet/, vae/, text_encoder/ (config.json + safetensors each), tokenizer/,
//    scheduler/ - what the loaders were written for;
//  - a single-file checkpoint (one .safetensors in the original CompVis / Stability layout, as most
//    fine-tunes are shared): model.diffusion_model.* (UNet), first_stage_model.* (VAE) and
//    cond_stage_model.* (CLIP text encoder, OpenAI layout for SD 1.x, OpenCLIP for SD 2.x).
//
// A single file is used in place: its tensors are renamed to diffusers names (OpenCLIP's fused
// q/k/v split by rows) without copying anything, and each component's config is worked out from
// the tensor sizes. So any SD 1.x / 2.x fine-tune (and its inpainting variant) loads without a
// conversion step. The tokenizer (not part of the file) comes from a folder next to it. What a
// file cannot tell can be set in "<file>.json" next to it: {"unet": {...}, "vae": {...},
// "text_encoder": {...}, "scheduler": {...}, "tokenizer": {"pad_token": "!"}}, each replacing the
// derived settings it names (e.g. "scheduler": {"timestep_spacing": "trailing"} for a Turbo model).

#include "Json.h"
#include "SafeTensors.h"

#include <cstddef>
#include <string>
#include <vector>

namespace nc {

struct CheckpointInfo {
    std::string family;          // "sd1", "sd2"; "sdxl" and others are recognised but not supported
    std::size_t inChannels = 4;  // 9 for inpainting models
    std::size_t contextDim = 0;  // text hidden size the UNet attends to (768 / 1024)
    bool hasVae = false, hasTextEncoder = false;
};

// True for a .safetensors file in the single-file layout (reads only the header).
bool isSingleFileCheckpoint(const std::string& path);
// What kind of model it is, from tensor names and shapes; throws if it is not one.
CheckpointInfo inspectCheckpoint(const std::string& path);

// Where a component ("unet", "vae", "text_encoder") of a model comes from: its config and the
// weights file to open with SafeTensors. For a single file the weights are "file#component" (the
// renamed view below) and the config is derived from the tensors. `what` names the caller in errors.
struct ModelComponent {
    json::Value config;
    std::string weights;
    std::string folder;  // the component's folder ("" for a single file)
};
ModelComponent modelComponent(const std::string& model, const std::string& component, const std::string& what);

// The renamed tensors of one component of a single-file checkpoint (SafeTensors("file#component")).
std::vector<SafeTensors::Rename> checkpointRenames(const SafeTensors& file, const std::string& component);

// Folder with the CLIP tokenizer (vocab.json + merges.txt) for a model: <model>/tokenizer, or for a
// single file a tokenizer/ next to it or in a sibling model folder (SD 1.x and 2.x share the
// vocabulary). The environment variable NEURACRAFT_TOKENIZER overrides the search.
std::string tokenizerDirectory(const std::string& model);
// True when prompts are padded with "!" (id 0, SD 2.x / Turbo) instead of <|endoftext|> (SD 1.x).
bool padsWithExclamation(const std::string& model);
// The scheduler config file of a model: scheduler/scheduler_config.json of a folder ("" if missing), or
// for a single file the SD 1.x / 2.x release schedule (scaled_linear, "leading", offset 1) with the
// sidecar's "scheduler" on top.
std::string schedulerConfig(const std::string& model);

} // namespace nc

#endif // NEURACRAFT_CHECKPOINT_H
