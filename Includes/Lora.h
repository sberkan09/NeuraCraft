#ifndef NEURACRAFT_LORA_H
#define NEURACRAFT_LORA_H

// LoRA (low-rank adaptation) files merged into Stable Diffusion weights as
// they load: W += scale * alpha / rank * up * down for every adapted layer.

#include "Matrix.h"

#include <map>
#include <string>
#include <vector>

namespace nc {

struct Lora {
    std::string file;   // .safetensors
    float scale = 1.0f; // strength (the "weight" in UIs)
};

// The adapted layers of one model component ("unet" or "text_encoder") from
// a set of LoRA files. Understands kohya / A1111 names
// (lora_unet_down_blocks_0_..._to_q.lora_down.weight, .lora_up.weight,
// .alpha; lora_te_text_model_...) and diffusers / PEFT names
// (unet.down_blocks.0....to_q.lora_A.weight / lora_B.weight). Linear and
// convolution layers (any kernel size on the down side, 1x1 on the up side).
class LoraMerger {
public:
    LoraMerger() = default;
    LoraMerger(const std::vector<Lora>& loras, const std::string& component);

    bool empty() const { return deltas_.empty(); }
    // Adds the deltas for weight tensor `name` (e.g. "down_blocks.0....to_q.weight")
    // to `weight`, PyTorch layout (out x everything else, flattened).
    void apply(const std::string& name, Matrix& weight);
    // Adapted layers found in the files / merged so far; layers never merged
    // after loading point at a file made for another model.
    std::size_t layers() const { return total_; }
    std::size_t merged() const { return merged_; }
    std::vector<std::string> unmerged() const;

private:
    struct Delta {
        Matrix up, down;  // out x rank, rank x in
        float scale;
        bool used = false;
    };
    std::map<std::string, std::vector<Delta>> deltas_;  // key: module name with '_' for '.'
    std::size_t merged_ = 0, total_ = 0;
};

} // namespace nc

#endif // NEURACRAFT_LORA_H
