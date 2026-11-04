#ifndef NEURACRAFT_DIFFUSION_H
#define NEURACRAFT_DIFFUSION_H

// Latent diffusion (Stable Diffusion family) components, loaded from
// Hugging Face diffusers checkpoints.

#include "Conv.h"
#include "Image.h"
#include "Network.h"
#include "Transformer.h"

#include <string>

namespace nc {

// The KL autoencoder that maps images (3 x H x W, values in [-1, 1]) to
// latents (4 x H/8 x W/8) and back. Layers are built for one image size.
struct Vae {
    Network encoder;  // image -> moments (2 * latent channels: mean, then log variance)
    Network decoder;  // unscaled latent -> image
    ImageShape image, latent;
    float scalingFactor = 0.18215f;  // latents the diffusion model sees are mean * scalingFactor

    // Image row (1 x 3HW in [-1, 1]) -> scaled latent mean (1 x 4hw), deterministic.
    Matrix encode(const Matrix& image);
    // Scaled latent -> image row in [-1, 1] (not clamped).
    Matrix decode(const Matrix& latent);
};

// Loads `directory`/vae (or `directory` itself if it holds config.json):
// config.json plus diffusion_pytorch_model(.fp16).safetensors, for images of
// width x height pixels (multiples of 8 x 2^(blocks - 1)). Either half can be
// skipped to save memory.
Vae loadVae(const std::string& directory, std::size_t width, std::size_t height, bool withEncoder = true,
            bool withDecoder = true);

// CLIP's text transformer (Hugging Face CLIPTextModel): token ids (padded to
// `length`, usually 77) -> one hidden state per position after the final
// layer norm, the conditioning Stable Diffusion's UNet cross-attends to.
struct ClipTextEncoder {
    Network net;
    SequenceContextPtr context;
    std::size_t length = 77, dim = 0, layers = 0;

    Matrix encode(const std::vector<int>& ids);  // length x dim
};

// Loads `directory`/text_encoder (or `directory` itself): config.json plus
// model(.fp16).safetensors. Supports hidden_act gelu (OpenCLIP, SD 2 / Turbo)
// and quick_gelu (OpenAI CLIP, SD 1.x).
ClipTextEncoder loadClipTextEncoder(const std::string& directory);

} // namespace nc

#endif // NEURACRAFT_DIFFUSION_H
