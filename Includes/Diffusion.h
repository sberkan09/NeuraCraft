#ifndef NEURACRAFT_DIFFUSION_H
#define NEURACRAFT_DIFFUSION_H

// Latent diffusion (Stable Diffusion family) components, loaded from
// Hugging Face diffusers checkpoints.

#include "Conv.h"
#include "Image.h"
#include "Network.h"

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

} // namespace nc

#endif // NEURACRAFT_DIFFUSION_H
