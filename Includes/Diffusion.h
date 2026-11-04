#ifndef NEURACRAFT_DIFFUSION_H
#define NEURACRAFT_DIFFUSION_H

// Latent diffusion (Stable Diffusion family) components, loaded from
// Hugging Face diffusers checkpoints.

#include "Conv.h"
#include "Image.h"
#include "Network.h"
#include "Transformer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

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

// Tiny autoencoder TAESD (diffusers AutoencoderTiny, madebyollin/taesd):
// ~1 M parameters of 3x3 convolutions, ReLU residual blocks and nearest
// upsampling that approximate the full VAE at a small fraction of the cost.
// Works in the diffusion model's (already scaled) latent space.
struct TinyVae {
    Network encoder;  // image in [0, 1] -> latent
    Network decoder;  // latent clamped to (-3, 3) -> image in [0, 1]
    ImageShape image, latent;

    Matrix encode(const Matrix& image);   // image row in [-1, 1] -> latent
    Matrix decode(const Matrix& latent);  // latent -> image row in [-1, 1]
};

// Loads `directory` (config.json + diffusion_pytorch_model.safetensors).
TinyVae loadTinyVae(const std::string& directory, std::size_t width, std::size_t height, bool withEncoder = true,
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
// and quick_gelu (OpenAI CLIP, SD 1.x). int8 quantizes its matrices.
ClipTextEncoder loadClipTextEncoder(const std::string& directory, bool int8 = false);

// Multi-head scaled dot-product attention without a mask: q (N x C), k and v
// (M x C) -> N x C, channels split into `heads` contiguous groups. Queries go
// in chunks, so memory stays O((N + M) C) rather than O(N M).
Matrix multiHeadAttention(const Matrix& q, const Matrix& k, const Matrix& v, std::size_t heads);

// Sinusoidal timestep embedding (diffusers' get_timestep_embedding): 1 x dim.
Matrix timestepEmbedding(float timestep, std::size_t dim, bool flipSinToCos = true, float freqShift = 0.0f);

// Stable Diffusion's denoiser (diffusers UNet2DConditionModel, SD 1.x / 2.x /
// Turbo layouts): predicts the noise in a latent at a timestep, conditioned
// on text hidden states through cross-attention. Built for one latent size.
// Inference only.
class UNet2DCondition {
public:
    UNet2DCondition();
    ~UNet2DCondition();
    UNet2DCondition(UNet2DCondition&&) noexcept;
    UNet2DCondition& operator=(UNet2DCondition&&) noexcept;

    // latent: 1 x (C h w); context: tokens x contextDim. Returns 1 x (C h w).
    Matrix forward(const Matrix& latent, float timestep, const Matrix& context);
    ImageShape latentShape() const;
    std::size_t contextDim() const;
    std::size_t parameterCount() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    friend UNet2DCondition loadUNet(const std::string&, std::size_t, std::size_t, bool);
};

// Loads `directory`/unet (or `directory` itself): config.json plus
// diffusion_pytorch_model(.fp16).safetensors, for latents of the given size
// (image size / 8). With int8, every convolution and linear layer is
// quantized as soon as it is loaded (about a quarter of the memory).
UNet2DCondition loadUNet(const std::string& directory, std::size_t latentWidth, std::size_t latentHeight, bool int8 = false);

// diffusers' EulerDiscreteScheduler (epsilon prediction; "scaled_linear" or
// "linear" betas; "trailing", "leading" or "linspace" timestep spacing).
struct EulerScheduler {
    std::vector<double> timesteps;  // model timesteps, from noisy to clean
    std::vector<double> sigmas;     // one per timestep, plus a final 0
    double initNoiseSigma = 1.0;    // scale of the initial Gaussian latent

    static EulerScheduler fromConfig(const std::string& schedulerConfigPath, std::size_t steps);
    // Uses the defaults of Stable Diffusion (1000 steps, scaled_linear 0.00085 .. 0.012, trailing).
    static EulerScheduler standard(std::size_t steps);
    double inputScale(std::size_t step) const;  // multiply the latent by this before the UNet
    // One Euler step from timestep `step` to the next, given the predicted noise.
    void step(std::size_t step, const Matrix& noise, Matrix& latent) const;
};

struct TextToImageOptions {
    std::size_t width = 512, height = 512;  // multiples of 64 for SD's 4-level UNet
    std::size_t steps = 1;                  // SD-Turbo: 1 to 4
    unsigned seed = 0;
    float guidance = 0.0f;                  // classifier-free guidance scale; <= 1 = off (SD-Turbo)
    std::string negativePrompt;             // used with guidance > 1
    std::string tinyVae;                    // TAESD directory: decode with it instead of the full VAE
    bool int8 = true;                       // int8 UNet and text encoder weights (false: float32)
    std::function<void(const std::string& stage, double seconds)> progress;  // after each stage / step
};

// The whole Stable Diffusion text-to-image pipeline on a diffusers directory
// (tokenizer, text_encoder, unet, vae, scheduler): CLIP encodes the prompt,
// the UNet denoises a seeded Gaussian latent with the Euler scheduler, the
// VAE decodes it. Each model is loaded only for its stage, so peak memory is
// about the UNet's.
Image textToImage(const std::string& directory, const std::string& prompt, const TextToImageOptions& options = {});
// The first two stages alone: the denoised latent (1 x 4 * H/8 * W/8), ...
Matrix textToLatent(const std::string& directory, const std::string& prompt, const TextToImageOptions& options = {});
// ... and the last: decode it with the full VAE of `directory`, or with the TAESD in options.tinyVae.
Image decodeLatent(const std::string& directory, const Matrix& latent, const TextToImageOptions& options = {});

} // namespace nc

#endif // NEURACRAFT_DIFFUSION_H
