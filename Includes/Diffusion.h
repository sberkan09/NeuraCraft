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
    float strength = 0.5f;                  // image to image: how much of the schedule runs (0 keeps the input)
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
// Image to image: the input is cropped / resized to width x height, encoded
// (TAESD encoder with options.tinyVae, else the full VAE), noised to the
// sigma of step steps - steps * strength, and denoised for the remaining
// steps with the prompt (SD-Turbo: steps 2, strength 0.5 = one step).
Image imageToImage(const std::string& directory, const Image& input, const std::string& prompt,
                   const TextToImageOptions& options = {});

// A Stable Diffusion model kept in memory for many generations (animations,
// interactive use): tokenizer, text encoder, UNet and a decoder (TAESD when
// options.tinyVae is set, else the full VAE), plus the matching encoder for
// image to image. Built for options.width x options.height; options.int8 and
// options.progress apply.
class StableDiffusion {
public:
    StableDiffusion(const std::string& directory, const TextToImageOptions& options = {}, bool withEncoder = true);
    ~StableDiffusion();
    StableDiffusion(StableDiffusion&&) noexcept;
    StableDiffusion& operator=(StableDiffusion&&) noexcept;

    std::size_t width() const;
    std::size_t height() const;
    Matrix encodePrompt(const std::string& prompt);  // tokens x dim hidden states
    Matrix noise(unsigned seed) const;               // a standard normal latent
    EulerScheduler scheduler(std::size_t steps) const;
    // Runs the scheduler's steps [first, end) on `latent`; guidance > 1 also uses `negative`.
    void denoise(Matrix& latent, const Matrix& context, const EulerScheduler& scheduler, std::size_t first = 0,
                 float guidance = 0.0f, const Matrix& negative = Matrix());
    Matrix encodeImage(const Image& image);  // cropped / resized to the size; needs the encoder
    Image decode(const Matrix& latent);

    // One picture from a prompt (text to image) ...
    Image generate(const std::string& prompt, unsigned seed, std::size_t steps = 1);
    // ... or from an image and a prompt (image to image, as imageToImage).
    Image transform(const Image& input, const std::string& prompt, unsigned seed, std::size_t steps = 2, float strength = 0.5f);

    const TextToImageOptions& impl_options() const;  // the options it was built with
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

struct MorphOptions {
    std::size_t framesPerSegment = 12;  // frames from one keyframe to the next
    std::size_t steps = 1;              // denoising steps per frame
    bool loop = true;                   // also morph from the last keyframe back to the first
};

// A smooth transition through keyframes (prompts[i], seeds[i]): text hidden
// states are interpolated linearly and the starting noise spherically, and
// every frame is generated from its interpolated pair. One seed for all
// keyframes is allowed (seeds of size 1).
std::vector<Image> morphAnimation(StableDiffusion& sd, const std::vector<std::string>& prompts, const std::vector<unsigned>& seeds,
                                  const MorphOptions& options = {});

struct ZoomOptions {
    std::size_t frames = 24;
    float zoom = 1.04f;           // per frame (> 1 zooms in)
    float dx = 0.0f, dy = 0.0f;   // pixels per frame
    float rotate = 0.0f;          // degrees per frame
    // Image to image per frame: a `steps` schedule of which steps * strength
    // run. The default is one UNet pass from timestep 249, one of the four
    // SD-Turbo was distilled at (lower noise blurs, higher noise flickers).
    std::size_t steps = 4;
    float strength = 0.25f;
    unsigned seed = 0;
    bool colorCoherence = true;   // match each frame's colour statistics to the first frame's
    // Before each redraw the moved frame is sharpened (unsharp mask of this
    // amount) and given Gaussian pixel noise (standard deviation, 1 = full
    // range): zooming blurs, and the model keeps blur it is shown.
    float sharpen = 1.0f;
    float noise = 0.04f;
};

// "Infinite zoom": starts from `start` (or a picture of the first prompt,
// redrawn once with image to image so the second frame does not jump) and
// makes each next frame by moving the previous one (zoom, pan, rotation) and
// redrawing it with image to image; prompts are spread evenly over the frames.
// Works best at the model's native size (512 for SD-Turbo).
std::vector<Image> zoomAnimation(StableDiffusion& sd, const std::vector<std::string>& prompts, const ZoomOptions& options = {},
                                 const Image* start = nullptr);

} // namespace nc

#endif // NEURACRAFT_DIFFUSION_H
