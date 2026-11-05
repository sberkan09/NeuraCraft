#ifndef NEURACRAFT_DIFFUSION_H
#define NEURACRAFT_DIFFUSION_H

// Latent diffusion (Stable Diffusion family) components, loaded from
// Hugging Face diffusers checkpoints.

#include "Conv.h"
#include "Image.h"
#include "IpAdapter.h"
#include "Lora.h"
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
    // SDXL: the hidden states after the second-to-last layer (no final norm), and - when the model has a
    // text projection (CLIPTextModelWithProjection) - the pooled embedding: the final-normed state at the
    // end-of-text token (the largest id) times the projection (1 x projection dim).
    struct Penultimate {
        Matrix hidden, pooled;
    };
    Penultimate encodePenultimate(const std::vector<int>& ids);
    Matrix projection;  // dim x projection dim; empty without one
};

// Loads `directory`/text_encoder (or `directory` itself): config.json plus
// model(.fp16).safetensors. Supports hidden_act gelu (OpenCLIP, SD 2 / Turbo)
// and quick_gelu (OpenAI CLIP, SD 1.x). int8 quantizes its matrices.
ClipTextEncoder loadClipTextEncoder(const std::string& directory, bool int8 = false, const std::vector<Lora>& loras = {});
// ... one of several: "text_encoder", or "text_encoder_2" (SDXL's OpenCLIP bigG with its projection).
ClipTextEncoder loadClipTextEncoder(const std::string& directory, bool int8, const std::vector<Lora>& loras, const std::string& which);

// Multi-head scaled dot-product attention without a mask: q (N x C), k and v
// (M x C) -> N x C, channels split into `heads` contiguous groups. Queries go
// in chunks, so memory stays O((N + M) C) rather than O(N M).
Matrix multiHeadAttention(const Matrix& q, const Matrix& k, const Matrix& v, std::size_t heads);

// Sinusoidal timestep embedding (diffusers' get_timestep_embedding): 1 x dim.
Matrix timestepEmbedding(float timestep, std::size_t dim, bool flipSinToCos = true, float freqShift = 0.0f);

// What a ControlNet adds to a UNet call: one residual per down-path skip and
// one for the mid block output (already multiplied by the conditioning scale).
struct ControlResiduals {
    std::vector<Matrix> down;
    Matrix mid;
};

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
    // ... with ControlNet residuals added to the skips and the mid block (null: none).
    Matrix forward(const Matrix& latent, float timestep, const Matrix& context, const ControlResiduals* residuals);
    // Several frames at once (AnimateDiff): latents has one frame per row and
    // the same text conditions every frame. With motion modules loaded,
    // temporal attention mixes the frames after every spatial block; without,
    // each row is forward() of that frame.
    Matrix forwardFrames(const Matrix& latents, float timestep, const Matrix& context);
    // ... with one text condition per frame (prompt travel), or a single one for all.
    Matrix forwardFrames(const Matrix& latents, float timestep, const std::vector<Matrix>& contexts);
    Matrix forwardFrames(const Matrix& latents, float timestep, const std::vector<Matrix>& contexts,
                         const std::vector<ControlResiduals>* residuals);  // one set per frame
    // ControlNet (a diffusers ControlNetModel directory loads with loadUNet as
    // a UNet without up blocks): the conditioning image (1 x C*H*W at the
    // image size, channel-major, values 0..1, e.g. canny edges) -> features,
    // computed once per image ...
    bool isControlNet() const;
    // IP-Adapter (after loadIpAdapter): image prompt tokens (tokens x contextDim,
    // see IpAdapter::tokens) attended in every cross-attention next to the
    // text, weighted by `scale`; empty tokens switch it off.
    bool hasImagePrompt() const;
    // SDXL ("addition_embed_type": "text_time"): the pooled text embedding and the six size numbers
    // (original height, width, crop top, left, target height, width), set before each call.
    bool needsAddedConditions() const;
    void setAddedConditions(const Matrix& textEmbeds, const std::vector<float>& timeIds);
    void setImagePrompt(const Matrix& tokens, float scale = 1.0f);
    Matrix controlEmbedding(const Matrix& conditionImage);
    // ... and per step, the residuals for the UNet, multiplied by `scale`.
    ControlResiduals control(const Matrix& latent, float timestep, const Matrix& context, const Matrix& embedding, float scale = 1.0f);
    ImageShape latentShape() const;  // the UNet's input (9 channels for inpainting models)
    std::size_t outputChannels() const;  // the predicted noise's channels (4)
    std::size_t contextDim() const;
    std::size_t parameterCount() const;
    bool hasMotionModules() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    friend UNet2DCondition loadUNet(const std::string&, std::size_t, std::size_t, bool, const std::vector<Lora>&);
    friend void loadMotionModules(UNet2DCondition&, const std::string&, bool);
    friend void loadIpAdapter(UNet2DCondition&, const std::string&, bool);
};

// Adds an IP-Adapter's image key / value projections (h94/IP-Adapter
// ip-adapter_sd15.safetensors: ip_adapter.N.to_k_ip / to_v_ip for every
// cross-attention, in diffusers' processor order) to an SD 1.x UNet.
void loadIpAdapter(UNet2DCondition& unet, const std::string& file, bool int8 = false);

// Adds an AnimateDiff motion adapter (diffusers MotionAdapter safetensors, such
// as AnimateDiff-Lightning or mm_sd_v15_v2) to an SD 1.x UNet: a temporal
// transformer after every resnet / attention of the down, mid and up blocks
// (GroupNorm over all frames, sinusoidal frame positions, two self-attention
// layers across frames per pixel, GEGLU feed-forward). Up to 32 frames.
void loadMotionModules(UNet2DCondition& unet, const std::string& file, bool int8 = false);

// Loads `directory`/unet (or `directory` itself): config.json plus
// diffusion_pytorch_model(.fp16).safetensors, for latents of the given size
// (image size / 8). With int8, every convolution and linear layer is
// quantized as soon as it is loaded (about a quarter of the memory).
// LoRA files in `loras` are merged into the weights first (before int8).
UNet2DCondition loadUNet(const std::string& directory, std::size_t latentWidth, std::size_t latentHeight, bool int8 = false,
                         const std::vector<Lora>& loras = {});

// diffusers' EulerDiscreteScheduler (epsilon prediction; "scaled_linear" or
// "linear" betas; "trailing", "leading" or "linspace" timestep spacing).
struct EulerScheduler {
    std::vector<double> timesteps;  // model timesteps, from noisy to clean
    std::vector<double> sigmas;     // one per timestep, plus a final 0
    double initNoiseSigma = 1.0;    // scale of the initial Gaussian latent
    // Latent consistency (LCMScheduler, for LCM / LCM-LoRA / AnimateLCM
    // models): each step predicts the clean latent through the consistency
    // boundary condition and re-noises it to the next timestep with fresh
    // noise from the library's random generator.
    bool consistency = false;

    static EulerScheduler fromConfig(const std::string& schedulerConfigPath, std::size_t steps);
    // diffusers' LCMScheduler timesteps (from 50 original steps) and step rule.
    static EulerScheduler lcm(std::size_t steps, const std::string& betaSchedule = "scaled_linear", double betaStart = 0.00085,
                              double betaEnd = 0.012, std::size_t originalSteps = 50, std::size_t trainSteps = 1000);
    static EulerScheduler lcmFromConfig(const std::string& schedulerConfigPath, std::size_t steps);
    // Explicit settings (diffusers names: "scaled_linear" / "linear", "trailing" / "leading" / "linspace").
    static EulerScheduler make(std::size_t steps, const std::string& betaSchedule, const std::string& spacing,
                               double betaStart = 0.00085, double betaEnd = 0.012, std::size_t trainSteps = 1000,
                               std::size_t offset = 1);
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
    std::string vae;                        // another VAE (a diffusers VAE folder, a model, or a VAE .safetensors); "" = the model's
    bool int8 = true;                       // int8 UNet and text encoder weights (false: float32)
    std::string sampler = "euler";          // "euler", or "lcm" for LCM models / LCM-LoRA (4 to 8 steps, guidance 1)
    float strength = 0.5f;                  // image to image: how much of the schedule runs (0 keeps the input)
    std::vector<Lora> loras;                // LoRA files merged into the UNet and text encoder
    // ControlNet: a ControlNetModel directory and its conditioning image (e.g.
    // cannyEdges of a photo), fitted to the output size; scale weights it.
    std::string controlNet;
    Image controlImage;
    float controlScale = 1.0f;
    // IP-Adapter image prompt: the adapter file, the picture and its weight;
    // the CLIP image encoder defaults to image_encoder/ next to the adapter.
    std::string ipAdapter, imageEncoder;
    Image imagePrompt;
    float ipScale = 0.6f;
    std::function<void(const std::string& stage, double seconds)> progress;  // after each stage / step
    // Live preview after each denoising step: the predicted clean picture at half size, decoded with
    // the TAESD in tinyVae (no preview without one). Costs about a quarter of a TAESD decode a step.
    std::function<void(const Image& picture)> preview;
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
// image to image. Built for options.width x options.height; options.int8,
// loras, sampler, guidance / negativePrompt, progress and an IP-Adapter image
// prompt (ipAdapter, imagePrompt, ipScale: used by every generation, e.g. to
// keep a style through a zoom or morph) apply.
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
    // Inpainting (needs an inpainting UNet: 4 latent + 1 mask + 4 masked-image
    // channels): redraws the white part of `mask` (grey, same picture area as
    // `input`) from the prompt and keeps the rest; the result is blended back
    // into the input with a mask feathered by `feather` pixels. Uses the
    // session's guidance / negative prompt (classifier-free guidance when > 1).
    Image inpaint(const Image& input, const Image& mask, const std::string& prompt, unsigned seed, std::size_t steps,
                  float strength = 1.0f, float feather = 4.0f);
    bool isInpainting() const;

    const TextToImageOptions& impl_options() const;  // the options it was built with
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Inpainting with an SD inpainting model directory (as diffusers'
// StableDiffusionInpaintPipeline): `mask` white = redraw. Steps, seed,
// sampler, LoRAs, tinyVae and int8 come from the options; strength 1 starts
// the masked area from pure noise.
Image inpaint(const std::string& directory, const Image& input, const Image& mask, const std::string& prompt,
              const TextToImageOptions& options = {});

struct OutpaintZoomOptions {
    std::size_t keyframes = 6;        // generated pictures, each outpainted around the previous one
    float shrink = 0.75f;             // the previous picture's size inside the next (0.4 .. 0.9)
    std::size_t framesPerKey = 12;    // rendered frames between two keyframes (no UNet work)
    bool zoomIn = false;              // play from the outermost picture inwards
    std::size_t steps = 4;
    // Below 1 the new border starts from the shrunk picture mirrored outwards
    // (noised), which continues the scene instead of letting the model frame
    // the old picture; 1 starts the border from pure noise.
    float strength = 0.9f;
    unsigned seed = 0;
};

// The classic infinite zoom: a first picture (or `start`), then each next
// keyframe is the previous one shrunk into the middle and its new border
// inpainted from the prompts (spread over the keyframes). The frames in
// between are rendered by zooming through the nested keyframes (geometric
// zoom, the inner picture drawn over the outer one), so a long smooth zoom
// costs one inpainting per keyframe. Needs an inpainting model.
std::vector<Image> outpaintZoom(StableDiffusion& sd, const std::vector<std::string>& prompts, const OutpaintZoomOptions& options = {},
                                const Image* start = nullptr);

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
    // run. The default runs SD-Turbo's distilled timesteps 499 and 249 (two
    // UNet passes): one pass from 249 keeps frames closer but cannot invent
    // the detail zooming needs, so the picture flattens after a dozen frames.
    std::size_t steps = 4;
    float strength = 0.5f;
    unsigned seed = 0;
    bool colorCoherence = true;   // match each frame's colour statistics to the first frame's
    // Before each redraw the moved frame is sharpened (unsharp mask of this
    // amount) and given Gaussian pixel noise (standard deviation, 1 = full
    // range): zooming blurs, and the model keeps blur it is shown.
    float sharpen = 0.5f;
    float noise = 0.04f;
};

// "Infinite zoom": starts from `start` (or a picture of the first prompt,
// redrawn once with image to image so the second frame does not jump) and
// makes each next frame by moving the previous one (zoom, pan, rotation) and
// redrawing it with image to image; prompts are spread evenly over the frames.
// Works best at the model's native size (512 for SD-Turbo).
std::vector<Image> zoomAnimation(StableDiffusion& sd, const std::vector<std::string>& prompts, const ZoomOptions& options = {},
                                 const Image* start = nullptr);

struct TextToVideoOptions {
    std::size_t width = 512, height = 512;  // multiples of 64
    std::size_t frames = 16;                // any count: more than `context` uses sliding windows
    // The motion module sees `context` frames at a time (AnimateDiff was
    // trained on 16, at most 32); longer clips are denoised in windows that
    // overlap by `overlap` frames, blended with pyramid weights.
    std::size_t context = 16, overlap = 4;
    // Image / video to video: one image animates (every frame starts from it),
    // or one image per frame restyles a clip. Each starts from the encoded
    // frame noised to the sigma of step steps - steps * strength.
    std::vector<Image> init;
    float strength = 0.75f;
    // Prompt travel: (frame, prompt) keyframes after the main prompt (frame
    // 0); each frame's text condition is interpolated between the keyframes
    // around it, so the scene changes over the clip.
    std::vector<std::pair<std::size_t, std::string>> promptAt;
    std::vector<Lora> loras;  // merged into the UNet and text encoder (not the motion module)
    // ControlNet for every frame: one conditioning image for all, or one per frame.
    std::string controlNet;
    std::vector<Image> controlImages;
    float controlScale = 1.0f;
    // IP-Adapter image prompt for every frame (keeps a subject or style across the clip).
    std::string ipAdapter, imageEncoder;
    Image imagePrompt;
    float ipScale = 0.6f;
    std::size_t steps = 4;                  // AnimateDiff-Lightning: the step count the module was distilled for
    unsigned seed = 0;
    float guidance = 1.0f;                  // <= 1: off (Lightning); else classifier-free guidance
    std::string negativePrompt;
    std::string motion;                     // motion adapter file; empty: the one .safetensors in DIR/motion
    std::string tinyVae;                    // TAESD directory: decode with it instead of the full VAE
    std::string vae;                        // another VAE (a diffusers VAE folder, a model, or a VAE .safetensors); "" = the model's
    bool int8 = true;
    // Lightning's schedule: Euler, linear betas, trailing timesteps. "config"
    // uses DIR/scheduler/scheduler_config.json instead (other motion modules).
    std::string betaSchedule = "linear";
    std::string sampler = "euler";  // "lcm" for AnimateLCM (with its LoRA; betaSchedule applies to both samplers)
    std::function<void(const std::string& stage, double seconds)> progress;
    std::function<void(const Image& picture)> preview;  // the middle frame, as TextToImageOptions::preview
};

// AnimateDiff text to video on an SD 1.x diffusers directory plus a motion
// adapter: CLIP encodes the prompt, the UNet with motion modules denoises all
// frames' latents together, each frame is decoded. Returns the frames. With
// options.init it is image to video or video to video instead.
std::vector<Image> textToVideo(const std::string& directory, const std::string& prompt, const TextToVideoOptions& options = {});

} // namespace nc

#endif // NEURACRAFT_DIFFUSION_H
