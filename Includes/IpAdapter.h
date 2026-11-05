#ifndef NEURACRAFT_IP_ADAPTER_H
#define NEURACRAFT_IP_ADAPTER_H

// Image prompts (IP-Adapter, h94/IP-Adapter for SD 1.5): a CLIP vision
// encoder turns a picture into an image embedding, the adapter projects it to
// a few context tokens, and the UNet attends to them next to the text
// (UNet2DCondition::setImagePrompt after loadIpAdapter).

#include "Image.h"
#include "Matrix.h"

#include <memory>
#include <string>

namespace nc {

// Hugging Face CLIPVisionModelWithProjection (ViT: patch embedding, class
// token, pre-LayerNorm, transformer layers, post-LayerNorm of the class token,
// projection).
class ClipImageEncoder {
public:
    explicit ClipImageEncoder(const std::string& directory);  // config.json + model.safetensors
    ~ClipImageEncoder();
    ClipImageEncoder(ClipImageEncoder&&) noexcept;
    ClipImageEncoder& operator=(ClipImageEncoder&&) noexcept;
    // CLIPImageProcessor preprocessing (shorter side to the input size,
    // centre crop, CLIP mean / std), then the image embedding (1 x projection_dim).
    Matrix embed(const Image& image);
    // The network on a preprocessed input (3 x S x S, normalized).
    Matrix forward(const std::vector<float>& chw);
    std::size_t imageSize() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// The adapter's image projection (ImageProjModel: Linear to tokens x context, LayerNorm per token).
class IpAdapter {
public:
    explicit IpAdapter(const std::string& file);  // ip-adapter_sd15.safetensors
    Matrix tokens(const Matrix& imageEmbedding) const;  // tokens x contextDim (4 x 768 for SD 1.5)
    const std::string& file() const { return file_; }

private:
    std::string file_;
    Matrix projT;  // embed x (tokens * dim)
    std::vector<float> projBias, normG, normB;
    std::size_t tokens_ = 4, dim_ = 768;
};

} // namespace nc

#endif // NEURACRAFT_IP_ADAPTER_H
