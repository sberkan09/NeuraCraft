#ifndef NEURACRAFT_DEPTH_H
#define NEURACRAFT_DEPTH_H

// Monocular depth estimation with Depth Anything V2 (Hugging Face
// DepthAnythingForDepthEstimation, e.g. depth-anything/Depth-Anything-V2-Small-hf):
// a DINOv2 ViT backbone, a DPT neck (reassemble + fusion) and a convolutional
// head. Output is relative inverse depth (larger = nearer).

#include "Image.h"
#include "Matrix.h"

#include <memory>
#include <string>
#include <vector>

namespace nc {

class DepthEstimator {
public:
    explicit DepthEstimator(const std::string& directory);  // config.json + model.safetensors
    ~DepthEstimator();
    DepthEstimator(DepthEstimator&&) noexcept;
    DepthEstimator& operator=(DepthEstimator&&) noexcept;

    // Depth for every pixel of `image` (row-major, image size). The picture is
    // resized so both sides are multiples of the patch size, the shorter one
    // about `inputSize` (518 = the training size; smaller is faster).
    std::vector<float> estimate(const Image& image, std::size_t inputSize = 518);
    // The same, as a picture: depth normalized to 0..255 (near = white), 3
    // channels, as the ControlNet depth preprocessors produce.
    Image depthImage(const Image& image, std::size_t inputSize = 518);
    // The network on an already normalized input (3 x H x W, ImageNet mean / std;
    // H, W multiples of 14): depth at H x W. Used by the tests and references.
    std::vector<float> forward(const std::vector<float>& chw, std::size_t height, std::size_t width);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

struct ParallaxOptions {
    std::size_t frames = 48;
    float amplitude = 0.03f;  // camera sway as a fraction of the width (near pixels move this much)
    float zoom = 0.06f;       // extra scale of the nearest pixels at the middle of the move (dolly in)
    bool loop = true;         // an ellipse that returns to the start (else a one-way dolly)
};

// A "3D photo": the picture re-rendered from a slightly moving camera using
// its depth (near pixels move and grow more than far ones). Pixels are
// forward-splatted with a depth test; holes behind foreground edges are
// filled from the farther neighbour along the row.
std::vector<Image> parallaxAnimation(const Image& image, const std::vector<float>& depth, const ParallaxOptions& options = {});

} // namespace nc

#endif // NEURACRAFT_DEPTH_H
