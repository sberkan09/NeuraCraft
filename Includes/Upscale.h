#ifndef NEURACRAFT_UPSCALE_H
#define NEURACRAFT_UPSCALE_H

// Image super-resolution: Real-ESRGAN's compact network (SRVGGNetCompact,
// e.g. realesr-general-x4v3, 1.2 M parameters): 3x3 convolutions with PReLU at
// the input size, a pixel shuffle, and the nearest-upscaled input added back.

#include "Conv.h"
#include "Image.h"

#include <memory>
#include <string>
#include <vector>

namespace nc {

class Upscaler {
public:
    // A .safetensors file with SRVGGNetCompact weights (body.0 ... body.N; convert
    // the released .pth with scripts/convert_pth.py). Scale and depth come from the shapes.
    explicit Upscaler(const std::string& file);
    std::size_t scale() const { return scale_; }
    // RGB in, RGB out at scale x the size; large pictures are processed in
    // tiles overlapping by the receptive field, so memory stays bounded and
    // the result equals the untiled one.
    Image upscale(const Image& image, std::size_t tile = 256) const;

private:
    struct Layer {
        Matrix weight;  // out x (in * 9), PyTorch layout
        Matrix bias;    // 1 x out
        Matrix prelu;   // 1 x out, empty for the last conv
    };
    std::vector<Layer> layers_;
    std::size_t scale_ = 4, features_ = 64;
    Image upscaleTile(const Image& tile) const;
};

} // namespace nc

#endif // NEURACRAFT_UPSCALE_H
