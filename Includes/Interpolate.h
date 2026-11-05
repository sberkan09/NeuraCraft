#ifndef NEURACRAFT_INTERPOLATE_H
#define NEURACRAFT_INTERPOLATE_H

// Video frame interpolation with RIFE (ECCV 2022 IFNet, v4 weights such as
// TensorForger/RIFE-safetensors flownet.safetensors): three IFBlocks estimate
// bidirectional optical flow and a blend mask coarse to fine (1/4, 1/2, full
// size), both frames are warped to the middle and blended.

#include "Image.h"
#include "Matrix.h"

#include <string>
#include <vector>

namespace nc {

class FrameInterpolator {
public:
    explicit FrameInterpolator(const std::string& file);
    // The frame halfway between a and b (RGB, same size; any size, padded to
    // a multiple of 32 internally).
    Image middle(const Image& a, const Image& b) const;
    // 2^passes - 1 new frames between every pair (passes = 1 doubles the frame
    // rate, 2 quadruples it); with loop, also between the last and the first.
    std::vector<Image> interpolate(const std::vector<Image>& frames, std::size_t passes = 1, bool loop = false) const;

    struct Conv {
        Matrix weight;  // PyTorch layout: conv out x (in k k); transposed conv in x (out k k)
        Matrix bias, prelu;
        std::size_t in = 0, out = 0, k = 3;
    };
    struct Block {
        Conv c0a, c0b;              // stride-2 convs
        Conv blocks[4][2];          // residual pairs
        Conv up1a, up1b, up2a, up2b;  // transposed convs (flow: 4 channels, mask: 1)
    };

private:
    std::vector<Block> blocks_;
};

} // namespace nc

#endif // NEURACRAFT_INTERPOLATE_H
