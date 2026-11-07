#ifndef NEURACRAFT_STEREO_H
#define NEURACRAFT_STEREO_H

// Stereoscopic 3D: a left and a right eye view of a picture from its depth (depth-image-based
// rendering), and the frame layouts 3D players, TVs and headsets read (side by side, top and bottom,
// red-cyan anaglyph). Video is converted frame by frame with the depth range smoothed over time,
// so the 3D effect does not pump.

#include "Image.h"

#include <string>
#include <utility>
#include <vector>

namespace nc {

enum class StereoLayout {
    SideBySide,      // left | right, twice as wide (full SBS)
    HalfSideBySide,  // left | right squeezed into the original width (half SBS: 3D TVs, most VR players)
    TopBottom,       // left over right, twice as tall
    Anaglyph,        // one picture for red (left) / cyan (right) glasses (Dubois colours)
};
StereoLayout stereoLayout(const std::string& name);  // "sbs", "half_sbs", "tb", "anaglyph"; throws otherwise
std::string stereoLayoutName(StereoLayout layout);

struct StereoOptions {
    float strength = 0.03f;  // largest disparity between near and far, as a fraction of the width
    float focus = 0.5f;      // nearness on the screen plane: nearer parts come out of it, farther ones go behind
                             // (0: everything in front of the screen, 1: everything behind it)
    float smooth = 0.8f;     // video: how much of the previous frames' depth range is kept (0 = each frame alone)
};

// `nearness`: one value per pixel, 0 (farthest) .. 1 (nearest). Each eye's view moves pixels by half
// the disparity (strength * width * (nearness - focus)) in opposite directions, forward-splatted with
// a depth test; uncovered pixels are filled from the farther side.
std::pair<Image, Image> stereoViews(const Image& picture, const std::vector<float>& nearness, const StereoOptions& options = {});
// One frame in `layout` from the two views (same size).
Image stereoFrame(const Image& left, const Image& right, StereoLayout layout);

// Depth (relative inverse depth, larger = nearer, as DepthEstimator gives it) -> nearness, frame after
// frame: robust range (2nd..98th percentile), smoothed over the frames by `smooth`.
class StereoDepthRange {
public:
    explicit StereoDepthRange(float smooth = 0.8f) : smooth_(smooth) {}
    std::vector<float> nearness(const std::vector<float>& depth);

private:
    float smooth_;
    bool started_ = false;
    float lo_ = 0.0f, hi_ = 1.0f;
};

} // namespace nc

#endif // NEURACRAFT_STEREO_H
