#ifndef NEURACRAFT_ANIMATION_H
#define NEURACRAFT_ANIMATION_H

// Skeletal animation of rigged models (Mesh::rig, from a glTF with a skin): posing them at any moment
// of their animation, reading motion capture (BVH) and moving a rigged humanoid with it.
//
// Retargeting needs no bone names: on both skeletons the head, both legs and both arms are found from
// the rest pose (the model stands with +Y up, facing +Z, its left at +X, as glTF and CMU's BVH files
// have it). Each bone of the model is pointed the way the motion's bone at the same place along the
// limb points, with the motion's twist; the hips also move, scaled by the ratio of hip heights.

#include "Mesh.h"

#include <string>
#include <vector>

namespace nc {

// Positions of a rigged mesh with each joint's local transform given (10 floats per joint, as
// MeshRig::rest: translation, rotation x y z w, scale). Parts with no weights stay where they are.
std::vector<float> skinnedPositions(const MeshRig& rig, const float* locals);
// The model `seconds` into its animation (held at the ends), normals recomputed; the rest pose without
// one. The result is a still: no rig, no glTF.
Mesh poseMesh(const Mesh& mesh, double seconds);
// Bounds of the model over its whole animation (the rest pose without one): a sphere around the box.
void animationBounds(const Mesh& mesh, float centre[3], float& radius);

// Motion capture: a skeleton (end sites included, as joints without channels) and its local rotations,
// frame by frame, with the root's position.
struct MotionCapture {
    std::vector<std::string> names;
    std::vector<int> parents;         // parents come first; -1 for the root
    std::vector<float> offsets;       // 3 per joint: position in the parent's frame at rest
    double fps = 30;
    std::vector<float> rotations;     // per frame and joint: x, y, z, w
    std::vector<float> rootPositions; // per frame: x, y, z

    std::size_t jointCount() const { return parents.size(); }
    std::size_t frameCount() const { return parents.empty() ? 0 : rotations.size() / (4 * parents.size()); }
};
// Biovision BVH (as CMU's motion capture database is shared). Throws std::runtime_error when malformed.
MotionCapture parseBvh(const std::string& text, const std::string& name = "BVH data");
MotionCapture loadBvh(const std::string& path);
bool isMotionFile(const std::string& path);  // .bvh

struct AnimateOptions {
    double fps = 30;       // frames a second of the result
    double start = 0;      // seconds into the motion
    double seconds = 0;    // how much of it (0: to the end)
    double speed = 1;      // playback speed
    bool inPlace = false;  // keep the hips over their rest place (no walking away)
};
// The rigged model moved by the motion: Mesh::glb gets the animation (everything else in it kept) and
// Mesh::rig its frames. `report`, when given, receives a line per limb found and matched. Throws
// std::invalid_argument when the model has no rig or its limbs cannot be found.
Mesh animateMesh(const Mesh& rigged, const MotionCapture& motion, const AnimateOptions& options = {}, std::vector<std::string>* report = nullptr);

} // namespace nc

#endif // NEURACRAFT_ANIMATION_H
