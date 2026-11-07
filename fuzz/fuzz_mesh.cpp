#include "Animation.h"
#include "Mesh.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// glTF binary, OBJ, PLY and BVH from the same bytes, each result checked and re-encoded (rigged
// models also posed), so a mesh a reader accepts is one the rest of the library can use.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    nc::setMaxMeshElements(1 << 16);
    nc::setMaxDecodePixels(1 << 20);
    const std::vector<std::uint8_t> bytes(data, data + size);
    auto use = [](const nc::Mesh& m) {
        nc::checkMesh(m);
        if (m.normals.size() != m.positions.size()) __builtin_trap();
        if (m.triangleCount() < 64) nc::encodeGlb(m);
        if (!m.rig.empty()) {
            const nc::Mesh posed = nc::poseMesh(m, m.rig.duration() / 2);
            if (posed.positions.size() != m.positions.size()) __builtin_trap();
            for (float x : posed.positions)
                if (std::isnan(x)) __builtin_trap();
        }
    };
    try {
        use(nc::decodeGlb(bytes));
    } catch (const std::exception&) {
    }
    try {
        use(nc::parseObj(std::string(bytes.begin(), bytes.end())));
    } catch (const std::exception&) {
    }
    try {
        use(nc::decodePly(bytes));
    } catch (const std::exception&) {
    }
    try {
        const nc::MotionCapture m = nc::parseBvh(std::string(bytes.begin(), bytes.end()));
        if (m.rotations.size() != m.frameCount() * m.jointCount() * 4 || m.rootPositions.size() != 3 * m.frameCount()) __builtin_trap();
    } catch (const std::exception&) {
    }
    return 0;
}
