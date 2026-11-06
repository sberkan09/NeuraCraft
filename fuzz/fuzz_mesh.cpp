#include "Mesh.h"

#include <stdexcept>
#include <string>
#include <vector>

// glTF binary, OBJ and PLY from the same bytes (the first byte picks the reader's guess first), each
// result checked and re-encoded, so a mesh a reader accepts is one the rest of the library can use.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    nc::setMaxMeshElements(1 << 16);
    nc::setMaxDecodePixels(1 << 20);
    const std::vector<std::uint8_t> bytes(data, data + size);
    auto use = [](const nc::Mesh& m) {
        nc::checkMesh(m);
        if (m.normals.size() != m.positions.size()) __builtin_trap();
        if (m.triangleCount() < 64) nc::encodeGlb(m);
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
    return 0;
}
