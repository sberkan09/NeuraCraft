#include "FuzzFile.h"
#include "Onnx.h"

#include <stdexcept>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string path = fuzzFile(data, size, ".pb");
    try {
        std::string name;
        nc::onnx::loadTensorProto(path, &name);
    } catch (const std::exception&) {
    }
    return 0;
}
