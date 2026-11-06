#include "FuzzFile.h"
#include "Onnx.h"

#include <stdexcept>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string path = fuzzFile(data, size, ".onnx");
    try {
        const nc::onnx::Model m = nc::onnx::Model::load(path);
        m.operatorCounts();
        m.parameterCount();
    } catch (const std::exception&) {
    }
    return 0;
}
