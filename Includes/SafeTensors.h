#ifndef NEURACRAFT_SAFETENSORS_H
#define NEURACRAFT_SAFETENSORS_H

#include "Matrix.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace nc {

// Reader for the safetensors format (Hugging Face): an 8-byte little-endian
// header length, a JSON header describing each tensor, then raw data.
// Tensors are loaded one at a time, so a model never needs two copies in memory.
class SafeTensors {
public:
    struct Tensor {
        std::string dtype;                // "F32", "F16" or "BF16" can be loaded
        std::vector<std::size_t> shape;
        std::size_t begin = 0, end = 0;   // byte range within the data section
        std::size_t elements() const;
    };

    explicit SafeTensors(const std::string& path);

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const Tensor& info(const std::string& name) const;
    std::vector<std::string> names() const;  // sorted
    const std::unordered_map<std::string, std::string>& metadata() const { return metadata_; }

    // Converted to float. Shape: 1-D -> 1 x n; otherwise all leading dimensions
    // are flattened into rows and the last dimension becomes the columns.
    Matrix load(const std::string& name) const;

private:
    std::string path_;
    std::size_t dataStart_ = 0;
    std::size_t fileSize_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
};

// Writes F32 tensors (used by tests and for exporting).
void saveSafeTensors(const std::string& path, const std::vector<std::pair<std::string, const Matrix*>>& tensors);

} // namespace nc

#endif // NEURACRAFT_SAFETENSORS_H
