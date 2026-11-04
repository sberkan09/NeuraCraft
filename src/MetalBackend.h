// Private interface to the Metal implementation (src/MetalBackend.mm).
#ifndef NEURACRAFT_METAL_BACKEND_H
#define NEURACRAFT_METAL_BACKEND_H

#include <cstddef>
#include <string>

namespace nc {
namespace metal {

bool available();
std::string deviceName();
std::string kernelName();  // "simdgroup" or "tiled"
// *MapBytes > 0: the operand lives in page-aligned memory of that (page-rounded)
// length and is used by the GPU in place; 0: it is copied.
void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
          const float* A, std::size_t lda, const float* B, std::size_t ldb,
          float* C, std::size_t ldc, bool accumulate,
          std::size_t aMapBytes = 0, std::size_t bMapBytes = 0, std::size_t cMapBytes = 0);

} // namespace metal
} // namespace nc

#endif // NEURACRAFT_METAL_BACKEND_H
