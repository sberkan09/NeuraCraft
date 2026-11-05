// Private interface to the Metal implementation (src/MetalBackend.mm).
#ifndef NEURACRAFT_METAL_BACKEND_H
#define NEURACRAFT_METAL_BACKEND_H

#include <cstddef>
#include <cstdint>
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

// C (M x N, row stride ldc) = op(A) * op(B) with one operand int8 weights
// (rows x K, row-major, one float scale per `group` inputs of each row,
// groupsPerRow scales per row) dequantized in the shader as tiles are loaded:
//   quantLeft:  C = Q (M x K) * X (K x N, row stride ldx)       (convolutions)
//   !quantLeft: C = X (M x K, row stride ldx) * Q^T, Q is N x K (dense layers)
bool int8Available();
void gemmQ8(bool quantLeft, std::size_t M, std::size_t N, std::size_t K,
            const std::int8_t* q, std::size_t qMapBytes, const float* scales, std::size_t scaleMapBytes,
            std::size_t groupsPerRow, std::size_t group, const float* x, std::size_t ldx, std::size_t xMapBytes,
            float* C, std::size_t ldc, std::size_t cMapBytes);

// out (N x C) = softmax(Q K^T / sqrt(C / heads)) V for each head (columns
// h*dh .. h*dh+dh-1); Q is N x C, K and V are M x C, all dense row-major.
bool attentionSupported(std::size_t headSize);
void attention(std::size_t N, std::size_t M, std::size_t C, std::size_t heads, const float* q, std::size_t qMapBytes,
               const float* k, std::size_t kMapBytes, const float* v, std::size_t vMapBytes, float* out,
               std::size_t outMapBytes);

} // namespace metal
} // namespace nc

#endif // NEURACRAFT_METAL_BACKEND_H
