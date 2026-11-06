// Private interface to the CUDA implementation (src/CudaBackend.cu): the same operations as the
// Metal backend (MetalBackend.h), on NVIDIA GPUs through cuBLAS and a few kernels of our own.
// Operands live in host memory; each call copies what it needs to the device, except weights that
// are marked constant (gemm's *Key arguments), which stay on the device between calls.
#ifndef NEURACRAFT_CUDA_BACKEND_H
#define NEURACRAFT_CUDA_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace nc {
namespace cuda {

bool available();
std::string deviceName();  // "NVIDIA GeForce RTX 3090 (sm_86, 24 GB)"
std::string kernelName();  // "cuBLAS TF32" or "cuBLAS FP32"

// Row-major C (M x N, stride ldc) = op(A) op(B) (+ C when accumulate).
void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K, const float* A, std::size_t lda, const float* B,
          std::size_t ldb, float* C, std::size_t ldc, bool accumulate);

// As MetalBackend.h's gemmQ8: one operand int8 weights with float scales, expanded on the device.
bool int8Available();
void gemmQ8(bool quantLeft, std::size_t M, std::size_t N, std::size_t K, const std::int8_t* q, const float* scales, std::size_t groupsPerRow,
            std::size_t group, const float* x, std::size_t ldx, float* C, std::size_t ldc);

// out (N x C) = softmax(Q K^T / sqrt(C / heads)) V per head; Q is N x C, K and V are M x C.
bool attentionSupported(std::size_t headSize);
void attention(std::size_t N, std::size_t M, std::size_t C, std::size_t heads, const float* q, const float* k, const float* v, float* out);

// gemm with one operand constant (constA: A, else B): inside a weight scope it is uploaded once and
// kept on the device, keyed by its host address and shape.
void gemmConst(bool constA, bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K, const float* A, std::size_t lda,
               const float* B, std::size_t ldb, float* C, std::size_t ldc, bool accumulate);

// A whole convolution on the device: `batch` NCHW pictures (C x H x W) -> batch x outC x OH x OW.
// weights: patch x outC (outByPatch false) or outC x patch, patch = C * k * k; bias may be null.
void conv2d(const float* in, std::size_t batch, std::size_t C, std::size_t H, std::size_t W, const float* weights, bool outByPatch,
            const float* bias, std::size_t outC, std::size_t k, std::size_t stride, std::size_t pad, float* out);

// Weight scopes (Backend.h, GpuWeightScope): copies live while at least one is open.
void beginWeightScope();
void endWeightScope();
// Host memory [p, p + bytes) was freed: drop device copies made from it.
void forgetHost(const void* p, std::size_t bytes);

// Device memory in use by the backend's buffers and resident weights (bytes), for diagnostics.
std::size_t deviceBytesInUse();

} // namespace cuda
} // namespace nc

#endif // NEURACRAFT_CUDA_BACKEND_H
