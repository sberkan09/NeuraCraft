#ifndef NEURACRAFT_BACKEND_H
#define NEURACRAFT_BACKEND_H

#include "Matrix.h"

#include <cstddef>
#include <string>

namespace nc {

// Where large matrix products run.
//   CPU - multi-threaded blocked kernel, or Apple Accelerate (AMX) on macOS
//   GPU - Metal compute shaders on Apple silicon (when built with Metal)
enum class Device { CPU, GPU };

bool gpuAvailable();
// Switches the active device. Returns false (and stays on CPU) if the GPU is
// not available in this build or on this machine.
bool setDevice(Device device);
Device currentDevice();
// e.g. "CPU (Accelerate, 10 threads)" or "GPU (Apple M4, Metal)".
std::string deviceDescription();
std::string cpuBackendName();  // "Accelerate" or "NeuraCraft blocked kernel"
std::string gpuName();         // empty if no GPU

// In GPU mode, products with fewer than this many multiply-adds (M*N*K) still
// run on the CPU, where they are faster than a GPU round trip.
void setGpuMinWork(std::size_t multiplyAdds);
std::size_t gpuMinWork();

// Row-major single-precision GEMM: C = op(A) * op(B) (+ C if accumulate).
// op(A) is M x K, op(B) is K x N. transA means A is stored K x M (lda >= M),
// transB means B is stored N x K (ldb >= K). Dispatches to the active device.
void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
          const float* A, std::size_t lda, const float* B, std::size_t ldb,
          float* C, std::size_t ldc, bool accumulate = false);

// C = op(A) * op(B) for whole matrices (C must already be M x N). This is what
// Matrix products use: same dispatch as gemm(), and on the GPU page-backed
// matrices (Matrix::pageBacked) are mapped in place instead of copied.
void matmul(const Matrix& a, bool transA, const Matrix& b, bool transB, Matrix& c);

// Direct access to each implementation (used by tests and benchmarks).
void gemmCpu(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
             const float* A, std::size_t lda, const float* B, std::size_t ldb,
             float* C, std::size_t ldc, bool accumulate = false);
// NeuraCraft's own multi-threaded, cache-blocked kernel (the CPU path when
// Accelerate is not available).
void gemmBlocked(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
                 const float* A, std::size_t lda, const float* B, std::size_t ldb,
                 float* C, std::size_t ldc, bool accumulate = false);
void gemmReference(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
                   const float* A, std::size_t lda, const float* B, std::size_t ldb,
                   float* C, std::size_t ldc, bool accumulate = false);
// Throws std::runtime_error if no GPU is available.
void gemmGpu(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
             const float* A, std::size_t lda, const float* B, std::size_t ldb,
             float* C, std::size_t ldc, bool accumulate = false);

} // namespace nc

#endif // NEURACRAFT_BACKEND_H
