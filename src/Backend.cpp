#include "Backend.h"
#include "Parallel.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NEURACRAFT_HAS_ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

#ifdef NEURACRAFT_HAS_METAL
#include "MetalBackend.h"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define NC_RESTRICT __restrict__
#else
#define NC_RESTRICT
#endif

namespace nc {

namespace {

std::atomic<Device> gDevice{Device::CPU};
std::atomic<std::size_t> gGpuMinWork{std::size_t(1) << 22};  // ~4M multiply-adds

// Register tile. 6x16 fits in 16 AVX or 32 NEON registers; plain SSE (16
// four-wide registers) does best with 4x16. Override with -DNC_GEMM_MR/NR.
#ifndef NC_GEMM_MR
#if defined(__AVX__) || defined(__aarch64__) || defined(_M_ARM64)
#define NC_GEMM_MR 6
#else
#define NC_GEMM_MR 4
#endif
#endif
#ifndef NC_GEMM_NR
#define NC_GEMM_NR 16
#endif
constexpr std::size_t MR = NC_GEMM_MR;   // rows of the register tile
constexpr std::size_t NR = NC_GEMM_NR;   // columns of the register tile
constexpr std::size_t KC = 256;          // K block: a B panel slice is KC x NR (16 KB)

// op(A) (M x K) -> panels of MR rows, k-major: packed[(panel*K + k)*MR + i].
void packA(bool trans, std::size_t M, std::size_t K, const float* A, std::size_t lda, float* packed) {
    const std::size_t panels = (M + MR - 1) / MR;
    parallelFor(0, panels, std::max<std::size_t>(1, 4096 / (K + 1)), [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            float* dst = packed + p * K * MR;
            const std::size_t r0 = p * MR;
            const std::size_t height = std::min(MR, M - r0);
            if (trans) {
                // A is stored K x M: each k reads `height` contiguous floats.
                for (std::size_t k = 0; k < K; ++k) {
                    const float* src = A + k * lda + r0;
                    float* d = dst + k * MR;
                    for (std::size_t i = 0; i < height; ++i) d[i] = src[i];
                    for (std::size_t i = height; i < MR; ++i) d[i] = 0.0f;
                }
            } else {
                // `height` sequential read streams, one contiguous write stream.
                const float* rows[MR];
                for (std::size_t i = 0; i < MR; ++i) rows[i] = A + (r0 + std::min(i, height - 1)) * lda;
                for (std::size_t k = 0; k < K; ++k) {
                    float* d = dst + k * MR;
                    for (std::size_t i = 0; i < MR; ++i) d[i] = i < height ? rows[i][k] : 0.0f;
                }
            }
        }
    });
}

// op(B) (K x N) -> panels of NR columns, k-major: packed[(panel*K + k)*NR + j].
void packB(bool trans, std::size_t N, std::size_t K, const float* B, std::size_t ldb, float* packed) {
    const std::size_t panels = (N + NR - 1) / NR;
    parallelFor(0, panels, std::max<std::size_t>(1, 4096 / (K + 1)), [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            float* dst = packed + p * K * NR;
            const std::size_t c0 = p * NR;
            const std::size_t width = std::min(NR, N - c0);
            if (trans) {
                // B is stored N x K: read each of the panel's columns contiguously.
                for (std::size_t j = 0; j < NR; ++j) {
                    if (j >= width) {
                        for (std::size_t k = 0; k < K; ++k) dst[k * NR + j] = 0.0f;
                        continue;
                    }
                    const float* src = B + (c0 + j) * ldb;
                    for (std::size_t k = 0; k < K; ++k) dst[k * NR + j] = src[k];
                }
            } else {
                for (std::size_t k = 0; k < K; ++k) {
                    float* row = dst + k * NR;
                    const float* src = B + k * ldb + c0;
                    for (std::size_t j = 0; j < width; ++j) row[j] = src[j];
                    for (std::size_t j = width; j < NR; ++j) row[j] = 0.0f;
                }
            }
        }
    });
}

// acc (MR x NR) = sum over kc of a-panel column * b-panel row. Fixed sizes let
// the compiler keep acc in vector registers.
#if defined(__GNUC__) || defined(__clang__)
// Explicit SIMD vectors (GCC/Clang extension; SSE/AVX on x86, NEON on ARM).
#if defined(__AVX__)
constexpr std::size_t VW = 8;
typedef float vfloat __attribute__((vector_size(32)));
#else
constexpr std::size_t VW = 4;
typedef float vfloat __attribute__((vector_size(16)));
#endif
static_assert(NR % VW == 0, "NR must be a multiple of the vector width");
constexpr std::size_t NV = NR / VW;

inline vfloat loadv(const float* p) {
    vfloat v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline void microKernel(std::size_t kc, const float* NC_RESTRICT a, const float* NC_RESTRICT b,
                        float (&out)[MR][NR]) {
    vfloat acc[MR][NV];
    for (std::size_t i = 0; i < MR; ++i)
        for (std::size_t v = 0; v < NV; ++v) acc[i][v] = vfloat{} ;
    for (std::size_t k = 0; k < kc; ++k) {
        vfloat bv[NV];
        for (std::size_t v = 0; v < NV; ++v) bv[v] = loadv(b + k * NR + v * VW);
        const float* ak = a + k * MR;
        for (std::size_t i = 0; i < MR; ++i) {
            const float ai = ak[i];
            for (std::size_t v = 0; v < NV; ++v) acc[i][v] += ai * bv[v];
        }
    }
    for (std::size_t i = 0; i < MR; ++i)
        for (std::size_t v = 0; v < NV; ++v) std::memcpy(&out[i][v * VW], &acc[i][v], sizeof(vfloat));
}
#else
inline void microKernel(std::size_t kc, const float* a, const float* b, float (&out)[MR][NR]) {
    float acc[MR][NR] = {};
    for (std::size_t k = 0; k < kc; ++k)
        for (std::size_t i = 0; i < MR; ++i)
            for (std::size_t j = 0; j < NR; ++j) acc[i][j] += a[k * MR + i] * b[k * NR + j];
    for (std::size_t i = 0; i < MR; ++i)
        for (std::size_t j = 0; j < NR; ++j) out[i][j] = acc[i][j];
}
#endif

void checkDims(std::size_t M, std::size_t N, std::size_t K, bool transA, bool transB,
               std::size_t lda, std::size_t ldb, std::size_t ldc) {
    if ((transA ? lda < M : lda < K) || (transB ? ldb < K : ldb < N) || ldc < N)
        throw std::invalid_argument("gemm: leading dimension too small");
}

} // namespace

void gemmBlocked(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
                 const float* A, std::size_t lda, const float* B, std::size_t ldb,
                 float* C, std::size_t ldc, bool accumulate) {
    checkDims(M, N, K, transA, transB, lda, ldb, ldc);
    if (M == 0 || N == 0) return;
    if (!accumulate)
        for (std::size_t i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
    if (K == 0) return;

    const std::size_t aPanels = (M + MR - 1) / MR;
    const std::size_t bPanels = (N + NR - 1) / NR;
    std::vector<float> packedA(aPanels * MR * K), packedB(bPanels * NR * K);
    packA(transA, M, K, A, lda, packedA.data());
    packB(transB, N, K, B, ldb, packedB.data());

    // Threads own disjoint row panels, so results do not depend on thread count.
    const std::size_t perPanel = MR * N * K;
    const std::size_t grain = std::max<std::size_t>(1, (std::size_t(1) << 18) / std::max<std::size_t>(1, perPanel));
    parallelFor(0, aPanels, grain, [&](std::size_t ap0, std::size_t ap1) {
        float acc[MR][NR];
        for (std::size_t k0 = 0; k0 < K; k0 += KC) {
            const std::size_t kc = std::min(KC, K - k0);
            for (std::size_t bp = 0; bp < bPanels; ++bp) {
                const float* bPanel = packedB.data() + (bp * K + k0) * NR;
                const std::size_t c0 = bp * NR;
                const std::size_t width = std::min(NR, N - c0);
                for (std::size_t ap = ap0; ap < ap1; ++ap) {
                    microKernel(kc, packedA.data() + (ap * K + k0) * MR, bPanel, acc);
                    const std::size_t r0 = ap * MR;
                    const std::size_t height = std::min(MR, M - r0);
                    for (std::size_t i = 0; i < height; ++i) {
                        float* c = C + (r0 + i) * ldc + c0;
                        for (std::size_t j = 0; j < width; ++j) c[j] += acc[i][j];
                    }
                }
            }
        }
    });
}

bool gpuAvailable() {
#ifdef NEURACRAFT_HAS_METAL
    return metal::available();
#else
    return false;
#endif
}

bool setDevice(Device device) {
    if (device == Device::GPU && !gpuAvailable()) {
        gDevice = Device::CPU;
        return false;
    }
    gDevice = device;
    return true;
}

Device currentDevice() { return gDevice.load(); }

std::string cpuBackendName() {
#ifdef NEURACRAFT_HAS_ACCELERATE
    return "Accelerate";
#else
    return "NeuraCraft blocked kernel";
#endif
}

std::string gpuName() {
#ifdef NEURACRAFT_HAS_METAL
    return metal::available() ? metal::deviceName() + ", " + metal::kernelName() + " kernel" : std::string();
#else
    return std::string();
#endif
}

std::string deviceDescription() {
    if (currentDevice() == Device::GPU) return "GPU (" + gpuName() + ", Metal)";
    return "CPU (" + cpuBackendName() + ", " + std::to_string(threadCount()) + " threads)";
}

void setGpuMinWork(std::size_t multiplyAdds) { gGpuMinWork = multiplyAdds; }
std::size_t gpuMinWork() { return gGpuMinWork.load(); }

void gemmReference(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
                   const float* A, std::size_t lda, const float* B, std::size_t ldb,
                   float* C, std::size_t ldc, bool accumulate) {
    checkDims(M, N, K, transA, transB, lda, ldb, ldc);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                const float a = transA ? A[k * lda + i] : A[i * lda + k];
                const float b = transB ? B[j * ldb + k] : B[k * ldb + j];
                acc += static_cast<double>(a) * b;
            }
            C[i * ldc + j] = static_cast<float>(acc) + (accumulate ? C[i * ldc + j] : 0.0f);
        }
}

void gemmCpu(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
             const float* A, std::size_t lda, const float* B, std::size_t ldb,
             float* C, std::size_t ldc, bool accumulate) {
    checkDims(M, N, K, transA, transB, lda, ldb, ldc);
    if (M == 0 || N == 0) return;
    if (K == 0) {
        if (!accumulate)
            for (std::size_t i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
        return;
    }
#ifdef NEURACRAFT_HAS_ACCELERATE
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                static_cast<__LAPACK_int>(M), static_cast<__LAPACK_int>(N), static_cast<__LAPACK_int>(K), 1.0f,
                A, static_cast<__LAPACK_int>(lda), B, static_cast<__LAPACK_int>(ldb), accumulate ? 1.0f : 0.0f,
                C, static_cast<__LAPACK_int>(ldc));
#else
    gemmBlocked(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
#endif
}

void gemmGpu(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
             const float* A, std::size_t lda, const float* B, std::size_t ldb,
             float* C, std::size_t ldc, bool accumulate) {
    checkDims(M, N, K, transA, transB, lda, ldb, ldc);
#ifdef NEURACRAFT_HAS_METAL
    if (!metal::available()) throw std::runtime_error("gemmGpu: no Metal device available");
    if (M == 0 || N == 0) return;
    metal::gemm(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
#else
    (void)transA; (void)transB; (void)M; (void)N; (void)K; (void)A; (void)lda; (void)B; (void)ldb;
    (void)C; (void)ldc; (void)accumulate;
    throw std::runtime_error("gemmGpu: NeuraCraft was built without GPU support");
#endif
}

void matmul(const Matrix& a, bool transA, const Matrix& b, bool transB, Matrix& c) {
    const std::size_t M = transA ? a.cols() : a.rows();
    const std::size_t K = transA ? a.rows() : a.cols();
    const std::size_t N = transB ? b.rows() : b.cols();
    if ((transB ? b.cols() : b.rows()) != K || c.rows() != M || c.cols() != N)
        throw std::invalid_argument("matmul: shape mismatch");
    if (M == 0 || N == 0) return;
#ifdef NEURACRAFT_HAS_METAL
    if (currentDevice() == Device::GPU && M * N * K >= gpuMinWork() && K > 0 && metal::available()) {
        metal::gemm(transA, transB, M, N, K, a.data(), a.cols(), b.data(), b.cols(), c.data(), N, false,
                    a.pageBacked() ? a.storageBytes() : 0, b.pageBacked() ? b.storageBytes() : 0,
                    c.pageBacked() ? c.storageBytes() : 0);
        return;
    }
#endif
    gemm(transA, transB, M, N, K, a.data(), a.cols(), b.data(), b.cols(), c.data(), N, false);
}

void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
          const float* A, std::size_t lda, const float* B, std::size_t ldb,
          float* C, std::size_t ldc, bool accumulate) {
    if (currentDevice() == Device::GPU && M * N * K >= gpuMinWork()) {
        gemmGpu(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
    } else {
        gemmCpu(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
    }
}

} // namespace nc
