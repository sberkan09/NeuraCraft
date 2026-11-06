#include "Backend.h"
#include "Profile.h"
#include "Parallel.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
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
#ifdef NEURACRAFT_HAS_CUDA
#include "CudaBackend.h"
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
#if defined(__AVX512F__)
#define NC_GEMM_MR 12  // 12 x 2 sixteen-wide accumulators of the 32 AVX-512 registers
#elif defined(__AVX__) || defined(__aarch64__) || defined(_M_ARM64)
#define NC_GEMM_MR 6
#else
#define NC_GEMM_MR 4
#endif
#endif
#ifndef NC_GEMM_NR
#if defined(__AVX512F__)
#define NC_GEMM_NR 32
#else
#define NC_GEMM_NR 16
#endif
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
#if defined(__AVX512F__)
constexpr std::size_t VW = 16;
typedef float vfloat __attribute__((vector_size(64)));
#elif defined(__AVX__)
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
// The same with A read in place: row i, column k at a[i * stride + off[k]].
inline void microKernelIndirect(std::size_t kc, const float* NC_RESTRICT a, std::size_t stride, const std::int64_t* NC_RESTRICT off,
                                const float* NC_RESTRICT b, float (&out)[MR][NR]) {
    vfloat acc[MR][NV];
    for (std::size_t i = 0; i < MR; ++i)
        for (std::size_t v = 0; v < NV; ++v) acc[i][v] = vfloat{};
    for (std::size_t k = 0; k < kc; ++k) {
        vfloat bv[NV];
        for (std::size_t v = 0; v < NV; ++v) bv[v] = loadv(b + k * NR + v * VW);
        const float* ak = a + off[k];
        for (std::size_t i = 0; i < MR; ++i) {
            const float ai = ak[i * stride];
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
inline void microKernelIndirect(std::size_t kc, const float* a, std::size_t stride, const std::int64_t* off, const float* b, float (&out)[MR][NR]) {
    float acc[MR][NR] = {};
    for (std::size_t k = 0; k < kc; ++k)
        for (std::size_t i = 0; i < MR; ++i)
            for (std::size_t j = 0; j < NR; ++j) acc[i][j] += a[i * stride + off[k]] * b[k * NR + j];
    for (std::size_t i = 0; i < MR; ++i)
        for (std::size_t j = 0; j < NR; ++j) out[i][j] = acc[i][j];
}
#endif

#if defined(__GNUC__) || defined(__clang__)
float dotProduct(const float* NC_RESTRICT a, const float* NC_RESTRICT b, std::size_t n) {
    vfloat acc0{}, acc1{};
    std::size_t k = 0;
    for (; k + 2 * VW <= n; k += 2 * VW) {
        acc0 += loadv(a + k) * loadv(b + k);
        acc1 += loadv(a + k + VW) * loadv(b + k + VW);
    }
    acc0 += acc1;
    float lanes[VW];
    std::memcpy(lanes, &acc0, sizeof(lanes));
    float sum = 0.0f;
    for (std::size_t l = 0; l < VW; ++l) sum += lanes[l];
    for (; k < n; ++k) sum += a[k] * b[k];
    return sum;
}
#else
float dotProduct(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t k = 0; k < n; ++k) sum += a[k] * b[k];
    return sum;
}
#endif

// A few rows of A (token-by-token inference): stream B once, no packing.
void gemmFewRows(bool transB, std::size_t M, std::size_t N, std::size_t K, const float* A, std::size_t lda,
                 const float* B, std::size_t ldb, float* C, std::size_t ldc) {
    const std::size_t grain = std::max<std::size_t>(16, (std::size_t(1) << 15) / std::max<std::size_t>(1, K * M));
    if (transB) {
        // C[i][j] += <A[i,:], B[j,:]>: both rows are contiguous.
        parallelFor(0, N, grain, [&](std::size_t j0, std::size_t j1) {
            for (std::size_t j = j0; j < j1; ++j)
                for (std::size_t i = 0; i < M; ++i) C[i * ldc + j] += dotProduct(A + i * lda, B + j * ldb, K);
        });
        return;
    }
    // C[i][j0..j1) += sum_k A[i][k] * B[k][j0..j1): each B row segment feeds all M rows.
    parallelFor(0, N, std::max<std::size_t>(grain, 64), [&](std::size_t j0, std::size_t j1) {
        const std::size_t width = j1 - j0;
        for (std::size_t k = 0; k < K; ++k) {
            const float* NC_RESTRICT b = B + k * ldb + j0;
            for (std::size_t i = 0; i < M; ++i) {
                const float a = A[i * lda + k];
                if (a == 0.0f) continue;
                float* NC_RESTRICT c = C + i * ldc + j0;
                for (std::size_t j = 0; j < width; ++j) c[j] += a * b[j];
            }
        }
    });
}

void checkDims(std::size_t M, std::size_t N, std::size_t K, bool transA, bool transB,
               std::size_t lda, std::size_t ldb, std::size_t ldc) {
    if ((transA ? lda < M : lda < K) || (transB ? ldb < K : ldb < N) || ldc < N)
        throw std::invalid_argument("gemm: leading dimension too small");
}

} // namespace

namespace {
// C (M x N) += packed A (panels of MR rows) x packed B (panels of NR columns), both k-major over all K.
// Threads own disjoint row panels, so results do not depend on thread count.
void blockedCore(std::size_t M, std::size_t N, std::size_t K, const float* packedA, const float* packedB, float* C, std::size_t ldc) {
    const std::size_t aPanels = (M + MR - 1) / MR;
    const std::size_t bPanels = (N + NR - 1) / NR;
    const std::size_t perPanel = MR * N * K;
    const std::size_t grain = std::max<std::size_t>(1, (std::size_t(1) << 18) / std::max<std::size_t>(1, perPanel));
    parallelFor(0, aPanels, grain, [&](std::size_t ap0, std::size_t ap1) {
        float acc[MR][NR];
        for (std::size_t k0 = 0; k0 < K; k0 += KC) {
            const std::size_t kc = std::min(KC, K - k0);
            for (std::size_t bp = 0; bp < bPanels; ++bp) {
                const float* bPanel = packedB + (bp * K + k0) * NR;
                const std::size_t c0 = bp * NR;
                const std::size_t width = std::min(NR, N - c0);
                for (std::size_t ap = ap0; ap < ap1; ++ap) {
                    microKernel(kc, packedA + (ap * K + k0) * MR, bPanel, acc);
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
} // namespace

void gemmBlocked(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
                 const float* A, std::size_t lda, const float* B, std::size_t ldb,
                 float* C, std::size_t ldc, bool accumulate) {
    checkDims(M, N, K, transA, transB, lda, ldb, ldc);
    if (M == 0 || N == 0) return;
    // Large products go in blocks of at most 2048 columns / rows, so the packed panels stay a few MB
    // (packing all of a 16384 x 1440 operand at once ran at a third of the speed). Each element's
    // sum is formed in the same order either way, so results do not change.
    constexpr std::size_t kBlock = 2048;
    if (N > 2 * kBlock && K > KC) {
        for (std::size_t n0 = 0; n0 < N; n0 += kBlock)
            gemmBlocked(transA, transB, M, std::min(kBlock, N - n0), K, A, lda, transB ? B + n0 * ldb : B + n0, ldb, C + n0, ldc, accumulate);
        return;
    }
    if (M > 2 * kBlock && K > KC) {
        for (std::size_t m0 = 0; m0 < M; m0 += kBlock)
            gemmBlocked(transA, transB, std::min(kBlock, M - m0), N, K, transA ? A + m0 : A + m0 * lda, lda, B, ldb, C + m0 * ldc, ldc, accumulate);
        return;
    }
    if (!accumulate)
        for (std::size_t i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
    if (K == 0) return;
    if (!transA && M <= 4) {
        gemmFewRows(transB, M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }

    const std::size_t aPanels = (M + MR - 1) / MR;
    const std::size_t bPanels = (N + NR - 1) / NR;
    std::vector<float> packedA(aPanels * MR * K), packedB(bPanels * NR * K);
    packA(transA, M, K, A, lda, packedA.data());
    packB(transB, N, K, B, ldb, packedB.data());
    blockedCore(M, N, K, packedA.data(), packedB.data(), C, ldc);
}

std::size_t gemmPanelRows() { return MR; }

void gemmIndirectA(std::size_t M, std::size_t N, std::size_t K, const float* A, std::size_t aStride, const std::int64_t* aOffsets, bool transB,
                   const float* B, std::size_t ldb, float* C, std::size_t ldc) {
    if (M == 0 || N == 0) return;
    for (std::size_t i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
    if (K == 0) return;
    const std::size_t aPanels = (M + MR - 1) / MR, bPanels = (N + NR - 1) / NR;
    std::vector<float> packedB(bPanels * NR * K);
    packB(transB, N, K, B, ldb, packedB.data());
    const std::size_t grain = std::max<std::size_t>(1, (std::size_t(1) << 18) / std::max<std::size_t>(1, MR * N * K));
    parallelFor(0, aPanels, grain, [&](std::size_t ap0, std::size_t ap1) {
        float acc[MR][NR];
        for (std::size_t k0 = 0; k0 < K; k0 += KC) {
            const std::size_t kc = std::min(KC, K - k0);
            for (std::size_t bp = 0; bp < bPanels; ++bp) {
                const float* bPanel = packedB.data() + (bp * K + k0) * NR;
                const std::size_t c0 = bp * NR, width = std::min(NR, N - c0);
                for (std::size_t ap = ap0; ap < ap1; ++ap) {
                    const std::size_t r0 = ap * MR, height = std::min(MR, M - r0);
                    microKernelIndirect(kc, A + r0 * aStride, aStride, aOffsets + k0, bPanel, acc);
                    for (std::size_t i = 0; i < height; ++i) {
                        float* c = C + (r0 + i) * ldc + c0;
                        for (std::size_t j = 0; j < width; ++j) c[j] += acc[i][j];
                    }
                }
            }
        }
    });
}

bool gemmPackedAvailable() {
#ifdef NEURACRAFT_HAS_ACCELERATE
    return false;
#else
    return currentDevice() == Device::CPU;
#endif
}

void gemmPackedA(std::size_t M, std::size_t N, std::size_t K, const float* packedA, bool transB, const float* B, std::size_t ldb, float* C,
                 std::size_t ldc) {
    if (M == 0 || N == 0) return;
    for (std::size_t i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
    if (K == 0) return;
    std::vector<float> packedB(((N + NR - 1) / NR) * NR * K);
    packB(transB, N, K, B, ldb, packedB.data());
    blockedCore(M, N, K, packedA, packedB.data(), C, ldc);
}

float dot(const float* a, const float* b, std::size_t n) { return dotProduct(a, b, n); }

namespace {
std::atomic<bool> gGpuAttention{false};
}
void setGpuAttention(bool enabled) { gGpuAttention = enabled && gpuAvailable(); }
bool gpuAttentionEnabled() { return gGpuAttention.load(); }

void setGpuAttentionKernel(const std::string& name) {
#ifdef NEURACRAFT_HAS_METAL
    if (metal::available()) metal::setAttentionKernel(name);
#else  // (CUDA has one attention implementation)
    (void)name;
#endif
}

std::string gpuAttentionKernel(std::size_t headSize) {
#if defined(NEURACRAFT_HAS_METAL)
    return metal::available() ? metal::attentionKernelName(headSize) : std::string();
#elif defined(NEURACRAFT_HAS_CUDA)
    return cuda::attentionSupported(headSize) ? "cublas" : std::string();
#else
    (void)headSize;
    return std::string();
#endif
}

bool gpuAttention(const Matrix& q, const Matrix& k, const Matrix& v, std::size_t heads, Matrix& out) {
#ifdef NEURACRAFT_HAS_METAL
    const std::size_t N = q.rows(), M = k.rows(), C = q.cols();
    if ((currentDevice() != Device::GPU && !gpuAttentionEnabled()) || heads == 0 || C % heads != 0 || M == 0 || N == 0) return false;
    if (k.cols() != C || v.cols() != C || v.rows() != M || out.rows() != N || out.cols() != C) return false;
    if (2 * N * M * C < gpuMinWork() || !metal::attentionSupported(C / heads)) return false;
    NC_PROFILE("attention gpu");
    auto map = [](const Matrix& m) { return m.pageBacked() ? m.storageBytes() : std::size_t(0); };
    metal::attention(N, M, C, heads, q.data(), map(q), k.data(), map(k), v.data(), map(v), out.data(), map(out));
    return true;
#elif defined(NEURACRAFT_HAS_CUDA)
    const std::size_t N = q.rows(), M = k.rows(), C = q.cols();
    if ((currentDevice() != Device::GPU && !gpuAttentionEnabled()) || heads == 0 || C % heads != 0 || M == 0 || N == 0) return false;
    if (k.cols() != C || v.cols() != C || v.rows() != M || out.rows() != N || out.cols() != C) return false;
    if (2 * N * M * C < gpuMinWork() || !cuda::attentionSupported(C / heads)) return false;
    NC_PROFILE("attention gpu");
    cuda::attention(N, M, C, heads, q.data(), k.data(), v.data(), out.data());
    return true;
#else
    (void)q; (void)k; (void)v; (void)heads; (void)out;
    return false;
#endif
}

bool gpuAvailable() {
#if defined(NEURACRAFT_HAS_METAL)
    return metal::available();
#elif defined(NEURACRAFT_HAS_CUDA)
    return cuda::available();
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
#if defined(NEURACRAFT_HAS_METAL)
    return metal::available() ? metal::deviceName() + ", " + metal::kernelName() + " kernel" : std::string();
#elif defined(NEURACRAFT_HAS_CUDA)
    return cuda::available() ? cuda::deviceName() + ", " + cuda::kernelName() : std::string();
#else
    return std::string();
#endif
}

std::string gpuBackendName() {
#if defined(NEURACRAFT_HAS_METAL)
    return "Metal";
#elif defined(NEURACRAFT_HAS_CUDA)
    return "CUDA";
#else
    return std::string();
#endif
}

bool gpuInt8Available() {
#if defined(NEURACRAFT_HAS_METAL)
    return metal::int8Available();
#elif defined(NEURACRAFT_HAS_CUDA)
    return cuda::int8Available();
#else
    return false;
#endif
}

std::string deviceDescription() {
    if (currentDevice() == Device::GPU) return "GPU (" + gpuName() + ", " + gpuBackendName() + ")";
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
    NC_PROFILE("gemm cpu");
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
    NC_PROFILE("gemm gpu");
    metal::gemm(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
#elif defined(NEURACRAFT_HAS_CUDA)
    if (!cuda::available()) throw std::runtime_error("gemmGpu: no CUDA device available");
    if (M == 0 || N == 0) return;
    NC_PROFILE("gemm gpu");
    cuda::gemm(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
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
        NC_PROFILE("gemm gpu");
        metal::gemm(transA, transB, M, N, K, a.data(), a.cols(), b.data(), b.cols(), c.data(), N, false,
                    a.pageBacked() ? a.storageBytes() : 0, b.pageBacked() ? b.storageBytes() : 0,
                    c.pageBacked() ? c.storageBytes() : 0);
        return;
    }
#endif
    gemm(transA, transB, M, N, K, a.data(), a.cols(), b.data(), b.cols(), c.data(), N, false);
}

GpuWeightScope::GpuWeightScope() {
#ifdef NEURACRAFT_HAS_CUDA
    if (cuda::available()) cuda::beginWeightScope();
#endif
}
GpuWeightScope::~GpuWeightScope() {
#ifdef NEURACRAFT_HAS_CUDA
    if (cuda::available()) cuda::endWeightScope();
#endif
}

bool preferFloatWeights() {
#ifdef NEURACRAFT_HAS_CUDA
    static const bool int8 = [] {
        const char* e = std::getenv("NEURACRAFT_GPU_INT8");
        return e && std::string(e) == "1";
    }();
    return !int8 && currentDevice() == Device::GPU && cuda::available();
#else
    return false;
#endif
}

void gpuHostMemoryFreed(const void* p, std::size_t bytes) {
#ifdef NEURACRAFT_HAS_CUDA
    cuda::forgetHost(p, bytes);
#else
    (void)p;
    (void)bytes;
#endif
}

void gemmConst(bool constA, bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K, const float* A, std::size_t lda,
               const float* B, std::size_t ldb, float* C, std::size_t ldc, bool accumulate) {
#ifdef NEURACRAFT_HAS_CUDA
    if (currentDevice() == Device::GPU && M * N * K >= gpuMinWork() && cuda::available()) {
        checkDims(M, N, K, transA, transB, lda, ldb, ldc);
        NC_PROFILE("gemm gpu");
        cuda::gemmConst(constA, transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
        return;
    }
#endif
    (void)constA;
    gemm(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, accumulate);
}

bool gpuConvolution(const float* in, std::size_t batch, std::size_t C, std::size_t H, std::size_t W, const float* weights, bool outByPatch,
                    const float* bias, std::size_t outC, std::size_t k, std::size_t stride, std::size_t pad, float* out) {
#ifdef NEURACRAFT_HAS_CUDA
    if (currentDevice() != Device::GPU || !cuda::available() || k == 0 || stride == 0 || H + 2 * pad < k || W + 2 * pad < k) return false;
    const std::size_t OH = (H + 2 * pad - k) / stride + 1, OW = (W + 2 * pad - k) / stride + 1;
    if (batch * OH * OW * outC * C * k * k < gpuMinWork()) return false;
    NC_PROFILE("conv gpu");
    cuda::conv2d(in, batch, C, H, W, weights, outByPatch, bias, outC, k, stride, pad, out);
    return true;
#else
    (void)in; (void)batch; (void)C; (void)H; (void)W; (void)weights; (void)outByPatch; (void)bias; (void)outC; (void)k; (void)stride; (void)pad; (void)out;
    return false;
#endif
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
