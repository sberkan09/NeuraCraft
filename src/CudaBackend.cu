// CUDA backend: cuBLAS GEMMs (TF32 tensor cores unless NEURACRAFT_CUDA_TF32=0), int8 weights
// expanded on the device, and attention as two strided-batched GEMMs around a softmax kernel.
#include "CudaBackend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace nc {
namespace cuda {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cublas: ") + what + " failed (" + std::to_string(static_cast<int>(s)) + ")");
}

// A device buffer that only grows.
struct Buffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    template <typename T>
    T* get(std::size_t count) {
        const std::size_t need = count * sizeof(T);
        if (need > bytes) {
            const std::size_t grown = std::max(need, bytes + bytes / 2);
            if (p) cudaFree(p);
            p = nullptr;
            bytes = 0;
            check(cudaMalloc(&p, grown), "cudaMalloc");
            bytes = grown;
        }
        return static_cast<T*>(p);
    }
};

struct State {
    bool ok = false;
    std::string name;
    bool tf32 = true;
    cublasHandle_t blas = nullptr;
    cudaStream_t stream = nullptr;
    Buffer a, b, c, s, q8, scales;
    std::mutex mutex;  // one caller at a time: the buffers are shared

    State() {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            cudaGetLastError();
            return;
        }
        int dev = 0;
        if (const char* e = std::getenv("NEURACRAFT_CUDA_DEVICE")) dev = std::atoi(e);
        if (cudaSetDevice(dev) != cudaSuccess) return;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return;
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return;
        cublasSetStream(blas, stream);
        const char* t = std::getenv("NEURACRAFT_CUDA_TF32");
        tf32 = !(t && std::string(t) == "0") && prop.major >= 8;
        cublasSetMathMode(blas, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
        name = std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + ", " +
               std::to_string((prop.totalGlobalMem + (1ull << 29)) >> 30) + " GB)";
        ok = true;
    }
};

State& state() {
    static State s;
    return s;
}

// Copies a host matrix of `rows` rows of `cols` floats (stride ld) to a dense device buffer.
void upload(State& st, float* dst, const float* src, std::size_t rows, std::size_t cols, std::size_t ld) {
    check(cudaMemcpy2DAsync(dst, cols * sizeof(float), src, ld * sizeof(float), cols * sizeof(float), rows, cudaMemcpyHostToDevice, st.stream),
          "upload");
}
void download(State& st, float* dst, std::size_t ld, const float* src, std::size_t rows, std::size_t cols) {
    check(cudaMemcpy2DAsync(dst, ld * sizeof(float), src, cols * sizeof(float), cols * sizeof(float), rows, cudaMemcpyDeviceToHost, st.stream),
          "download");
}

// Row-major C = op(A) op(B) on device operands (dense: lda = A's columns, ...), as a column-major
// product of the transposes: C^T = op(B)^T op(A)^T.
void deviceGemm(State& st, bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K, const float* A, std::size_t lda,
                const float* B, std::size_t ldb, float* C, std::size_t ldc, float beta) {
    const float alpha = 1.0f;
    check(cublasSgemm(st.blas, transB ? CUBLAS_OP_T : CUBLAS_OP_N, transA ? CUBLAS_OP_T : CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
                      static_cast<int>(K), &alpha, B, static_cast<int>(ldb), A, static_cast<int>(lda), &beta, C, static_cast<int>(ldc)),
          "sgemm");
}

__global__ void expandQ8(const signed char* q, const float* scales, std::size_t rows, std::size_t K, std::size_t groupsPerRow, std::size_t group,
                         float* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= rows * K) return;
    const std::size_t r = i / K, k = i % K;
    out[i] = static_cast<float>(q[i]) * scales[r * groupsPerRow + k / group];
}

// Row-wise softmax of x * scale over `cols` values, in place; one block per row.
__global__ void softmaxRows(float* s, std::size_t cols, float scale) {
    float* row = s + blockIdx.x * cols;
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    float m = -INFINITY;
    for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, row[j]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) shared[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < warps ? shared[lane] : -INFINITY;
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) shared[0] = m;
    }
    __syncthreads();
    m = shared[0];
    __syncthreads();
    float sum = 0.0f;
    for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x) {
        const float e = __expf((row[j] - m) * scale);
        row[j] = e;
        sum += e;
    }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < warps ? shared[lane] : 0.0f;
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) shared[0] = sum;
    }
    __syncthreads();
    const float inv = 1.0f / shared[0];
    for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x) row[j] *= inv;
}

} // namespace

bool available() { return state().ok; }
std::string deviceName() { return state().ok ? state().name : std::string(); }
std::string kernelName() { return state().tf32 ? "cuBLAS TF32" : "cuBLAS FP32"; }
bool int8Available() { return state().ok; }
bool attentionSupported(std::size_t headSize) { return state().ok && headSize > 0 && headSize <= 256; }

std::size_t deviceBytesInUse() {
    State& st = state();
    return st.a.bytes + st.b.bytes + st.c.bytes + st.s.bytes + st.q8.bytes + st.scales.bytes;
}

void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K, const float* A, std::size_t lda, const float* B,
          std::size_t ldb, float* C, std::size_t ldc, bool accumulate) {
    State& st = state();
    if (!st.ok) throw std::runtime_error("cuda: no device");
    if (M == 0 || N == 0) return;
    std::lock_guard<std::mutex> lock(st.mutex);
    const std::size_t aRows = transA ? K : M, aCols = transA ? M : K, bRows = transB ? N : K, bCols = transB ? K : N;
    float* dA = st.a.get<float>(aRows * aCols);
    float* dB = st.b.get<float>(bRows * bCols);
    float* dC = st.c.get<float>(M * N);
    upload(st, dA, A, aRows, aCols, lda);
    upload(st, dB, B, bRows, bCols, ldb);
    if (accumulate) upload(st, dC, C, M, N, ldc);
    if (K == 0) check(cudaMemsetAsync(dC, 0, M * N * sizeof(float), st.stream), "memset");
    else deviceGemm(st, transA, transB, M, N, K, dA, aCols, dB, bCols, dC, N, accumulate ? 1.0f : 0.0f);
    download(st, C, ldc, dC, M, N);
    check(cudaStreamSynchronize(st.stream), "gemm");
}

void gemmQ8(bool quantLeft, std::size_t M, std::size_t N, std::size_t K, const std::int8_t* q, const float* scales, std::size_t groupsPerRow,
            std::size_t group, const float* x, std::size_t ldx, float* C, std::size_t ldc) {
    State& st = state();
    if (!st.ok) throw std::runtime_error("cuda: no device");
    if (M == 0 || N == 0) return;
    std::lock_guard<std::mutex> lock(st.mutex);
    const std::size_t qRows = quantLeft ? M : N;
    signed char* dq = st.q8.get<signed char>(qRows * K);
    float* ds = st.scales.get<float>(qRows * groupsPerRow);
    float* dW = st.b.get<float>(qRows * K);
    const std::size_t xRows = quantLeft ? K : M, xCols = quantLeft ? N : K;
    float* dX = st.a.get<float>(xRows * xCols);
    float* dC = st.c.get<float>(M * N);
    check(cudaMemcpyAsync(dq, q, qRows * K, cudaMemcpyHostToDevice, st.stream), "upload q8");
    check(cudaMemcpyAsync(ds, scales, qRows * groupsPerRow * sizeof(float), cudaMemcpyHostToDevice, st.stream), "upload scales");
    upload(st, dX, x, xRows, xCols, ldx);
    const std::size_t total = qRows * K;
    expandQ8<<<static_cast<unsigned>((total + 255) / 256), 256, 0, st.stream>>>(dq, ds, qRows, K, groupsPerRow, group, dW);
    check(cudaGetLastError(), "expandQ8");
    if (quantLeft) deviceGemm(st, false, false, M, N, K, dW, K, dX, N, dC, N, 0.0f);  // C = W (M x K) * X (K x N)
    else deviceGemm(st, false, true, M, N, K, dX, K, dW, K, dC, N, 0.0f);              // C = X (M x K) * W^T, W is N x K
    download(st, C, ldc, dC, M, N);
    check(cudaStreamSynchronize(st.stream), "gemmQ8");
}

void attention(std::size_t N, std::size_t M, std::size_t C, std::size_t heads, const float* q, const float* k, const float* v, float* out) {
    State& st = state();
    if (!st.ok) throw std::runtime_error("cuda: no device");
    std::lock_guard<std::mutex> lock(st.mutex);
    const std::size_t dh = C / heads;
    float* dQ = st.a.get<float>(N * C);
    float* dKV = st.b.get<float>(2 * M * C);
    float* dK = dKV;
    float* dV = dKV + M * C;
    float* dO = st.c.get<float>(N * C);
    check(cudaMemcpyAsync(dQ, q, N * C * sizeof(float), cudaMemcpyHostToDevice, st.stream), "upload q");
    check(cudaMemcpyAsync(dK, k, M * C * sizeof(float), cudaMemcpyHostToDevice, st.stream), "upload k");
    check(cudaMemcpyAsync(dV, v, M * C * sizeof(float), cudaMemcpyHostToDevice, st.stream), "upload v");
    // Heads in groups whose scores (N x M each) fit in about 1 GB.
    const std::size_t perHead = N * M;
    const std::size_t group = std::max<std::size_t>(1, std::min(heads, (std::size_t(1) << 28) / std::max<std::size_t>(1, perHead)));
    float* dS = st.s.get<float>(group * perHead);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh)), one = 1.0f, zero = 0.0f;
    for (std::size_t h0 = 0; h0 < heads; h0 += group) {
        const int g = static_cast<int>(std::min(group, heads - h0));
        // S_h (N x M, row-major) = Q_h K_h^T: column-major S_h^T = K_h Q_h^T.
        check(cublasSgemmStridedBatched(st.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(M), static_cast<int>(N), static_cast<int>(dh), &one,
                                        dK + h0 * dh, static_cast<int>(C), static_cast<long long>(dh), dQ + h0 * dh, static_cast<int>(C),
                                        static_cast<long long>(dh), &zero, dS, static_cast<int>(M), static_cast<long long>(perHead), g),
              "attention scores");
        softmaxRows<<<static_cast<unsigned>(N * static_cast<std::size_t>(g)), 256, 0, st.stream>>>(dS, M, scale);
        check(cudaGetLastError(), "softmax");
        // O_h (N x dh, row stride C) = P_h V_h: column-major O_h^T = V_h^T P_h^T.
        check(cublasSgemmStridedBatched(st.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(dh), static_cast<int>(N), static_cast<int>(M), &one,
                                        dV + h0 * dh, static_cast<int>(C), static_cast<long long>(dh), dS, static_cast<int>(M),
                                        static_cast<long long>(perHead), &zero, dO + h0 * dh, static_cast<int>(C), static_cast<long long>(dh), g),
              "attention values");
    }
    check(cudaMemcpyAsync(out, dO, N * C * sizeof(float), cudaMemcpyDeviceToHost, st.stream), "download");
    check(cudaStreamSynchronize(st.stream), "attention");
}

} // namespace cuda
} // namespace nc
