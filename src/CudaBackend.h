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

// ---- Device graphs: whole passes (a UNet step, a VAE decode) with the activations on the device ----
//
// A Session holds the backend for one pass: buffers come from the stream's memory pool, the host
// parameters it uploads (norm scales, biases) live until it ends, and its destructor waits for the
// stream. Layer weights of 64 KB or more are resident as with gemmConst. All pointers below are
// device pointers except the ones named host*.
namespace dev {
class Session {
public:
    Session();
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
};
float* alloc(std::size_t floats);
void release(float* p);
void upload(float* dst, const float* host, std::size_t floats);
void download(float* host, const float* src, std::size_t floats);
void copy(float* dst, const float* src, std::size_t floats);
void fill(float* dst, float value, std::size_t floats);

// NCHW pictures (B x C x H x W).
void conv2d(const float* in, std::size_t B, std::size_t C, std::size_t H, std::size_t W, const float* hostWeights, bool outByPatch,
            const float* hostBias, std::size_t outC, std::size_t k, std::size_t stride, std::size_t pad, float* out);
void groupNorm(const float* in, float* out, std::size_t B, std::size_t C, std::size_t HW, std::size_t groups, float eps, const float* hostGamma,
               const float* hostBeta, bool silu);
void silu(float* x, std::size_t n);
void add(float* y, const float* x, std::size_t n);                                          // y += x
void addChannels(float* x, const float* hostPerChannel, std::size_t B, std::size_t C, std::size_t HW);  // x[b, c, :] += v[c]
void upsample(const float* in, float* out, std::size_t B, std::size_t C, std::size_t H, std::size_t W, std::size_t factor);
void pad(const float* in, float* out, std::size_t B, std::size_t C, std::size_t H, std::size_t W, std::size_t top, std::size_t bottom,
         std::size_t left, std::size_t right);
// One picture's channels <-> tokens: C x HW <-> HW x C.
void toTokens(const float* in, float* out, std::size_t C, std::size_t HW);
void fromTokens(const float* in, float* out, std::size_t HW, std::size_t C);

// Rows of tokens (N x features).
// y (N x out) = x W + b with W host in x out (Dense); transW: W is out x in (PyTorch Linear).
void linear(const float* x, std::size_t N, std::size_t in, const float* hostW, bool transW, const float* hostBias, std::size_t out, float* y);
void layerNorm(const float* x, float* y, std::size_t N, std::size_t C, float eps, const float* hostGamma, const float* hostBeta);
// out (N x C) = softmax(Q K^T / sqrt(dh)) V per head; Q is N x C, K and V are M x C.
void attention(const float* q, const float* k, const float* v, float* out, std::size_t N, std::size_t M, std::size_t C, std::size_t heads);
// out (N x H) = g[:, :H] * gelu(g[:, H:]) for g N x 2H (diffusers GEGLU, exact erf GELU).
void geglu(const float* g, float* out, std::size_t N, std::size_t H);
// y (N x C) = a * x + y
void axpy(float* y, const float* x, float a, std::size_t n);
} // namespace dev

} // namespace cuda
} // namespace nc

#endif // NEURACRAFT_CUDA_BACKEND_H
