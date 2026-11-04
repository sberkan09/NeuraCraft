// Metal (Apple GPU) implementation of NeuraCraft's GEMM. Compiled only on Apple
// platforms when NEURACRAFT_USE_METAL is ON. The shader is compiled at runtime
// from source, so building does not need the offline Metal compiler (full Xcode).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBackend.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace nc {
namespace metal {

namespace {

const char* const kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint M; uint N; uint K;
    uint lda; uint ldb; uint ldc;
    uint transA; uint transB; uint accumulate;
};

#define TS 32   // output tile per threadgroup (TS x TS)
#define TK 16   // K slice loaded per iteration
#define TPD 8   // threads per dimension (8 x 8 = 64 threads)
#define WPT 4   // outputs per thread per dimension (TS / TPD)

kernel void gemm_tiled(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device float* C [[buffer(2)]],
                       constant GemmParams& p [[buffer(3)]],
                       uint2 tid [[thread_position_in_threadgroup]],
                       uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float As[TK][TS + 1];
    threadgroup float Bs[TK][TS + 1];

    const uint row0 = group.y * TS;
    const uint col0 = group.x * TS;
    const uint lid = tid.y * TPD + tid.x;

    float acc[WPT][WPT];
    for (uint i = 0; i < WPT; ++i)
        for (uint j = 0; j < WPT; ++j) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < p.K; k0 += TK) {
        for (uint e = lid; e < TK * TS; e += TPD * TPD) {
            // Index order chosen so neighbouring threads read neighbouring memory.
            uint ka, ma;
            if (p.transA) { ka = e / TS; ma = e % TS; } else { ma = e / TK; ka = e % TK; }
            const uint gr = row0 + ma, gka = k0 + ka;
            float va = 0.0f;
            if (gr < p.M && gka < p.K) va = p.transA ? A[gka * p.lda + gr] : A[gr * p.lda + gka];
            As[ka][ma] = va;

            uint kb, nb;
            if (p.transB) { nb = e / TK; kb = e % TK; } else { kb = e / TS; nb = e % TS; }
            const uint gc = col0 + nb, gkb = k0 + kb;
            float vb = 0.0f;
            if (gc < p.N && gkb < p.K) vb = p.transB ? B[gc * p.ldb + gkb] : B[gkb * p.ldb + gc];
            Bs[kb][nb] = vb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TK; ++kk) {
            float a[WPT], b[WPT];
            for (uint i = 0; i < WPT; ++i) a[i] = As[kk][tid.y + i * TPD];
            for (uint j = 0; j < WPT; ++j) b[j] = Bs[kk][tid.x + j * TPD];
            for (uint i = 0; i < WPT; ++i)
                for (uint j = 0; j < WPT; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < WPT; ++i) {
        const uint r = row0 + tid.y + i * TPD;
        if (r >= p.M) continue;
        for (uint j = 0; j < WPT; ++j) {
            const uint c = col0 + tid.x + j * TPD;
            if (c >= p.N) continue;
            const uint idx = r * p.ldc + c;
            C[idx] = p.accumulate ? C[idx] + acc[i][j] : acc[i][j];
        }
    }
}
)METAL";

const char* const kSimdShaderSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct GemmParams {
    uint M; uint N; uint K;
    uint lda; uint ldb; uint ldc;
    uint transA; uint transB; uint accumulate;
};

// 64x64 output tile per threadgroup, 4 simdgroups (128 threads), each owning a
// 32x32 block = 4x4 hardware 8x8 matrices. Needs Apple7+ (M1 and later).
#define BM 64
#define BN 64
#define BK 16

kernel void gemm_simd(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]],
                      constant GemmParams& p [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint tidx [[thread_index_in_threadgroup]],
                      uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];
    threadgroup float Cs[BM * BN];

    const uint row0 = group.y * BM;
    const uint col0 = group.x * BN;
    const uint sgRow = (sgid / 2) * 32;
    const uint sgCol = (sgid % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint e = tidx; e < BM * BK; e += 128) {
            uint r, k;
            if (p.transA) { r = e % BM; k = e / BM; } else { r = e / BK; k = e % BK; }
            const uint gr = row0 + r, gk = k0 + k;
            float v = 0.0f;
            if (gr < p.M && gk < p.K) v = p.transA ? A[gk * p.lda + gr] : A[gr * p.lda + gk];
            As[r * BK + k] = v;
        }
        for (uint e = tidx; e < BK * BN; e += 128) {
            uint k, c;
            if (p.transB) { c = e / BK; k = e % BK; } else { k = e / BN; c = e % BN; }
            const uint gc = col0 + c, gk = k0 + k;
            float v = 0.0f;
            if (gc < p.N && gk < p.K) v = p.transB ? B[gc * p.ldb + gk] : B[gk * p.ldb + gc];
            Bs[k * BN + c] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sgRow + i * 8) * BK + kk, BK);
            for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], Bs + kk * BN + sgCol + j * 8, BN);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], Cs + (sgRow + i * 8) * BN + sgCol + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tidx; e < BM * BN; e += 128) {
        const uint r = row0 + e / BN, c = col0 + e % BN;
        if (r < p.M && c < p.N) {
            const uint idx = r * p.ldc + c;
            C[idx] = p.accumulate ? C[idx] + Cs[e] : Cs[e];
        }
    }
}
)METAL";

struct GemmParams {
    std::uint32_t M, N, K, lda, ldb, ldc, transA, transB, accumulate;
};

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> gemm = nil;       // portable tiled kernel
    id<MTLComputePipelineState> gemmSimd = nil;   // simdgroup-matrix kernel (Apple7+)
    bool useSimd = false;
    std::string name;
    std::string error;
    bool ok = false;
};

Context& context() {
    static Context ctx;
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            ctx.device = MTLCreateSystemDefaultDevice();
            if (ctx.device == nil) {
                ctx.error = "no Metal device";
                return;
            }
            ctx.name = [[ctx.device name] UTF8String];
            ctx.queue = [ctx.device newCommandQueue];
            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:kShaderSource];
            id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:&err];
            if (lib == nil) {
                ctx.error = std::string("shader compilation failed: ") +
                            (err ? [[err localizedDescription] UTF8String] : "unknown error");
                return;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_tiled"];
            if (fn == nil) {
                ctx.error = "gemm_tiled function missing";
                return;
            }
            ctx.gemm = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
            if (ctx.gemm == nil) {
                ctx.error = std::string("pipeline creation failed: ") +
                            (err ? [[err localizedDescription] UTF8String] : "unknown error");
                return;
            }
            ctx.ok = ctx.queue != nil;
            if (!ctx.ok) {
                ctx.error = "could not create a command queue";
                return;
            }

            // Faster kernel on M1 and later; any failure just keeps the tiled one.
            bool wantSimd = true;
            if (@available(macOS 11.0, iOS 14.0, *)) {
                wantSimd = [ctx.device supportsFamily:MTLGPUFamilyApple7];
            } else {
                wantSimd = false;
            }
            const char* forced = std::getenv("NEURACRAFT_METAL_KERNEL");
            if (forced && std::string(forced) == "tiled") wantSimd = false;
            if (wantSimd) {
                NSError* simdErr = nil;
                NSString* simdSrc = [NSString stringWithUTF8String:kSimdShaderSource];
                id<MTLLibrary> simdLib = [ctx.device newLibraryWithSource:simdSrc options:nil error:&simdErr];
                id<MTLFunction> simdFn = simdLib ? [simdLib newFunctionWithName:@"gemm_simd"] : nil;
                if (simdFn) ctx.gemmSimd = [ctx.device newComputePipelineStateWithFunction:simdFn error:&simdErr];
                ctx.useSimd = ctx.gemmSimd != nil && [ctx.gemmSimd maxTotalThreadsPerThreadgroup] >= 128;
            }
        }
    });
    return ctx;
}

std::uint32_t to32(std::size_t v) {
    if (v > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("Metal gemm: dimension too large");
    return static_cast<std::uint32_t>(v);
}

} // namespace

bool available() { return context().ok; }

std::string deviceName() { return context().name; }

std::string kernelName() { return context().useSimd ? "simdgroup" : "tiled"; }

void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
          const float* A, std::size_t lda, const float* B, std::size_t ldb,
          float* C, std::size_t ldc, bool accumulate) {
    Context& ctx = context();
    if (!ctx.ok) throw std::runtime_error("Metal unavailable: " + ctx.error);
    if (M == 0 || N == 0) return;
    if (K == 0) {
        if (!accumulate)
            for (std::size_t i = 0; i < M; ++i) std::memset(C + i * ldc, 0, N * sizeof(float));
        return;
    }

    GemmParams params{to32(M), to32(N), to32(K), to32(lda), to32(ldb), to32(N),
                      transA ? 1u : 0u, transB ? 1u : 0u, accumulate ? 1u : 0u};
    // Bytes actually spanned by each strided operand.
    const std::size_t aRows = transA ? K : M, aCols = transA ? M : K;
    const std::size_t bRows = transB ? N : K, bCols = transB ? K : N;
    const std::size_t aBytes = ((aRows - 1) * lda + aCols) * sizeof(float);
    const std::size_t bBytes = ((bRows - 1) * ldb + bCols) * sizeof(float);
    const std::size_t cBytes = M * N * sizeof(float);

    std::string failure;
    @autoreleasepool {
        id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A length:aBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B length:bBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [ctx.device newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
        if (bufA == nil || bufB == nil || bufC == nil) {
            failure = "could not allocate GPU buffers";
        } else {
            float* out = static_cast<float*>([bufC contents]);
            if (accumulate)
                for (std::size_t i = 0; i < M; ++i) std::memcpy(out + i * N, C + i * ldc, N * sizeof(float));

            id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:(ctx.useSimd ? ctx.gemmSimd : ctx.gemm)];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBytes:&params length:sizeof(params) atIndex:3];
            if (ctx.useSimd) {
                [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1)
                    threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            if ([cb status] == MTLCommandBufferStatusError) {
                NSError* err = [cb error];
                failure = std::string("command buffer failed: ") +
                          (err ? [[err localizedDescription] UTF8String] : "unknown error");
            } else {
                for (std::size_t i = 0; i < M; ++i) std::memcpy(C + i * ldc, out + i * N, N * sizeof(float));
            }
        }
    }
    if (!failure.empty()) throw std::runtime_error("Metal gemm: " + failure);
}

} // namespace metal
} // namespace nc
