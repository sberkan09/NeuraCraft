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

// Int8 weight GEMM: tiles of the int8 operand are dequantized (one scale per
// `group` inputs of each weight row) while they are loaded into threadgroup
// memory; the rest is the float kernel. Weights cross the bus as int8.
const char* const kQ8Common = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Q8Params {
    uint M; uint N; uint K;
    uint ldx; uint ldc;
    uint gpr; uint group; uint quantLeft;
};

// Element (r, k) of the left operand and (k, c) of the right one.
inline float q8Left(device const char* Q, device const float* S, device const float* X,
                    constant Q8Params& p, uint r, uint k) {
    return p.quantLeft ? S[r * p.gpr + k / p.group] * float(Q[ulong(r) * p.K + k]) : X[ulong(r) * p.ldx + k];
}
inline float q8Right(device const char* Q, device const float* S, device const float* X,
                     constant Q8Params& p, uint k, uint c) {
    return p.quantLeft ? X[ulong(k) * p.ldx + c] : S[c * p.gpr + k / p.group] * float(Q[ulong(c) * p.K + k]);
}
)METAL";

const char* const kQ8TiledSource = R"METAL(
#define TS 32
#define TK 16
#define TPD 8
#define WPT 4

kernel void gemm_q8_tiled(device const char* Q [[buffer(0)]],
                          device const float* S [[buffer(1)]],
                          device const float* X [[buffer(2)]],
                          device float* C [[buffer(3)]],
                          constant Q8Params& p [[buffer(4)]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float As[TK][TS + 1];
    threadgroup float Bs[TK][TS + 1];

    const uint row0 = tg.y * TS;
    const uint col0 = tg.x * TS;
    const uint lid = tid.y * TPD + tid.x;

    float acc[WPT][WPT];
    for (uint i = 0; i < WPT; ++i)
        for (uint j = 0; j < WPT; ++j) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < p.K; k0 += TK) {
        for (uint e = lid; e < TK * TS; e += TPD * TPD) {
            // Both left operands are row-major M x K; the right one is K x N
            // (quantLeft: activations) or N x K (int8 weight rows).
            const uint ma = e / TK, ka = e % TK;
            const uint gr = row0 + ma, gka = k0 + ka;
            As[ka][ma] = (gr < p.M && gka < p.K) ? q8Left(Q, S, X, p, gr, gka) : 0.0f;

            uint kb, nb;
            if (p.quantLeft) { kb = e / TS; nb = e % TS; } else { nb = e / TK; kb = e % TK; }
            const uint gc = col0 + nb, gkb = k0 + kb;
            Bs[kb][nb] = (gc < p.N && gkb < p.K) ? q8Right(Q, S, X, p, gkb, gc) : 0.0f;
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
            if (c < p.N) C[ulong(r) * p.ldc + c] = acc[i][j];
        }
    }
}
)METAL";

const char* const kQ8SimdSource = R"METAL(
#include <metal_simdgroup_matrix>
#define BM 64
#define BN 64
#define BK 16

kernel void gemm_q8_simd(device const char* Q [[buffer(0)]],
                         device const float* S [[buffer(1)]],
                         device const float* X [[buffer(2)]],
                         device float* C [[buffer(3)]],
                         constant Q8Params& p [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tidx [[thread_index_in_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];
    threadgroup float Cs[BM * BN];

    const uint row0 = tg.y * BM;
    const uint col0 = tg.x * BN;
    const uint sgRow = (sgid / 2) * 32;
    const uint sgCol = (sgid % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint e = tidx; e < BM * BK; e += 128) {
            const uint r = e / BK, k = e % BK;
            const uint gr = row0 + r, gk = k0 + k;
            As[r * BK + k] = (gr < p.M && gk < p.K) ? q8Left(Q, S, X, p, gr, gk) : 0.0f;
        }
        for (uint e = tidx; e < BK * BN; e += 128) {
            uint k, c;
            if (p.quantLeft) { k = e / BN; c = e % BN; } else { c = e / BK; k = e % BK; }
            const uint gc = col0 + c, gk = k0 + k;
            Bs[k * BN + c] = (gc < p.N && gk < p.K) ? q8Right(Q, S, X, p, gk, gc) : 0.0f;
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
        if (r < p.M && c < p.N) C[ulong(r) * p.ldc + c] = Cs[e];
    }
}
)METAL";

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
    id<MTLComputePipelineState> gemmQ8 = nil;     // int8 weights, same kernel family as the float one
    bool useSimd = false;
    std::string name;
    std::string error;
    std::string q8Error;
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
            // NEURACRAFT_METAL_KERNEL=tiled|simdgroup overrides the automatic choice.
            const char* forced = std::getenv("NEURACRAFT_METAL_KERNEL");
            if (forced && std::string(forced) == "tiled") wantSimd = false;
            if (forced && std::string(forced) == "simdgroup") wantSimd = true;
            if (wantSimd) {
                NSError* simdErr = nil;
                NSString* simdSrc = [NSString stringWithUTF8String:kSimdShaderSource];
                id<MTLLibrary> simdLib = [ctx.device newLibraryWithSource:simdSrc options:nil error:&simdErr];
                id<MTLFunction> simdFn = simdLib ? [simdLib newFunctionWithName:@"gemm_simd"] : nil;
                if (simdFn) ctx.gemmSimd = [ctx.device newComputePipelineStateWithFunction:simdFn error:&simdErr];
                ctx.useSimd = ctx.gemmSimd != nil && [ctx.gemmSimd maxTotalThreadsPerThreadgroup] >= 128;
            }

            // Int8 kernel of the same family; if it fails to build, int8 layers
            // keep expanding their weights for the float kernel.
            {
                const std::string q8 = std::string(kQ8Common) + (ctx.useSimd ? kQ8SimdSource : kQ8TiledSource);
                NSError* q8Err = nil;
                id<MTLLibrary> q8Lib = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:q8.c_str()]
                                                                options:nil
                                                                  error:&q8Err];
                id<MTLFunction> q8Fn = q8Lib ? [q8Lib newFunctionWithName:(ctx.useSimd ? @"gemm_q8_simd" : @"gemm_q8_tiled")] : nil;
                if (q8Fn) ctx.gemmQ8 = [ctx.device newComputePipelineStateWithFunction:q8Fn error:&q8Err];
                if (ctx.gemmQ8 != nil && [ctx.gemmQ8 maxTotalThreadsPerThreadgroup] < (ctx.useSimd ? 128u : 64u)) ctx.gemmQ8 = nil;
                if (ctx.gemmQ8 == nil && q8Err != nil) ctx.q8Error = [[q8Err localizedDescription] UTF8String];
            }
        }
    });
    return ctx;
}

std::uint32_t to32(std::size_t v) {
    if (v > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("Metal gemm: dimension too large");
    return static_cast<std::uint32_t>(v);
}

// Wraps page-aligned host memory without copying (unified memory), or copies.
id<MTLBuffer> inputBuffer(Context& ctx, const void* p, std::size_t bytes, std::size_t mapBytes) {
    if (mapBytes >= bytes && mapBytes > 0) {
        id<MTLBuffer> b = [ctx.device newBufferWithBytesNoCopy:const_cast<void*>(p)
                                                        length:mapBytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        if (b != nil) return b;
    }
    return [ctx.device newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
}

struct Q8Params {
    std::uint32_t M, N, K, ldx, ldc, gpr, group, quantLeft;
};

} // namespace

bool available() { return context().ok; }

std::string deviceName() { return context().name; }

std::string kernelName() { return context().useSimd ? "simdgroup" : "tiled"; }

void gemm(bool transA, bool transB, std::size_t M, std::size_t N, std::size_t K,
          const float* A, std::size_t lda, const float* B, std::size_t ldb,
          float* C, std::size_t ldc, bool accumulate,
          std::size_t aMapBytes, std::size_t bMapBytes, std::size_t cMapBytes) {
    Context& ctx = context();
    if (!ctx.ok) throw std::runtime_error("Metal unavailable: " + ctx.error);
    if (M == 0 || N == 0) return;
    if (K == 0) {
        if (!accumulate)
            for (std::size_t i = 0; i < M; ++i) std::memset(C + i * ldc, 0, N * sizeof(float));
        return;
    }

    // C can be written in place only when it is dense (ldc == N).
    const bool mapC = cMapBytes > 0 && ldc == N;
    GemmParams params{to32(M), to32(N), to32(K), to32(lda), to32(ldb), to32(mapC ? ldc : N),
                      transA ? 1u : 0u, transB ? 1u : 0u, accumulate ? 1u : 0u};
    // Bytes actually spanned by each strided operand.
    const std::size_t aRows = transA ? K : M, aCols = transA ? M : K;
    const std::size_t bRows = transB ? N : K, bCols = transB ? K : N;
    const std::size_t aBytes = ((aRows - 1) * lda + aCols) * sizeof(float);
    const std::size_t bBytes = ((bRows - 1) * ldb + bCols) * sizeof(float);
    const std::size_t cBytes = M * N * sizeof(float);

    auto input = [&](const float* p, std::size_t bytes, std::size_t mapBytes) { return inputBuffer(ctx, p, bytes, mapBytes); };

    std::string failure;
    @autoreleasepool {
        id<MTLBuffer> bufA = input(A, aBytes, aMapBytes);
        id<MTLBuffer> bufB = input(B, bBytes, bMapBytes);
        id<MTLBuffer> bufC = nil;
        bool inPlace = false;
        if (mapC && cMapBytes >= cBytes) {
            bufC = [ctx.device newBufferWithBytesNoCopy:C length:cMapBytes options:MTLResourceStorageModeShared deallocator:nil];
            inPlace = bufC != nil;
        }
        if (!inPlace) {
            bufC = [ctx.device newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
            params.ldc = to32(N);
        }
        if (bufA == nil || bufB == nil || bufC == nil) {
            failure = "could not allocate GPU buffers";
        } else {
            float* out = static_cast<float*>([bufC contents]);
            if (accumulate && !inPlace)
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
            } else if (!inPlace) {
                for (std::size_t i = 0; i < M; ++i) std::memcpy(C + i * ldc, out + i * N, N * sizeof(float));
            }
        }
    }
    if (!failure.empty()) throw std::runtime_error("Metal gemm: " + failure);
}

bool int8Available() {
    Context& ctx = context();
    return ctx.ok && ctx.gemmQ8 != nil;
}

void gemmQ8(bool quantLeft, std::size_t M, std::size_t N, std::size_t K,
            const std::int8_t* q, std::size_t qMapBytes, const float* scales, std::size_t scaleMapBytes,
            std::size_t groupsPerRow, std::size_t group, const float* x, std::size_t ldx, std::size_t xMapBytes,
            float* C, std::size_t ldc, std::size_t cMapBytes) {
    Context& ctx = context();
    if (!ctx.ok) throw std::runtime_error("Metal unavailable: " + ctx.error);
    if (ctx.gemmQ8 == nil) throw std::runtime_error("Metal int8 GEMM unavailable: " + ctx.q8Error);
    if (M == 0 || N == 0) return;
    if (K == 0) {
        for (std::size_t i = 0; i < M; ++i) std::memset(C + i * ldc, 0, N * sizeof(float));
        return;
    }
    if (group == 0 || groupsPerRow * group < K) throw std::invalid_argument("Metal gemmQ8: bad weight groups");

    const std::size_t qRows = quantLeft ? M : N;
    const std::size_t qBytes = qRows * K;
    const std::size_t sBytes = qRows * groupsPerRow * sizeof(float);
    const std::size_t xRows = quantLeft ? K : M, xCols = quantLeft ? N : K;
    const std::size_t xBytes = ((xRows - 1) * ldx + xCols) * sizeof(float);
    const std::size_t cBytes = M * N * sizeof(float);
    const bool mapC = cMapBytes >= cBytes && cMapBytes > 0 && ldc == N;
    Q8Params params{to32(M), to32(N), to32(K), to32(ldx), to32(N), to32(groupsPerRow), to32(group), quantLeft ? 1u : 0u};

    std::string failure;
    @autoreleasepool {
        id<MTLBuffer> bufQ = inputBuffer(ctx, q, qBytes, qMapBytes);
        id<MTLBuffer> bufS = inputBuffer(ctx, scales, sBytes, scaleMapBytes);
        id<MTLBuffer> bufX = inputBuffer(ctx, x, xBytes, xMapBytes);
        id<MTLBuffer> bufC = nil;
        bool inPlace = false;
        if (mapC) {
            bufC = [ctx.device newBufferWithBytesNoCopy:C length:cMapBytes options:MTLResourceStorageModeShared deallocator:nil];
            inPlace = bufC != nil;
        }
        if (!inPlace) bufC = [ctx.device newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
        if (bufQ == nil || bufS == nil || bufX == nil || bufC == nil) {
            failure = "could not allocate GPU buffers";
        } else {
            id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx.gemmQ8];
            [enc setBuffer:bufQ offset:0 atIndex:0];
            [enc setBuffer:bufS offset:0 atIndex:1];
            [enc setBuffer:bufX offset:0 atIndex:2];
            [enc setBuffer:bufC offset:0 atIndex:3];
            [enc setBytes:&params length:sizeof(params) atIndex:4];
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
            } else if (!inPlace) {
                const float* out = static_cast<const float*>([bufC contents]);
                for (std::size_t i = 0; i < M; ++i) std::memcpy(C + i * ldc, out + i * N, N * sizeof(float));
            }
        }
    }
    if (!failure.empty()) throw std::runtime_error("Metal gemmQ8: " + failure);
}

} // namespace metal
} // namespace nc
