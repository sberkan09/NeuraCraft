// Metal (Apple GPU) implementation of NeuraCraft's GEMM. Compiled only on Apple
// platforms when NEURACRAFT_USE_METAL is ON. The shader is compiled at runtime
// from source, so building does not need the offline Metal compiler (full Xcode).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBackend.h"

#include <cstdint>
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

struct GemmParams {
    std::uint32_t M, N, K, lda, ldb, ldc, transA, transB, accumulate;
};

struct Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> gemm = nil;
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
            if (!ctx.ok) ctx.error = "could not create a command queue";
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
            [enc setComputePipelineState:ctx.gemm];
            [enc setBuffer:bufA offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            [enc setBytes:&params length:sizeof(params) atIndex:3];
            const MTLSize groups = MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1);
            const MTLSize threads = MTLSizeMake(8, 8, 1);
            [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
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
