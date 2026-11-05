// Metal (Apple GPU) implementation of NeuraCraft's GEMM. Compiled only on Apple
// platforms when NEURACRAFT_USE_METAL is ON. The shader is compiled at runtime
// from source, so building does not need the offline Metal compiler (full Xcode).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalBackend.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
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

// Fused multi-head attention, flash-attention style: per head, blocks of 32
// query rows stream the keys and values through threadgroup memory 16 at a
// time with an online softmax, so the (queries x keys) score matrix never
// exists. Four threads share a query row, each holding a quarter of its head
// dimensions; DQ (= head size / 4) is a function constant so loops unroll.
const char* const kAttentionSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct AttnParams { uint N; uint M; uint C; uint dh; float scale; };
constant uint DQ [[function_constant(0)]];

#define BR 32
#define BC 16
#define DMAX 160
#define DPT 40

kernel void attention_flash(device const float* Q [[buffer(0)]],
                            device const float* K [[buffer(1)]],
                            device const float* V [[buffer(2)]],
                            device float* O [[buffer(3)]],
                            constant AttnParams& p [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float Ks[BC * DMAX];
    threadgroup float Vs[BC * DMAX];
    const uint base = tg.y * p.dh;
    const uint row = tg.x * BR + tidx / 4;
    const uint e0 = (tidx % 4) * DQ;
    const bool live = row < p.N;

    float q[DPT], o[DPT];
    for (uint e = 0; e < DQ; ++e) {
        q[e] = live ? Q[ulong(row) * p.C + base + e0 + e] * p.scale : 0.0f;
        o[e] = 0.0f;
    }
    float m = -1e30f, l = 0.0f;  // finite: Metal compiles with fast math

    for (uint j0 = 0; j0 < p.M; j0 += BC) {
        const uint nk = min(uint(BC), p.M - j0);
        for (uint t = tidx; t < BC * p.dh; t += 128) {
            const uint j = t / p.dh, e = t % p.dh;
            const bool in = j < nk;
            Ks[j * DMAX + e] = in ? K[ulong(j0 + j) * p.C + base + e] : 0.0f;
            Vs[j * DMAX + e] = in ? V[ulong(j0 + j) * p.C + base + e] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[BC];
        float mt = m;
        for (uint j = 0; j < BC; ++j) {
            float d = 0.0f;
            for (uint e = 0; e < DQ; ++e) d = fma(q[e], Ks[j * DMAX + e0 + e], d);
            d += simd_shuffle_xor(d, ushort(1));
            d += simd_shuffle_xor(d, ushort(2));
            s[j] = j < nk ? d : -1e30f;
            mt = max(mt, s[j]);
        }
        const float corr = exp(m - mt);
        l *= corr;
        for (uint e = 0; e < DQ; ++e) o[e] *= corr;
        for (uint j = 0; j < BC; ++j) {
            if (j >= nk) break;
            const float pj = exp(s[j] - mt);
            l += pj;
            for (uint e = 0; e < DQ; ++e) o[e] = fma(pj, Vs[j * DMAX + e0 + e], o[e]);
        }
        m = mt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (live) {
        const float inv = 1.0f / l;
        for (uint e = 0; e < DQ; ++e) O[ulong(row) * p.C + base + e0 + e] = o[e] * inv;
    }
}
)METAL";

// The same attention with simdgroup matrices (Apple7+, M1 and later). Four
// simdgroups per threadgroup each own 8 query rows; keys and values stream
// through threadgroup memory 16 at a time. Per tile: S = Q K^T (8 x 16) with
// 8x8 hardware products, an online softmax in a small per-simdgroup scratch
// (four lanes per row), O = diag(correction) O + P V. Its scratch is handed
// between lanes with threadgroup barriers: with simdgroup barriers the macOS CI
// GPU returned wrong rows now and then once several threadgroups shared a
// core. DB = head size / 8 is a
// function constant so the loops over head dimensions unroll.
const char* const kAttentionSimdSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct AttnParams { uint N; uint M; uint C; uint dh; float scale; };
constant uint DB [[function_constant(0)]];

#define BC 16
#define DMAX 160
#define DBMAX 20

kernel void attention_simd(device const float* Q [[buffer(0)]],
                           device const float* K [[buffer(1)]],
                           device const float* V [[buffer(2)]],
                           device float* O [[buffer(3)]],
                           constant AttnParams& p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tidx [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float Ks[BC * DMAX];
    threadgroup float Vs[BC * DMAX];
    threadgroup float scratch[4][8 * BC + 64];  // per simdgroup: S / P (8 x 16), then an 8 x 8 matrix
    threadgroup float* Ss = scratch[sg];
    threadgroup float* Ds = scratch[sg] + 8 * BC;

    const uint base = tg.y * p.dh;
    const uint row0 = tg.x * 32 + sg * 8;
    const uint r = lane / 4, c0 = (lane % 4) * 4;  // softmax: four lanes per row, four columns each

    // Q (8 x dh) into simdgroup matrices through the 8 x 8 scratch; rows past N are zero.
    simdgroup_float8x8 q[DBMAX], o[DBMAX];
    for (uint d = 0; d < DB; ++d) {
        for (uint e = lane; e < 64; e += 32) {
            const uint qr = row0 + e / 8;
            Ds[e] = qr < p.N ? Q[ulong(qr) * p.C + base + d * 8 + e % 8] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(q[d], Ds, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = -1e30f, l = 0.0f;  // this lane's row; finite: Metal compiles with fast math

    for (uint j0 = 0; j0 < p.M; j0 += BC) {
        const uint nk = min(uint(BC), p.M - j0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = tidx; t < BC * p.dh; t += 128) {
            const uint j = t / p.dh, e = t % p.dh;
            const bool in = j < nk;
            Ks[j * DMAX + e] = in ? K[ulong(j0 + j) * p.C + base + e] : 0.0f;
            Vs[j * DMAX + e] = in ? V[ulong(j0 + j) * p.C + base + e] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T for the two 8-key halves of the tile.
        for (uint h = 0; h < 2; ++h) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < DB; ++d) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, Ks + (h * 8) * DMAX + d * 8, DMAX, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, q[d], kt, acc);
            }
            simdgroup_store(acc, Ss + h * 8, BC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax of this lane's four scores.
        float s[4];
        float mt = -1e30f;
        for (uint i = 0; i < 4; ++i) {
            s[i] = j0 + c0 + i < p.M ? Ss[r * BC + c0 + i] * p.scale : -1e30f;
            mt = max(mt, s[i]);
        }
        mt = max(mt, simd_shuffle_xor(mt, ushort(1)));
        mt = max(mt, simd_shuffle_xor(mt, ushort(2)));
        const float mNew = max(m, mt);
        const float corr = exp(m - mNew);
        float sum = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const float e = j0 + c0 + i < p.M ? exp(s[i] - mNew) : 0.0f;
            Ss[r * BC + c0 + i] = e;
            sum += e;
        }
        sum += simd_shuffle_xor(sum, ushort(1));
        sum += simd_shuffle_xor(sum, ushort(2));
        l = l * corr + sum;
        m = mNew;
        // diag(corr) in the 8 x 8 scratch: entries 2 lane, 2 lane + 1 lie in this lane's row.
        for (uint t = 0; t < 2; ++t) Ds[2 * lane + t] = (2 * lane + t) % 8 == r ? corr : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 dcorr, p0, p1;
        simdgroup_load(dcorr, Ds, 8);
        simdgroup_load(p0, Ss, BC);
        simdgroup_load(p1, Ss + 8, BC);
        for (uint d = 0; d < DB; ++d) {
            simdgroup_float8x8 scaled, v0, v1;
            simdgroup_multiply(scaled, dcorr, o[d]);
            simdgroup_load(v0, Vs + d * 8, DMAX);
            simdgroup_load(v1, Vs + 8 * DMAX + d * 8, DMAX);
            simdgroup_multiply_accumulate(scaled, p0, v0, scaled);
            simdgroup_multiply_accumulate(o[d], p1, v1, scaled);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l, out through the 8 x 8 scratch for rows < N.
    for (uint t = 0; t < 2; ++t) Ds[2 * lane + t] = (2 * lane + t) % 8 == r ? 1.0f / l : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dinv;
    simdgroup_load(dinv, Ds, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 0; d < DB; ++d) {
        simdgroup_float8x8 res;
        simdgroup_multiply(res, dinv, o[d]);
        simdgroup_store(res, Ss, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            const uint orow = row0 + e / 8;
            if (orow < p.N) O[ulong(orow) * p.C + base + d * 8 + e % 8] = Ss[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    id<MTLLibrary> attentionLib = nil;      // attention_flash, specialized per head size
    id<MTLLibrary> attentionSimdLib = nil;  // attention_simd (Apple7+)
    std::map<unsigned, bool> attentionSimd;  // head size -> the pipeline is the simdgroup one
    std::string attentionChoice;             // "", "scalar" or "simdgroup" (setAttentionKernel)
    std::map<unsigned, id<MTLComputePipelineState>> attention;
    std::mutex attentionMutex;
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

struct AttnParams {
    std::uint32_t N, M, C, dh;
    float scale;
};

// The attention pipeline for one head size (built on first use), or nil: the simdgroup-matrix
// kernel with the simdgroup GEMM (head sizes divisible by 8), else the portable one.
// Which one is preferred: setAttentionKernel(), else NEURACRAFT_METAL_ATTENTION=scalar|simdgroup,
// else the simdgroup one where it runs (M4: 34.1 against 37.8 ms for 4096 tokens x 320, 8 heads).
id<MTLComputePipelineState> attentionPipeline(Context& ctx, unsigned dh) {
    std::lock_guard<std::mutex> lock(ctx.attentionMutex);
    auto it = ctx.attention.find(dh);
    if (it != ctx.attention.end()) return it->second;
    id<MTLComputePipelineState> pipe = nil;
    bool simd = false;
    @autoreleasepool {
        NSError* err = nil;
        std::string choice = ctx.attentionChoice;
        if (choice.empty()) {
            const char* env = std::getenv("NEURACRAFT_METAL_ATTENTION");
            choice = env ? env : "simdgroup";
        }
        if (ctx.useSimd && choice == "simdgroup" && dh % 8 == 0 && dh <= 160) {
            if (ctx.attentionSimdLib == nil)
                ctx.attentionSimdLib = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:kAttentionSimdSource]
                                                                options:nil
                                                                  error:&err];
            if (ctx.attentionSimdLib != nil) {
                MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
                const std::uint32_t db = dh / 8;
                [values setConstantValue:&db type:MTLDataTypeUInt atIndex:0];
                id<MTLFunction> fn = [ctx.attentionSimdLib newFunctionWithName:@"attention_simd" constantValues:values error:&err];
                if (fn) pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
                if (pipe != nil && [pipe maxTotalThreadsPerThreadgroup] < 128) pipe = nil;
                simd = pipe != nil;
            }
        }
        if (pipe == nil) {
            if (ctx.attentionLib == nil)
                ctx.attentionLib = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:kAttentionSource] options:nil error:&err];
            if (ctx.attentionLib != nil) {
                MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
                const std::uint32_t dq = dh / 4;
                [values setConstantValue:&dq type:MTLDataTypeUInt atIndex:0];
                id<MTLFunction> fn = [ctx.attentionLib newFunctionWithName:@"attention_flash" constantValues:values error:&err];
                if (fn) pipe = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
                if (pipe != nil && [pipe maxTotalThreadsPerThreadgroup] < 128) pipe = nil;
            }
        }
    }
    ctx.attention[dh] = pipe;
    ctx.attentionSimd[dh] = simd;
    return pipe;
}

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

void setAttentionKernel(const std::string& name) {
    if (!name.empty() && name != "scalar" && name != "simdgroup")
        throw std::invalid_argument("Metal attention kernel must be scalar or simdgroup, not " + name);
    Context& ctx = context();
    std::lock_guard<std::mutex> lock(ctx.attentionMutex);
    ctx.attentionChoice = name;
    ctx.attention.clear();  // rebuilt for the new choice on next use
    ctx.attentionSimd.clear();
}

std::string attentionKernelName(std::size_t headSize) {
    Context& ctx = context();
    if (!ctx.ok || headSize == 0 || headSize % 4 != 0 || headSize > 160) return "";
    if (attentionPipeline(ctx, static_cast<unsigned>(headSize)) == nil) return "";
    std::lock_guard<std::mutex> lock(ctx.attentionMutex);
    return ctx.attentionSimd[static_cast<unsigned>(headSize)] ? "simdgroup" : "scalar";
}

bool attentionSupported(std::size_t headSize) {
    Context& ctx = context();
    return ctx.ok && headSize > 0 && headSize % 4 == 0 && headSize <= 160 && attentionPipeline(ctx, static_cast<unsigned>(headSize)) != nil;
}

void attention(std::size_t N, std::size_t M, std::size_t C, std::size_t heads, const float* q, std::size_t qMapBytes,
               const float* k, std::size_t kMapBytes, const float* v, std::size_t vMapBytes, float* out,
               std::size_t outMapBytes) {
    Context& ctx = context();
    if (!ctx.ok) throw std::runtime_error("Metal unavailable: " + ctx.error);
    if (heads == 0 || C % heads != 0) throw std::invalid_argument("Metal attention: heads must divide the width");
    const std::size_t dh = C / heads;
    id<MTLComputePipelineState> pipe = attentionSupported(dh) ? attentionPipeline(ctx, static_cast<unsigned>(dh)) : nil;
    if (pipe == nil) throw std::runtime_error("Metal attention: head size " + std::to_string(dh) + " not supported");
    if (N == 0) return;
    if (M == 0) throw std::invalid_argument("Metal attention: no keys");

    AttnParams params{to32(N), to32(M), to32(C), to32(dh), 1.0f / std::sqrt(static_cast<float>(dh))};
    const std::size_t qBytes = N * C * sizeof(float), kvBytes = M * C * sizeof(float);
    std::string failure;
    @autoreleasepool {
        id<MTLBuffer> bufQ = inputBuffer(ctx, q, qBytes, qMapBytes);
        id<MTLBuffer> bufK = inputBuffer(ctx, k, kvBytes, kMapBytes);
        id<MTLBuffer> bufV = inputBuffer(ctx, v, kvBytes, vMapBytes);
        id<MTLBuffer> bufO = nil;
        bool inPlace = false;
        if (outMapBytes >= qBytes && outMapBytes > 0) {
            bufO = [ctx.device newBufferWithBytesNoCopy:out length:outMapBytes options:MTLResourceStorageModeShared deallocator:nil];
            inPlace = bufO != nil;
        }
        if (!inPlace) bufO = [ctx.device newBufferWithLength:qBytes options:MTLResourceStorageModeShared];
        if (bufQ == nil || bufK == nil || bufV == nil || bufO == nil) {
            failure = "could not allocate GPU buffers";
        } else {
            id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:bufQ offset:0 atIndex:0];
            [enc setBuffer:bufK offset:0 atIndex:1];
            [enc setBuffer:bufV offset:0 atIndex:2];
            [enc setBuffer:bufO offset:0 atIndex:3];
            [enc setBytes:&params length:sizeof(params) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb status] == MTLCommandBufferStatusError) {
                NSError* err = [cb error];
                failure = std::string("command buffer failed: ") +
                          (err ? [[err localizedDescription] UTF8String] : "unknown error");
            } else if (!inPlace) {
                std::memcpy(out, [bufO contents], qBytes);
            }
        }
    }
    if (!failure.empty()) throw std::runtime_error("Metal attention: " + failure);
}

} // namespace metal
} // namespace nc
