#include "Quantized.h"
#include "Backend.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NEURACRAFT_HAS_METAL
#include "MetalBackend.h"
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace nc {

namespace {
#ifdef NEURACRAFT_HAS_METAL
// Bytes of a page-backed buffer the GPU may map (0: too small, it is copied).
std::size_t mapBytes(std::size_t bytes) { return bytes >= kPageBackedBytes ? detail::storageBytesFor(bytes) : 0; }
#endif

std::int8_t quantizeValue(float v, float inv) {
    const long q = std::lround(v * inv);
    return static_cast<std::int8_t>(std::max(-127L, std::min(127L, q)));
}
} // namespace

QuantizedWeights QuantizedWeights::fromOutIn(const Matrix& w, std::size_t group) {
    QuantizedWeights q;
    q.out_ = w.rows();
    q.in_ = w.cols();
    q.group_ = std::max<std::size_t>(1, std::min(group, std::max<std::size_t>(1, q.in_)));
    q.groupsPerRow_ = (q.in_ + q.group_ - 1) / q.group_;
    q.q_.resize(q.out_ * q.in_);
    q.scale_.resize(q.out_ * q.groupsPerRow_);
    parallelFor(0, q.out_, 16, [&](std::size_t j0, std::size_t j1) {
        for (std::size_t j = j0; j < j1; ++j) {
            const float* row = w.data() + j * q.in_;
            std::int8_t* dst = q.q_.data() + j * q.in_;
            for (std::size_t g = 0; g < q.groupsPerRow_; ++g) {
                const std::size_t k0 = g * q.group_, k1 = std::min(q.in_, k0 + q.group_);
                float maxAbs = 0.0f;
                for (std::size_t k = k0; k < k1; ++k) maxAbs = std::max(maxAbs, std::fabs(row[k]));
                const float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
                q.scale_[j * q.groupsPerRow_ + g] = scale;
                const float inv = 1.0f / scale;
                for (std::size_t k = k0; k < k1; ++k) dst[k] = quantizeValue(row[k], inv);
            }
        }
    });
    return q;
}

QuantizedWeights QuantizedWeights::fromInOut(const Matrix& w, std::size_t group) {
    return fromOutIn(w.transpose(), group);
}

void QuantizedWeights::expandRow(std::size_t j, float* dst) const {
    if (j >= out_) throw std::out_of_range("QuantizedWeights::expandRow: row " + std::to_string(j) + " out of range");
    const std::int8_t* src = q_.data() + j * in_;
    const float* scales = scale_.data() + j * groupsPerRow_;
    for (std::size_t g = 0; g < groupsPerRow_; ++g) {
        const std::size_t k0 = g * group_, k1 = std::min(in_, k0 + group_);
        for (std::size_t k = k0; k < k1; ++k) dst[k] = scales[g] * static_cast<float>(src[k]);
    }
}

namespace {
// Sum over one row pair of int8 values: w . x for k in [0, n).
// Each kernel handles whole 32 (x86) / 16 (ARM) chunks and leaves the rest to the caller.
#if defined(__AVX2__)
// |w| as unsigned and x with w's sign, so the unsigned x signed products
// equal w * x and never saturate (|w|, |x| <= 127).
inline __m256i dotChunk(__m256i acc, __m256i w, __m256i x) {
    const __m256i aw = _mm256_sign_epi8(w, w), sx = _mm256_sign_epi8(x, w);
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    return _mm256_dpbusd_epi32(acc, aw, sx);
#else
    const __m256i pairs = _mm256_maddubs_epi16(aw, sx);  // <= 2 * 127 * 127 per lane
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
#endif
}
constexpr std::size_t kChunk = 32;
#elif defined(__ARM_NEON)
inline int32x4_t dotChunk(int32x4_t acc, int8x16_t w, int8x16_t x) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, w, x);
#else
    const int16x8_t lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
    const int16x8_t hi = vmull_s8(vget_high_s8(w), vget_high_s8(x));
    return vpadalq_s16(vpadalq_s16(acc, lo), hi);
#endif
}
constexpr std::size_t kChunk = 16;
#else
constexpr std::size_t kChunk = 1;
#endif
constexpr std::size_t kInputBlock = 32;

// sum_b scale_w[b's group] * scale_x[b] * (w_b . x_b) over one weight row and
// one input row; inputs have one scale per block of `block` columns, and a
// block never straddles two weight groups.
float rowDot(const std::int8_t* w, const float* ws, const std::int8_t* x, const float* xs, std::size_t in, std::size_t group,
             std::size_t block) {
    const std::size_t perGroup = group / block;  // block divides group
    float tail = 0.0f;
    // Two float accumulators, alternated per block, keep the FMA latency off the critical path.
#if defined(__AVX2__)
    __m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
#elif defined(__ARM_NEON)
    float32x4_t acc[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
#endif
    std::size_t k0 = 0, b = 0, g = 0;
#if defined(__AVX2__) || defined(__ARM_NEON)
    // The default layout (64-column weight groups, 32-column input blocks):
    // whole groups without bounds checks or branches.
    if (group == 64 && block == 32) {
        for (; k0 + 64 <= in; k0 += 64, b += 2, ++g) {
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(w + k0 + 4096);
#endif
            const float s0 = ws[g] * xs[b], s1 = ws[g] * xs[b + 1];
#if defined(__AVX2__)
            const __m256i i0 = dotChunk(_mm256_setzero_si256(), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k0)),
                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k0)));
            const __m256i i1 = dotChunk(_mm256_setzero_si256(), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k0 + 32)),
                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k0 + 32)));
            acc[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_set1_ps(s0), acc[0]);
            acc[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), _mm256_set1_ps(s1), acc[1]);
#else
            const int32x4_t i0 = dotChunk(dotChunk(vdupq_n_s32(0), vld1q_s8(w + k0), vld1q_s8(x + k0)), vld1q_s8(w + k0 + 16),
                                          vld1q_s8(x + k0 + 16));
            const int32x4_t i1 = dotChunk(dotChunk(vdupq_n_s32(0), vld1q_s8(w + k0 + 32), vld1q_s8(x + k0 + 32)),
                                          vld1q_s8(w + k0 + 48), vld1q_s8(x + k0 + 48));
            acc[0] = vfmaq_n_f32(acc[0], vcvtq_f32_s32(i0), s0);
            acc[1] = vfmaq_n_f32(acc[1], vcvtq_f32_s32(i1), s1);
#endif
        }
    }
#endif
    for (; k0 < in; ++g) {
        const float wg = ws[g];
        for (std::size_t s = 0; s < perGroup && k0 < in; ++s, ++b, k0 += block) {
            const std::size_t k1 = std::min(in, k0 + block);
            const float scale = wg * xs[b];
            std::size_t k = k0;
#if defined(__GNUC__) || defined(__clang__)
            // Weight rows are streamed from memory once per token; asking for the
            // bytes 4 KB ahead roughly doubles the bandwidth reached (11 -> 20 GB/s
            // on a 2-core Xeon).
            if ((k0 & 63) == 0) __builtin_prefetch(w + k0 + 4096);
#endif
#if defined(__AVX2__)
            __m256i isum = _mm256_setzero_si256();
            for (; k + kChunk <= k1; k += kChunk)
                isum = dotChunk(isum, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k)),
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k)));
            acc[b & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_set1_ps(scale), acc[b & 1]);
#elif defined(__ARM_NEON)
            int32x4_t isum = vdupq_n_s32(0);
            for (; k + kChunk <= k1; k += kChunk) isum = dotChunk(isum, vld1q_s8(w + k), vld1q_s8(x + k));
            acc[b & 1] = vfmaq_n_f32(acc[b & 1], vcvtq_f32_s32(isum), scale);
#endif
            if (k < k1) {
                std::int32_t rest = 0;
                for (; k < k1; ++k) rest += static_cast<std::int32_t>(w[k]) * static_cast<std::int32_t>(x[k]);
                tail += scale * static_cast<float>(rest);
            }
        }
    }
#if defined(__AVX2__)
    const __m256 sum = _mm256_add_ps(acc[0], acc[1]);
    const __m128 h = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 h2 = _mm_add_ps(h, _mm_movehl_ps(h, h));
    tail += _mm_cvtss_f32(_mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1)));
#elif defined(__ARM_NEON)
    tail += vaddvq_f32(vaddq_f32(acc[0], acc[1]));
#endif
    return tail;
}

// Four consecutive weight rows (stride `in`) against one input row: each input
// chunk is loaded once for all four (prompt processing, n > 1).
void rowDot4(const std::int8_t* w, const float* ws, std::size_t wsStride, const std::int8_t* x, const float* xs,
             std::size_t in, std::size_t group, std::size_t block, float* out) {
#if defined(__AVX2__) || defined(__ARM_NEON)
    const std::size_t blocks = (in + block - 1) / block;
    float tail[4] = {0, 0, 0, 0};
#if defined(__AVX2__)
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
#else
    float32x4_t acc[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
#endif
    const std::size_t perGroup = group / block;
    for (std::size_t b = 0, g = 0, s = 0; b < blocks; ++b, s = s + 1 == perGroup ? (++g, 0) : s + 1) {
        const std::size_t k0 = b * block, k1 = std::min(in, k0 + block);
        std::size_t k = k0;
#if defined(__AVX2__)
        __m256i isum[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        for (; k + kChunk <= k1; k += kChunk) {
            const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k));
            for (int r = 0; r < 4; ++r)
                isum[r] = dotChunk(isum[r], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + r * in + k)), xv);
        }
        for (int r = 0; r < 4; ++r)
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum[r]), _mm256_set1_ps(ws[r * wsStride + g] * xs[b]), acc[r]);
#else
        int32x4_t isum[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
        for (; k + kChunk <= k1; k += kChunk) {
            const int8x16_t xv = vld1q_s8(x + k);
            for (int r = 0; r < 4; ++r) isum[r] = dotChunk(isum[r], vld1q_s8(w + r * in + k), xv);
        }
        for (int r = 0; r < 4; ++r) acc[r] = vfmaq_n_f32(acc[r], vcvtq_f32_s32(isum[r]), ws[r * wsStride + g] * xs[b]);
#endif
        for (; k < k1; ++k)
            for (int r = 0; r < 4; ++r)
                tail[r] += ws[r * wsStride + g] * xs[b] *
                           static_cast<float>(static_cast<std::int32_t>(w[r * in + k]) * static_cast<std::int32_t>(x[k]));
    }
    for (int r = 0; r < 4; ++r) {
#if defined(__AVX2__)
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[r]), _mm256_extractf128_ps(acc[r], 1));
        const __m128 h2 = _mm_add_ps(h, _mm_movehl_ps(h, h));
        out[r] = tail[r] + _mm_cvtss_f32(_mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 1)));
#else
        out[r] = tail[r] + vaddvq_f32(acc[r]);
#endif
    }
#else
    for (int r = 0; r < 4; ++r) out[r] = rowDot(w + r * in, ws + r * wsStride, x, xs, in, group, block);
#endif
}
} // namespace

namespace {
std::atomic<bool> gInt8Inputs{true};
}
void setInt8Inputs(bool enabled) { gInt8Inputs = enabled; }
bool int8Inputs() { return gInt8Inputs; }

const char* QuantizedWeights::kernelName() {
#if defined(__AVX2__) && (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
    return "int8 AVX-VNNI";
#elif defined(__AVX2__)
    return "int8 AVX2";
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    return "int8 NEON dot product";
#elif defined(__ARM_NEON)
    return "int8 NEON";
#else
    return "int8 portable";
#endif
}

void QuantizedWeights::expandRows(std::size_t j0, std::size_t j1, float* dst) const {
    parallelFor(j0, j1, 8, [&](std::size_t a, std::size_t b) {
        for (std::size_t j = a; j < b; ++j) expandRow(j, dst + (j - j0) * in_);
    });
}

Matrix QuantizedWeights::multiply(const Matrix& x) const {
    if (x.cols() != in_)
        throw std::invalid_argument("QuantizedWeights::multiply: expected " + std::to_string(in_) + " columns, got " +
                                    std::to_string(x.cols()));
    const std::size_t n = x.rows();
#ifdef NEURACRAFT_HAS_METAL
    if (gpuProducts(n)) {
        // out (n x out) = x * W^T on the GPU, dequantizing tiles of W as they are loaded.
        Matrix out(n, out_);
        metal::gemmQ8(false, n, out_, in_, q_.data(), mapBytes(q_.capacity()), scale_.data(),
                      mapBytes(scale_.capacity() * sizeof(float)), groupsPerRow_, group_, x.data(), in_,
                      x.pageBacked() ? x.storageBytes() : 0, out.data(), out_, out.pageBacked() ? out.storageBytes() : 0);
        return out;
    }
#endif
    if (n >= kGemmRows) {
        // Many rows: expand tiles of weight rows (at most ~2M floats each) and use the float GEMM.
        Matrix out(n, out_);
        const std::size_t tileRows = std::max<std::size_t>(1, std::min(out_, (std::size_t(1) << 21) / std::max<std::size_t>(1, in_)));
        std::vector<float> tile(tileRows * in_);
        for (std::size_t j0 = 0; j0 < out_; j0 += tileRows) {
            const std::size_t j1 = std::min(out_, j0 + tileRows);
            expandRows(j0, j1, tile.data());
            gemm(false, true, n, j1 - j0, in_, x.data(), in_, tile.data(), in_, out.data() + j0, out_, false);
        }
        return out;
    }
    if (!int8Inputs()) return multiplyFloat(x);
    Matrix out(n, out_);
    if (n == 0 || out_ == 0) return out;

    // Inputs to int8 with one scale per block of 32 columns (llama.cpp's q8_0
    // block), or per weight group when groups are not a multiple of 32.
    const std::size_t block = group_ % kInputBlock == 0 ? kInputBlock : group_;
    const std::size_t blocks = (in_ + block - 1) / block;
    std::vector<std::int8_t> xq(n * in_);
    std::vector<float> xs(n * blocks);
    parallelFor(0, n, std::max<std::size_t>(1, 4096 / std::max<std::size_t>(1, in_)), [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            const float* row = x.data() + i * in_;
            for (std::size_t b = 0; b < blocks; ++b) {
                const std::size_t k0 = b * block, k1 = std::min(in_, k0 + block);
                float maxAbs = 0.0f;
                for (std::size_t k = k0; k < k1; ++k) maxAbs = std::max(maxAbs, std::fabs(row[k]));
                const float scale = maxAbs / 127.0f, inv = maxAbs > 0.0f ? 127.0f / maxAbs : 0.0f;
                xs[i * blocks + b] = scale;
                for (std::size_t k = k0; k < k1; ++k) xq[i * in_ + k] = quantizeValue(row[k], inv);
            }
        }
    });

    // Each weight row stays in L1 while it meets all n inputs.
    const std::size_t grain = std::max<std::size_t>(4, (std::size_t(1) << 16) / std::max<std::size_t>(1, in_ * n));
    const std::size_t rowBlocks = (out_ + 3) / 4;  // rows are handed out in blocks of four
    parallelFor(0, rowBlocks, std::max<std::size_t>(1, grain / 4), [&](std::size_t b0, std::size_t b1) {
        for (std::size_t rb = b0; rb < b1; ++rb) {
            const std::size_t j0 = 4 * rb, j1 = std::min(out_, j0 + 4);
            if (n > 1 && j1 - j0 == 4) {
                float r[4];
                for (std::size_t i = 0; i < n; ++i) {
                    rowDot4(q_.data() + j0 * in_, scale_.data() + j0 * groupsPerRow_, groupsPerRow_, xq.data() + i * in_,
                            xs.data() + i * blocks, in_, group_, block, r);
                    for (int k = 0; k < 4; ++k) out(i, j0 + static_cast<std::size_t>(k)) = r[k];
                }
                continue;
            }
            for (std::size_t j = j0; j < j1; ++j) {
                const std::int8_t* w = q_.data() + j * in_;
                const float* ws = scale_.data() + j * groupsPerRow_;
                for (std::size_t i = 0; i < n; ++i)
                    out(i, j) = rowDot(w, ws, xq.data() + i * in_, xs.data() + i * blocks, in_, group_, block);
            }
        }
    });
    return out;
}

bool QuantizedWeights::gpuProducts(std::size_t n) const {
#ifdef NEURACRAFT_HAS_METAL
    return n >= kGemmRows && currentDevice() == Device::GPU && n * out_ * in_ >= gpuMinWork() && !q_.empty() &&
           metal::int8Available();
#else
    (void)n;
    return false;
#endif
}

void QuantizedWeights::multiplyLeft(std::size_t n, const float* b, std::size_t ldb, float* c, std::size_t ldc,
                                    std::size_t bMapBytes, std::size_t cMapBytes) const {
    if (n == 0 || out_ == 0) return;
#ifdef NEURACRAFT_HAS_METAL
    if (gpuProducts(n)) {
        metal::gemmQ8(true, out_, n, in_, q_.data(), mapBytes(q_.capacity()), scale_.data(),
                      mapBytes(scale_.capacity() * sizeof(float)), groupsPerRow_, group_, b, ldb, bMapBytes, c, ldc,
                      cMapBytes);
        return;
    }
#endif
    (void)bMapBytes;
    (void)cMapBytes;
    const std::size_t tileRows = std::max<std::size_t>(1, std::min(out_, (std::size_t(1) << 21) / std::max<std::size_t>(1, in_)));
    std::vector<float> tile(tileRows * in_);
    for (std::size_t j0 = 0; j0 < out_; j0 += tileRows) {
        const std::size_t j1 = std::min(out_, j0 + tileRows);
        expandRows(j0, j1, tile.data());
        gemm(false, false, j1 - j0, n, in_, tile.data(), in_, b, ldb, c + j0 * ldc, ldc, false);
    }
}

Matrix QuantizedWeights::multiplyFloat(const Matrix& x) const {
    if (x.cols() != in_)
        throw std::invalid_argument("QuantizedWeights::multiplyFloat: expected " + std::to_string(in_) + " columns, got " +
                                    std::to_string(x.cols()));
    const std::size_t n = x.rows();
    Matrix out(n, out_);
    if (n == 0 || out_ == 0) return out;
    const std::size_t grain = std::max<std::size_t>(8, (std::size_t(1) << 16) / std::max<std::size_t>(1, in_ * n));
    parallelFor(0, out_, grain, [&](std::size_t j0, std::size_t j1) {
        std::vector<float> row(in_);
        for (std::size_t j = j0; j < j1; ++j) {
            // Expand the row once (with its group scales), then reuse it for all n inputs.
            const std::int8_t* src = q_.data() + j * in_;
            const float* scales = scale_.data() + j * groupsPerRow_;
            for (std::size_t g = 0; g < groupsPerRow_; ++g) {
                const std::size_t k0 = g * group_, k1 = std::min(in_, k0 + group_);
                const float s = scales[g];
                for (std::size_t k = k0; k < k1; ++k) row[k] = s * static_cast<float>(src[k]);
            }
            for (std::size_t i = 0; i < n; ++i) out(i, j) = dot(x.data() + i * in_, row.data(), in_);
        }
    });
    return out;
}

Matrix QuantizedWeights::dequantize() const {
    Matrix w(in_, out_);
    for (std::size_t j = 0; j < out_; ++j)
        for (std::size_t k = 0; k < in_; ++k)
            w(k, j) = scale_[j * groupsPerRow_ + k / group_] * static_cast<float>(q_[j * in_ + k]);
    return w;
}

} // namespace nc
