#include "VectorMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace nc {

namespace {
inline float bitsToFloat(std::int32_t i) {
    float f;
    std::memcpy(&f, &i, sizeof f);
    return f;
}

inline std::int32_t floatToBits(float f) {
    std::int32_t i;
    std::memcpy(&i, &f, sizeof i);
    return i;
}

// exp for one value, written so a loop over it vectorizes: no branches, no calls.
inline float expOne(float x) {
    // Clamped so the result rounds to exactly 0 below (exp(-110) < 2^-150) and overflows to +inf above
    // (exp(89) > FLT_MAX) through the arithmetic itself: selects would stop SSE2 vectorization.
    x = std::min(89.0f, std::max(-110.0f, x));
    // n = round(x / ln 2), r = x - n ln 2 (in two parts for accuracy), exp(x) = 2^n exp(r).
    // Rounding by adding 1.5 * 2^23 keeps n in the float's low bits: no float -> int conversion
    // (which compilers will not vectorize when it may trap).
    const float shifted = x * 1.44269504088896341f + 12582912.0f;
    const float fn = shifted - 12582912.0f;
    const std::int32_t n = floatToBits(shifted) - 0x4B400000;
    float r = x - fn * 0.693359375f;
    r = r - fn * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    const float e = p * r * r + r + 1.0f;
    // 2^n as two factors, so n from -150 (denormal results) to 128 (just below the overflow) fit.
    const std::int32_t h = n >> 1;
    return e * bitsToFloat((h + 127) << 23) * bitsToFloat((n - h + 127) << 23);
}
} // namespace

void expArray(const float* x, float* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = expOne(x[i]);
    }
}

void erfArray(const float* x, float* y, std::size_t n) {
    // erf(|x|) = 1 - t * exp(-x^2 + P(t)), t = 1 / (1 + |x| / 2) (Numerical Recipes' erfc, ~1.2e-7),
    // with exp from expOne; the sign is restored at the end.
    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        const float z = std::min(std::fabs(v), 10.0f);
        const float t = 1.0f / (1.0f + 0.5f * z);
        float p = 0.17087277f;
        p = p * t - 0.82215223f;
        p = p * t + 1.48851587f;
        p = p * t - 1.13520398f;
        p = p * t + 0.27886807f;
        p = p * t - 0.18628806f;
        p = p * t + 0.09678418f;
        p = p * t + 0.37409196f;
        p = p * t + 1.00002368f;
        p = p * t - 1.26551223f;
        const float erfcz = t * expOne(-z * z + p);
        y[i] = std::copysign(1.0f - erfcz, v);
    }
}

void softmaxRow(float* row, std::size_t n, float scale) {
    if (n == 0) return;
    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < n; ++j) {
        row[j] *= scale;
        m = std::max(m, row[j]);
    }
    for (std::size_t j = 0; j < n; ++j) row[j] -= m;
    expArray(row, row, n);
    float sum = 0.0f;
    for (std::size_t j = 0; j < n; ++j) sum += row[j];
    const float inv = 1.0f / sum;
    for (std::size_t j = 0; j < n; ++j) row[j] *= inv;
}

} // namespace nc
