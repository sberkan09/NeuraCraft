#include "Quantized.h"
#include "Backend.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nc {

namespace {
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

Matrix QuantizedWeights::multiply(const Matrix& x) const {
    if (x.cols() != in_)
        throw std::invalid_argument("QuantizedWeights::multiply: expected " + std::to_string(in_) + " columns, got " +
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
