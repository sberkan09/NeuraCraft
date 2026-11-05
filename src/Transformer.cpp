#include "Transformer.h"
#include "Profile.h"
#include "VectorMath.h"
#include "Activations.h"
#include "Backend.h"
#include "Dense.h"
#include "Parallel.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace nc {

namespace {
std::size_t requireLength(const SequenceContextPtr& ctx, std::size_t rows, const char* who) {
    if (!ctx || ctx->length == 0)
        throw std::logic_error(std::string(who) + ": sequence length unknown (run a TokenEmbedding first)");
    if (rows % ctx->length != 0)
        throw std::invalid_argument(std::string(who) + ": " + std::to_string(rows) +
                                    " rows is not a multiple of the sequence length " + std::to_string(ctx->length));
    return ctx->length;
}
} // namespace

// ---- TokenEmbedding ---------------------------------------------------------------------

TokenEmbedding::TokenEmbedding(std::size_t vocabSize, std::size_t dim, std::size_t maxContext,
                               SequenceContextPtr context, bool learnedPositions)
    : vocab_(vocabSize), dim_(dim), maxContext_(maxContext), context_(std::move(context)),
      learnedPositions_(learnedPositions) {
    if (vocabSize == 0 || dim == 0 || maxContext == 0) throw std::invalid_argument("TokenEmbedding: sizes must be > 0");
    if (!context_) throw std::invalid_argument("TokenEmbedding: context must not be null");
    if (SkipParameterInit::active()) return;  // a loader assigns the tables
    tokens_ = Matrix::normal(vocabSize, dim, 0.0f, 0.02f);
    tokenGrad_ = Matrix(vocabSize, dim);
    if (learnedPositions_) {
        positions_ = Matrix::normal(maxContext, dim, 0.0f, 0.02f);
        positionGrad_ = Matrix(maxContext, dim);
    }
}

Matrix TokenEmbedding::forward(const Matrix& input, bool) {
    batch_ = input.rows();
    length_ = input.cols();
    const std::size_t offset = context_->incremental ? context_->offset : 0;
    if (context_->incremental && batch_ != 1)
        throw std::invalid_argument("TokenEmbedding: incremental inference handles one sequence at a time");
    // Learned positions exist only up to maxContext; rotary positions are unbounded.
    const std::size_t lastPosition = learnedPositions_ ? offset + length_ : length_;
    if (length_ == 0 || lastPosition > maxContext_)
        throw std::invalid_argument("TokenEmbedding: positions " + std::to_string(offset) + ".." +
                                    std::to_string(offset + length_ - 1) + " exceed the context of " +
                                    std::to_string(maxContext_));
    context_->length = length_;
    ids_.resize(batch_ * length_);
    Matrix out(batch_ * length_, dim_);
    for (std::size_t r = 0; r < ids_.size(); ++r) {
        const long id = std::lround(input.data()[r]);
        if (id < 0 || static_cast<std::size_t>(id) >= vocab_)
            throw std::out_of_range("TokenEmbedding: token id " + std::to_string(id) + " outside the vocabulary");
        ids_[r] = static_cast<std::size_t>(id);
        float* o = out.data() + r * dim_;
        if (quantized()) quantizedTokens_.expandRow(ids_[r], o);
        else std::copy(tokens_.data() + ids_[r] * dim_, tokens_.data() + (ids_[r] + 1) * dim_, o);
        if (learnedPositions_) {
            const float* pos = positions_.data() + (offset + r % length_) * dim_;
            for (std::size_t j = 0; j < dim_; ++j) o[j] += pos[j];
        }
    }
    return out;
}

std::size_t TokenEmbedding::quantize() {
    if (quantized() || tokens_.empty()) return 0;
    quantizedTokens_ = QuantizedWeights::fromOutIn(tokens_);  // vocab x dim: one row per token
    tokens_ = Matrix();
    tokenGrad_ = Matrix();
    return 1;
}

Matrix TokenEmbedding::backward(const Matrix& gradOutput) {
    if (quantized()) throw std::logic_error("TokenEmbedding: a quantized table is inference-only");
    if (gradOutput.rows() != ids_.size() || gradOutput.cols() != dim_)
        throw std::invalid_argument("TokenEmbedding::backward: gradient shape does not match the last forward");
    // Tied: TiedOutput already wrote its share. Buffers may have been released.
    if (!tied_ || !tokenGrad_.sameShape(tokens_)) tokenGrad_ = Matrix(vocab_, dim_);
    if (learnedPositions_) positionGrad_ = Matrix(maxContext_, dim_);
    for (std::size_t r = 0; r < ids_.size(); ++r) {
        const float* g = gradOutput.data() + r * dim_;
        float* tg = tokenGrad_.data() + ids_[r] * dim_;
        for (std::size_t j = 0; j < dim_; ++j) tg[j] += g[j];
        if (learnedPositions_) {
            float* pg = positionGrad_.data() + (r % length_) * dim_;
            for (std::size_t j = 0; j < dim_; ++j) pg[j] += g[j];
        }
    }
    return Matrix(batch_, length_);
}

std::vector<Param> TokenEmbedding::parameters() {
    if (quantized()) return learnedPositions_ ? std::vector<Param>{{&positions_, &positionGrad_}} : std::vector<Param>{};
    if (!learnedPositions_) return {{&tokens_, &tokenGrad_}};
    return {{&tokens_, &tokenGrad_}, {&positions_, &positionGrad_}};
}

std::string TokenEmbedding::describe() const {
    return "TokenEmbedding(" + std::to_string(vocab_) + " x " + std::to_string(dim_) + ", ctx " +
           std::to_string(maxContext_) + (learnedPositions_ ? "" : ", no positions") + (quantized() ? ", int8" : "") + ")";
}

// ---- TiedOutput ---------------------------------------------------------------------------

TiedOutput::TiedOutput(TokenEmbedding& embedding) : embedding_(embedding) { embedding_.markTied(); }

Matrix TiedOutput::forward(const Matrix& input, bool training) {
    if (embedding_.quantized()) return embedding_.quantizedTable().multiply(input);
    input_ = training ? input : Matrix();
    return input.matmulTranspose(embedding_.tokenTable());  // (N x D) * (V x D)^T
}

Matrix TiedOutput::backward(const Matrix& gradOutput) {
    if (embedding_.quantized()) throw std::logic_error("TiedOutput: quantized layers are inference-only");
    // Runs before the embedding's backward, which then adds the lookup gradient.
    embedding_.tokenGradient() = gradOutput.transposeMatmul(input_);  // V x D
    return gradOutput * embedding_.tokenTable();
}

std::string TiedOutput::describe() const {
    return "TiedOutput(" + std::to_string(embedding_.outputSize(0)) + " -> " + std::to_string(embedding_.vocabSize()) +
           ", shares embedding" + (embedding_.quantized() ? ", int8" : "") + ")";
}

// ---- RMSNorm ----------------------------------------------------------------------------

RMSNorm::RMSNorm(std::size_t dim, float epsilon) : dim_(dim), epsilon_(epsilon), gamma_(1, dim, 1.0f), gammaGrad_(1, dim) {
    if (dim == 0) throw std::invalid_argument("RMSNorm: dim must be > 0");
}

Matrix RMSNorm::forward(const Matrix& input, bool training) {
    if (input.cols() != dim_)
        throw std::invalid_argument("RMSNorm: expected " + std::to_string(dim_) + " features, got " +
                                    std::to_string(input.cols()));
    const std::size_t n = input.rows();
    input_ = training ? input : Matrix();
    invRms_.assign(n, 0.0f);
    Matrix out(n, dim_);
    parallelFor(0, n, 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* x = input.data() + r * dim_;
            double sq = 0.0;
            for (std::size_t j = 0; j < dim_; ++j) sq += static_cast<double>(x[j]) * x[j];
            const float inv = static_cast<float>(1.0 / std::sqrt(sq / static_cast<double>(dim_) + epsilon_));
            invRms_[r] = inv;
            float* o = out.data() + r * dim_;
            for (std::size_t j = 0; j < dim_; ++j) o[j] = x[j] * inv * gamma_.data()[j];
        }
    });
    return out;
}

Matrix RMSNorm::backward(const Matrix& gradOutput) {
    if (!gradOutput.sameShape(input_))
        throw std::invalid_argument("RMSNorm::backward: gradient shape does not match the last forward");
    const std::size_t n = gradOutput.rows();
    gammaGrad_ = Matrix(1, dim_);
    for (std::size_t r = 0; r < n; ++r) {
        const float* g = gradOutput.data() + r * dim_;
        const float* x = input_.data() + r * dim_;
        for (std::size_t j = 0; j < dim_; ++j) gammaGrad_.data()[j] += g[j] * x[j] * invRms_[r];
    }
    Matrix dx(n, dim_);
    const float d = static_cast<float>(dim_);
    parallelFor(0, n, 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* g = gradOutput.data() + r * dim_;
            const float* x = input_.data() + r * dim_;
            const float inv = invRms_[r];
            float dot = 0.0f;  // sum_j g_j * gamma_j * x_j
            for (std::size_t j = 0; j < dim_; ++j) dot += g[j] * gamma_.data()[j] * x[j];
            float* out = dx.data() + r * dim_;
            for (std::size_t j = 0; j < dim_; ++j)
                out[j] = inv * (g[j] * gamma_.data()[j] - x[j] * inv * inv * dot / d);
        }
    });
    return dx;
}

std::vector<Param> RMSNorm::parameters() { return {{&gamma_, &gammaGrad_}}; }

std::string RMSNorm::describe() const { return "RMSNorm(" + std::to_string(dim_) + ")"; }

// ---- SwiGLU -------------------------------------------------------------------------------

Matrix SwiGLUGate::forward(const Matrix& input, bool training) {
    if (input.cols() % 2 != 0) throw std::invalid_argument("SwiGLU: input needs an even number of columns");
    input_ = training ? input : Matrix();
    const std::size_t h = input.cols() / 2;
    Matrix out(input.rows(), h);
    parallelFor(0, input.rows(), 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* a = input.data() + r * 2 * h;
            const float* b = a + h;
            float* o = out.data() + r * h;
            for (std::size_t j = 0; j < h; ++j) o[j] = a[j] * sigmoid(a[j]) * b[j];
        }
    });
    return out;
}

Matrix SwiGLUGate::backward(const Matrix& gradOutput) {
    const std::size_t h = input_.cols() / 2;
    if (gradOutput.rows() != input_.rows() || gradOutput.cols() != h)
        throw std::invalid_argument("SwiGLU::backward: gradient shape does not match the last forward");
    Matrix dx(input_.rows(), 2 * h);
    parallelFor(0, input_.rows(), 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* a = input_.data() + r * 2 * h;
            const float* b = a + h;
            const float* g = gradOutput.data() + r * h;
            float* da = dx.data() + r * 2 * h;
            float* db = da + h;
            for (std::size_t j = 0; j < h; ++j) {
                const float s = sigmoid(a[j]);
                const float silu = a[j] * s;
                da[j] = g[j] * b[j] * s * (1.0f + a[j] * (1.0f - s));
                db[j] = g[j] * silu;
            }
        }
    });
    return dx;
}

// ---- LayerNorm --------------------------------------------------------------------------

LayerNorm::LayerNorm(std::size_t dim, float epsilon)
    : dim_(dim), epsilon_(epsilon), gamma_(1, dim, 1.0f), beta_(1, dim, 0.0f), gammaGrad_(1, dim), betaGrad_(1, dim) {
    if (dim == 0) throw std::invalid_argument("LayerNorm: dim must be > 0");
}

Matrix LayerNorm::forward(const Matrix& input, bool training) {
    NC_PROFILE("layer norm");
    if (input.cols() != dim_)
        throw std::invalid_argument("LayerNorm: expected " + std::to_string(dim_) + " features, got " +
                                    std::to_string(input.cols()));
    const std::size_t n = input.rows();
    // Inference keeps nothing for backward.
    normalized_ = training ? Matrix(n, dim_) : Matrix();
    invStd_.assign(n, 0.0f);
    Matrix out(n, dim_);
    parallelFor(0, n, 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* x = input.data() + r * dim_;
            double mean = 0.0, var = 0.0;
            meanAndVariance(x, dim_, mean, var);
            const float inv = static_cast<float>(1.0 / std::sqrt(var + epsilon_));
            invStd_[r] = inv;
            const float m = static_cast<float>(mean);
            const float* g = gamma_.data();
            const float* b = beta_.data();
            float* o = out.data() + r * dim_;
            if (training) {
                float* xh = normalized_.data() + r * dim_;
                for (std::size_t j = 0; j < dim_; ++j) {
                    xh[j] = (x[j] - m) * inv;
                    o[j] = xh[j] * g[j] + b[j];
                }
            } else {
                for (std::size_t j = 0; j < dim_; ++j) o[j] = (x[j] - m) * inv * g[j] + b[j];
            }
        }
    });
    return out;
}

Matrix LayerNorm::backward(const Matrix& gradOutput) {
    if (!gradOutput.sameShape(normalized_))
        throw std::invalid_argument("LayerNorm::backward: gradient shape does not match the last forward");
    const std::size_t n = gradOutput.rows();
    gammaGrad_ = gradOutput.hadamard(normalized_).sumRows();
    betaGrad_ = gradOutput.sumRows();
    Matrix dx(n, dim_);
    const float d = static_cast<float>(dim_);
    parallelFor(0, n, 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* g = gradOutput.data() + r * dim_;
            const float* xh = normalized_.data() + r * dim_;
            float sumDxh = 0.0f, sumDxhXh = 0.0f;
            for (std::size_t j = 0; j < dim_; ++j) {
                const float dxh = g[j] * gamma_.data()[j];
                sumDxh += dxh;
                sumDxhXh += dxh * xh[j];
            }
            float* out = dx.data() + r * dim_;
            for (std::size_t j = 0; j < dim_; ++j) {
                const float dxh = g[j] * gamma_.data()[j];
                out[j] = invStd_[r] / d * (d * dxh - sumDxh - xh[j] * sumDxhXh);
            }
        }
    });
    return dx;
}

std::vector<Param> LayerNorm::parameters() {
    return {{&gamma_, &gammaGrad_}, {&beta_, &betaGrad_}};
}

std::string LayerNorm::describe() const { return "LayerNorm(" + std::to_string(dim_) + ")"; }

// ---- GELU ---------------------------------------------------------------------------------

namespace {
constexpr float kGeluC = 0.7978845608f;  // sqrt(2 / pi)
constexpr float kGeluA = 0.044715f;
} // namespace

Matrix GELU::forward(const Matrix& input, bool training) {
    Matrix out(input.rows(), input.cols());
    if (approximation_ != Approximation::Tanh) {
        const bool erf = approximation_ == Approximation::Erf;
        parallelFor(0, input.size(), 16384, [&](std::size_t i0, std::size_t i1) {
            for (std::size_t i = i0; i < i1; ++i) {
                const float x = input.data()[i];
                out.data()[i] = erf ? 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)) : x / (1.0f + std::exp(-1.702f * x));
            }
        });
        input_ = training ? input : Matrix();
        tanh_ = Matrix();
        return out;
    }
    input_ = input;
    tanh_ = Matrix(input.rows(), input.cols());
    parallelFor(0, input.size(), 16384, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            const float x = input.data()[i];
            const float t = std::tanh(kGeluC * (x + kGeluA * x * x * x));
            tanh_.data()[i] = t;
            out.data()[i] = 0.5f * x * (1.0f + t);
        }
    });
    if (!training) {
        input_ = Matrix();
        tanh_ = Matrix();
    }
    return out;
}

Matrix GELU::backward(const Matrix& gradOutput) {
    if (!gradOutput.sameShape(input_)) throw std::invalid_argument("GELU::backward: gradient shape does not match the last forward");
    Matrix grad(gradOutput);
    parallelFor(0, grad.size(), 16384, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            const float x = input_.data()[i];
            float d;
            if (approximation_ == Approximation::Tanh) {
                const float t = tanh_.data()[i];
                const float dt = (1.0f - t * t) * kGeluC * (1.0f + 3.0f * kGeluA * x * x);
                d = 0.5f * (1.0f + t) + 0.5f * x * dt;
            } else if (approximation_ == Approximation::Erf) {
                d = 0.5f * (1.0f + std::erf(x * 0.70710678118654752f)) + x * 0.39894228040143268f * std::exp(-0.5f * x * x);
            } else {
                const float s = 1.0f / (1.0f + std::exp(-1.702f * x));
                d = s + 1.702f * x * s * (1.0f - s);
            }
            grad.data()[i] *= d;
        }
    });
    return grad;
}

std::string GELU::describe() const {
    switch (approximation_) {
    case Approximation::Erf: return "GELU(erf)";
    case Approximation::Sigmoid: return "GELU(quick)";
    default: return "GELU";
    }
}

// ---- CausalSelfAttention ------------------------------------------------------------------

CausalSelfAttention::CausalSelfAttention(std::size_t dim, std::size_t heads, SequenceContextPtr context,
                                         bool rotary, float ropeBase, std::size_t kvHeads)
    : dim_(dim), heads_(heads), kvHeads_(kvHeads ? kvHeads : heads), headDim_(heads ? dim / heads : 0),
      context_(std::move(context)), rotary_(rotary), ropeBase_(ropeBase) {
    if (dim == 0 || heads == 0 || dim % heads != 0)
        throw std::invalid_argument("CausalSelfAttention: dim must be a positive multiple of heads");
    if (heads % kvHeads_ != 0)
        throw std::invalid_argument("CausalSelfAttention: heads must be a multiple of the key/value heads");
    if (rotary && headDim_ % 2 != 0) throw std::invalid_argument("CausalSelfAttention: rotary needs an even head size");
    if (!context_) throw std::invalid_argument("CausalSelfAttention: context must not be null");
    kvDim_ = kvHeads_ * headDim_;
    const std::size_t cols = dim + 2 * kvDim_;
    if (SkipParameterInit::active()) return;  // a loader assigns the parameters
    const float limitQkv = std::sqrt(6.0f / static_cast<float>(dim + cols));
    const float limitOut = std::sqrt(6.0f / static_cast<float>(2 * dim));
    wQkv_ = Matrix::uniform(dim, cols, -limitQkv, limitQkv);
    bQkv_ = Matrix(1, cols);
    wOut_ = Matrix::uniform(dim, dim, -limitOut, limitOut);
    bOut_ = Matrix(1, dim);
    wQkvGrad_ = Matrix(dim, cols);
    bQkvGrad_ = Matrix(1, cols);
    wOutGrad_ = Matrix(dim, dim);
    bOutGrad_ = Matrix(1, dim);
}

std::vector<double> ropeFrequencies(std::size_t headDim, double base, const RopeScaling& s) {
    const double pi = 3.14159265358979323846;
    std::vector<double> freq(headDim / 2);
    for (std::size_t i = 0; i < freq.size(); ++i) {
        double f = std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(headDim));
        if (s.enabled()) {
            const double context = static_cast<double>(s.originalContext), wavelength = 2.0 * pi / f;
            const double longest = context / s.lowFreqFactor, shortest = context / s.highFreqFactor;
            if (wavelength > longest) {
                f /= s.factor;
            } else if (wavelength >= shortest) {
                const double smooth = (context / wavelength - s.lowFreqFactor) / (s.highFreqFactor - s.lowFreqFactor);
                f = (1.0 - smooth) * f / s.factor + smooth * f;
            }
        }
        freq[i] = f;
    }
    return freq;
}

void CausalSelfAttention::setRopeScaling(const RopeScaling& scaling) {
    if (scaling.enabled() && (scaling.factor <= 0.0f || scaling.highFreqFactor <= scaling.lowFreqFactor || scaling.originalContext == 0))
        throw std::invalid_argument("CausalSelfAttention: invalid RoPE scaling");
    ropeScaling_ = scaling;
    ropeCos_.clear();
    ropeSin_.clear();
}

void CausalSelfAttention::ensureRopeTable(std::size_t positions) {
    const std::size_t half = headDim_ / 2;
    const std::size_t have = half ? ropeCos_.size() / half : 0;
    if (positions <= have) return;
    const std::size_t target = std::max(positions, have * 2);
    ropeCos_.resize(target * half);
    ropeSin_.resize(target * half);
    const std::vector<double> freq = ropeFrequencies(headDim_, ropeBase_, ropeScaling_);
    for (std::size_t p = have; p < target; ++p)
        for (std::size_t i = 0; i < half; ++i) {
            const double angle = static_cast<double>(p) * freq[i];
            ropeCos_[p * half + i] = static_cast<float>(std::cos(angle));
            ropeSin_[p * half + i] = static_cast<float>(std::sin(angle));
        }
}

template <typename PositionOf>
void CausalSelfAttention::rotate(float* qkv, std::size_t rows, std::size_t ld, PositionOf positionOf, bool inverse,
                                 bool keys) {
    const std::size_t half = headDim_ / 2;
    std::size_t maxPos = 0;
    for (std::size_t r = 0; r < rows; ++r) maxPos = std::max(maxPos, positionOf(r));
    ensureRopeTable(maxPos + 1);
    const float sign = inverse ? -1.0f : 1.0f;
    parallelFor(0, rows, 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* c = ropeCos_.data() + positionOf(r) * half;
            const float* s = ropeSin_.data() + positionOf(r) * half;
            for (std::size_t part = 0; part < (keys ? 2u : 1u); ++part) {
                const std::size_t count = part == 0 ? heads_ : kvHeads_;   // Q heads, then K heads
                for (std::size_t h = 0; h < count; ++h) {
                    float* x = qkv + r * ld + part * dim_ + h * headDim_;
                    for (std::size_t i = 0; i < half; ++i) {
                        const float x1 = x[i], x2 = x[i + half], sn = sign * s[i];
                        x[i] = x1 * c[i] - x2 * sn;
                        x[i + half] = x1 * sn + x2 * c[i];
                    }
                }
            }
        }
    });
}

Matrix CausalSelfAttention::projectQkv(const Matrix& input) const {
    if (!qQkv_.empty()) return qQkv_.multiply(input).addRowVector(bQkv_);
    return (input * wQkv_).addRowVector(bQkv_);
}

Matrix CausalSelfAttention::projectOut(const Matrix& merged) const {
    if (!qOut_.empty()) return qOut_.multiply(merged).addRowVector(bOut_);
    return (merged * wOut_).addRowVector(bOut_);
}

std::size_t CausalSelfAttention::quantize() {
    if (!qQkv_.empty() || wQkv_.empty()) return 0;
    qQkv_ = QuantizedWeights::fromInOut(wQkv_);
    qOut_ = QuantizedWeights::fromInOut(wOut_);
    wQkv_ = Matrix();
    wOut_ = Matrix();
    wQkvGrad_ = Matrix();
    wOutGrad_ = Matrix();
    bQkvGrad_ = Matrix();
    bOutGrad_ = Matrix();
    input_ = Matrix();
    return 2;
}

void CausalSelfAttention::setWeights(Matrix qkv, Matrix out) {
    if (qkv.rows() != dim_ || qkv.cols() != qkvColumns() || out.rows() != dim_ || out.cols() != dim_)
        throw std::invalid_argument("CausalSelfAttention::setWeights: wrong shapes");
    wQkv_ = std::move(qkv);
    wOut_ = std::move(out);
    bQkv_ = Matrix(1, qkvColumns());
    bOut_ = Matrix(1, dim_);
    qQkv_ = QuantizedWeights();
    qOut_ = QuantizedWeights();
}

void CausalSelfAttention::setQuantized(QuantizedWeights qkv, QuantizedWeights out) {
    if (qkv.inputs() != dim_ || qkv.outputs() != qkvColumns() || out.inputs() != dim_ || out.outputs() != dim_)
        throw std::invalid_argument("CausalSelfAttention::setQuantized: wrong shapes");
    qQkv_ = std::move(qkv);
    qOut_ = std::move(out);
    wQkv_ = wOut_ = wQkvGrad_ = wOutGrad_ = Matrix();
    bQkv_ = Matrix(1, qkvColumns());
    bOut_ = Matrix(1, dim_);
}

Matrix CausalSelfAttention::forward(const Matrix& input, bool training) {
    if (input.cols() != dim_)
        throw std::invalid_argument("CausalSelfAttention: expected " + std::to_string(dim_) + " features, got " +
                                    std::to_string(input.cols()));
    if (context_ && context_->incremental) return forwardIncremental(input);
    length_ = requireLength(context_, input.rows(), "CausalSelfAttention");
    batch_ = input.rows() / length_;
    const std::size_t L = length_, D = dim_, hd = headDim_, ld = D + 2 * kvDim_, group = heads_ / kvHeads_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    if (qQkv_.empty()) input_ = input;
    qkv_ = projectQkv(input);  // (B*L) x (D + 2*kvDim): [Q | K | V]
    if (rotary_) rotate(qkv_.data(), qkv_.rows(), ld, [L](std::size_t r) { return r % L; }, false);
    attention_ = Matrix(batch_ * heads_ * L, L);  // softmax probabilities per (sequence, head)
    merged_ = Matrix(batch_ * L, D);              // concatenated head outputs

    // Products per (sequence, head) are small: run them on the CPU, in parallel.
    parallelFor(0, batch_ * heads_, 1, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const std::size_t b = p / heads_, h = p % heads_, g = h / group;
            const float* q = qkv_.data() + b * L * ld + h * hd;
            const float* k = qkv_.data() + b * L * ld + D + g * hd;
            const float* v = qkv_.data() + b * L * ld + D + kvDim_ + g * hd;
            float* s = attention_.data() + p * L * L;
            gemmCpu(false, true, L, L, hd, q, ld, k, ld, s, L);
            for (std::size_t i = 0; i < L; ++i) {
                float* row = s + i * L;
                float maxVal = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j <= i; ++j) {
                    row[j] *= scale;
                    maxVal = std::max(maxVal, row[j]);
                }
                float total = 0.0f;
                for (std::size_t j = 0; j <= i; ++j) {
                    row[j] = std::exp(row[j] - maxVal);
                    total += row[j];
                }
                for (std::size_t j = 0; j <= i; ++j) row[j] /= total;
                for (std::size_t j = i + 1; j < L; ++j) row[j] = 0.0f;  // causal mask
            }
            gemmCpu(false, false, L, hd, L, s, L, v, ld, merged_.data() + b * L * D + h * hd, D);
        }
    });
    Matrix out = projectOut(merged_);
    if (!training) input_ = qkv_ = attention_ = merged_ = Matrix();  // only backward needs them
    return out;
}

Matrix CausalSelfAttention::forwardIncremental(const Matrix& input) {
    const std::size_t n = input.rows(), D = dim_, hd = headDim_, kv = kvDim_, ld = D + 2 * kv, group = heads_ / kvHeads_;
    const std::size_t past = context_->offset;
    if (past == 0) {
        cacheK_.clear();
        cacheV_.clear();
        cacheStart_ = 0;
    }
    const std::size_t cachedRows = cacheK_.size() / kv;
    if (cacheStart_ + cachedRows != past)
        throw std::logic_error("CausalSelfAttention: KV cache ends at position " + std::to_string(cacheStart_ + cachedRows) +
                               " but the context offset is " + std::to_string(past));
    Matrix qkv = projectQkv(input);
    if (rotary_) rotate(qkv.data(), n, ld, [past](std::size_t r) { return past + r; }, false);

    // Sliding window: keep at most `window` positions including the new ones.
    const std::size_t window = context_->window;
    if (window > 0 && cachedRows + n > window) {
        const std::size_t drop = std::min(cachedRows, cachedRows + n - window);
        cacheK_.erase(cacheK_.begin(), cacheK_.begin() + static_cast<long>(drop * kv));
        cacheV_.erase(cacheV_.begin(), cacheV_.begin() + static_cast<long>(drop * kv));
        cacheStart_ += drop;
    }
    for (std::size_t r = 0; r < n; ++r) {
        const float* row = qkv.data() + r * ld;
        cacheK_.insert(cacheK_.end(), row + D, row + D + kv);
        cacheV_.insert(cacheV_.end(), row + D + kv, row + D + 2 * kv);
    }
    const std::size_t total = cacheK_.size() / kv;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    Matrix merged(n, D);
    parallelFor(0, heads_, 1, [&](std::size_t h0, std::size_t h1) {
        std::vector<float> scores(n * total);
        for (std::size_t h = h0; h < h1; ++h) {
            const std::size_t g = h / group;
            const float* q = qkv.data() + h * hd;
            gemmCpu(false, true, n, total, hd, q, ld, cacheK_.data() + g * hd, kv, scores.data(), total);
            for (std::size_t i = 0; i < n; ++i) {
                float* row = scores.data() + i * total;
                const std::size_t visible = past + i + 1 - cacheStart_;  // causal: earlier positions and itself
                float maxVal = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j < visible; ++j) {
                    row[j] *= scale;
                    maxVal = std::max(maxVal, row[j]);
                }
                float sum = 0.0f;
                for (std::size_t j = 0; j < visible; ++j) {
                    row[j] = std::exp(row[j] - maxVal);
                    sum += row[j];
                }
                for (std::size_t j = 0; j < visible; ++j) row[j] /= sum;
                for (std::size_t j = visible; j < total; ++j) row[j] = 0.0f;
            }
            gemmCpu(false, false, n, hd, total, scores.data(), total, cacheV_.data() + g * hd, kv, merged.data() + h * hd, D);
        }
    });
    qkv_ = Matrix();  // inference only: backward is not available after this pass
    return projectOut(merged);
}

Matrix CausalSelfAttention::backward(const Matrix& gradOutput) {
    if (!qQkv_.empty()) throw std::logic_error("CausalSelfAttention: quantized layers are inference-only");
    if (qkv_.empty()) throw std::logic_error("CausalSelfAttention::backward called before forward");
    if (gradOutput.rows() != batch_ * length_ || gradOutput.cols() != dim_)
        throw std::invalid_argument("CausalSelfAttention::backward: gradient shape does not match the last forward");
    const std::size_t L = length_, D = dim_, hd = headDim_, ld = D + 2 * kvDim_, group = heads_ / kvHeads_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    wOutGrad_ = merged_.transposeMatmul(gradOutput);
    bOutGrad_ = gradOutput.sumRows();
    const Matrix dMerged = gradOutput.matmulTranspose(wOut_);
    Matrix dQkv(batch_ * L, ld);

    // One task per (sequence, key/value head): the query heads sharing a K/V
    // head run in order and accumulate into its dK / dV, so nothing races.
    parallelFor(0, batch_ * kvHeads_, 1, [&](std::size_t p0, std::size_t p1) {
        std::vector<float> dP(L * L);
        for (std::size_t p = p0; p < p1; ++p) {
            const std::size_t b = p / kvHeads_, g = p % kvHeads_;
            const float* k = qkv_.data() + b * L * ld + D + g * hd;
            const float* v = k + kvDim_;
            float* dk = dQkv.data() + b * L * ld + D + g * hd;
            float* dv = dk + kvDim_;
            for (std::size_t h = g * group; h < (g + 1) * group; ++h) {
                const bool first = h == g * group;
                const float* q = qkv_.data() + b * L * ld + h * hd;
                const float* prob = attention_.data() + (b * heads_ + h) * L * L;
                const float* dO = dMerged.data() + b * L * D + h * hd;
                float* dq = dQkv.data() + b * L * ld + h * hd;

                gemmCpu(false, true, L, L, hd, dO, D, v, ld, dP.data(), L);    // dP = dO V^T
                gemmCpu(true, false, L, hd, L, prob, L, dO, D, dv, ld, !first);  // dV += P^T dO
                for (std::size_t i = 0; i < L; ++i) {                           // softmax backward
                    float* row = dP.data() + i * L;
                    const float* pr = prob + i * L;
                    float dotp = 0.0f;
                    for (std::size_t j = 0; j <= i; ++j) dotp += row[j] * pr[j];
                    for (std::size_t j = 0; j < L; ++j) row[j] = j <= i ? pr[j] * (row[j] - dotp) * scale : 0.0f;
                }
                gemmCpu(false, false, L, hd, L, dP.data(), L, k, ld, dq, ld);           // dQ = dS K
                gemmCpu(true, false, L, hd, L, dP.data(), L, q, ld, dk, ld, !first);    // dK += dS^T Q
            }
        }
    });

    // Gradients so far are w.r.t. the rotated Q and K: rotate them back.
    if (rotary_) rotate(dQkv.data(), dQkv.rows(), ld, [L](std::size_t r) { return r % L; }, true);
    wQkvGrad_ = input_.transposeMatmul(dQkv);
    bQkvGrad_ = dQkv.sumRows();
    return dQkv.matmulTranspose(wQkv_);
}

std::vector<Param> CausalSelfAttention::parameters() {
    if (!qQkv_.empty()) return {};
    return {{&wQkv_, &wQkvGrad_}, {&bQkv_, &bQkvGrad_}, {&wOut_, &wOutGrad_}, {&bOut_, &bOutGrad_}};
}

std::string CausalSelfAttention::describe() const {
    std::string s = "CausalSelfAttention(" + std::to_string(dim_) + ", " + std::to_string(heads_) + " heads";
    if (kvHeads_ != heads_) s += ", " + std::to_string(kvHeads_) + " kv";
    if (rotary_) s += ", RoPE";
    if (!qQkv_.empty()) s += ", int8";
    return s + ")";
}

// ---- Residual -------------------------------------------------------------------------------

Matrix Residual::forward(const Matrix& input, bool training) {
    return input + inner_.forward(input, training);
}

Matrix Residual::backward(const Matrix& gradOutput) {
    return gradOutput + inner_.backward(gradOutput);
}

std::string Residual::describe() const {
    std::string s = "Residual[";
    for (std::size_t i = 0; i < inner_.layerCount(); ++i) {
        if (i) s += ", ";
        s += inner_.layer(i).name();
    }
    return s + "]";
}

// ---- GPT ------------------------------------------------------------------------------------

void saveGPTConfig(const std::string& path, const GPTConfig& c) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("saveGPTConfig: cannot open " + path);
    out << "vocab_size = " << c.vocabSize << "\ncontext_length = " << c.contextLength << "\ndim = " << c.dim
        << "\nheads = " << c.heads << "\nlayers = " << c.layers << "\nmlp_ratio = " << c.mlpRatio
        << "\ndropout = " << c.dropout
        << "\nposition = " << (c.position == PositionEncoding::Rotary ? "rotary" : "learned")
        << "\nnorm = " << (c.norm == Normalization::RMSNorm ? "rmsnorm" : "layernorm")
        << "\nmlp = " << (c.mlp == FeedForward::SwiGLU ? "swiglu" : "gelu")
        << "\ntie_embeddings = " << (c.tieEmbeddings ? 1 : 0) << "\nrope_base = " << c.ropeBase
        << "\nkv_heads = " << c.kvHeads << "\nnorm_eps = " << c.normEpsilon << "\nmlp_hidden = " << c.mlpHiddenSize << '\n';
    if (c.ropeScaling.enabled())
        out << "rope_scaling_factor = " << c.ropeScaling.factor << "\nrope_low_freq_factor = " << c.ropeScaling.lowFreqFactor
            << "\nrope_high_freq_factor = " << c.ropeScaling.highFreqFactor
            << "\nrope_original_context = " << c.ropeScaling.originalContext << '\n';
    if (!out) throw std::runtime_error("saveGPTConfig: write failed for " + path);
}

GPTConfig loadGPTConfig(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("loadGPTConfig: cannot open " + path);
    GPTConfig c;
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) {
            const auto b = s.find_first_not_of(" \t\r"), e = s.find_last_not_of(" \t\r");
            return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
        };
        const std::string key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
        try {
            if (key == "vocab_size") c.vocabSize = std::stoul(value);
            else if (key == "context_length") c.contextLength = std::stoul(value);
            else if (key == "dim") c.dim = std::stoul(value);
            else if (key == "heads") c.heads = std::stoul(value);
            else if (key == "layers") c.layers = std::stoul(value);
            else if (key == "mlp_ratio") c.mlpRatio = std::stoul(value);
            else if (key == "dropout") c.dropout = std::stof(value);
            else if (key == "position") {
                if (value != "rotary" && value != "learned") throw std::invalid_argument(value);
                c.position = value == "rotary" ? PositionEncoding::Rotary : PositionEncoding::Learned;
            } else if (key == "norm") {
                if (value != "rmsnorm" && value != "layernorm") throw std::invalid_argument(value);
                c.norm = value == "rmsnorm" ? Normalization::RMSNorm : Normalization::LayerNorm;
            } else if (key == "mlp") {
                if (value != "swiglu" && value != "gelu") throw std::invalid_argument(value);
                c.mlp = value == "swiglu" ? FeedForward::SwiGLU : FeedForward::GELU;
            } else if (key == "tie_embeddings") c.tieEmbeddings = std::stoi(value) != 0;
            else if (key == "rope_base") c.ropeBase = std::stof(value);
            else if (key == "kv_heads") c.kvHeads = std::stoul(value);
            else if (key == "norm_eps") c.normEpsilon = std::stof(value);
            else if (key == "mlp_hidden") c.mlpHiddenSize = std::stoul(value);
            else if (key == "rope_scaling_factor") c.ropeScaling.factor = std::stof(value);
            else if (key == "rope_low_freq_factor") c.ropeScaling.lowFreqFactor = std::stof(value);
            else if (key == "rope_high_freq_factor") c.ropeScaling.highFreqFactor = std::stof(value);
            else if (key == "rope_original_context") c.ropeScaling.originalContext = std::stoul(value);
        } catch (const std::exception&) {
            throw std::runtime_error("loadGPTConfig: bad value for " + key + " in " + path);
        }
    }
    return c;
}

std::size_t GPTConfig::mlpHidden() const {
    if (mlpHiddenSize) return mlpHiddenSize;
    if (mlp == FeedForward::GELU) return mlpRatio * dim;
    // SwiGLU has three matrices instead of two: 2/3 of the width keeps the
    // parameter count equal; rounded up to a multiple of 8.
    const std::size_t h = (2 * mlpRatio * dim + 2) / 3;
    return (h + 7) / 8 * 8;
}

SequenceContextPtr buildGPT(Network& net, const GPTConfig& c) {
    if (c.vocabSize == 0 || c.contextLength == 0 || c.dim == 0 || c.heads == 0 || c.dim % c.heads != 0)
        throw std::invalid_argument("buildGPT: invalid configuration");
    const bool rotary = c.position == PositionEncoding::Rotary;
    auto context = std::make_shared<SequenceContext>();
    context->slidingCacheSupported = rotary;
    auto& embedding = net.add<TokenEmbedding>(c.vocabSize, c.dim, c.contextLength, context, !rotary);
    if (c.dropout > 0.0f) net.add<Dropout>(c.dropout);
    auto addNorm = [&](Residual& block) {
        if (c.norm == Normalization::RMSNorm) block.add<RMSNorm>(c.dim, c.normEpsilon);
        else block.add<LayerNorm>(c.dim, c.normEpsilon);
    };
    const std::size_t hidden = c.mlpHidden();
    for (std::size_t i = 0; i < c.layers; ++i) {
        auto& attention = net.add<Residual>();
        addNorm(attention);
        auto& selfAttention = attention.add<CausalSelfAttention>(c.dim, c.heads, context, rotary, c.ropeBase, c.kvHeads);
        if (rotary) selfAttention.setRopeScaling(c.ropeScaling);
        if (c.dropout > 0.0f) attention.add<Dropout>(c.dropout);

        auto& mlp = net.add<Residual>();
        addNorm(mlp);
        if (c.mlp == FeedForward::SwiGLU) {
            mlp.add<Dense>(c.dim, 2 * hidden);
            mlp.add<SwiGLUGate>();
        } else {
            mlp.add<Dense>(c.dim, hidden);
            mlp.add<GELU>();
        }
        mlp.add<Dense>(hidden, c.dim);
        if (c.dropout > 0.0f) mlp.add<Dropout>(c.dropout);
    }
    if (c.norm == Normalization::RMSNorm) net.add<RMSNorm>(c.dim, c.normEpsilon);
    else net.add<LayerNorm>(c.dim, c.normEpsilon);
    if (c.tieEmbeddings) net.add<TiedOutput>(embedding);
    else net.add<Dense>(c.dim, c.vocabSize);
    return context;
}

} // namespace nc
