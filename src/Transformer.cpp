#include "Transformer.h"
#include "Activations.h"
#include "Backend.h"
#include "Dense.h"
#include "Parallel.h"

#include <cmath>
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
                               SequenceContextPtr context)
    : vocab_(vocabSize), dim_(dim), maxContext_(maxContext), context_(std::move(context)) {
    if (vocabSize == 0 || dim == 0 || maxContext == 0) throw std::invalid_argument("TokenEmbedding: sizes must be > 0");
    if (!context_) throw std::invalid_argument("TokenEmbedding: context must not be null");
    tokens_ = Matrix::normal(vocabSize, dim, 0.0f, 0.02f);
    positions_ = Matrix::normal(maxContext, dim, 0.0f, 0.02f);
    tokenGrad_ = Matrix(vocabSize, dim);
    positionGrad_ = Matrix(maxContext, dim);
}

Matrix TokenEmbedding::forward(const Matrix& input, bool) {
    batch_ = input.rows();
    length_ = input.cols();
    const std::size_t offset = context_->incremental ? context_->offset : 0;
    if (context_->incremental && batch_ != 1)
        throw std::invalid_argument("TokenEmbedding: incremental inference handles one sequence at a time");
    if (length_ == 0 || offset + length_ > maxContext_)
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
        const float* tok = tokens_.data() + ids_[r] * dim_;
        const float* pos = positions_.data() + (offset + r % length_) * dim_;
        float* o = out.data() + r * dim_;
        for (std::size_t j = 0; j < dim_; ++j) o[j] = tok[j] + pos[j];
    }
    return out;
}

Matrix TokenEmbedding::backward(const Matrix& gradOutput) {
    if (gradOutput.rows() != ids_.size() || gradOutput.cols() != dim_)
        throw std::invalid_argument("TokenEmbedding::backward: gradient shape does not match the last forward");
    tokenGrad_.fill(0.0f);
    positionGrad_.fill(0.0f);
    for (std::size_t r = 0; r < ids_.size(); ++r) {
        const float* g = gradOutput.data() + r * dim_;
        float* tg = tokenGrad_.data() + ids_[r] * dim_;
        float* pg = positionGrad_.data() + (r % length_) * dim_;
        for (std::size_t j = 0; j < dim_; ++j) {
            tg[j] += g[j];
            pg[j] += g[j];
        }
    }
    return Matrix(batch_, length_);
}

std::vector<Param> TokenEmbedding::parameters() {
    return {{&tokens_, &tokenGrad_}, {&positions_, &positionGrad_}};
}

std::string TokenEmbedding::describe() const {
    return "TokenEmbedding(" + std::to_string(vocab_) + " x " + std::to_string(dim_) + ", ctx " +
           std::to_string(maxContext_) + ")";
}

// ---- LayerNorm --------------------------------------------------------------------------

LayerNorm::LayerNorm(std::size_t dim, float epsilon)
    : dim_(dim), epsilon_(epsilon), gamma_(1, dim, 1.0f), beta_(1, dim, 0.0f), gammaGrad_(1, dim), betaGrad_(1, dim) {
    if (dim == 0) throw std::invalid_argument("LayerNorm: dim must be > 0");
}

Matrix LayerNorm::forward(const Matrix& input, bool) {
    if (input.cols() != dim_)
        throw std::invalid_argument("LayerNorm: expected " + std::to_string(dim_) + " features, got " +
                                    std::to_string(input.cols()));
    const std::size_t n = input.rows();
    normalized_ = Matrix(n, dim_);
    invStd_.assign(n, 0.0f);
    Matrix out(n, dim_);
    parallelFor(0, n, 64, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* x = input.data() + r * dim_;
            double mean = 0.0;
            for (std::size_t j = 0; j < dim_; ++j) mean += x[j];
            mean /= static_cast<double>(dim_);
            double var = 0.0;
            for (std::size_t j = 0; j < dim_; ++j) var += (x[j] - mean) * (x[j] - mean);
            var /= static_cast<double>(dim_);
            const float inv = static_cast<float>(1.0 / std::sqrt(var + epsilon_));
            invStd_[r] = inv;
            float* xh = normalized_.data() + r * dim_;
            float* o = out.data() + r * dim_;
            for (std::size_t j = 0; j < dim_; ++j) {
                xh[j] = static_cast<float>(x[j] - mean) * inv;
                o[j] = xh[j] * gamma_.data()[j] + beta_.data()[j];
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

Matrix GELU::forward(const Matrix& input, bool) {
    input_ = input;
    tanh_ = Matrix(input.rows(), input.cols());
    Matrix out(input.rows(), input.cols());
    parallelFor(0, input.size(), 16384, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            const float x = input.data()[i];
            const float t = std::tanh(kGeluC * (x + kGeluA * x * x * x));
            tanh_.data()[i] = t;
            out.data()[i] = 0.5f * x * (1.0f + t);
        }
    });
    return out;
}

Matrix GELU::backward(const Matrix& gradOutput) {
    if (!gradOutput.sameShape(tanh_)) throw std::invalid_argument("GELU::backward: gradient shape does not match the last forward");
    Matrix grad(gradOutput);
    parallelFor(0, grad.size(), 16384, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            const float x = input_.data()[i];
            const float t = tanh_.data()[i];
            const float dt = (1.0f - t * t) * kGeluC * (1.0f + 3.0f * kGeluA * x * x);
            grad.data()[i] *= 0.5f * (1.0f + t) + 0.5f * x * dt;
        }
    });
    return grad;
}

// ---- CausalSelfAttention ------------------------------------------------------------------

CausalSelfAttention::CausalSelfAttention(std::size_t dim, std::size_t heads, SequenceContextPtr context)
    : dim_(dim), heads_(heads), headDim_(heads ? dim / heads : 0), context_(std::move(context)) {
    if (dim == 0 || heads == 0 || dim % heads != 0)
        throw std::invalid_argument("CausalSelfAttention: dim must be a positive multiple of heads");
    if (!context_) throw std::invalid_argument("CausalSelfAttention: context must not be null");
    const float limitQkv = std::sqrt(6.0f / static_cast<float>(dim + 3 * dim));
    const float limitOut = std::sqrt(6.0f / static_cast<float>(2 * dim));
    wQkv_ = Matrix::uniform(dim, 3 * dim, -limitQkv, limitQkv);
    bQkv_ = Matrix(1, 3 * dim);
    wOut_ = Matrix::uniform(dim, dim, -limitOut, limitOut);
    bOut_ = Matrix(1, dim);
    wQkvGrad_ = Matrix(dim, 3 * dim);
    bQkvGrad_ = Matrix(1, 3 * dim);
    wOutGrad_ = Matrix(dim, dim);
    bOutGrad_ = Matrix(1, dim);
}

Matrix CausalSelfAttention::forward(const Matrix& input, bool) {
    if (input.cols() != dim_)
        throw std::invalid_argument("CausalSelfAttention: expected " + std::to_string(dim_) + " features, got " +
                                    std::to_string(input.cols()));
    if (context_ && context_->incremental) return forwardIncremental(input);
    length_ = requireLength(context_, input.rows(), "CausalSelfAttention");
    batch_ = input.rows() / length_;
    const std::size_t L = length_, D = dim_, hd = headDim_, ld = 3 * D;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    input_ = input;
    qkv_ = (input * wQkv_).addRowVector(bQkv_);   // (B*L) x 3D: [Q | K | V]
    attention_ = Matrix(batch_ * heads_ * L, L);  // softmax probabilities per (sequence, head)
    merged_ = Matrix(batch_ * L, D);              // concatenated head outputs

    // Products per (sequence, head) are small: run them on the CPU, in parallel.
    parallelFor(0, batch_ * heads_, 1, [&](std::size_t p0, std::size_t p1) {
        for (std::size_t p = p0; p < p1; ++p) {
            const std::size_t b = p / heads_, h = p % heads_;
            const float* q = qkv_.data() + b * L * ld + h * hd;
            const float* k = q + D;
            const float* v = q + 2 * D;
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
    return (merged_ * wOut_).addRowVector(bOut_);
}

Matrix CausalSelfAttention::forwardIncremental(const Matrix& input) {
    const std::size_t n = input.rows(), D = dim_, hd = headDim_, ld = 3 * D;
    const std::size_t past = context_->offset;
    if (past == 0) {
        cacheK_.clear();
        cacheV_.clear();
        cachedPositions_ = 0;
    }
    if (cachedPositions_ != past)
        throw std::logic_error("CausalSelfAttention: KV cache holds " + std::to_string(cachedPositions_) +
                               " positions but the context offset is " + std::to_string(past));
    const Matrix qkv = (input * wQkv_).addRowVector(bQkv_);
    for (std::size_t r = 0; r < n; ++r) {
        const float* row = qkv.data() + r * ld;
        cacheK_.insert(cacheK_.end(), row + D, row + 2 * D);
        cacheV_.insert(cacheV_.end(), row + 2 * D, row + 3 * D);
    }
    cachedPositions_ += n;
    const std::size_t total = cachedPositions_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    Matrix merged(n, D);
    std::vector<float> scores(n * total);
    for (std::size_t h = 0; h < heads_; ++h) {
        const float* q = qkv.data() + h * hd;
        gemmCpu(false, true, n, total, hd, q, ld, cacheK_.data() + h * hd, D, scores.data(), total);
        for (std::size_t i = 0; i < n; ++i) {
            float* row = scores.data() + i * total;
            const std::size_t visible = past + i + 1;  // causal: earlier positions and itself
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
        gemmCpu(false, false, n, hd, total, scores.data(), total, cacheV_.data() + h * hd, D, merged.data() + h * hd, D);
    }
    qkv_ = Matrix();  // inference only: backward is not available after this pass
    return (merged * wOut_).addRowVector(bOut_);
}

Matrix CausalSelfAttention::backward(const Matrix& gradOutput) {
    if (qkv_.empty()) throw std::logic_error("CausalSelfAttention::backward called before forward");
    if (gradOutput.rows() != batch_ * length_ || gradOutput.cols() != dim_)
        throw std::invalid_argument("CausalSelfAttention::backward: gradient shape does not match the last forward");
    const std::size_t L = length_, D = dim_, hd = headDim_, ld = 3 * D;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    wOutGrad_ = merged_.transposeMatmul(gradOutput);
    bOutGrad_ = gradOutput.sumRows();
    const Matrix dMerged = gradOutput.matmulTranspose(wOut_);
    Matrix dQkv(batch_ * L, 3 * D);

    parallelFor(0, batch_ * heads_, 1, [&](std::size_t p0, std::size_t p1) {
        std::vector<float> dP(L * L);
        for (std::size_t p = p0; p < p1; ++p) {
            const std::size_t b = p / heads_, h = p % heads_;
            const std::size_t base = b * L * ld + h * hd;
            const float* q = qkv_.data() + base;
            const float* k = q + D;
            const float* v = q + 2 * D;
            const float* prob = attention_.data() + p * L * L;
            const float* dO = dMerged.data() + b * L * D + h * hd;
            float* dq = dQkv.data() + base;
            float* dk = dq + D;
            float* dv = dq + 2 * D;

            gemmCpu(false, true, L, L, hd, dO, D, v, ld, dP.data(), L);  // dP = dO V^T
            gemmCpu(true, false, L, hd, L, prob, L, dO, D, dv, ld);      // dV = P^T dO
            for (std::size_t i = 0; i < L; ++i) {                         // softmax backward
                float* row = dP.data() + i * L;
                const float* pr = prob + i * L;
                float dot = 0.0f;
                for (std::size_t j = 0; j <= i; ++j) dot += row[j] * pr[j];
                for (std::size_t j = 0; j < L; ++j) row[j] = j <= i ? pr[j] * (row[j] - dot) * scale : 0.0f;
            }
            gemmCpu(false, false, L, hd, L, dP.data(), L, k, ld, dq, ld);  // dQ = dS K
            gemmCpu(true, false, L, hd, L, dP.data(), L, q, ld, dk, ld);   // dK = dS^T Q
        }
    });

    wQkvGrad_ = input_.transposeMatmul(dQkv);
    bQkvGrad_ = dQkv.sumRows();
    return dQkv.matmulTranspose(wQkv_);
}

std::vector<Param> CausalSelfAttention::parameters() {
    return {{&wQkv_, &wQkvGrad_}, {&bQkv_, &bQkvGrad_}, {&wOut_, &wOutGrad_}, {&bOut_, &bOutGrad_}};
}

std::string CausalSelfAttention::describe() const {
    return "CausalSelfAttention(" + std::to_string(dim_) + ", " + std::to_string(heads_) + " heads)";
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

SequenceContextPtr buildGPT(Network& net, const GPTConfig& c) {
    if (c.vocabSize == 0 || c.contextLength == 0 || c.dim == 0 || c.heads == 0 || c.dim % c.heads != 0)
        throw std::invalid_argument("buildGPT: invalid configuration");
    auto context = std::make_shared<SequenceContext>();
    net.add<TokenEmbedding>(c.vocabSize, c.dim, c.contextLength, context);
    if (c.dropout > 0.0f) net.add<Dropout>(c.dropout);
    for (std::size_t i = 0; i < c.layers; ++i) {
        auto& attention = net.add<Residual>();
        attention.add<LayerNorm>(c.dim);
        attention.add<CausalSelfAttention>(c.dim, c.heads, context);
        if (c.dropout > 0.0f) attention.add<Dropout>(c.dropout);

        auto& mlp = net.add<Residual>();
        mlp.add<LayerNorm>(c.dim);
        mlp.add<Dense>(c.dim, c.mlpRatio * c.dim);
        mlp.add<GELU>();
        mlp.add<Dense>(c.mlpRatio * c.dim, c.dim);
        if (c.dropout > 0.0f) mlp.add<Dropout>(c.dropout);
    }
    net.add<LayerNorm>(c.dim);
    net.add<Dense>(c.dim, c.vocabSize);
    return context;
}

} // namespace nc
