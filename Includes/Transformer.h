#ifndef NEURACRAFT_TRANSFORMER_H
#define NEURACRAFT_TRANSFORMER_H

#include "Layer.h"
#include "Quantized.h"
#include "Network.h"

#include <memory>
#include <string>
#include <vector>

namespace nc {

// Language-model layout: the network input is (batch x seqLen) token ids, one
// sequence per row. TokenEmbedding turns it into (batch * seqLen) x dim rows,
// ordered sequence by sequence; every later layer works on that row layout.
// Layers that mix positions (attention) read the current sequence length from
// a SequenceContext shared with the embedding.
//
// Incremental (KV-cached) inference: with `incremental` set, a forward pass
// carries only new tokens of a single sequence that start at absolute position
// `offset`; each attention layer keeps the keys/values of earlier positions.
// generate() manages this; offset 0 starts a fresh cache. With `window` > 0
// attention keeps only the most recent `window` positions (a sliding cache,
// meaningful for rotary embeddings, which only depend on relative distance).
struct SequenceContext {
    std::size_t length = 0;
    bool incremental = false;
    std::size_t offset = 0;
    std::size_t window = 0;
    bool slidingCacheSupported = false;  // set by buildGPT for rotary models
};
using SequenceContextPtr = std::shared_ptr<SequenceContext>;

// Token embedding, plus a learned positional embedding unless
// `learnedPositions` is false (rotary models encode positions in attention).
// Input ids are stored as floats.
class TokenEmbedding : public Layer {
public:
    TokenEmbedding(std::size_t vocabSize, std::size_t dim, std::size_t maxContext, SequenceContextPtr context,
                   bool learnedPositions = true);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;  // returns zeros (ids are not differentiable)
    std::vector<Param> parameters() override;
    std::string name() const override { return "TokenEmbedding"; }
    std::string describe() const override;
    std::size_t outputSize(std::size_t) const override { return dim_; }

    std::size_t vocabSize() const { return vocab_; }
    std::size_t maxContext() const { return maxContext_; }
    Matrix& tokenTable() { return tokens_; }
    Matrix& positionTable() { return positions_; }
    bool learnedPositions() const { return learnedPositions_; }

    // For weight tying: TiedOutput writes its part of the token-table gradient
    // during backward, and this layer then adds instead of overwriting.
    void markTied() { tied_ = true; }
    Matrix& tokenGradient() { return tokenGrad_; }

private:
    std::size_t vocab_, dim_, maxContext_;
    SequenceContextPtr context_;
    bool learnedPositions_ = true;
    bool tied_ = false;
    Matrix tokens_, positions_, tokenGrad_, positionGrad_;
    std::vector<std::size_t> ids_;
    std::size_t batch_ = 0, length_ = 0;
};

// Per-row layer normalization with learned scale and shift.
class LayerNorm : public Layer {
public:
    explicit LayerNorm(std::size_t dim, float epsilon = 1e-5f);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "LayerNorm"; }
    std::string describe() const override;

    Matrix& gamma() { return gamma_; }
    Matrix& beta() { return beta_; }

private:
    std::size_t dim_;
    float epsilon_;
    Matrix gamma_, beta_, gammaGrad_, betaGrad_;
    Matrix normalized_;
    std::vector<float> invStd_;
};

// Root-mean-square normalization (Llama): y = x / rms(x) * gamma, no shift.
class RMSNorm : public Layer {
public:
    explicit RMSNorm(std::size_t dim, float epsilon = 1e-5f);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "RMSNorm"; }
    std::string describe() const override;

    Matrix& gamma() { return gamma_; }

private:
    std::size_t dim_;
    float epsilon_;
    Matrix gamma_, gammaGrad_;
    Matrix input_;
    std::vector<float> invRms_;
};

// SwiGLU gate: input rows are [a | b] (2H columns), output is silu(a) * b (H
// columns). Between Dense(D, 2H) and Dense(H, D) it forms a SwiGLU MLP.
class SwiGLUGate : public Layer {
public:
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "SwiGLU"; }
    std::size_t outputSize(std::size_t inputSize) const override { return inputSize / 2; }

private:
    Matrix input_;
};

// Gaussian error linear unit (tanh approximation, as in GPT-2).
class GELU : public Layer {
public:
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "GELU"; }

private:
    Matrix input_;
    Matrix tanh_;  // tanh term from forward, reused by backward
};

// Multi-head causal self-attention: each position attends to itself and
// earlier positions of the same sequence.
//
// With `rotary`, queries and keys are rotated by their absolute position
// (RoPE, "rotate half" layout as in Llama/GPT-NeoX), so scores depend only on
// relative distance.
class CausalSelfAttention : public Layer {
public:
    // kvHeads < heads gives grouped-query attention (Llama 2/3, TinyLlama):
    // each key/value head is shared by heads / kvHeads query heads.
    CausalSelfAttention(std::size_t dim, std::size_t heads, SequenceContextPtr context, bool rotary = false,
                        float ropeBase = 10000.0f, std::size_t kvHeads = 0);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "CausalSelfAttention"; }
    std::string describe() const override;

    std::size_t quantize() override;  // int8 projections; inference-only afterwards
    std::size_t keyValueHeads() const { return kvHeads_; }
    std::size_t qkvColumns() const { return dim_ + 2 * kvDim_; }
    // Install weights from a checkpoint, with zero biases: float (in x out)
    // matrices, or ready-made int8 ones.
    void setWeights(Matrix qkv, Matrix out);
    void setQuantized(QuantizedWeights qkv, QuantizedWeights out);

    // Fused [Q | K | V] projection (dim x (dim + 2 * kvHeads * headDim), heads
    // contiguous within each block) and the output projection (dim x dim), as in
    // GPT-2's c_attn / c_proj.
    Matrix& qkvWeight() { return wQkv_; }
    Matrix& qkvBias() { return bQkv_; }
    Matrix& outWeight() { return wOut_; }
    Matrix& outBias() { return bOut_; }

private:
    std::size_t dim_, heads_, kvHeads_, headDim_, kvDim_ = 0;
    SequenceContextPtr context_;
    Matrix wQkv_, bQkv_, wOut_, bOut_;
    Matrix wQkvGrad_, bQkvGrad_, wOutGrad_, bOutGrad_;
    Matrix input_, qkv_, attention_, merged_;  // caches for backward
    std::size_t batch_ = 0, length_ = 0;

    Matrix forwardIncremental(const Matrix& input);
    Matrix projectQkv(const Matrix& input) const;
    Matrix projectOut(const Matrix& merged) const;
    QuantizedWeights qQkv_, qOut_;
    std::vector<float> cacheK_, cacheV_;  // (positions x dim) keys / values for incremental inference
    std::size_t cacheStart_ = 0;          // absolute position of the first cached row

    // Rotates the Q and K parts of `qkv` rows; row r has position positionOf(r).
    // inverse = true applies the transpose (used for gradients).
    template <typename PositionOf>
    void rotate(float* qkv, std::size_t rows, std::size_t ld, PositionOf positionOf, bool inverse, bool keys = true);
    void ensureRopeTable(std::size_t positions);
    bool rotary_ = false;
    float ropeBase_ = 10000.0f;
    std::vector<float> ropeCos_, ropeSin_;  // positions x headDim/2
};

// Output projection that reuses the token embedding table (weight tying):
// logits = x * E^T. Saves vocab x dim parameters.
class TiedOutput : public Layer {
public:
    explicit TiedOutput(TokenEmbedding& embedding);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "TiedOutput"; }
    std::string describe() const override;
    // Keeps an int8 copy of the table for the output product (lookups stay float).
    std::size_t quantize() override;

private:
    TokenEmbedding& embedding_;
    Matrix input_;
    QuantizedWeights quantized_;
};

// y = x + f(x), where f is a stack of layers (a pre-norm transformer sub-block).
class Residual : public Layer {
public:
    Residual() = default;

    template <typename L, typename... Args>
    L& add(Args&&... args) {
        return inner_.add<L>(std::forward<Args>(args)...);
    }

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override { return inner_.parameters(); }
    std::size_t quantize() override { return inner_.quantize(); }
    std::string name() const override { return "Residual"; }
    std::string describe() const override;
    Network& inner() { return inner_; }

private:
    Network inner_;
};

enum class PositionEncoding { Learned, Rotary };
enum class Normalization { LayerNorm, RMSNorm };
enum class FeedForward { GELU, SwiGLU };

struct GPTConfig {
    std::size_t vocabSize = 0;
    std::size_t contextLength = 64;
    std::size_t dim = 128;
    std::size_t heads = 4;
    std::size_t layers = 4;
    std::size_t mlpRatio = 4;   // GELU MLP hidden = ratio * dim; SwiGLU uses ~2/3 of that (same parameters)
    float dropout = 0.0f;
    // Defaults are the GPT-2 style; modern() switches to the Llama style.
    PositionEncoding position = PositionEncoding::Learned;
    Normalization norm = Normalization::LayerNorm;
    FeedForward mlp = FeedForward::GELU;
    bool tieEmbeddings = false;
    float ropeBase = 10000.0f;
    std::size_t kvHeads = 0;         // grouped-query attention; 0 = same as heads
    float normEpsilon = 1e-5f;
    std::size_t mlpHiddenSize = 0;   // explicit MLP width (e.g. Llama's intermediate_size); 0 = derived

    GPTConfig& modern() {
        position = PositionEncoding::Rotary;
        norm = Normalization::RMSNorm;
        mlp = FeedForward::SwiGLU;
        tieEmbeddings = true;
        return *this;
    }
    std::size_t mlpHidden() const;
};

// key = value text files, so a saved model can be rebuilt without code changes.
void saveGPTConfig(const std::string& path, const GPTConfig& config);
GPTConfig loadGPTConfig(const std::string& path);

// Appends a GPT-style decoder (pre-norm blocks) to `net`. The network maps
// (batch x seqLen) token ids to (batch * seqLen) x vocab logits; train it with
// SparseSoftmaxCrossEntropy against the (batch x seqLen) next-token ids.
// Returns the shared sequence context; pass it to generate() for KV caching.
SequenceContextPtr buildGPT(Network& net, const GPTConfig& config);

} // namespace nc

#endif // NEURACRAFT_TRANSFORMER_H
