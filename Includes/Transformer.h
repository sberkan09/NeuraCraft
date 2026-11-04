#ifndef NEURACRAFT_TRANSFORMER_H
#define NEURACRAFT_TRANSFORMER_H

#include "Layer.h"
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
// generate() manages this; offset 0 starts a fresh cache.
struct SequenceContext {
    std::size_t length = 0;
    bool incremental = false;
    std::size_t offset = 0;
};
using SequenceContextPtr = std::shared_ptr<SequenceContext>;

// Token + learned positional embedding. Input ids are stored as floats.
class TokenEmbedding : public Layer {
public:
    TokenEmbedding(std::size_t vocabSize, std::size_t dim, std::size_t maxContext, SequenceContextPtr context);

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

private:
    std::size_t vocab_, dim_, maxContext_;
    SequenceContextPtr context_;
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
class CausalSelfAttention : public Layer {
public:
    CausalSelfAttention(std::size_t dim, std::size_t heads, SequenceContextPtr context);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::vector<Param> parameters() override;
    std::string name() const override { return "CausalSelfAttention"; }
    std::string describe() const override;

private:
    std::size_t dim_, heads_, headDim_;
    SequenceContextPtr context_;
    Matrix wQkv_, bQkv_, wOut_, bOut_;
    Matrix wQkvGrad_, bQkvGrad_, wOutGrad_, bOutGrad_;
    Matrix input_, qkv_, attention_, merged_;  // caches for backward
    std::size_t batch_ = 0, length_ = 0;

    Matrix forwardIncremental(const Matrix& input);
    std::vector<float> cacheK_, cacheV_;  // (positions x dim) keys / values for incremental inference
    std::size_t cachedPositions_ = 0;
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
    std::string name() const override { return "Residual"; }
    std::string describe() const override;

private:
    Network inner_;
};

struct GPTConfig {
    std::size_t vocabSize = 0;
    std::size_t contextLength = 64;
    std::size_t dim = 128;
    std::size_t heads = 4;
    std::size_t layers = 4;
    std::size_t mlpRatio = 4;
    float dropout = 0.0f;
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
