#ifndef NEURACRAFT_TEXT_H
#define NEURACRAFT_TEXT_H

#include "Data.h"
#include "Network.h"
#include "Transformer.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nc {

// Common interface for tokenizers (text <-> token ids).
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& ids) const = 0;
    virtual std::size_t vocabSize() const = 0;
};

// Character-level tokenizer: the vocabulary is the sorted set of bytes in the
// text it was built from, so the same text always gives the same ids.
class CharTokenizer : public Tokenizer {
public:
    CharTokenizer() = default;
    explicit CharTokenizer(const std::string& text);

    std::size_t vocabSize() const override { return idToChar_.size(); }
    bool contains(char c) const { return charToId_[static_cast<unsigned char>(c)] >= 0; }
    std::vector<int> encode(const std::string& text) const override;  // throws on unknown characters
    std::string decode(const std::vector<int>& ids) const override;

private:
    std::vector<char> idToChar_;
    std::array<int, 256> charToId_{};
};

// Byte-level byte-pair encoding (GPT-2 style). Ids 0..255 are raw bytes, so any
// input can be encoded; each learned merge adds one id. Text is first split
// into words (letters, digits, punctuation runs, each with an optional leading
// space, and whitespace runs) and merges never cross word boundaries.
// encode() caches word encodings, so share one instance between threads only
// for decoding.
class BpeTokenizer : public Tokenizer {
public:
    BpeTokenizer();  // bytes only, no merges

    // Learns merges until the vocabulary has `vocabSize` (>= 256) entries or
    // no pair occurs at least twice.
    static BpeTokenizer train(const std::string& text, std::size_t vocabSize);

    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& ids) const override;
    std::size_t vocabSize() const override { return pieces_.size(); }
    std::size_t mergeCount() const { return merges_.size(); }
    const std::string& piece(int id) const { return pieces_.at(static_cast<std::size_t>(id)); }

    void save(const std::string& path) const;
    static BpeTokenizer load(const std::string& path);

private:
    void addMerge(int a, int b);
    std::vector<int> encodeWord(const std::string& word) const;

    std::vector<std::pair<int, int>> merges_;   // merge i creates id 256 + i
    std::vector<std::string> pieces_;           // bytes of every id
    std::unordered_map<std::uint64_t, int> rank_;  // pair -> merge index
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

// Splits text into the words BPE works on (concatenating them gives the text back).
std::vector<std::string> splitWords(const std::string& text);

std::string loadText(const std::string& path);

// Windows of seqLen + 1 tokens starting every `stride` tokens (0 = seqLen):
// x row = tokens[i .. i+seqLen), y row = tokens[i+1 .. i+seqLen] (next tokens).
Dataset makeNextTokenDataset(const std::vector<int>& tokens, std::size_t seqLen, std::size_t stride = 0);

struct SamplingOptions {
    float temperature = 1.0f;  // 0 = always pick the most likely token
    std::size_t topK = 0;      // 0 = sample from the whole vocabulary
    float topP = 1.0f;         // nucleus sampling: smallest set of tokens with this much probability (1 = off)
    // Tokens seen in the last `repetitionWindow` positions get their logits
    // divided (if positive) or multiplied (if negative) by this; 1 = off.
    float repetitionPenalty = 1.0f;
    std::size_t repetitionWindow = 64;
    std::function<void(int)> onToken;  // called with each new token as it is produced (streaming)
    int stopToken = -1;        // stop after producing this token (e.g. <|endoftext|>), -1 = never
    std::vector<int> stopTokens;  // further stop tokens (Llama 3 ends turns with <|eot_id|> or <|eom_id|>)
};

// Picks the next token from one row of logits; `history` (the context so far)
// is only used for the repetition penalty.
int sampleToken(const float* logits, std::size_t vocab, const SamplingOptions& options,
                const std::vector<int>& history = {});

// Autoregressive sampling from a language model built with buildGPT (or any
// network mapping 1 x L token ids to L x vocab logits). Uses at most the last
// `contextLength` tokens as context. Returns only the newly generated ids.
//
// Passing the SequenceContext returned by buildGPT enables the KV cache: each
// step then runs only the new token through the network. Rotary models slide
// the cache by one token per step (sliding-window attention: exact for one
// layer; deeper layers also carry information from beyond the window, which
// on the Shakespeare model changes next-token loss by < 0.01 %). With learned
// positions a full cache is rebuilt from the last contextLength / 2 tokens.
std::vector<int> generate(Network& net, const std::vector<int>& prompt, std::size_t newTokens,
                          std::size_t contextLength, const SamplingOptions& options = SamplingOptions{},
                          const std::shared_ptr<SequenceContext>& kvCache = nullptr);

// Generation that keeps its KV cache between calls, for multi-turn chat: each
// call feeds only tokens the model has not seen yet (the new input, plus the
// previous reply's last token), then samples until a stop token or maxTokens.
// The network must not run other incremental passes in between (attention
// layers check this and throw).
class Generator {
public:
    Generator(Network& net, SequenceContextPtr context, std::size_t contextLength);

    // Appends `input` to the conversation and returns the continuation.
    std::vector<int> generate(const std::vector<int>& input, std::size_t maxTokens, const SamplingOptions& options);
    void reset();  // forget the conversation
    const std::vector<int>& history() const { return tokens_; }  // every token so far, prompts and replies
    std::size_t cachedTokens() const { return fed_ - start_; }   // positions currently held by the KV cache

private:
    std::vector<float> feed(std::size_t from);  // runs tokens_[from..] on top of the cache; returns the last logits

    Network& net_;
    SequenceContextPtr context_;
    std::size_t contextLength_;
    std::vector<int> tokens_;
    std::size_t fed_ = 0;     // tokens_[0..fed_) have been through the network
    std::size_t start_ = 0;   // first token still covered by the cache (after a rebuild)
    std::size_t offset_ = 0;  // the cache's next position
};

} // namespace nc

#endif // NEURACRAFT_TEXT_H
