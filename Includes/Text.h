#ifndef NEURACRAFT_TEXT_H
#define NEURACRAFT_TEXT_H

#include "Data.h"
#include "Network.h"
#include "Transformer.h"

#include <array>
#include <cstdint>
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
};

// Autoregressive sampling from a language model built with buildGPT (or any
// network mapping 1 x L token ids to L x vocab logits). Uses at most the last
// `contextLength` tokens as context. Returns only the newly generated ids.
//
// Passing the SequenceContext returned by buildGPT enables the KV cache: each
// step then runs only the new token through the network. Rotary models slide
// the cache by one token per step, which matches the uncached path exactly.
// With learned positions a full cache is rebuilt from the last
// contextLength / 2 tokens, so the window is then between half and full length.
std::vector<int> generate(Network& net, const std::vector<int>& prompt, std::size_t newTokens,
                          std::size_t contextLength, const SamplingOptions& options = SamplingOptions{},
                          const std::shared_ptr<SequenceContext>& kvCache = nullptr);

} // namespace nc

#endif // NEURACRAFT_TEXT_H
