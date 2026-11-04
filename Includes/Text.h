#ifndef NEURACRAFT_TEXT_H
#define NEURACRAFT_TEXT_H

#include "Data.h"
#include "Network.h"

#include <array>
#include <string>
#include <vector>

namespace nc {

// Character-level tokenizer: the vocabulary is the sorted set of bytes in the
// text it was built from, so the same text always gives the same ids.
class CharTokenizer {
public:
    CharTokenizer() = default;
    explicit CharTokenizer(const std::string& text);

    std::size_t vocabSize() const { return idToChar_.size(); }
    bool contains(char c) const { return charToId_[static_cast<unsigned char>(c)] >= 0; }
    std::vector<int> encode(const std::string& text) const;  // throws on unknown characters
    std::string decode(const std::vector<int>& ids) const;

private:
    std::vector<char> idToChar_;
    std::array<int, 256> charToId_{};
};

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
std::vector<int> generate(Network& net, const std::vector<int>& prompt, std::size_t newTokens,
                          std::size_t contextLength, const SamplingOptions& options = SamplingOptions{});

} // namespace nc

#endif // NEURACRAFT_TEXT_H
