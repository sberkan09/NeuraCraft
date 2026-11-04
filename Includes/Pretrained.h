#ifndef NEURACRAFT_PRETRAINED_H
#define NEURACRAFT_PRETRAINED_H

#include "Network.h"
#include "Text.h"
#include "Transformer.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nc {

// GPT-2's byte-level BPE tokenizer, loaded from Hugging Face `vocab.json` and
// `merges.txt`. Text is split with (an implementation of) GPT-2's
// pre-tokenization pattern, bytes are mapped to printable characters, and
// merges are applied by rank, so ids match the original tokenizer.
// encode() caches per-word results (not thread-safe).
class Gpt2Tokenizer : public Tokenizer {
public:
    Gpt2Tokenizer(const std::string& vocabJsonPath, const std::string& mergesPath);

    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& ids) const override;
    std::size_t vocabSize() const override { return idToToken_.size(); }
    int endOfText() const { return endOfText_; }  // id of <|endoftext|>, or -1

    // GPT-2 pre-tokenization (exposed for tests).
    static std::vector<std::string> splitWords(const std::string& text);

private:
    std::vector<int> encodeWord(const std::string& word) const;

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int> ranks_;  // "left right" -> merge rank
    std::array<std::string, 256> byteToUnicode_;
    std::unordered_map<std::uint32_t, unsigned char> unicodeToByte_;
    int endOfText_ = -1;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

struct PretrainedModel {
    Network net;
    SequenceContextPtr context;  // pass to generate() for the KV cache
    GPTConfig config;
};

// Loads a Hugging Face GPT-2 checkpoint directory (config.json +
// model.safetensors; any size: gpt2, gpt2-medium, ...) into a NeuraCraft
// network built with buildGPT. Gradient buffers are released, so the model
// takes about as much memory as its weights; training it still works.
PretrainedModel loadGPT2(const std::string& directory);

} // namespace nc

#endif // NEURACRAFT_PRETRAINED_H
