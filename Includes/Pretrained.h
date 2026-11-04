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

// Llama / Mistral SentencePiece-style BPE tokenizer from a Hugging Face
// `tokenizer.json`: special tokens are split out first, each piece of text is
// normalized (prepend "▁", spaces -> "▁"), BPE merges run over characters, and
// characters missing from the vocabulary fall back to <0xXX> byte tokens.
class LlamaTokenizer : public Tokenizer {
public:
    explicit LlamaTokenizer(const std::string& tokenizerJsonPath);

    // Special token names in the text (e.g. "</s>") become their ids.
    std::vector<int> encode(const std::string& text) const override { return encode(text, false); }
    std::vector<int> encode(const std::string& text, bool addBos) const;
    std::string decode(const std::vector<int>& ids) const override;  // skips special tokens
    std::string decodePiece(int id) const;  // one token, for streaming (no leading-space strip)
    std::size_t vocabSize() const override { return idToToken_.size(); }
    int bos() const { return bos_; }
    int eos() const { return eos_; }
    bool isSpecial(int id) const;

private:
    std::vector<int> encodeSegment(const std::string& text) const;

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int> ranks_;  // "left right" -> merge rank
    std::vector<std::pair<std::string, int>> specials_;  // added tokens matched verbatim
    std::array<int, 256> byteTokens_{};          // <0xXX> ids, -1 if absent
    std::string prepend_ = "\xe2\x96\x81";      // "▁"
    bool replaceSpaces_ = true;
    int bos_ = -1, eos_ = -1, unk_ = -1;
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

// Loads a Hugging Face Llama-architecture checkpoint (Llama 2, TinyLlama,
// Mistral without sliding windows, ...): config.json plus model.safetensors
// or a sharded model.safetensors.index.json, BF16/F16/F32. With int8 the
// linear layers are quantized while loading, so the float weights are never
// all in memory (TinyLlama-1.1B: ~1.5 GB instead of 4.4 GB).
PretrainedModel loadLlama(const std::string& directory, bool int8 = true);

} // namespace nc

#endif // NEURACRAFT_PRETRAINED_H
