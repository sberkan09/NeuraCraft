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

// Tokenizer from a Hugging Face `tokenizer.json`, in either of the two forms
// Llama-family models use:
//  - SentencePiece style (Llama 1/2, TinyLlama, Mistral): each piece of text is
//    normalized (prepend "▁", spaces -> "▁"), BPE merges run over characters,
//    and characters missing from the vocabulary fall back to <0xXX> tokens.
//  - Byte-level BPE (Llama 3, also GPT-2-style tokenizer.json files): text is
//    split with the model's regex pre-tokenizer (implemented by hand for the
//    Llama 3 and GPT-2 patterns), bytes are mapped to printable characters,
//    and whole words already in the vocabulary skip the merges when the model
//    sets `ignore_merges`.
// In both, added (special) tokens are split out of the text first.
// encode() caches per-word results in byte-level mode (not thread-safe).
class LlamaTokenizer : public Tokenizer {
public:
    explicit LlamaTokenizer(const std::string& tokenizerJsonPath);

    // Special token names in the text (e.g. "</s>", "<|eot_id|>") become their ids.
    std::vector<int> encode(const std::string& text) const override { return encode(text, false); }
    // addBos prepends the token the tokenizer's post-processor adds (<s>, <|begin_of_text|>).
    std::vector<int> encode(const std::string& text, bool addBos) const;
    std::string decode(const std::vector<int>& ids) const override;  // skips special tokens
    std::string decodePiece(int id) const;  // one token, for streaming (no leading-space strip)
    std::size_t vocabSize() const override { return idToToken_.size(); }
    int bos() const { return bos_; }
    int eos() const { return eos_; }
    int tokenId(const std::string& token) const;  // -1 if absent
    bool isSpecial(int id) const;
    bool byteLevel() const { return byteLevel_; }

    // Llama 3 pre-tokenization (exposed for tests): the regex
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    // maxDigits = 1 gives the Qwen 2 variant (\p{N} instead of \p{N}{1,3}).
    static std::vector<std::string> splitLlama3(const std::string& text, std::size_t maxDigits = 3);

private:
    enum class Split { Gpt2, Llama3 };
    std::vector<int> encodeSegment(const std::string& text) const;
    std::vector<int> encodeByteLevelWord(const std::string& word) const;
    void applyMerges(std::vector<std::string>& symbols) const;

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int> ranks_;  // "left right" -> merge rank
    std::vector<std::pair<std::string, int>> specials_;  // added tokens matched verbatim
    std::vector<bool> special_;                  // by id
    std::array<int, 256> byteTokens_{};          // <0xXX> ids, -1 if absent
    std::string prepend_ = "\xe2\x96\x81";      // "▁"
    bool replaceSpaces_ = true;
    bool byteLevel_ = false;
    bool ignoreMerges_ = false;
    bool addPrefixSpace_ = false;
    Split split_ = Split::Llama3;
    std::size_t maxDigits_ = 3;
    std::array<std::string, 256> byteToUnicode_;
    std::unordered_map<std::uint32_t, unsigned char> unicodeToByte_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
    int bos_ = -1, eos_ = -1, unk_ = -1;
};

// CLIP's BPE tokenizer (Stable Diffusion text encoders), from `vocab.json` and
// `merges.txt`, matching transformers' CLIPTokenizer without ftfy: text is
// cleaned (control characters dropped, whitespace collapsed, spaces around
// CJK ideographs), lowercased, split with CLIP's pattern, mapped to bytes and
// merged with an end-of-word marker "</w>". Lowercasing covers ASCII, Latin,
// Greek and Cyrillic. With a tokenizer_config.json, its extra added tokens are
// split out of the raw text first, as transformers does (Stable Diffusion 2's
// pad token "!" then always becomes id 0).
class ClipTokenizer : public Tokenizer {
public:
    ClipTokenizer(const std::string& vocabJsonPath, const std::string& mergesPath,
                  const std::string& tokenizerConfigPath = "");
    // Loads vocab.json, merges.txt and (if present) tokenizer_config.json from a directory.
    static ClipTokenizer fromDirectory(const std::string& directory);

    std::vector<int> encode(const std::string& text) const override;  // no bos / eos / padding
    std::string decode(const std::vector<int>& ids) const override;   // "</w>" becomes a space; specials dropped
    std::size_t vocabSize() const override { return idToToken_.size(); }
    // <|startoftext|> text <|endoftext|>, truncated to `length` (keeping the
    // end token) and padded with `pad` (Stable Diffusion 2 / Turbo pad with "!", id 0).
    std::vector<int> encodeForModel(const std::string& text, std::size_t length = 77, int pad = -1) const;
    int startOfText() const { return bos_; }
    int endOfText() const { return eos_; }

    static std::vector<std::string> splitWords(const std::string& cleanedText);  // CLIP's pattern (for tests)
    static std::string clean(const std::string& text);                           // cleanup + lowercase (for tests)

private:
    std::vector<int> encodeWord(const std::string& word) const;
    std::vector<int> encodeText(const std::string& text) const;  // no added-token splitting

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, int> ranks_;
    std::vector<std::pair<std::string, int>> added_;  // matched verbatim before cleanup, longest first
    std::array<std::string, 256> byteToUnicode_;
    std::unordered_map<std::uint32_t, unsigned char> unicodeToByte_;
    int bos_ = -1, eos_ = -1;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

struct PretrainedModel {
    Network net;
    SequenceContextPtr context;  // pass to generate() for the KV cache
    GPTConfig config;
    std::vector<int> stopTokens;  // end-of-turn ids from generation_config.json / config.json (Llama)
    float temperature = -1.0f;    // the model's recommended sampling (generation_config.json), -1 = none
    float topP = -1.0f;
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
