#include "Pretrained.h"
#include "Activations.h"
#include "Dense.h"
#include "Json.h"
#include "SafeTensors.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void appendUtf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Decodes one UTF-8 sequence at s[i]; invalid bytes come back as themselves
// with length 1 (so pre-tokenization never loses input).
std::uint32_t decodeUtf8(const std::string& s, std::size_t i, std::size_t& len) {
    const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char c = b(i);
    len = 1;
    if (c < 0x80) return c;
    std::size_t need = 0;
    std::uint32_t cp = 0;
    if ((c & 0xE0) == 0xC0) { need = 1; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { need = 2; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { need = 3; cp = c & 0x07; }
    else return 0x110000 + c;  // invalid lead byte
    if (i + need >= s.size()) return 0x110000 + c;  // truncated sequence
    for (std::size_t k = 1; k <= need; ++k) {
        if ((b(i + k) & 0xC0) != 0x80) return 0x110000 + c;
        cp = (cp << 6) | (b(i + k) & 0x3F);
    }
    len = need + 1;
    return cp;
}

bool isSpace(std::uint32_t c) {
    return c == ' ' || (c >= '\t' && c <= '\r') || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool isNumber(std::uint32_t c) {
    return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) || (c >= 0x966 && c <= 0x96F) ||
           (c >= 0x2070 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x2189) ||
           (c >= 0x2460 && c <= 0x249B) || (c >= 0xFF10 && c <= 0xFF19);
}

// Approximation of \p{L}: the letter blocks of the scripts GPT-2 users meet in
// practice. Symbols, punctuation and emoji fall outside and count as "other".
bool isLetter(std::uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
    if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;  // Latin-1 / Extended-A/B (Turkish ç ğ ı ö ş ü)
    if (c >= 0x250 && c <= 0x2C1) return true;                      // IPA, spacing modifiers (letter part)
    if (c >= 0x2C6 && c <= 0x2D1) return true;
    if (c >= 0x370 && c <= 0x3FF) return c != 0x37E && c != 0x387 && c != 0x375 && c != 0x3F6;  // Greek
    if (c >= 0x400 && c <= 0x52F) return !(c >= 0x482 && c <= 0x489);  // Cyrillic
    if (c >= 0x531 && c <= 0x587) return true;                          // Armenian
    if (c >= 0x5D0 && c <= 0x5F2) return true;                          // Hebrew
    if ((c >= 0x620 && c <= 0x64A) || (c >= 0x671 && c <= 0x6D3)) return true;  // Arabic
    if (c >= 0x904 && c <= 0x939) return true;                          // Devanagari letters
    if (c >= 0xE01 && c <= 0xE30) return true;                          // Thai
    if (c >= 0x10A0 && c <= 0x10FF) return true;                        // Georgian
    if (c >= 0x1100 && c <= 0x11FF) return true;                        // Hangul Jamo
    if (c >= 0x1E00 && c <= 0x1FFF) return !(c >= 0x1FBD && c <= 0x1FC1) && !(c >= 0x1FCD && c <= 0x1FCF) &&
                                           !(c >= 0x1FDD && c <= 0x1FDF) && !(c >= 0x1FED && c <= 0x1FEF) &&
                                           c != 0x1FFD && c != 0x1FFE;
    if (c >= 0x3041 && c <= 0x3096) return true;                        // Hiragana
    if (c >= 0x30A1 && c <= 0x30FA) return true;                        // Katakana
    if ((c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF)) return true;  // CJK
    if (c >= 0xAC00 && c <= 0xD7A3) return true;                        // Hangul syllables
    if (c >= 0xFF21 && c <= 0xFF3A) return true;                        // Fullwidth Latin
    if (c >= 0xFF41 && c <= 0xFF5A) return true;
    return false;
}

enum class Kind { Space, Letter, Number, Other };

Kind kindOf(std::uint32_t c) {
    if (isSpace(c)) return Kind::Space;
    if (isLetter(c)) return Kind::Letter;
    if (isNumber(c)) return Kind::Number;
    return Kind::Other;
}

} // namespace

// Implements 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> Gpt2Tokenizer::splitWords(const std::string& text) {
    struct Cp {
        std::uint32_t cp;
        std::size_t pos, len;
        Kind kind;
    };
    std::vector<Cp> cps;
    for (std::size_t i = 0; i < text.size();) {
        std::size_t len = 1;
        const std::uint32_t cp = decodeUtf8(text, i, len);
        cps.push_back({cp, i, len, kindOf(cp)});
        i += len;
    }
    std::vector<std::string> words;
    const std::size_t n = cps.size();
    auto emit = [&](std::size_t a, std::size_t b) {
        const std::size_t begin = cps[a].pos, end = b < n ? cps[b].pos : text.size();
        words.push_back(text.substr(begin, end - begin));
    };
    std::size_t i = 0;
    while (i < n) {
        const std::uint32_t c = cps[i].cp;
        // Contractions.
        if (c == '\'' && i + 1 < n) {
            const std::uint32_t a = cps[i + 1].cp, b = i + 2 < n ? cps[i + 2].cp : 0;
            std::size_t take = 0;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) take = 3;
            else if (a == 's' || a == 't' || a == 'm' || a == 'd') take = 2;
            if (take) {
                emit(i, i + take);
                i += take;
                continue;
            }
        }
        // ' ?X+' for X in letters, numbers, other.
        std::size_t j = i;
        if (c == ' ' && i + 1 < n && cps[i + 1].kind != Kind::Space) j = i + 1;
        const Kind k = cps[j].kind;
        if (k != Kind::Space) {
            std::size_t e = j;
            while (e < n && cps[e].kind == k) ++e;
            emit(i, e);
            i = e;
            continue;
        }
        // Whitespace: \s+(?!\S) leaves the last space before a non-space, else \s+.
        std::size_t e = i;
        while (e < n && cps[e].kind == Kind::Space) ++e;
        if (e == n) {
            emit(i, e);
            i = e;
        } else if (e - i >= 2) {
            emit(i, e - 1);
            i = e - 1;
        } else {
            emit(i, i + 1);
            i += 1;
        }
    }
    return words;
}

Gpt2Tokenizer::Gpt2Tokenizer(const std::string& vocabJsonPath, const std::string& mergesPath) {
    // bytes_to_unicode(): printable bytes map to themselves, the rest to 256+.
    std::vector<int> printable;
    for (int b = '!'; b <= '~'; ++b) printable.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) printable.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) printable.push_back(b);
    std::array<bool, 256> isPrintable{};
    for (int b : printable) isPrintable[static_cast<std::size_t>(b)] = true;
    std::uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
        const std::uint32_t cp = isPrintable[static_cast<std::size_t>(b)] ? static_cast<std::uint32_t>(b) : next++;
        std::string s;
        appendUtf8(s, cp);
        byteToUnicode_[static_cast<std::size_t>(b)] = s;
        unicodeToByte_[cp] = static_cast<unsigned char>(b);
    }

    const json::Value vocab = json::parse(readFile(vocabJsonPath));
    if (!vocab.isObject() || vocab.object.empty()) throw std::runtime_error("Gpt2Tokenizer: bad vocab.json");
    idToToken_.resize(vocab.object.size());
    for (const auto& [token, id] : vocab.object) {
        const auto i = static_cast<std::size_t>(id.number);
        if (!id.isNumber() || i >= idToToken_.size()) throw std::runtime_error("Gpt2Tokenizer: bad id for " + token);
        vocab_[token] = static_cast<int>(i);
        idToToken_[i] = token;
    }
    auto eot = vocab_.find("<|endoftext|>");
    if (eot != vocab_.end()) endOfText_ = eot->second;

    std::istringstream merges(readFile(mergesPath));
    std::string line;
    int rank = 0;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.rfind("#version", 0) == 0) continue;
        if (line.find(' ') == std::string::npos) throw std::runtime_error("Gpt2Tokenizer: bad merges line: " + line);
        ranks_.emplace(line, rank++);
    }
}

std::vector<int> Gpt2Tokenizer::encodeWord(const std::string& word) const {
    auto hit = cache_.find(word);
    if (hit != cache_.end()) return hit->second;
    std::vector<std::string> symbols;
    for (unsigned char b : word) symbols.push_back(byteToUnicode_[b]);
    while (symbols.size() > 1) {
        int best = std::numeric_limits<int>::max();
        std::size_t at = 0;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = ranks_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == std::numeric_limits<int>::max()) break;
        const std::string left = symbols[at], right = symbols[at + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            } else {
                merged.push_back(symbols[i]);
            }
        }
        symbols = std::move(merged);
    }
    std::vector<int> ids;
    ids.reserve(symbols.size());
    for (const auto& s : symbols) {
        auto it = vocab_.find(s);
        if (it == vocab_.end()) throw std::runtime_error("Gpt2Tokenizer: token missing from vocab");
        ids.push_back(it->second);
    }
    if (cache_.size() < 200000) cache_.emplace(word, ids);
    return ids;
}

std::vector<int> Gpt2Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const auto& w : splitWords(text)) {
        const auto part = encodeWord(w);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

std::string Gpt2Tokenizer::decode(const std::vector<int>& ids) const {
    std::string mapped;
    for (int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= idToToken_.size())
            throw std::out_of_range("Gpt2Tokenizer: id " + std::to_string(id) + " out of range");
        mapped += idToToken_[static_cast<std::size_t>(id)];
    }
    std::string out;
    for (std::size_t i = 0; i < mapped.size();) {
        std::size_t len = 1;
        const std::uint32_t cp = decodeUtf8(mapped, i, len);
        auto it = unicodeToByte_.find(cp);
        if (it != unicodeToByte_.end()) out += static_cast<char>(it->second);
        else out += mapped.substr(i, len);  // not produced by the byte mapping: keep as-is
        i += len;
    }
    return out;
}

// ---- Llama tokenizer ---------------------------------------------------------------------

LlamaTokenizer::LlamaTokenizer(const std::string& tokenizerJsonPath) {
    byteTokens_.fill(-1);
    const json::Value root = json::parse(readFile(tokenizerJsonPath));
    const json::Value& model = root.at("model");
    if (model.at("type").string != "BPE") throw std::runtime_error("LlamaTokenizer: only BPE tokenizer.json is supported");

    const json::Value& vocab = model.at("vocab");
    idToToken_.resize(vocab.object.size());
    for (const auto& [token, id] : vocab.object) {
        const auto i = static_cast<std::size_t>(id.number);
        if (i >= idToToken_.size()) idToToken_.resize(i + 1);
        idToToken_[i] = token;
        vocab_[token] = static_cast<int>(i);
    }
    int rank = 0;
    for (const auto& m : model.at("merges").array) {
        // Either "a b" strings or ["a", "b"] pairs, depending on the tokenizers version.
        if (m.isString()) ranks_.emplace(m.string, rank++);
        else if (m.isArray() && m.array.size() == 2) ranks_.emplace(m.array[0].string + " " + m.array[1].string, rank++);
        else throw std::runtime_error("LlamaTokenizer: bad merge entry");
    }
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        auto it = vocab_.find(name);
        if (it != vocab_.end()) byteTokens_[static_cast<std::size_t>(b)] = it->second;
    }
    if (const json::Value* added = root.find("added_tokens")) {
        for (const auto& a : added->array) {
            const int id = static_cast<int>(a.at("id").number);
            const std::string& content = a.at("content").string;
            specials_.emplace_back(content, id);
            if (static_cast<std::size_t>(id) >= idToToken_.size()) idToToken_.resize(static_cast<std::size_t>(id) + 1);
            idToToken_[static_cast<std::size_t>(id)] = content;
            vocab_[content] = id;
        }
    }
    // Longest first, so overlapping names match greedily.
    std::sort(specials_.begin(), specials_.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    auto findId = [&](const char* name) { auto it = vocab_.find(name); return it == vocab_.end() ? -1 : it->second; };
    bos_ = findId("<s>");
    eos_ = findId("</s>");
    unk_ = findId("<unk>");

    // Normalizer: the Llama/Mistral one is Sequence[Prepend("▁"), Replace(" ", "▁")].
    prepend_.clear();
    replaceSpaces_ = false;
    const json::Value* norm = root.find("normalizer");
    std::vector<const json::Value*> steps;
    if (norm && norm->isObject()) {
        if (norm->at("type").string == "Sequence") for (const auto& s : norm->at("normalizers").array) steps.push_back(&s);
        else steps.push_back(norm);
    }
    for (const auto* s : steps) {
        const std::string& type = s->at("type").string;
        if (type == "Prepend") prepend_ = s->at("prepend").string;
        else if (type == "Replace" && s->at("pattern").find("String") && s->at("pattern").at("String").string == " ")
            replaceSpaces_ = true;
        else throw std::runtime_error("LlamaTokenizer: unsupported normalizer step '" + type + "'");
    }
    const json::Value* pre = root.find("pre_tokenizer");
    if (pre && pre->isObject())
        throw std::runtime_error("LlamaTokenizer: tokenizers with a pre_tokenizer (e.g. Llama 3) are not supported");
}

bool LlamaTokenizer::isSpecial(int id) const {
    for (const auto& s : specials_)
        if (s.second == id) return true;
    return false;
}

std::vector<int> LlamaTokenizer::encodeSegment(const std::string& text) const {
    std::vector<int> ids;
    if (text.empty()) return ids;
    std::string normalized = prepend_;
    for (char c : text) {
        if (replaceSpaces_ && c == ' ') normalized += "\xe2\x96\x81";
        else normalized += c;
    }
    // BPE over UTF-8 characters.
    std::vector<std::string> symbols;
    for (std::size_t i = 0; i < normalized.size();) {
        std::size_t len = 1;
        decodeUtf8(normalized, i, len);
        symbols.push_back(normalized.substr(i, len));
        i += len;
    }
    for (;;) {
        int best = std::numeric_limits<int>::max();
        std::size_t at = 0;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = ranks_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == std::numeric_limits<int>::max()) break;
        const std::string left = symbols[at], right = symbols[at + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            } else {
                merged.push_back(symbols[i]);
            }
        }
        symbols = std::move(merged);
    }
    for (const auto& s : symbols) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
            continue;
        }
        for (unsigned char b : s) {  // byte fallback
            const int id = byteTokens_[b];
            ids.push_back(id >= 0 ? id : unk_);
        }
    }
    return ids;
}

std::vector<int> LlamaTokenizer::encode(const std::string& text, bool addBos) const {
    std::vector<int> ids;
    if (addBos && bos_ >= 0) ids.push_back(bos_);
    std::size_t start = 0, i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto& [content, id] : specials_) {
            if (!content.empty() && text.compare(i, content.size(), content) == 0) {
                const auto part = encodeSegment(text.substr(start, i - start));
                ids.insert(ids.end(), part.begin(), part.end());
                ids.push_back(id);
                i += content.size();
                start = i;
                matched = true;
                break;
            }
        }
        if (!matched) ++i;
    }
    const auto part = encodeSegment(text.substr(start));
    ids.insert(ids.end(), part.begin(), part.end());
    return ids;
}

std::string LlamaTokenizer::decodePiece(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= idToToken_.size())
        throw std::out_of_range("LlamaTokenizer: id " + std::to_string(id) + " out of range");
    if (isSpecial(id)) return std::string();
    const std::string& tok = idToToken_[static_cast<std::size_t>(id)];
    if (tok.size() == 6 && tok.compare(0, 3, "<0x") == 0 && tok[5] == '>')  // byte token
        return std::string(1, static_cast<char>(std::stoi(tok.substr(3, 2), nullptr, 16)));
    std::string out;
    for (std::size_t i = 0; i < tok.size();) {
        if (tok.compare(i, 3, "\xe2\x96\x81") == 0) {
            out += ' ';
            i += 3;
        } else {
            out += tok[i++];
        }
    }
    return out;
}

std::string LlamaTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += decodePiece(id);
    if (!prepend_.empty() && !out.empty() && out[0] == ' ') out.erase(0, 1);  // undo the prepended "▁"
    return out;
}

// ---- Model -----------------------------------------------------------------------------------

namespace {
// Layers are built without parameters (SkipParameterInit), so shapes are
// checked against what the configuration implies.
void assign(Matrix& dst, Matrix src, std::size_t rows, std::size_t cols, const std::string& name) {
    if (src.rows() != rows || src.cols() != cols)
        throw std::runtime_error("loader: " + name + " is " + std::to_string(src.rows()) + "x" +
                                 std::to_string(src.cols()) + ", expected " + std::to_string(rows) + "x" +
                                 std::to_string(cols));
    dst = std::move(src);
}

template <typename T>
T& layerAs(Network& net, std::size_t i, const char* what) {
    auto* p = dynamic_cast<T*>(&net.layer(i));
    if (!p) throw std::logic_error(std::string("loadGPT2: unexpected layer layout (") + what + ")");
    return *p;
}
} // namespace

PretrainedModel loadGPT2(const std::string& directory) {
    const json::Value cfgJson = json::parse(readFile(directory + "/config.json"));
    const std::string type = cfgJson.find("model_type") ? cfgJson.at("model_type").string : "gpt2";
    if (type != "gpt2") throw std::runtime_error("loadGPT2: model_type is '" + type + "', expected gpt2");

    GPTConfig cfg;
    cfg.vocabSize = static_cast<std::size_t>(cfgJson.numberOr("vocab_size", 50257));
    cfg.contextLength = static_cast<std::size_t>(cfgJson.numberOr("n_positions", cfgJson.numberOr("n_ctx", 1024)));
    cfg.dim = static_cast<std::size_t>(cfgJson.numberOr("n_embd", 768));
    cfg.heads = static_cast<std::size_t>(cfgJson.numberOr("n_head", 12));
    cfg.layers = static_cast<std::size_t>(cfgJson.numberOr("n_layer", 12));
    const json::Value* inner = cfgJson.find("n_inner");
    if (inner && inner->isNumber() && static_cast<std::size_t>(inner->number) != 4 * cfg.dim)
        throw std::runtime_error("loadGPT2: n_inner other than 4 * n_embd is not supported");
    cfg.mlpRatio = 4;
    cfg.tieEmbeddings = true;
    cfg.dropout = 0.0f;
    if (std::fabs(cfgJson.numberOr("layer_norm_epsilon", 1e-5) - 1e-5) > 1e-9)
        throw std::runtime_error("loadGPT2: only layer_norm_epsilon = 1e-5 is supported");
    const json::Value* act = cfgJson.find("activation_function");
    if (act && act->isString() && act->string != "gelu_new" && act->string != "gelu_pytorch_tanh")
        throw std::runtime_error("loadGPT2: activation '" + act->string + "' is not supported (tanh GELU only)");

    PretrainedModel model;
    model.config = cfg;
    {
        SkipParameterInit skip;  // no random weights or gradient buffers: the checkpoint fills everything
        model.context = buildGPT(model.net, cfg);
    }

    const SafeTensors st(directory + "/model.safetensors");
    const std::string prefix = st.has("wte.weight") ? "" : (st.has("transformer.wte.weight") ? "transformer." : "");
    auto load = [&](const std::string& name) { return st.load(prefix + name); };

    Network& net = model.net;
    auto& emb = layerAs<TokenEmbedding>(net, 0, "embedding");
    assign(emb.tokenTable(), load("wte.weight"), cfg.vocabSize, cfg.dim, "wte.weight");
    assign(emb.positionTable(), load("wpe.weight"), cfg.contextLength, cfg.dim, "wpe.weight");
    for (std::size_t i = 0; i < cfg.layers; ++i) {
        const std::string p = "h." + std::to_string(i) + ".";
        Network& att = layerAs<Residual>(net, 1 + 2 * i, "attention block").inner();
        auto& ln1 = layerAs<LayerNorm>(att, 0, "ln_1");
        assign(ln1.gamma(), load(p + "ln_1.weight"), 1, cfg.dim, p + "ln_1.weight");
        assign(ln1.beta(), load(p + "ln_1.bias"), 1, cfg.dim, p + "ln_1.bias");
        auto& sa = layerAs<CausalSelfAttention>(att, 1, "attention");
        assign(sa.qkvWeight(), load(p + "attn.c_attn.weight"), cfg.dim, 3 * cfg.dim, p + "attn.c_attn.weight");
        assign(sa.qkvBias(), load(p + "attn.c_attn.bias"), 1, 3 * cfg.dim, p + "attn.c_attn.bias");
        assign(sa.outWeight(), load(p + "attn.c_proj.weight"), cfg.dim, cfg.dim, p + "attn.c_proj.weight");
        assign(sa.outBias(), load(p + "attn.c_proj.bias"), 1, cfg.dim, p + "attn.c_proj.bias");

        Network& mlp = layerAs<Residual>(net, 2 + 2 * i, "MLP block").inner();
        auto& ln2 = layerAs<LayerNorm>(mlp, 0, "ln_2");
        assign(ln2.gamma(), load(p + "ln_2.weight"), 1, cfg.dim, p + "ln_2.weight");
        assign(ln2.beta(), load(p + "ln_2.bias"), 1, cfg.dim, p + "ln_2.bias");
        auto& fc = layerAs<Dense>(mlp, 1, "c_fc");
        assign(fc.weights(), load(p + "mlp.c_fc.weight"), cfg.dim, 4 * cfg.dim, p + "mlp.c_fc.weight");
        assign(fc.bias(), load(p + "mlp.c_fc.bias"), 1, 4 * cfg.dim, p + "mlp.c_fc.bias");
        auto& proj = layerAs<Dense>(mlp, 3, "mlp c_proj");
        assign(proj.weights(), load(p + "mlp.c_proj.weight"), 4 * cfg.dim, cfg.dim, p + "mlp.c_proj.weight");
        assign(proj.bias(), load(p + "mlp.c_proj.bias"), 1, cfg.dim, p + "mlp.c_proj.bias");
    }
    auto& lnf = layerAs<LayerNorm>(net, 1 + 2 * cfg.layers, "ln_f");
    assign(lnf.gamma(), load("ln_f.weight"), 1, cfg.dim, "ln_f.weight");
    assign(lnf.beta(), load("ln_f.bias"), 1, cfg.dim, "ln_f.bias");
    net.releaseGradients();
    return model;
}

namespace {
// Tensor access across a single-file or sharded safetensors checkpoint.
class CheckpointFiles {
public:
    explicit CheckpointFiles(const std::string& dir) : dir_(dir) {
        std::ifstream index(dir + "/model.safetensors.index.json");
        if (index) {
            const json::Value root = json::parse(readFile(dir + "/model.safetensors.index.json"));
            for (const auto& [name, file] : root.at("weight_map").object) fileOf_[name] = file.string;
        }
    }
    Matrix load(const std::string& name) {
        const std::string file = fileOf_.count(name) ? fileOf_.at(name) : "model.safetensors";
        auto it = open_.find(file);
        if (it == open_.end()) it = open_.emplace(file, std::make_unique<SafeTensors>(dir_ + "/" + file)).first;
        return it->second->load(name);
    }

private:
    std::string dir_;
    std::unordered_map<std::string, std::string> fileOf_;
    std::unordered_map<std::string, std::unique_ptr<SafeTensors>> open_;
};

Matrix stackRows(const std::vector<const Matrix*>& parts) {
    std::size_t rows = 0;
    const std::size_t cols = parts.front()->cols();
    for (const Matrix* m : parts) {
        if (m->cols() != cols) throw std::runtime_error("loadLlama: projections disagree in width");
        rows += m->rows();
    }
    FloatVector data;
    data.reserve(rows * cols);
    for (const Matrix* m : parts) data.insert(data.end(), m->values().begin(), m->values().end());
    return Matrix(rows, cols, std::move(data));
}

void expectShape(const Matrix& m, std::size_t rows, std::size_t cols, const std::string& name) {
    if (m.rows() != rows || m.cols() != cols)
        throw std::runtime_error("loadLlama: " + name + " is " + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) +
                                 ", expected " + std::to_string(rows) + "x" + std::to_string(cols));
}

// out x in (PyTorch Linear) -> layer weights, quantized or as float in x out.
void setDense(Dense& d, const Matrix& outIn, bool int8) {
    if (int8) {
        d.setQuantized(QuantizedWeights::fromOutIn(outIn));
    } else {
        d.weights() = outIn.transpose();
        d.bias() = Matrix(1, d.outputSize());
    }
}
} // namespace

PretrainedModel loadLlama(const std::string& directory, bool int8) {
    const json::Value c = json::parse(readFile(directory + "/config.json"));
    const std::string type = c.find("model_type") ? c.at("model_type").string : "llama";
    if (type != "llama" && type != "mistral")
        throw std::runtime_error("loadLlama: model_type is '" + type + "', expected llama or mistral");
    if (const json::Value* act = c.find("hidden_act"); act && act->string != "silu")
        throw std::runtime_error("loadLlama: activation '" + act->string + "' is not supported (silu only)");
    if (const json::Value* rs = c.find("rope_scaling"); rs && rs->type != json::Value::Type::Null)
        throw std::runtime_error("loadLlama: rope_scaling is not supported");
    if (const json::Value* ab = c.find("attention_bias"); ab && ab->boolean)
        throw std::runtime_error("loadLlama: attention biases are not supported");
    if (const json::Value* sw = c.find("sliding_window"); sw && sw->isNumber())
        throw std::runtime_error("loadLlama: sliding-window attention is not supported");

    GPTConfig cfg;
    cfg.modern();
    cfg.vocabSize = static_cast<std::size_t>(c.at("vocab_size").number);
    cfg.dim = static_cast<std::size_t>(c.at("hidden_size").number);
    cfg.heads = static_cast<std::size_t>(c.at("num_attention_heads").number);
    cfg.kvHeads = static_cast<std::size_t>(c.numberOr("num_key_value_heads", static_cast<double>(cfg.heads)));
    cfg.layers = static_cast<std::size_t>(c.at("num_hidden_layers").number);
    cfg.mlpHiddenSize = static_cast<std::size_t>(c.at("intermediate_size").number);
    cfg.contextLength = static_cast<std::size_t>(c.numberOr("max_position_embeddings", 2048));
    cfg.normEpsilon = static_cast<float>(c.numberOr("rms_norm_eps", 1e-6));
    cfg.ropeBase = static_cast<float>(c.numberOr("rope_theta", 10000.0));
    cfg.tieEmbeddings = c.find("tie_word_embeddings") && c.at("tie_word_embeddings").boolean;
    cfg.dropout = 0.0f;
    if (const json::Value* hd = c.find("head_dim"); hd && hd->isNumber() && static_cast<std::size_t>(hd->number) * cfg.heads != cfg.dim)
        throw std::runtime_error("loadLlama: head_dim * heads must equal hidden_size");

    PretrainedModel model;
    model.config = cfg;
    {
        SkipParameterInit skip;
        model.context = buildGPT(model.net, cfg);
    }
    CheckpointFiles files(directory);
    const std::size_t D = cfg.dim, kv = cfg.kvHeads * (cfg.dim / cfg.heads), H = cfg.mlpHidden(), V = cfg.vocabSize;
    Network& net = model.net;

    auto& emb = layerAs<TokenEmbedding>(net, 0, "embedding");
    Matrix table = files.load("model.embed_tokens.weight");
    expectShape(table, V, D, "embed_tokens");
    emb.tokenTable() = std::move(table);

    for (std::size_t i = 0; i < cfg.layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        Network& att = layerAs<Residual>(net, 1 + 2 * i, "attention block").inner();
        Matrix g = files.load(p + "input_layernorm.weight");
        expectShape(g, 1, D, p + "input_layernorm");
        layerAs<RMSNorm>(att, 0, "input_layernorm").gamma() = std::move(g);
        auto& sa = layerAs<CausalSelfAttention>(att, 1, "attention");
        {
            const Matrix q = files.load(p + "self_attn.q_proj.weight"), k = files.load(p + "self_attn.k_proj.weight"),
                         v = files.load(p + "self_attn.v_proj.weight"), o = files.load(p + "self_attn.o_proj.weight");
            expectShape(q, D, D, p + "q_proj");
            expectShape(k, kv, D, p + "k_proj");
            expectShape(v, kv, D, p + "v_proj");
            expectShape(o, D, D, p + "o_proj");
            const Matrix qkv = stackRows({&q, &k, &v});  // (D + 2kv) x D, out x in
            if (int8) sa.setQuantized(QuantizedWeights::fromOutIn(qkv), QuantizedWeights::fromOutIn(o));
            else sa.setWeights(qkv.transpose(), o.transpose());
        }

        Network& mlp = layerAs<Residual>(net, 2 + 2 * i, "MLP block").inner();
        Matrix g2 = files.load(p + "post_attention_layernorm.weight");
        expectShape(g2, 1, D, p + "post_attention_layernorm");
        layerAs<RMSNorm>(mlp, 0, "post_attention_layernorm").gamma() = std::move(g2);
        {
            const Matrix gate = files.load(p + "mlp.gate_proj.weight"), up = files.load(p + "mlp.up_proj.weight");
            expectShape(gate, H, D, p + "gate_proj");
            expectShape(up, H, D, p + "up_proj");
            setDense(layerAs<Dense>(mlp, 1, "gate/up"), stackRows({&gate, &up}), int8);  // SwiGLU input [gate | up]
        }
        {
            const Matrix down = files.load(p + "mlp.down_proj.weight");
            expectShape(down, D, H, p + "down_proj");
            setDense(layerAs<Dense>(mlp, 3, "down_proj"), down, int8);
        }
    }
    Matrix gf = files.load("model.norm.weight");
    expectShape(gf, 1, D, "model.norm");
    layerAs<RMSNorm>(net, 1 + 2 * cfg.layers, "final norm").gamma() = std::move(gf);
    if (cfg.tieEmbeddings) {
        if (int8) net.layer(2 + 2 * cfg.layers).quantize();
    } else {
        const Matrix head = files.load("lm_head.weight");
        expectShape(head, V, D, "lm_head");
        setDense(layerAs<Dense>(net, 2 + 2 * cfg.layers, "lm_head"), head, int8);
    }
    net.releaseGradients();
    return model;
}

} // namespace nc
