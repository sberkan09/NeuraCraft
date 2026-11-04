#include "Pretrained.h"
#include "Activations.h"
#include "Dense.h"
#include "Json.h"
#include "SafeTensors.h"

#include <cmath>
#include <fstream>
#include <limits>
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

} // namespace nc
