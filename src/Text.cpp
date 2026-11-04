#include "Text.h"
#include "Random.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace nc {

CharTokenizer::CharTokenizer(const std::string& text) {
    charToId_.fill(-1);
    std::array<bool, 256> seen{};
    for (unsigned char c : text) seen[c] = true;
    for (int c = 0; c < 256; ++c)
        if (seen[static_cast<std::size_t>(c)]) {
            charToId_[static_cast<std::size_t>(c)] = static_cast<int>(idToChar_.size());
            idToChar_.push_back(static_cast<char>(c));
        }
    if (idToChar_.empty()) throw std::invalid_argument("CharTokenizer: empty text");
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        const int id = charToId_[c];
        if (id < 0) {
            std::ostringstream msg;
            msg << "CharTokenizer: character '" << static_cast<char>(c) << "' (byte " << static_cast<int>(c)
                << ") is not in the vocabulary";
            throw std::invalid_argument(msg.str());
        }
        ids.push_back(id);
    }
    return ids;
}

std::string CharTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    out.reserve(ids.size());
    for (int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= idToChar_.size())
            throw std::out_of_range("CharTokenizer: id " + std::to_string(id) + " out of range");
        out.push_back(idToChar_[static_cast<std::size_t>(id)]);
    }
    return out;
}

// ---- BPE ------------------------------------------------------------------------

namespace {
enum class CharClass { Space, Newline, Letter, Digit, Other };

CharClass classify(unsigned char c) {
    if (c == ' ' || c == '\t') return CharClass::Space;
    if (c == '\n' || c == '\r') return CharClass::Newline;
    if (std::isalpha(c) || c >= 128) return CharClass::Letter;  // bytes >= 128: UTF-8 letters
    if (std::isdigit(c)) return CharClass::Digit;
    return CharClass::Other;
}

std::uint64_t pairKey(int a, int b) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
}

constexpr const char* kBpeMagic = "NEURACRAFT-BPE";
} // namespace

std::vector<std::string> splitWords(const std::string& text) {
    std::vector<std::string> words;
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
        const std::size_t start = i;
        CharClass c = classify(static_cast<unsigned char>(text[i]));
        if (c == CharClass::Space && i + 1 < n) {
            const CharClass next = classify(static_cast<unsigned char>(text[i + 1]));
            if (next == CharClass::Letter || next == CharClass::Digit || next == CharClass::Other) {
                ++i;  // a single leading space belongs to the word
                c = next;
            }
        }
        if (c == CharClass::Space || c == CharClass::Newline) {
            while (i < n && classify(static_cast<unsigned char>(text[i])) == c) ++i;
            // Like GPT-2, leave the last space of a run to lead the next word.
            if (c == CharClass::Space && i - start > 1 && i < n) {
                const CharClass next = classify(static_cast<unsigned char>(text[i]));
                if (next == CharClass::Letter || next == CharClass::Digit || next == CharClass::Other) --i;
            }
        } else {
            while (i < n && classify(static_cast<unsigned char>(text[i])) == c) ++i;
        }
        if (i == start) ++i;  // safety: always make progress
        words.push_back(text.substr(start, i - start));
    }
    return words;
}

BpeTokenizer::BpeTokenizer() {
    pieces_.reserve(256);
    for (int b = 0; b < 256; ++b) pieces_.push_back(std::string(1, static_cast<char>(b)));
}

void BpeTokenizer::addMerge(int a, int b) {
    rank_[pairKey(a, b)] = static_cast<int>(merges_.size());
    merges_.push_back({a, b});
    pieces_.push_back(pieces_[static_cast<std::size_t>(a)] + pieces_[static_cast<std::size_t>(b)]);
}

BpeTokenizer BpeTokenizer::train(const std::string& text, std::size_t vocabSize) {
    if (vocabSize < 256) throw std::invalid_argument("BpeTokenizer::train: vocabSize must be >= 256");
    BpeTokenizer tok;

    // Work on unique words weighted by frequency.
    std::unordered_map<std::string, std::size_t> counts;
    for (auto& w : splitWords(text)) ++counts[w];
    std::vector<std::vector<int>> words;
    std::vector<std::size_t> freq;
    words.reserve(counts.size());
    for (const auto& [w, c] : counts) {
        std::vector<int> ids;
        for (unsigned char ch : w) ids.push_back(ch);
        if (ids.size() < 2) continue;  // nothing to merge
        words.push_back(std::move(ids));
        freq.push_back(c);
    }

    std::unordered_map<std::uint64_t, std::size_t> pairCounts;
    while (tok.vocabSize() < vocabSize) {
        pairCounts.clear();
        for (std::size_t w = 0; w < words.size(); ++w)
            for (std::size_t i = 0; i + 1 < words[w].size(); ++i) pairCounts[pairKey(words[w][i], words[w][i + 1])] += freq[w];

        std::uint64_t best = 0;
        std::size_t bestCount = 0;
        for (const auto& [key, c] : pairCounts)
            if (c > bestCount || (c == bestCount && key < best)) {  // deterministic tie-break
                best = key;
                bestCount = c;
            }
        if (bestCount < 2) break;

        const int a = static_cast<int>(best >> 32), b = static_cast<int>(best & 0xffffffffu);
        const int merged = static_cast<int>(tok.vocabSize());
        tok.addMerge(a, b);
        for (auto& ids : words) {
            std::size_t out = 0;
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
                    ids[out++] = merged;
                    ++i;
                } else {
                    ids[out++] = ids[i];
                }
            }
            ids.resize(out);
        }
    }
    return tok;
}

std::vector<int> BpeTokenizer::encodeWord(const std::string& word) const {
    auto hit = cache_.find(word);
    if (hit != cache_.end()) return hit->second;
    std::vector<int> ids;
    ids.reserve(word.size());
    for (unsigned char ch : word) ids.push_back(ch);
    // Repeatedly apply the earliest-learned merge present in the word.
    for (;;) {
        int bestRank = std::numeric_limits<int>::max();
        for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
            auto it = rank_.find(pairKey(ids[i], ids[i + 1]));
            if (it != rank_.end() && it->second < bestRank) bestRank = it->second;
        }
        if (bestRank == std::numeric_limits<int>::max()) break;
        const auto [a, b] = merges_[static_cast<std::size_t>(bestRank)];
        const int merged = 256 + bestRank;
        std::size_t out = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
                ids[out++] = merged;
                ++i;
            } else {
                ids[out++] = ids[i];
            }
        }
        ids.resize(out);
    }
    if (cache_.size() < 100000) cache_.emplace(word, ids);
    return ids;
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    ids.reserve(text.size() / 2);
    for (const auto& w : splitWords(text)) {
        const auto part = encodeWord(w);
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size())
            throw std::out_of_range("BpeTokenizer: id " + std::to_string(id) + " out of range");
        out += pieces_[static_cast<std::size_t>(id)];
    }
    return out;
}

void BpeTokenizer::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("BpeTokenizer::save: cannot open " + path);
    out << kBpeMagic << " 1\n" << merges_.size() << '\n';
    for (const auto& [a, b] : merges_) out << a << ' ' << b << '\n';
    if (!out) throw std::runtime_error("BpeTokenizer::save: write failed for " + path);
}

BpeTokenizer BpeTokenizer::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("BpeTokenizer::load: cannot open " + path);
    std::string magic;
    int version = 0;
    std::size_t count = 0;
    if (!(in >> magic >> version >> count) || magic != kBpeMagic || version != 1)
        throw std::runtime_error("BpeTokenizer::load: " + path + " is not a NeuraCraft BPE file");
    BpeTokenizer tok;
    for (std::size_t i = 0; i < count; ++i) {
        int a = -1, b = -1;
        if (!(in >> a >> b)) throw std::runtime_error("BpeTokenizer::load: truncated file");
        const int limit = static_cast<int>(tok.vocabSize());
        if (a < 0 || b < 0 || a >= limit || b >= limit) throw std::runtime_error("BpeTokenizer::load: invalid merge");
        tok.addMerge(a, b);
    }
    return tok;
}

std::string loadText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("loadText: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

Dataset makeNextTokenDataset(const std::vector<int>& tokens, std::size_t seqLen, std::size_t stride) {
    if (seqLen == 0) throw std::invalid_argument("makeNextTokenDataset: seqLen must be > 0");
    if (stride == 0) stride = seqLen;
    if (tokens.size() < seqLen + 1)
        throw std::invalid_argument("makeNextTokenDataset: need at least seqLen + 1 tokens");
    const std::size_t count = (tokens.size() - seqLen - 1) / stride + 1;
    Dataset d{Matrix(count, seqLen), Matrix(count, seqLen)};
    for (std::size_t s = 0; s < count; ++s)
        for (std::size_t t = 0; t < seqLen; ++t) {
            d.x(s, t) = static_cast<float>(tokens[s * stride + t]);
            d.y(s, t) = static_cast<float>(tokens[s * stride + t + 1]);
        }
    return d;
}

int sampleToken(const float* raw, std::size_t vocab, const SamplingOptions& options, const std::vector<int>& history) {
    const float* logits = raw;
    std::vector<float> penalized;
    if (options.repetitionPenalty != 1.0f && !history.empty()) {
        penalized.assign(raw, raw + vocab);
        const std::size_t from = history.size() > options.repetitionWindow ? history.size() - options.repetitionWindow : 0;
        std::vector<bool> seen(vocab, false);
        for (std::size_t i = from; i < history.size(); ++i) {
            const int id = history[i];
            if (id < 0 || static_cast<std::size_t>(id) >= vocab || seen[static_cast<std::size_t>(id)]) continue;
            seen[static_cast<std::size_t>(id)] = true;
            float& v = penalized[static_cast<std::size_t>(id)];
            v = v > 0.0f ? v / options.repetitionPenalty : v * options.repetitionPenalty;
        }
        logits = penalized.data();
    }
    if (options.temperature <= 0.0f) return static_cast<int>(std::max_element(logits, logits + vocab) - logits);
    std::vector<std::size_t> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::size_t keep = vocab;
    auto byLogit = [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; };
    if (options.topK > 0 && options.topK < vocab) {
        keep = options.topK;
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep), order.end(), byLogit);
    } else if (options.topP < 1.0f) {
        std::sort(order.begin(), order.end(), byLogit);
    }
    float maxVal = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < keep; ++i) maxVal = std::max(maxVal, logits[order[i]]);
    std::vector<double> probs(keep);
    double total = 0.0;
    for (std::size_t i = 0; i < keep; ++i) total += probs[i] = std::exp((logits[order[i]] - maxVal) / options.temperature);
    if (options.topP < 1.0f) {
        // `order` is sorted here: keep the most likely tokens until their mass reaches topP.
        double mass = 0.0;
        std::size_t n = 0;
        while (n < keep && (n == 0 || mass < options.topP * total)) mass += probs[n++];
        probs.resize(n);
    }
    std::discrete_distribution<std::size_t> pick(probs.begin(), probs.end());
    return static_cast<int>(order[pick(rng())]);
}

namespace {
Matrix tokensRow(const std::vector<int>& tokens, std::size_t from) {
    Matrix x(1, tokens.size() - from);
    for (std::size_t t = from; t < tokens.size(); ++t) x(0, t - from) = static_cast<float>(tokens[t]);
    return x;
}

struct IncrementalScope {
    SequenceContext& ctx;
    explicit IncrementalScope(SequenceContext& c) : ctx(c) {
        ctx.incremental = true;
        ctx.offset = 0;
    }
    ~IncrementalScope() {
        ctx.incremental = false;
        ctx.offset = 0;
        ctx.window = 0;
    }
};
} // namespace

namespace {
bool isStop(int token, const SamplingOptions& o) {
    return token == o.stopToken || std::find(o.stopTokens.begin(), o.stopTokens.end(), token) != o.stopTokens.end();
}
} // namespace

std::vector<int> generate(Network& net, const std::vector<int>& prompt, std::size_t newTokens,
                          std::size_t contextLength, const SamplingOptions& options,
                          const std::shared_ptr<SequenceContext>& kvCache) {
    if (prompt.empty()) throw std::invalid_argument("generate: prompt must contain at least one token");
    if (contextLength == 0) throw std::invalid_argument("generate: contextLength must be > 0");
    if (newTokens == 0) return {};
    if (kvCache) {
        Generator g(net, kvCache, contextLength);
        return g.generate(prompt, newTokens, options);
    }
    std::vector<int> tokens = prompt, produced;
    produced.reserve(newTokens);
    for (std::size_t step = 0; step < newTokens; ++step) {
        const std::size_t len = std::min(contextLength, tokens.size());
        const Matrix logits = net.predict(tokensRow(tokens, tokens.size() - len));
        const int next = sampleToken(logits.data() + (logits.rows() - 1) * logits.cols(), logits.cols(), options, tokens);
        tokens.push_back(next);
        produced.push_back(next);
        if (options.onToken) options.onToken(next);
        if (isStop(next, options)) break;
    }
    return produced;
}

Generator::Generator(Network& net, SequenceContextPtr context, std::size_t contextLength)
    : net_(net), context_(std::move(context)), contextLength_(contextLength) {
    if (!context_) throw std::invalid_argument("Generator: a SequenceContext (from buildGPT) is required");
    if (contextLength_ == 0) throw std::invalid_argument("Generator: contextLength must be > 0");
}

void Generator::reset() {
    tokens_.clear();
    fed_ = start_ = offset_ = 0;
}

std::vector<float> Generator::feed(std::size_t from) {
    context_->offset = offset_;
    const Matrix logits = net_.predict(tokensRow(tokens_, from));
    offset_ += tokens_.size() - from;
    fed_ = tokens_.size();
    return std::vector<float>(logits.data() + (logits.rows() - 1) * logits.cols(), logits.data() + logits.size());
}

std::vector<int> Generator::generate(const std::vector<int>& input, std::size_t maxTokens, const SamplingOptions& options) {
    tokens_.insert(tokens_.end(), input.begin(), input.end());
    if (tokens_.size() == fed_) throw std::invalid_argument("Generator: nothing new to feed");
    std::vector<int> produced;
    if (maxTokens == 0) return produced;

    IncrementalScope scope(*context_);
    // Rotary models slide the cache by one token per step (exactly like the
    // uncached sliding window); learned positions need a rebuild when full.
    const bool sliding = context_->slidingCacheSupported;
    context_->window = sliding ? contextLength_ : 0;
    auto rebuild = [&](std::size_t keep) {  // restart the cache from the last `keep` tokens
        start_ = tokens_.size() - std::min(keep, tokens_.size());
        offset_ = 0;
        return feed(start_);
    };
    std::vector<float> last;
    if (fed_ == 0) last = rebuild(contextLength_);
    else if (!sliding && offset_ + (tokens_.size() - fed_) > contextLength_) last = rebuild(std::max<std::size_t>(1, contextLength_ / 2));
    else last = feed(fed_);
    for (std::size_t step = 0; step < maxTokens; ++step) {
        const int next = sampleToken(last.data(), last.size(), options, tokens_);
        tokens_.push_back(next);
        produced.push_back(next);
        if (options.onToken) options.onToken(next);
        if (step + 1 == maxTokens || isStop(next, options)) break;
        if (!sliding && offset_ >= contextLength_) last = rebuild(std::max<std::size_t>(1, contextLength_ / 2));
        else last = feed(tokens_.size() - 1);
    }
    return produced;
}

} // namespace nc
