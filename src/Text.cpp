#include "Text.h"
#include "Random.h"

#include <algorithm>
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

std::vector<int> generate(Network& net, const std::vector<int>& prompt, std::size_t newTokens,
                          std::size_t contextLength, const SamplingOptions& options) {
    if (prompt.empty()) throw std::invalid_argument("generate: prompt must contain at least one token");
    if (contextLength == 0) throw std::invalid_argument("generate: contextLength must be > 0");
    std::vector<int> tokens = prompt, produced;
    produced.reserve(newTokens);
    std::vector<float> probs;
    for (std::size_t step = 0; step < newTokens; ++step) {
        const std::size_t len = std::min(contextLength, tokens.size());
        Matrix x(1, len);
        for (std::size_t t = 0; t < len; ++t) x(0, t) = static_cast<float>(tokens[tokens.size() - len + t]);
        const Matrix logits = net.predict(x);
        const std::size_t vocab = logits.cols();
        const float* last = logits.data() + (logits.rows() - 1) * vocab;

        int next = 0;
        if (options.temperature <= 0.0f) {
            next = static_cast<int>(std::max_element(last, last + vocab) - last);
        } else {
            std::vector<std::size_t> order(vocab);
            std::iota(order.begin(), order.end(), 0);
            std::size_t keep = vocab;
            if (options.topK > 0 && options.topK < vocab) {
                keep = options.topK;
                std::partial_sort(order.begin(), order.begin() + static_cast<long>(keep), order.end(),
                                  [&](std::size_t a, std::size_t b) { return last[a] > last[b]; });
            }
            float maxVal = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < keep; ++i) maxVal = std::max(maxVal, last[order[i]]);
            probs.assign(keep, 0.0f);
            for (std::size_t i = 0; i < keep; ++i) probs[i] = std::exp((last[order[i]] - maxVal) / options.temperature);
            std::discrete_distribution<std::size_t> pick(probs.begin(), probs.end());
            next = static_cast<int>(order[pick(rng())]);
        }
        tokens.push_back(next);
        produced.push_back(next);
    }
    return produced;
}

} // namespace nc
