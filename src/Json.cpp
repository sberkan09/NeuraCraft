#include "Json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace nc {
namespace json {

const Value* Value::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& kv : object)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

const Value& Value::at(const std::string& key) const {
    const Value* v = find(key);
    if (!v) throw std::runtime_error("json: missing key '" + key + "'");
    return *v;
}

double Value::numberOr(const std::string& key, double fallback) const {
    const Value* v = find(key);
    return v && v->isNumber() ? v->number : fallback;
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Value document() {
        Value v = value(0);
        skipSpace();
        if (pos_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("json: " + what + " at byte " + std::to_string(pos_));
    }

    void skipSpace() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) ++pos_;
    }

    bool consume(const char* literal) {
        std::size_t n = 0;
        while (literal[n]) ++n;
        if (s_.compare(pos_, n, literal) != 0) return false;
        pos_ += n;
        return true;
    }

    Value value(int depth) {
        if (depth > 256) fail("nesting too deep");
        skipSpace();
        if (pos_ >= s_.size()) fail("unexpected end");
        Value v;
        const char c = s_[pos_];
        if (c == '{') {
            v.type = Value::Type::Object;
            ++pos_;
            skipSpace();
            if (pos_ < s_.size() && s_[pos_] == '}') {
                ++pos_;
                return v;
            }
            for (;;) {
                skipSpace();
                if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected a key");
                std::string key = string();
                skipSpace();
                if (pos_ >= s_.size() || s_[pos_] != ':') fail("expected ':'");
                ++pos_;
                v.object.emplace_back(std::move(key), value(depth + 1));
                skipSpace();
                if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return v; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type = Value::Type::Array;
            ++pos_;
            skipSpace();
            if (pos_ < s_.size() && s_[pos_] == ']') {
                ++pos_;
                return v;
            }
            for (;;) {
                v.array.push_back(value(depth + 1));
                skipSpace();
                if (pos_ < s_.size() && s_[pos_] == ',') { ++pos_; continue; }
                if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return v; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.type = Value::Type::String;
            v.string = string();
            return v;
        }
        if (consume("true")) { v.type = Value::Type::Bool; v.boolean = true; return v; }
        if (consume("false")) { v.type = Value::Type::Bool; return v; }
        if (consume("null")) return v;
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* begin = s_.c_str() + pos_;
            char* end = nullptr;
            v.number = std::strtod(begin, &end);
            if (end == begin) fail("bad number");
            pos_ += static_cast<std::size_t>(end - begin);
            v.type = Value::Type::Number;
            return v;
        }
        fail(std::string("unexpected character '") + c + "'");
    }

    static void appendUtf8(std::string& out, unsigned cp) {
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

    unsigned hex4() {
        if (pos_ + 4 > s_.size()) fail("short \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            const char h = s_[pos_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<unsigned>(h - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }

    std::string string() {
        ++pos_;  // opening quote
        std::string out;
        for (;;) {
            if (pos_ >= s_.size()) fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"') return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size()) fail("unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                unsigned cp = hex4();
                if (cp >= 0xD800 && cp <= 0xDBFF) {  // surrogate pair
                    if (s_.compare(pos_, 2, "\\u") != 0) fail("lone high surrogate");
                    pos_ += 2;
                    const unsigned low = hex4();
                    if (low < 0xDC00 || low > 0xDFFF) fail("bad low surrogate");
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                }
                appendUtf8(out, cp);
                break;
            }
            default:
                fail("bad escape");
            }
        }
    }

    const std::string& s_;
    std::size_t pos_ = 0;
};

} // namespace

Value parse(const std::string& text) { return Parser(text).document(); }

std::string quote(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out + "\"";
}

namespace {
void dumpTo(const Value& v, std::string& out) {
    switch (v.type) {
    case Value::Type::Null: out += "null"; break;
    case Value::Type::Bool: out += v.boolean ? "true" : "false"; break;
    case Value::Type::Number: {
        if (!std::isfinite(v.number)) {
            out += "null";
        } else if (v.number == std::floor(v.number) && std::fabs(v.number) < 1e15) {
            out += std::to_string(static_cast<long long>(v.number));
        } else {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.9g", v.number);
            out += buf;
        }
        break;
    }
    case Value::Type::String: out += quote(v.string); break;
    case Value::Type::Array:
        out += '[';
        for (std::size_t i = 0; i < v.array.size(); ++i) {
            if (i) out += ',';
            dumpTo(v.array[i], out);
        }
        out += ']';
        break;
    case Value::Type::Object:
        out += '{';
        for (std::size_t i = 0; i < v.object.size(); ++i) {
            if (i) out += ',';
            out += quote(v.object[i].first);
            out += ':';
            dumpTo(v.object[i].second, out);
        }
        out += '}';
        break;
    }
}
} // namespace

std::string dump(const Value& value) {
    std::string out;
    dumpTo(value, out);
    return out;
}

Value makeString(const std::string& s) {
    Value v;
    v.type = Value::Type::String;
    v.string = s;
    return v;
}
Value makeNumber(double n) {
    Value v;
    v.type = Value::Type::Number;
    v.number = n;
    return v;
}
Value makeBool(bool b) {
    Value v;
    v.type = Value::Type::Bool;
    v.boolean = b;
    return v;
}
Value makeArray(std::vector<Value> items) {
    Value v;
    v.type = Value::Type::Array;
    v.array = std::move(items);
    return v;
}
Value makeObject(std::vector<std::pair<std::string, Value>> members) {
    Value v;
    v.type = Value::Type::Object;
    v.object = std::move(members);
    return v;
}

} // namespace json
} // namespace nc
