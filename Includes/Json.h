#ifndef NEURACRAFT_JSON_H
#define NEURACRAFT_JSON_H

#include <string>
#include <utility>
#include <vector>

namespace nc {
namespace json {

// Minimal JSON value, enough for model configs, safetensors headers and
// tokenizer vocabularies. Objects keep their key order.
struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::pair<std::string, Value>> object;

    bool isObject() const { return type == Type::Object; }
    bool isArray() const { return type == Type::Array; }
    bool isString() const { return type == Type::String; }
    bool isNumber() const { return type == Type::Number; }

    const Value* find(const std::string& key) const;  // nullptr if missing or not an object
    const Value& at(const std::string& key) const;    // throws std::runtime_error if missing
    double numberOr(const std::string& key, double fallback) const;
};

// Parses a complete JSON document (UTF-8). Throws std::runtime_error with the
// byte offset on malformed input.
Value parse(const std::string& text);

// Compact JSON text of a value (numbers that are integers print without a
// fraction; others with up to 15 significant digits), and of a string literal.
std::string dump(const Value& value);
std::string quote(const std::string& text);

// Builders for writing JSON.
Value makeString(const std::string& s);
Value makeNumber(double n);
Value makeBool(bool b);
Value makeArray(std::vector<Value> items = {});
Value makeObject(std::vector<std::pair<std::string, Value>> members = {});

} // namespace json
} // namespace nc

#endif // NEURACRAFT_JSON_H
