#ifndef NEURACRAFT_ONNX_OPS_H
#define NEURACRAFT_ONNX_OPS_H

// Internal to the ONNX executor: graph nodes and the operator table.

#include "Onnx.h"

#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace nc::onnx {

struct Graph;

struct Attribute {
    std::string name;
    bool has = false;
    float f = 0.0f;
    std::int64_t i = 0;
    std::string s;
    std::shared_ptr<Tensor> t;
    std::vector<float> floats;
    std::vector<std::int64_t> ints;
    std::vector<std::string> strings;
    std::shared_ptr<Graph> g;  // a subgraph (If branches, Loop / Scan bodies)
};

struct Node {
    std::string op, name, domain;
    std::vector<std::string> inputs, outputs;
    std::map<std::string, Attribute> attrs;

    // The operator table key: "Op" in the default domain, "domain::Op" elsewhere (com.microsoft::MatMulNBits).
    std::string key() const { return domain.empty() || domain == "ai.onnx" ? op : domain + "::" + op; }

    const Attribute* attr(const std::string& n) const {
        auto it = attrs.find(n);
        return it == attrs.end() ? nullptr : &it->second;
    }
    std::int64_t i(const std::string& n, std::int64_t def) const {
        const Attribute* a = attr(n);
        return a ? a->i : def;
    }
    float f(const std::string& n, float def) const {
        const Attribute* a = attr(n);
        return a ? a->f : def;
    }
    std::string s(const std::string& n, const std::string& def) const {
        const Attribute* a = attr(n);
        return a ? a->s : def;
    }
    std::vector<std::int64_t> ints(const std::string& n) const {
        const Attribute* a = attr(n);
        return a ? a->ints : std::vector<std::int64_t>{};
    }
};

// A graph: the model's, or a subgraph of an If / Loop / Scan, whose nodes may also read values of the
// graphs around it ("outer" lists those).
struct Graph {
    std::vector<Node> nodes;
    std::unordered_map<std::string, std::shared_ptr<const Tensor>> initializers;
    std::vector<ValueInfo> inputs, outputs;  // as declared (inputs may include initializers)
    std::unordered_map<std::string, std::size_t> lastUse;  // value -> index of the last node reading it
    std::set<std::string> outer;
};

// Inputs (nullptr for a missing optional one) -> outputs.
using OpFn = std::function<std::vector<Tensor>(const Node&, const std::vector<const Tensor*>&, int opset)>;
const std::map<std::string, OpFn>& operatorTable();

bool isFloatType(int elemType);

} // namespace nc::onnx

#endif // NEURACRAFT_ONNX_OPS_H
