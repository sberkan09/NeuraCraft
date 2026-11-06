#include "Onnx.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "OnnxOps.h"

namespace nc::onnx {

// ---------------------------------------------------------------------------------------------
// Tensor

std::size_t Tensor::size() const {
    std::size_t n = 1;
    for (std::int64_t d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

Tensor Tensor::floats(std::vector<std::int64_t> shape, std::vector<float> values) {
    Tensor t;
    t.kind = Kind::Float;
    t.elemType = 1;
    t.shape = std::move(shape);
    t.f = std::move(values);
    if (t.f.size() != t.size()) throw std::invalid_argument("onnx::Tensor::floats: shape and value count disagree");
    return t;
}

Tensor Tensor::ints(std::vector<std::int64_t> shape, std::vector<std::int64_t> values, int elemType) {
    Tensor t;
    t.kind = Kind::Int;
    t.elemType = elemType;
    t.shape = std::move(shape);
    t.i = std::move(values);
    if (t.i.size() != t.size()) throw std::invalid_argument("onnx::Tensor::ints: shape and value count disagree");
    return t;
}

bool isFloatType(int t) { return t == 1 || t == 10 || t == 11 || t == 16; }

// ---------------------------------------------------------------------------------------------
// Protobuf wire format (what ONNX files are): fields are (number, wire type) keys followed by a
// varint, a fixed 32 / 64-bit value or a length-delimited byte string (strings, bytes, nested
// messages, packed repeated numbers).

namespace {

struct Reader {
    const std::uint8_t* p;
    const std::uint8_t* end;

    bool done() const { return p >= end; }
    std::uint64_t varint() {
        std::uint64_t v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p >= end) throw std::runtime_error("onnx: truncated varint");
            const std::uint8_t b = *p++;
            v |= static_cast<std::uint64_t>(b & 0x7f) << shift;
            if (!(b & 0x80)) return v;
        }
        throw std::runtime_error("onnx: bad varint");
    }
    std::uint32_t fixed32() {
        if (end - p < 4) throw std::runtime_error("onnx: truncated fixed32");
        std::uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    std::uint64_t fixed64() {
        if (end - p < 8) throw std::runtime_error("onnx: truncated fixed64");
        std::uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    Reader bytes() {
        const std::uint64_t n = varint();
        if (static_cast<std::uint64_t>(end - p) < n) throw std::runtime_error("onnx: truncated field");
        Reader r{p, p + n};
        p += n;
        return r;
    }
    std::string string() {
        const Reader r = bytes();
        return std::string(reinterpret_cast<const char*>(r.p), static_cast<std::size_t>(r.end - r.p));
    }
    void skip(int wire) {
        switch (wire) {
        case 0: varint(); break;
        case 1: fixed64(); break;
        case 2: bytes(); break;
        case 5: fixed32(); break;
        default: throw std::runtime_error("onnx: unsupported wire type " + std::to_string(wire));
        }
    }
    // Repeated numbers: packed (one length-delimited run) or one key per value.
    template <typename F>
    void repeated(int wire, F&& one) {
        if (wire == 2) {
            Reader r = bytes();
            while (!r.done()) one(r);
        } else {
            one(*this);
        }
    }
};

float halfToFloat(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {  // subnormal
            int e = -1;
            std::uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while (!(m & 0x400));
            bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
float bf16ToFloat(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

std::size_t elemBytes(int type) {
    switch (type) {
    case 1: case 6: case 12: return 4;
    case 2: case 3: case 9: return 1;
    case 4: case 5: case 10: case 16: return 2;
    case 7: case 11: case 13: return 8;
    default: throw std::runtime_error("onnx: unsupported tensor type " + std::to_string(type));
    }
}

// Raw little-endian values of `type` -> the tensor's storage.
void fromRaw(Tensor& t, const std::uint8_t* data, std::size_t bytes) {
    const std::size_t n = t.size(), w = elemBytes(t.elemType);
    if (bytes != n * w)
        throw std::runtime_error("onnx: tensor data has " + std::to_string(bytes) + " bytes, expected " + std::to_string(n * w));
    if (isFloatType(t.elemType)) {
        t.kind = Tensor::Kind::Float;
        t.f.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint8_t* q = data + k * w;
            switch (t.elemType) {
            case 1: std::memcpy(&t.f[k], q, 4); break;
            case 10: { std::uint16_t h; std::memcpy(&h, q, 2); t.f[k] = halfToFloat(h); break; }
            case 16: { std::uint16_t h; std::memcpy(&h, q, 2); t.f[k] = bf16ToFloat(h); break; }
            case 11: { double d; std::memcpy(&d, q, 8); t.f[k] = static_cast<float>(d); break; }
            }
        }
    } else {
        t.kind = Tensor::Kind::Int;
        t.i.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const std::uint8_t* q = data + k * w;
            switch (t.elemType) {
            case 2: case 9: t.i[k] = *q; break;
            case 3: t.i[k] = static_cast<std::int8_t>(*q); break;
            case 4: { std::uint16_t v; std::memcpy(&v, q, 2); t.i[k] = v; break; }
            case 5: { std::int16_t v; std::memcpy(&v, q, 2); t.i[k] = v; break; }
            case 6: { std::int32_t v; std::memcpy(&v, q, 4); t.i[k] = v; break; }
            case 12: { std::uint32_t v; std::memcpy(&v, q, 4); t.i[k] = v; break; }
            case 7: case 13: { std::int64_t v; std::memcpy(&v, q, 8); t.i[k] = v; break; }
            }
        }
    }
}

std::string readFileBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("onnx: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

Tensor parseTensor(Reader r, const std::string& baseDir, std::string* nameOut = nullptr) {
    Tensor t;
    std::string name;
    const std::uint8_t* raw = nullptr;
    std::size_t rawBytes = 0;
    std::vector<float> floatData;
    std::vector<double> doubleData;
    std::vector<std::int64_t> intData;  // int32_data (also fp16 / bool / int8 ... as numbers) and int64_data
    bool external = false;
    std::map<std::string, std::string> ext;
    while (!r.done()) {
        const std::uint64_t key = r.varint();
        const int field = static_cast<int>(key >> 3), wire = static_cast<int>(key & 7);
        switch (field) {
        case 1: r.repeated(wire, [&](Reader& x) { t.shape.push_back(static_cast<std::int64_t>(x.varint())); }); break;
        case 2: t.elemType = static_cast<int>(r.varint()); break;
        case 4: r.repeated(wire, [&](Reader& x) { const std::uint32_t b = x.fixed32(); float f; std::memcpy(&f, &b, 4); floatData.push_back(f); }); break;
        case 5: r.repeated(wire, [&](Reader& x) { intData.push_back(static_cast<std::int32_t>(x.varint())); }); break;
        case 7: r.repeated(wire, [&](Reader& x) { intData.push_back(static_cast<std::int64_t>(x.varint())); }); break;
        case 8: name = r.string(); break;
        case 9: { const Reader b = r.bytes(); raw = b.p; rawBytes = static_cast<std::size_t>(b.end - b.p); break; }
        case 10: r.repeated(wire, [&](Reader& x) { const std::uint64_t b = x.fixed64(); double d; std::memcpy(&d, &b, 8); doubleData.push_back(d); }); break;
        case 11: r.repeated(wire, [&](Reader& x) { intData.push_back(static_cast<std::int64_t>(x.varint())); }); break;
        case 13: {
            Reader e = r.bytes();
            std::string k, v;
            while (!e.done()) {
                const std::uint64_t kk = e.varint();
                if ((kk >> 3) == 1) k = e.string();
                else if ((kk >> 3) == 2) v = e.string();
                else e.skip(static_cast<int>(kk & 7));
            }
            ext[k] = v;
            break;
        }
        case 14: external = r.varint() == 1; break;
        default: r.skip(wire);
        }
    }
    if (nameOut) *nameOut = name;
    if (t.elemType == 8) throw std::runtime_error("onnx: string tensors are not supported (" + name + ")");
    if (external) {
        const std::string file = (std::filesystem::path(baseDir) / ext["location"]).string();
        std::ifstream in(file, std::ios::binary);
        if (!in) throw std::runtime_error("onnx: cannot open external data " + file + " for " + name);
        const std::size_t offset = ext.count("offset") ? std::stoull(ext["offset"]) : 0;
        std::size_t length = ext.count("length") ? std::stoull(ext["length"]) : t.size() * elemBytes(t.elemType);
        std::vector<std::uint8_t> buf(length);
        in.seekg(static_cast<std::streamoff>(offset));
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(length));
        if (static_cast<std::size_t>(in.gcount()) != length) throw std::runtime_error("onnx: short external data for " + name);
        fromRaw(t, buf.data(), length);
        return t;
    }
    if (raw) {
        fromRaw(t, raw, rawBytes);
        return t;
    }
    const std::size_t n = t.size();
    if (isFloatType(t.elemType)) {
        t.kind = Tensor::Kind::Float;
        if (t.elemType == 1) t.f = std::move(floatData);
        else if (t.elemType == 11) t.f.assign(doubleData.begin(), doubleData.end());
        else {  // fp16 / bf16 stored as uint16 in int32_data
            t.f.resize(intData.size());
            for (std::size_t k = 0; k < intData.size(); ++k)
                t.f[k] = t.elemType == 10 ? halfToFloat(static_cast<std::uint16_t>(intData[k])) : bf16ToFloat(static_cast<std::uint16_t>(intData[k]));
        }
        if (t.f.size() != n) throw std::runtime_error("onnx: tensor " + name + " has " + std::to_string(t.f.size()) + " values, expected " + std::to_string(n));
    } else {
        t.kind = Tensor::Kind::Int;
        t.i = std::move(intData);
        if (t.i.size() != n) throw std::runtime_error("onnx: tensor " + name + " has " + std::to_string(t.i.size()) + " values, expected " + std::to_string(n));
    }
    return t;
}

ValueInfo parseValueInfo(Reader r) {
    ValueInfo v;
    while (!r.done()) {
        const std::uint64_t key = r.varint();
        const int field = static_cast<int>(key >> 3), wire = static_cast<int>(key & 7);
        if (field == 1) v.name = r.string();
        else if (field == 2) {
            Reader type = r.bytes();
            while (!type.done()) {
                const std::uint64_t tk = type.varint();
                if ((tk >> 3) != 1) {
                    type.skip(static_cast<int>(tk & 7));
                    continue;
                }
                Reader tensor = type.bytes();
                while (!tensor.done()) {
                    const std::uint64_t k = tensor.varint();
                    if ((k >> 3) == 1) v.elemType = static_cast<int>(tensor.varint());
                    else if ((k >> 3) == 2) {
                        Reader shape = tensor.bytes();
                        while (!shape.done()) {
                            const std::uint64_t sk = shape.varint();
                            if ((sk >> 3) != 1) {
                                shape.skip(static_cast<int>(sk & 7));
                                continue;
                            }
                            Reader dim = shape.bytes();
                            std::int64_t value = -1;
                            std::string param;
                            while (!dim.done()) {
                                const std::uint64_t dk = dim.varint();
                                if ((dk >> 3) == 1) value = static_cast<std::int64_t>(dim.varint());
                                else if ((dk >> 3) == 2) param = dim.string();
                                else dim.skip(static_cast<int>(dk & 7));
                            }
                            v.shape.push_back(param.empty() ? value : -1);
                            v.dimNames.push_back(param);
                        }
                    } else {
                        tensor.skip(static_cast<int>(k & 7));
                    }
                }
            }
        } else {
            r.skip(wire);
        }
    }
    return v;
}

Attribute parseAttribute(Reader r, const std::string& baseDir) {
    Attribute a;
    while (!r.done()) {
        const std::uint64_t key = r.varint();
        const int field = static_cast<int>(key >> 3), wire = static_cast<int>(key & 7);
        switch (field) {
        case 1: a.name = r.string(); break;
        case 2: { const std::uint32_t b = r.fixed32(); std::memcpy(&a.f, &b, 4); a.has = true; break; }
        case 3: a.i = static_cast<std::int64_t>(r.varint()); a.has = true; break;
        case 4: a.s = r.string(); a.has = true; break;
        case 5: a.t = std::make_shared<Tensor>(parseTensor(r.bytes(), baseDir)); a.has = true; break;
        case 6: r.skip(wire); a.graph = true; break;
        case 7: r.repeated(wire, [&](Reader& x) { const std::uint32_t b = x.fixed32(); float f; std::memcpy(&f, &b, 4); a.floats.push_back(f); }); a.has = true; break;
        case 8: r.repeated(wire, [&](Reader& x) { a.ints.push_back(static_cast<std::int64_t>(x.varint())); }); a.has = true; break;
        case 9: a.strings.push_back(r.string()); a.has = true; break;
        default: r.skip(wire);
        }
    }
    return a;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Model

struct Model::Impl {
    std::vector<Node> nodes;
    std::unordered_map<std::string, std::shared_ptr<const Tensor>> initializers;
    std::vector<ValueInfo> inputs, outputs;
    int opset = 1;
    std::unordered_map<std::string, std::size_t> lastUse;  // value -> index of the last node reading it
};

Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

const std::vector<ValueInfo>& Model::inputs() const { return impl_->inputs; }
const std::vector<ValueInfo>& Model::outputs() const { return impl_->outputs; }
int Model::opset() const { return impl_->opset; }
std::size_t Model::nodeCount() const { return impl_->nodes.size(); }
std::map<std::string, std::size_t> Model::operatorCounts() const {
    std::map<std::string, std::size_t> c;
    for (const Node& n : impl_->nodes) ++c[n.op];
    return c;
}
std::size_t Model::parameterCount() const {
    std::size_t n = 0;
    for (const auto& [_, t] : impl_->initializers) n += t->size();
    return n;
}

Model Model::load(const std::string& path) {
    const std::string bytes = readFileBytes(path);
    const std::string baseDir = std::filesystem::path(path).parent_path().string();
    Model model;
    Impl& m = *model.impl_;
    Reader r{reinterpret_cast<const std::uint8_t*>(bytes.data()), reinterpret_cast<const std::uint8_t*>(bytes.data()) + bytes.size()};
    bool haveGraph = false;
    std::vector<ValueInfo> declaredInputs;
    while (!r.done()) {
        const std::uint64_t key = r.varint();
        const int field = static_cast<int>(key >> 3), wire = static_cast<int>(key & 7);
        if (field == 8) {  // opset_import
            Reader o = r.bytes();
            std::string domain;
            std::int64_t version = 0;
            while (!o.done()) {
                const std::uint64_t k = o.varint();
                if ((k >> 3) == 1) domain = o.string();
                else if ((k >> 3) == 2) version = static_cast<std::int64_t>(o.varint());
                else o.skip(static_cast<int>(k & 7));
            }
            if (domain.empty() || domain == "ai.onnx") m.opset = static_cast<int>(version);
        } else if (field == 7) {  // graph
            haveGraph = true;
            Reader g = r.bytes();
            while (!g.done()) {
                const std::uint64_t k = g.varint();
                const int f = static_cast<int>(k >> 3), w = static_cast<int>(k & 7);
                if (f == 1) {
                    Reader nr = g.bytes();
                    Node node;
                    while (!nr.done()) {
                        const std::uint64_t nk = nr.varint();
                        const int nf = static_cast<int>(nk >> 3), nw = static_cast<int>(nk & 7);
                        if (nf == 1) node.inputs.push_back(nr.string());
                        else if (nf == 2) node.outputs.push_back(nr.string());
                        else if (nf == 3) node.name = nr.string();
                        else if (nf == 4) node.op = nr.string();
                        else if (nf == 5) {
                            Attribute a = parseAttribute(nr.bytes(), baseDir);
                            node.attrs[a.name] = std::move(a);
                        } else if (nf == 7) node.domain = nr.string();
                        else nr.skip(nw);
                    }
                    m.nodes.push_back(std::move(node));
                } else if (f == 5) {
                    std::string name;
                    Tensor t = parseTensor(g.bytes(), baseDir, &name);
                    m.initializers[name] = std::make_shared<const Tensor>(std::move(t));
                } else if (f == 11) {
                    declaredInputs.push_back(parseValueInfo(g.bytes()));
                } else if (f == 12) {
                    m.outputs.push_back(parseValueInfo(g.bytes()));
                } else {
                    g.skip(w);
                }
            }
        } else {
            r.skip(wire);
        }
    }
    if (!haveGraph) throw std::runtime_error("onnx: " + path + " has no graph");
    for (ValueInfo& v : declaredInputs)
        if (!m.initializers.count(v.name)) m.inputs.push_back(std::move(v));
    // Every operator must be one we run; nodes are in topological order (the ONNX rule).
    const auto& ops = operatorTable();
    for (std::size_t k = 0; k < m.nodes.size(); ++k) {
        const Node& n = m.nodes[k];
        if (!n.domain.empty() && n.domain != "ai.onnx")
            throw std::runtime_error("onnx: " + path + ": operator " + n.domain + "::" + n.op + " (node '" + n.name + "') is not supported");
        if (!ops.count(n.op))
            throw std::runtime_error("onnx: " + path + ": operator " + n.op + " (node '" + n.name + "') is not supported");
        for (const auto& [name, a] : n.attrs)
            if (a.graph) throw std::runtime_error("onnx: " + path + ": control flow (" + n.op + " node '" + n.name + "') is not supported");
        for (const std::string& in : n.inputs)
            if (!in.empty()) m.lastUse[in] = k;
    }
    return model;
}

std::map<std::string, Tensor> Model::run(const std::map<std::string, Tensor>& feeds, const std::vector<std::string>& wanted) const {
    const Impl& m = *impl_;
    std::unordered_map<std::string, std::shared_ptr<const Tensor>> values = m.initializers;
    for (const ValueInfo& in : m.inputs) {
        auto it = feeds.find(in.name);
        if (it == feeds.end()) throw std::invalid_argument("onnx: missing input '" + in.name + "'");
        if (it->second.shape.size() != in.shape.size() && !in.shape.empty())
            throw std::invalid_argument("onnx: input '" + in.name + "' has rank " + std::to_string(it->second.shape.size()) + ", the model expects " +
                                        std::to_string(in.shape.size()));
        for (std::size_t d = 0; d < in.shape.size() && d < it->second.shape.size(); ++d)
            if (in.shape[d] > 0 && it->second.shape[d] != in.shape[d])
                throw std::invalid_argument("onnx: input '" + in.name + "' dimension " + std::to_string(d) + " is " + std::to_string(it->second.shape[d]) +
                                            ", the model expects " + std::to_string(in.shape[d]));
        values[in.name] = std::make_shared<const Tensor>(it->second);
    }
    std::set<std::string> keep;
    if (wanted.empty())
        for (const ValueInfo& o : m.outputs) keep.insert(o.name);
    else
        keep.insert(wanted.begin(), wanted.end());
    const auto& ops = operatorTable();
    // NEURACRAFT_ONNX_PROFILE=1: time per operator type, printed to stderr after the run.
    static const bool profile = [] {
        const char* e = std::getenv("NEURACRAFT_ONNX_PROFILE");
        return e && *e && std::string(e) != "0";
    }();
    std::map<std::string, std::pair<double, std::size_t>> spent;
    for (std::size_t k = 0; k < m.nodes.size(); ++k) {
        const Node& n = m.nodes[k];
        const auto started = std::chrono::steady_clock::now();
        std::vector<const Tensor*> ins;
        for (const std::string& name : n.inputs) {
            if (name.empty()) {
                ins.push_back(nullptr);
                continue;
            }
            auto it = values.find(name);
            if (it == values.end()) throw std::runtime_error("onnx: node '" + n.name + "' (" + n.op + ") reads '" + name + "', which nothing made");
            ins.push_back(it->second.get());
        }
        std::vector<Tensor> outs;
        try {
            outs = ops.at(n.op)(n, ins, m.opset);
        } catch (const std::exception& e) {
            throw std::runtime_error("onnx: " + n.op + " node '" + n.name + "': " + e.what());
        }
        for (std::size_t o = 0; o < n.outputs.size() && o < outs.size(); ++o)
            if (!n.outputs[o].empty()) values[n.outputs[o]] = std::make_shared<const Tensor>(std::move(outs[o]));
        if (profile) {
            auto& sp = spent[n.op];
            sp.first += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            ++sp.second;
        }
        // Free what no later node reads.
        for (const std::string& name : n.inputs) {
            if (name.empty() || keep.count(name) || m.initializers.count(name)) continue;
            auto lu = m.lastUse.find(name);
            if (lu != m.lastUse.end() && lu->second == k) values.erase(name);
        }
    }
    if (profile) {
        std::vector<std::pair<double, std::string>> rows;
        for (const auto& [op, sp] : spent) rows.push_back({sp.first, op + " x" + std::to_string(sp.second)});
        std::sort(rows.rbegin(), rows.rend());
        for (const auto& [sec, what] : rows) std::fprintf(stderr, "  onnx %-28s %8.3f s\n", what.c_str(), sec);
    }
    std::map<std::string, Tensor> out;
    for (const std::string& name : keep) {
        auto it = values.find(name);
        if (it == values.end()) throw std::runtime_error("onnx: output '" + name + "' was not computed");
        out[name] = *it->second;
    }
    return out;
}

Tensor loadTensorProto(const std::string& path, std::string* name) {
    const std::string bytes = readFileBytes(path);
    Reader r{reinterpret_cast<const std::uint8_t*>(bytes.data()), reinterpret_cast<const std::uint8_t*>(bytes.data()) + bytes.size()};
    return parseTensor(r, std::filesystem::path(path).parent_path().string(), name);
}

namespace {
void putVarint(std::string& out, std::uint64_t v) {
    while (v >= 0x80) {
        out += static_cast<char>((v & 0x7f) | 0x80);
        v >>= 7;
    }
    out += static_cast<char>(v);
}
void putKey(std::string& out, int field, int wire) { putVarint(out, static_cast<std::uint64_t>(field) << 3 | static_cast<std::uint64_t>(wire)); }
} // namespace

void saveTensorProto(const std::string& path, const Tensor& t, const std::string& name) {
    std::string out;
    for (std::int64_t d : t.shape) {
        putKey(out, 1, 0);
        putVarint(out, static_cast<std::uint64_t>(d));
    }
    const bool fl = t.kind == Tensor::Kind::Float;
    const int type = fl ? 1 : (t.elemType == 9 ? 9 : 7);
    putKey(out, 2, 0);
    putVarint(out, static_cast<std::uint64_t>(type));
    if (!name.empty()) {
        putKey(out, 8, 2);
        putVarint(out, name.size());
        out += name;
    }
    std::string raw;
    if (fl) raw.assign(reinterpret_cast<const char*>(t.f.data()), t.f.size() * 4);
    else if (type == 9)
        for (std::int64_t v : t.i) raw += static_cast<char>(v != 0);
    else raw.assign(reinterpret_cast<const char*>(t.i.data()), t.i.size() * 8);
    putKey(out, 9, 2);
    putVarint(out, raw.size());
    out += raw;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("onnx: cannot write " + path);
    f.write(out.data(), static_cast<std::streamsize>(out.size()));
}

std::vector<std::string> supportedOperators() {
    std::vector<std::string> names;
    for (const auto& [name, _] : operatorTable()) names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
}

} // namespace nc::onnx
