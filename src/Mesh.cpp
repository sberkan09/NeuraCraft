#include "Mesh.h"

#include "Animation.h"
#include "Transform.h"

#include "Json.h"
#include "Parallel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
thread_local std::size_t gMaxElements = std::size_t(1) << 24;

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string extension(const std::string& path) { return lower(std::filesystem::path(path).extension().string()); }

std::vector<std::uint8_t> readBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadMesh: cannot open " + path);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void writeBytes(const std::string& path, const void* data, std::size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("saveMesh: cannot open " + path);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("saveMesh: write failed for " + path);
}

// Byte (0..255) of a value scaled to 0..255; NaN gives 0. Integers from doubles go through these, so
// hostile files (NaN, 1e300) never reach an out-of-range conversion.
std::uint8_t toByte(double x) { return x >= 0.0 ? static_cast<std::uint8_t>(std::min(x + 0.5, 255.0)) : 0; }
bool inRange(double x, double lo, double hi) { return x >= lo && x <= hi; }  // false for NaN

void checkCount(std::size_t n, const std::string& what) {
    if (n > gMaxElements) throw std::runtime_error(what + ": more than " + std::to_string(gMaxElements) + " elements");
}

// Colour of the texture at (u, v), bilinear, wrapping.
std::array<float, 3> sampleTexture(const Image& t, float u, float v) {
    const float x = (u - std::floor(u)) * static_cast<float>(t.width) - 0.5f, y = (v - std::floor(v)) * static_cast<float>(t.height) - 0.5f;
    const long x0 = static_cast<long>(std::floor(x)), y0 = static_cast<long>(std::floor(y));
    const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    const long W = static_cast<long>(t.width), H = static_cast<long>(t.height);
    auto px = [&](long xx, long yy, std::size_t c) {
        xx = ((xx % W) + W) % W;
        yy = ((yy % H) + H) % H;
        const std::size_t i = (static_cast<std::size_t>(yy) * t.width + static_cast<std::size_t>(xx)) * t.channels;
        return static_cast<float>(t.channels >= 3 ? t.pixels[i + c] : t.pixels[i]);
    };
    std::array<float, 3> out{};
    for (std::size_t c = 0; c < 3; ++c)
        out[c] = (px(x0, y0, c) * (1 - fx) + px(x0 + 1, y0, c) * fx) * (1 - fy) + (px(x0, y0 + 1, c) * (1 - fx) + px(x0 + 1, y0 + 1, c) * fx) * fy;
    return out;
}

// Per-vertex colours: the vertex colours, else the texture at each vertex's uv, else grey.
std::vector<std::uint8_t> vertexColours(const Mesh& m) {
    if (!m.colors.empty()) return m.colors;
    std::vector<std::uint8_t> out(m.vertexCount() * 3, 200);
    if (m.texture.pixels.empty() || m.uvs.empty()) return out;
    for (std::size_t i = 0; i < m.vertexCount(); ++i) {
        const auto c = sampleTexture(m.texture, m.uvs[2 * i], m.uvs[2 * i + 1]);
        for (std::size_t k = 0; k < 3; ++k) out[3 * i + k] = toByte(c[k]);
    }
    return out;
}
} // namespace

std::size_t Mesh::bytes() const {
    return positions.size() * 4 + normals.size() * 4 + uvs.size() * 4 + colors.size() + indices.size() * 4 + texture.pixels.size() + glb.size() +
           (rig.rest.size() + rig.rootParent.size() + rig.inverseBind.size() + rig.weights.size() + rig.bindPositions.size() + rig.frames.size()) * 4 +
           rig.joints.size() * 2;
}

void setMaxMeshElements(std::size_t elements) { gMaxElements = elements ? elements : std::size_t(1) << 24; }
std::size_t maxMeshElements() { return gMaxElements; }

void checkMesh(const Mesh& m, const std::string& what) {
    if (m.positions.size() % 3) throw std::invalid_argument(what + ": positions are not x, y, z triples");
    const std::size_t n = m.vertexCount();
    if (!m.normals.empty() && m.normals.size() != 3 * n) throw std::invalid_argument(what + ": one normal per vertex expected");
    if (!m.uvs.empty() && m.uvs.size() != 2 * n) throw std::invalid_argument(what + ": one uv per vertex expected");
    if (!m.colors.empty() && m.colors.size() != 3 * n) throw std::invalid_argument(what + ": one colour per vertex expected");
    if (m.indices.size() % 3) throw std::invalid_argument(what + ": indices are not triangles");
    for (const std::vector<float>* v : {&m.positions, &m.normals, &m.uvs})
        for (float x : *v)
            if (!std::isfinite(x) || std::fabs(x) > 1e15f) throw std::invalid_argument(what + ": coordinates must be finite numbers");
    for (std::uint32_t i : m.indices)
        if (i >= n) throw std::invalid_argument(what + ": index out of range");
    if (!m.texture.pixels.empty() && m.texture.pixels.size() != m.texture.width * m.texture.height * m.texture.channels)
        throw std::invalid_argument(what + ": texture size mismatch");
}

void computeNormals(Mesh& m) {
    m.normals.assign(m.positions.size(), 0.0f);
    for (std::size_t t = 0; t < m.triangleCount(); ++t) {
        const std::uint32_t a = m.indices[3 * t], b = m.indices[3 * t + 1], c = m.indices[3 * t + 2];
        const float* p = m.positions.data();
        const float e1[3] = {p[3 * b] - p[3 * a], p[3 * b + 1] - p[3 * a + 1], p[3 * b + 2] - p[3 * a + 2]};
        const float e2[3] = {p[3 * c] - p[3 * a], p[3 * c + 1] - p[3 * a + 1], p[3 * c + 2] - p[3 * a + 2]};
        const float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        for (std::uint32_t v : {a, b, c})
            for (int k = 0; k < 3; ++k) m.normals[3 * v + k] += n[k];
    }
    for (std::size_t i = 0; i < m.vertexCount(); ++i) {
        float* n = m.normals.data() + 3 * i;
        const float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (l > 0) {
            n[0] /= l;
            n[1] /= l;
            n[2] /= l;
        } else n[2] = 1.0f;
    }
}

bool isMeshFile(const std::string& path) {
    const std::string e = extension(path);
    return e == ".glb" || e == ".obj" || e == ".ply";
}

// ---- glTF binary ----

namespace {
void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
std::uint32_t get32(const std::uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<std::uint32_t>(p[3]) << 24); }
} // namespace

std::vector<std::uint8_t> encodeGlb(const Mesh& m) {
    checkMesh(m, "encodeGlb");
    if (m.empty()) throw std::invalid_argument("encodeGlb: the mesh has no triangles");
    std::vector<std::uint8_t> bin;
    std::vector<json::Value> views, accessors;
    auto align = [&] {
        while (bin.size() % 4) bin.push_back(0);
    };
    // A buffer view of raw bytes; returns its index.
    auto view = [&](const void* data, std::size_t bytes, int target) {
        align();
        const std::size_t offset = bin.size();
        const auto* p = static_cast<const std::uint8_t*>(data);
        bin.insert(bin.end(), p, p + bytes);
        std::vector<std::pair<std::string, json::Value>> v{{"buffer", json::makeNumber(0)}, {"byteOffset", json::makeNumber(static_cast<double>(offset))},
                                                           {"byteLength", json::makeNumber(static_cast<double>(bytes))}};
        if (target) v.emplace_back("target", json::makeNumber(target));
        views.push_back(json::makeObject(std::move(v)));
        return views.size() - 1;
    };
    auto accessor = [&](std::size_t viewIndex, int componentType, std::size_t count, const char* type, bool normalized = false,
                        const std::vector<float>* minMax = nullptr) {
        std::vector<std::pair<std::string, json::Value>> a{{"bufferView", json::makeNumber(static_cast<double>(viewIndex))},
                                                           {"componentType", json::makeNumber(componentType)},
                                                           {"count", json::makeNumber(static_cast<double>(count))},
                                                           {"type", json::makeString(type)}};
        if (normalized) a.emplace_back("normalized", json::makeBool(true));
        if (minMax) {
            std::vector<json::Value> lo, hi;
            for (int k = 0; k < 3; ++k) {
                lo.push_back(json::makeNumber((*minMax)[k]));
                hi.push_back(json::makeNumber((*minMax)[3 + k]));
            }
            a.emplace_back("min", json::makeArray(std::move(lo)));
            a.emplace_back("max", json::makeArray(std::move(hi)));
        }
        accessors.push_back(json::makeObject(std::move(a)));
        return accessors.size() - 1;
    };
    constexpr int kArray = 34962, kElements = 34963, kFloat = 5126, kUByte = 5121, kUInt = 5125;
    const std::size_t n = m.vertexCount();
    std::vector<float> bounds{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                              -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    for (std::size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            bounds[k] = std::min(bounds[k], m.positions[3 * i + k]);
            bounds[3 + k] = std::max(bounds[3 + k], m.positions[3 * i + k]);
        }
    std::vector<std::pair<std::string, json::Value>> attributes;
    attributes.emplace_back("POSITION", json::makeNumber(static_cast<double>(accessor(view(m.positions.data(), m.positions.size() * 4, kArray), kFloat, n, "VEC3", false, &bounds))));
    if (!m.normals.empty())
        attributes.emplace_back("NORMAL", json::makeNumber(static_cast<double>(accessor(view(m.normals.data(), m.normals.size() * 4, kArray), kFloat, n, "VEC3"))));
    const bool textured = !m.texture.pixels.empty() && !m.uvs.empty();
    if (textured)
        attributes.emplace_back("TEXCOORD_0", json::makeNumber(static_cast<double>(accessor(view(m.uvs.data(), m.uvs.size() * 4, kArray), kFloat, n, "VEC2"))));
    if (!m.colors.empty()) {
        // COLOR_0 as normalized bytes needs 4-byte aligned elements: RGBA.
        std::vector<std::uint8_t> rgba(n * 4, 255);
        for (std::size_t i = 0; i < n; ++i) std::memcpy(&rgba[4 * i], &m.colors[3 * i], 3);
        attributes.emplace_back("COLOR_0", json::makeNumber(static_cast<double>(accessor(view(rgba.data(), rgba.size(), kArray), kUByte, n, "VEC4", true))));
    }
    const std::size_t indexAccessor = accessor(view(m.indices.data(), m.indices.size() * 4, kElements), kUInt, m.indices.size(), "SCALAR");

    std::vector<std::pair<std::string, json::Value>> pbr{{"metallicFactor", json::makeNumber(0)}, {"roughnessFactor", json::makeNumber(1)}};
    std::vector<std::pair<std::string, json::Value>> root{{"asset", json::makeObject({{"version", json::makeString("2.0")}, {"generator", json::makeString("NeuraCraft")}})},
                                                          {"scene", json::makeNumber(0)},
                                                          {"scenes", json::makeArray({json::makeObject({{"nodes", json::makeArray({json::makeNumber(0)})}})})},
                                                          {"nodes", json::makeArray({json::makeObject({{"mesh", json::makeNumber(0)}})})}};
    if (textured) {
        Image tex = m.texture;
        const std::vector<std::uint8_t> png = encodePng(tex);
        const std::size_t imageView = view(png.data(), png.size(), 0);
        root.emplace_back("images", json::makeArray({json::makeObject({{"bufferView", json::makeNumber(static_cast<double>(imageView))}, {"mimeType", json::makeString("image/png")}})}));
        root.emplace_back("samplers", json::makeArray({json::makeObject({{"magFilter", json::makeNumber(9729)}, {"minFilter", json::makeNumber(9987)}})}));
        root.emplace_back("textures", json::makeArray({json::makeObject({{"source", json::makeNumber(0)}, {"sampler", json::makeNumber(0)}})}));
        pbr.emplace_back("baseColorTexture", json::makeObject({{"index", json::makeNumber(0)}}));
    }
    root.emplace_back("materials", json::makeArray({json::makeObject({{"pbrMetallicRoughness", json::makeObject(std::move(pbr))}, {"doubleSided", json::makeBool(true)}})}));
    root.emplace_back("meshes", json::makeArray({json::makeObject({{"primitives", json::makeArray({json::makeObject(
                                                    {{"attributes", json::makeObject(std::move(attributes))},
                                                     {"indices", json::makeNumber(static_cast<double>(indexAccessor))},
                                                     {"material", json::makeNumber(0)},
                                                     {"mode", json::makeNumber(4)}})})}})}));
    align();
    root.emplace_back("buffers", json::makeArray({json::makeObject({{"byteLength", json::makeNumber(static_cast<double>(bin.size()))}})}));
    root.emplace_back("bufferViews", json::makeArray(std::move(views)));
    root.emplace_back("accessors", json::makeArray(std::move(accessors)));
    std::string text = json::dump(json::makeObject(std::move(root)));
    while (text.size() % 4) text += ' ';

    std::vector<std::uint8_t> out;
    put32(out, 0x46546C67);  // "glTF"
    put32(out, 2);
    put32(out, static_cast<std::uint32_t>(12 + 8 + text.size() + 8 + bin.size()));
    put32(out, static_cast<std::uint32_t>(text.size()));
    put32(out, 0x4E4F534A);  // "JSON"
    out.insert(out.end(), text.begin(), text.end());
    put32(out, static_cast<std::uint32_t>(bin.size()));
    put32(out, 0x004E4942);  // "BIN\0"
    out.insert(out.end(), bin.begin(), bin.end());
    return out;
}

namespace {
// A 4x4 column-major matrix (glTF's layout).
using Mat4 = std::array<double, 16>;
Mat4 identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }
Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + rr] * b[c * 4 + k];
            r[c * 4 + rr] = s;
        }
    return r;
}

struct GlbReader {
    const json::Value& doc;
    const std::uint8_t* bin;
    std::size_t binSize;
    const std::string& name;

    [[noreturn]] void fail(const std::string& why) const { throw std::runtime_error("loadMesh: " + name + ": " + why); }
    const json::Value& item(const char* list, double index) const {
        const json::Value* l = doc.find(list);
        if (!l || !l->isArray() || index < 0 || index >= static_cast<double>(l->array.size()) || index != std::floor(index))
            fail(std::string("bad reference into ") + list);
        return l->array[static_cast<std::size_t>(index)];
    }

    // An accessor's elements as floats (normalized integers scaled to 0..1); comps per element.
    std::vector<double> read(double index, std::size_t& count, std::size_t& comps) const {
        const json::Value& a = item("accessors", index);
        if (a.find("sparse")) fail("sparse accessors are not supported");
        const std::string type = a.find("type") && a.at("type").isString() ? a.at("type").string : "";
        comps = type == "SCALAR" ? 1 : type == "VEC2" ? 2 : type == "VEC3" ? 3 : type == "VEC4" ? 4 : type == "MAT4" ? 16 : 0;
        if (!comps) fail("unsupported accessor type " + type);
        const double countD = a.numberOr("count", -1);
        if (countD < 0 || countD != std::floor(countD) || countD > static_cast<double>(gMaxElements)) fail("bad accessor count");
        count = static_cast<std::size_t>(countD);
        const double ctD = a.numberOr("componentType", 0);
        const int ct = inRange(ctD, 5120, 5126) ? static_cast<int>(ctD) : 0;
        const std::size_t size = ct == 5126 || ct == 5125 ? 4 : ct == 5123 || ct == 5122 ? 2 : ct == 5121 || ct == 5120 ? 1 : 0;
        if (!size) fail("unsupported component type");
        const bool normalized = a.find("normalized") && a.at("normalized").boolean;
        if (!a.find("bufferView")) fail("accessors without buffer views are not supported");
        const json::Value& v = item("bufferViews", a.at("bufferView").number);
        if (v.numberOr("buffer", 0) != 0) fail("only the GLB's own buffer is supported");
        const double vOff = v.numberOr("byteOffset", 0), vLen = v.numberOr("byteLength", -1), aOff = a.numberOr("byteOffset", 0);
        const double strideD = v.numberOr("byteStride", 0);
        if (!inRange(vOff, 0, 1e15) || !inRange(vLen, 0, 1e15) || !inRange(aOff, 0, 1e15) || !inRange(strideD, 0, 255) ||
            vOff + vLen > static_cast<double>(binSize))
            fail("buffer view out of range");
        const std::size_t elem = size * comps, stride = strideD > 0 ? static_cast<std::size_t>(strideD) : elem;
        if (stride < elem) fail("byte stride smaller than an element");
        if (count && aOff + static_cast<double>((count - 1) * stride + elem) > vLen) fail("accessor out of range");
        const std::uint8_t* base = bin + static_cast<std::size_t>(vOff + aOff);
        std::vector<double> out(count * comps);
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t c = 0; c < comps; ++c) {
                const std::uint8_t* p = base + i * stride + c * size;
                double x = 0;
                switch (ct) {
                case 5126: {
                    float f;
                    std::memcpy(&f, p, 4);
                    x = f;
                    break;
                }
                case 5125: x = get32(p); break;
                case 5123: x = p[0] | (p[1] << 8); if (normalized) x /= 65535.0; break;
                case 5122: x = static_cast<std::int16_t>(p[0] | (p[1] << 8)); if (normalized) x = std::max(x / 32767.0, -1.0); break;
                case 5121: x = p[0]; if (normalized) x /= 255.0; break;
                case 5120: x = static_cast<std::int8_t>(p[0]); if (normalized) x = std::max(x / 127.0, -1.0); break;
                }
                out[i * comps + c] = x;
            }
        return out;
    }
};

// A node's local transform as translation, rotation, scale (10 floats).
void nodeTrs(const json::Value& node, float* out) {
    double t[3] = {0, 0, 0}, s[3] = {1, 1, 1};
    xf::Quat q{0, 0, 0, 1};
    if (const json::Value* mtx = node.find("matrix"); mtx && mtx->isArray() && mtx->array.size() == 16) {
        xf::Mat4 m;
        for (int k = 0; k < 16; ++k) m[static_cast<std::size_t>(k)] = mtx->array[static_cast<std::size_t>(k)].number;
        xf::decompose(m, t, q, s);
    } else {
        if (const json::Value* v = node.find("translation"); v && v->isArray() && v->array.size() == 3)
            for (int k = 0; k < 3; ++k) t[k] = v->array[static_cast<std::size_t>(k)].number;
        if (const json::Value* v = node.find("rotation"); v && v->isArray() && v->array.size() == 4)
            q = xf::normalized({v->array[0].number, v->array[1].number, v->array[2].number, v->array[3].number});
        if (const json::Value* v = node.find("scale"); v && v->isArray() && v->array.size() == 3)
            for (int k = 0; k < 3; ++k) s[k] = v->array[static_cast<std::size_t>(k)].number;
    }
    for (int k = 0; k < 3; ++k) {
        out[k] = static_cast<float>(t[k]);
        out[7 + k] = static_cast<float>(s[k]);
    }
    for (int k = 0; k < 4; ++k) out[3 + k] = static_cast<float>(q[static_cast<std::size_t>(k)]);
    for (int k = 0; k < 10; ++k)
        if (!std::isfinite(out[k]) || std::fabs(out[k]) > 1e15f) throw std::runtime_error("loadMesh: a joint's transform is not finite");
}

// The skeleton, skin and first animation of a model whose primitives were read with their joints and
// weights; positions become the rest pose.
void readRig(Mesh& out, const GlbReader& r, const json::Value& doc, const std::vector<int>& jointNodes, const std::vector<int>& jointOfNode,
             const std::vector<int>& parentOfNode, const std::vector<xf::Mat4>& world, const std::vector<bool>& seen) {
    const std::size_t J = jointNodes.size();
    std::vector<int> parentJoint(J, -1);
    for (std::size_t j = 0; j < J; ++j) {
        int p = parentOfNode[static_cast<std::size_t>(jointNodes[j])];
        for (std::size_t guard = 0; p >= 0 && jointOfNode[static_cast<std::size_t>(p)] < 0 && guard < jointOfNode.size(); ++guard)
            p = parentOfNode[static_cast<std::size_t>(p)];
        parentJoint[j] = p >= 0 ? jointOfNode[static_cast<std::size_t>(p)] : -1;
    }
    // Parents before children.
    std::vector<int> order, newIndex(J, -1);
    while (order.size() < J) {
        const std::size_t before = order.size();
        for (std::size_t j = 0; j < J; ++j)
            if (newIndex[j] < 0 && (parentJoint[j] < 0 || newIndex[static_cast<std::size_t>(parentJoint[j])] >= 0)) {
                newIndex[j] = static_cast<int>(order.size());
                order.push_back(static_cast<int>(j));
            }
        if (order.size() == before) r.fail("the joints form a cycle");
    }
    MeshRig& g = out.rig;
    g.names.resize(J);
    g.parents.resize(J);
    g.nodes.resize(J);
    g.rest.resize(10 * J);
    for (std::size_t i = 0; i < J; ++i) {
        const auto old = static_cast<std::size_t>(order[i]);
        const json::Value& node = r.item("nodes", jointNodes[old]);
        g.names[i] = node.find("name") && node.at("name").isString() ? node.at("name").string : "joint_" + std::to_string(i);
        g.parents[i] = parentJoint[old] < 0 ? -1 : newIndex[static_cast<std::size_t>(parentJoint[old])];
        g.nodes[i] = jointNodes[old];
        nodeTrs(node, g.rest.data() + 10 * i);
    }
    const int rootParentNode = parentOfNode[static_cast<std::size_t>(g.nodes[0])];
    const xf::Mat4 rp = rootParentNode >= 0 && seen[static_cast<std::size_t>(rootParentNode)] ? world[static_cast<std::size_t>(rootParentNode)] : xf::identity();
    g.rootParent.assign(rp.begin(), rp.end());
    g.inverseBind.assign(16 * J, 0.0f);
    for (std::size_t i = 0; i < J; ++i)
        for (int k = 0; k < 4; ++k) g.inverseBind[16 * i + 5 * static_cast<std::size_t>(k)] = 1.0f;
    const json::Value& skin = doc.at("skins").array[0];
    if (const json::Value* ib = skin.find("inverseBindMatrices"); ib && ib->isNumber()) {
        std::size_t n = 0, comps = 0;
        const std::vector<double> m = r.read(ib->number, n, comps);
        if (n != J || comps != 16) r.fail("inverseBindMatrices do not match the joints");
        for (std::size_t old = 0; old < J; ++old)
            for (int k = 0; k < 16; ++k) {
                const double v = m[16 * old + static_cast<std::size_t>(k)];
                if (!std::isfinite(v) || std::fabs(v) > 1e15) r.fail("inverseBindMatrices are not finite");
                g.inverseBind[16 * static_cast<std::size_t>(newIndex[old]) + static_cast<std::size_t>(k)] = static_cast<float>(v);
            }
    }
    for (std::uint16_t& j : g.joints) j = static_cast<std::uint16_t>(newIndex[j]);
    for (std::size_t v = 0; 4 * v < g.weights.size(); ++v) {
        float* w = g.weights.data() + 4 * v;
        const float sum = w[0] + w[1] + w[2] + w[3];
        if (sum > 0)
            for (int k = 0; k < 4; ++k) w[k] /= sum;
    }
    g.bindPositions = out.positions;

    // The first animation, sampled at the rate of its keyframes (1..60 a second; rotations by slerp, the
    // rest linearly).
    if (const json::Value* anims = doc.find("animations"); anims && anims->isArray() && !anims->array.empty()) {
        const json::Value& a = anims->array[0];
        const json::Value* channels = a.find("channels");
        const json::Value* samplers = a.find("samplers");
        struct Track {
            std::size_t joint, offset, comps;
            std::vector<double> times, values;
            int mode;  // 0 linear, 1 step, 2 cubic spline (values only, tangents skipped)
        };
        std::vector<Track> tracks;
        double duration = 0, minStep = 1e30;
        if (channels && channels->isArray() && samplers && samplers->isArray())
            for (const json::Value& c : channels->array) {
                const json::Value* target = c.find("target");
                if (!target || !target->find("node") || !target->find("path") || !c.find("sampler")) continue;
                const double nodeD = target->at("node").number;
                if (!(nodeD >= 0 && nodeD < static_cast<double>(jointOfNode.size()))) continue;
                const int old = jointOfNode[static_cast<std::size_t>(nodeD)];
                const std::string path = target->at("path").isString() ? target->at("path").string : "";
                if (old < 0 || (path != "translation" && path != "rotation" && path != "scale")) continue;
                const double si = c.at("sampler").number;
                if (!(si >= 0 && si < static_cast<double>(samplers->array.size()))) r.fail("bad animation sampler");
                const json::Value& s = samplers->array[static_cast<std::size_t>(si)];
                if (!s.find("input") || !s.find("output")) r.fail("animation sampler without input or output");
                Track t;
                t.joint = static_cast<std::size_t>(newIndex[static_cast<std::size_t>(old)]);
                t.offset = path == "translation" ? 0 : path == "rotation" ? 3 : 7;
                const std::string interp = s.find("interpolation") && s.at("interpolation").isString() ? s.at("interpolation").string : "LINEAR";
                t.mode = interp == "STEP" ? 1 : interp == "CUBICSPLINE" ? 2 : 0;
                std::size_t tn = 0, tc = 0, vn = 0;
                t.times = r.read(s.at("input").number, tn, tc);
                t.values = r.read(s.at("output").number, vn, t.comps);
                if (tc != 1 || tn == 0 || t.comps != (path == "rotation" ? 4u : 3u) || vn != tn * (t.mode == 2 ? 3 : 1)) r.fail("animation sampler sizes do not match");
                for (std::size_t k = 0; k < tn; ++k)
                    if (!std::isfinite(t.times[k]) || t.times[k] < 0 || t.times[k] > 3600 || (k && t.times[k] < t.times[k - 1])) r.fail("bad animation times");
                for (double v : t.values)
                    if (!std::isfinite(v) || std::fabs(v) > 1e15) r.fail("animation values are not finite");
                duration = std::max(duration, t.times.back());
                for (std::size_t k = 1; k < tn; ++k)
                    if (t.times[k] - t.times[k - 1] > 1e-6) minStep = std::min(minStep, t.times[k] - t.times[k - 1]);
                tracks.push_back(std::move(t));
            }
        if (!tracks.empty()) {
            g.fps = minStep < 1e29 ? std::clamp(std::round(1.0 / minStep), 1.0, 60.0) : 30.0;
            const std::size_t frames = std::min<std::size_t>(static_cast<std::size_t>(std::floor(duration * g.fps + 0.5)) + 1, 18000);
            if (frames * J * 10 > 4 * gMaxElements) r.fail("the animation is too long");
            g.frames.resize(frames * J * 10);
            for (std::size_t f = 0; f < frames; ++f) std::copy(g.rest.begin(), g.rest.end(), g.frames.begin() + static_cast<long>(f * J * 10));
            for (const Track& t : tracks) {
                const std::size_t n = t.times.size(), step = t.mode == 2 ? 3 : 1, shift = t.mode == 2 ? 1 : 0;
                auto value = [&](std::size_t k, std::size_t c) { return t.values[(k * step + shift) * t.comps + c]; };
                for (std::size_t f = 0; f < frames; ++f) {
                    const double time = static_cast<double>(f) / g.fps;
                    const std::size_t hi = static_cast<std::size_t>(std::upper_bound(t.times.begin(), t.times.end(), time) - t.times.begin());
                    const std::size_t k0 = hi == 0 ? 0 : hi - 1, k1 = std::min(hi, n - 1);
                    double u = 0;
                    if (t.mode != 1 && k1 != k0 && t.times[k1] > t.times[k0]) u = std::clamp((time - t.times[k0]) / (t.times[k1] - t.times[k0]), 0.0, 1.0);
                    float* out10 = g.frames.data() + (f * J + t.joint) * 10;
                    if (t.comps == 4) {
                        const xf::Quat q = xf::slerp(xf::normalized({value(k0, 0), value(k0, 1), value(k0, 2), value(k0, 3)}),
                                                     xf::normalized({value(k1, 0), value(k1, 1), value(k1, 2), value(k1, 3)}), u);
                        for (int c = 0; c < 4; ++c) out10[3 + c] = static_cast<float>(q[static_cast<std::size_t>(c)]);
                    } else {
                        for (std::size_t c = 0; c < 3; ++c) out10[t.offset + c] = static_cast<float>(value(k0, c) + u * (value(k1, c) - value(k0, c)));
                    }
                }
            }
        }
    }
    out.positions = skinnedPositions(g, g.rest.data());
}
} // namespace

Mesh decodeGlb(const std::vector<std::uint8_t>& d, const std::string& name) {
    auto fail = [&](const std::string& why) -> void { throw std::runtime_error("loadMesh: " + name + ": " + why); };
    if (d.size() < 20 || get32(d.data()) != 0x46546C67) fail("not a glTF binary file");
    if (get32(d.data() + 4) != 2) fail("only glTF 2.0 is supported");
    const std::size_t jsonLen = get32(d.data() + 12);
    if (get32(d.data() + 16) != 0x4E4F534A || 20 + jsonLen > d.size()) fail("bad JSON chunk");
    const json::Value doc = json::parse(std::string(d.begin() + 20, d.begin() + 20 + static_cast<long>(jsonLen)));
    const std::uint8_t* bin = nullptr;
    std::size_t binSize = 0;
    const std::size_t at = 20 + ((jsonLen + 3) & ~std::size_t(3));
    if (at + 8 <= d.size() && get32(d.data() + at + 4) == 0x004E4942) {
        binSize = get32(d.data() + at);
        if (at + 8 + binSize > d.size()) fail("truncated binary chunk");
        bin = d.data() + at + 8;
    }
    if (const json::Value* req = doc.find("extensionsRequired"); req && req->isArray() && !req->array.empty())
        fail("needs extension " + (req->array[0].isString() ? req->array[0].string : std::string("?")) + " (compressed meshes are not supported)");
    GlbReader r{doc, bin, binSize, name};

    // A rigged model: the first skin's joints (other skins' meshes are read as they stand).
    const json::Value* nodes = doc.find("nodes");
    const std::size_t nodeCount = nodes && nodes->isArray() ? nodes->array.size() : 0;
    std::vector<int> jointNodes;
    std::vector<int> jointOfNode(nodeCount, -1), parentOfNode(nodeCount, -1);
    for (std::size_t i = 0; i < nodeCount; ++i)
        if (const json::Value* kids = nodes->array[i].find("children"); kids && kids->isArray())
            for (const json::Value& k : kids->array)
                if (k.isNumber() && k.number >= 0 && k.number < static_cast<double>(nodeCount)) parentOfNode[static_cast<std::size_t>(k.number)] = static_cast<int>(i);
    const json::Value* skins = doc.find("skins");
    const json::Value* skin = skins && skins->isArray() && !skins->array.empty() ? &skins->array[0] : nullptr;
    if (skin)
        if (const json::Value* js = skin->find("joints"); js && js->isArray()) {
            if (js->array.size() > 1024) r.fail("too many joints");
            for (const json::Value& j : js->array) {
                if (!j.isNumber() || j.number < 0 || j.number >= static_cast<double>(nodeCount) || j.number != std::floor(j.number)) r.fail("bad joint");
                const auto n = static_cast<std::size_t>(j.number);
                if (jointOfNode[n] >= 0) r.fail("a joint listed twice");
                jointOfNode[n] = static_cast<int>(jointNodes.size());
                jointNodes.push_back(static_cast<int>(n));
            }
        }
    std::vector<xf::Mat4> nodeWorld(nodeCount, xf::identity());
    std::vector<bool> nodeSeen(nodeCount, false);
    bool anySkinned = false;

    Mesh out;
    bool textureTaken = false;
    auto addPrimitive = [&](const json::Value& prim, const Mat4& M, bool skinnedNode) {
        if (prim.numberOr("mode", 4) != 4) return;  // triangles only (points and lines are skipped)
        const json::Value* attrs = prim.find("attributes");
        if (!attrs || !attrs->find("POSITION")) return;
        std::size_t n = 0, comps = 0;
        const std::vector<double> pos = r.read(attrs->at("POSITION").number, n, comps);
        if (comps != 3) r.fail("POSITION must be VEC3");
        checkCount(out.vertexCount() + n, "loadMesh: " + name);
        const std::uint32_t first = static_cast<std::uint32_t>(out.vertexCount());
        // Skinned vertices stay in bind space (placed by their joints later); others go where the node puts them.
        const bool skinned = skinnedNode && attrs->find("JOINTS_0") && attrs->find("WEIGHTS_0");
        for (std::size_t i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k)
                out.positions.push_back(skinned ? static_cast<float>(pos[3 * i + k])
                                                : static_cast<float>(M[k] * pos[3 * i] + M[4 + k] * pos[3 * i + 1] + M[8 + k] * pos[3 * i + 2] + M[12 + k]));
        if (!jointNodes.empty()) {
            std::vector<std::uint16_t> jv(4 * n, 0);
            std::vector<float> wv(4 * n, 0.0f);
            if (skinned) {
                std::size_t jn = 0, jc = 0, wn = 0, wc = 0;
                const std::vector<double> js = r.read(attrs->at("JOINTS_0").number, jn, jc), ws = r.read(attrs->at("WEIGHTS_0").number, wn, wc);
                if (jn != n || wn != n || jc != 4 || wc != 4) r.fail("JOINTS_0 / WEIGHTS_0 do not match POSITION");
                for (std::size_t i = 0; i < 4 * n; ++i) {
                    if (!(js[i] >= 0 && js[i] < static_cast<double>(jointNodes.size()))) r.fail("joint index out of range");
                    jv[i] = static_cast<std::uint16_t>(js[i]);
                    wv[i] = std::isfinite(ws[i]) && ws[i] > 0 ? static_cast<float>(ws[i]) : 0.0f;
                }
                anySkinned = true;
            }
            out.rig.joints.insert(out.rig.joints.end(), jv.begin(), jv.end());
            out.rig.weights.insert(out.rig.weights.end(), wv.begin(), wv.end());
        }
        // uvs and colours are kept when every primitive has them (else dropped at the end)
        std::vector<float> uv(2 * n, 0.0f);
        bool hasUv = false;
        if (const json::Value* t = attrs->find("TEXCOORD_0")) {
            std::size_t tn = 0, tc = 0;
            const std::vector<double> x = r.read(t->number, tn, tc);
            if (tn != n || tc != 2) r.fail("TEXCOORD_0 does not match POSITION");
            for (std::size_t i = 0; i < 2 * n; ++i) uv[i] = static_cast<float>(x[i]);
            hasUv = true;
        }
        std::vector<std::uint8_t> col(3 * n, 255);
        bool hasCol = false;
        if (const json::Value* c = attrs->find("COLOR_0")) {
            std::size_t cn = 0, cc = 0;
            const std::vector<double> x = r.read(c->number, cn, cc);
            if (cn != n || cc < 3) r.fail("COLOR_0 does not match POSITION");
            for (std::size_t i = 0; i < n; ++i)
                for (int k = 0; k < 3; ++k) col[3 * i + k] = toByte(x[i * cc + k] * 255.0);
            hasCol = true;
        }
        // The first textured material gives the mesh its texture; uvs of other materials are kept as they are.
        if (!textureTaken && hasUv && prim.find("material")) {
            const json::Value& mat = r.item("materials", prim.at("material").number);
            const json::Value* pbr = mat.find("pbrMetallicRoughness");
            const json::Value* bct = pbr ? pbr->find("baseColorTexture") : nullptr;
            if (bct && bct->find("index")) {
                const json::Value& tex = r.item("textures", bct->at("index").number);
                if (tex.find("source")) {
                    const json::Value& img = r.item("images", tex.at("source").number);
                    const std::string mime = img.find("mimeType") && img.at("mimeType").isString() ? img.at("mimeType").string : "";
                    if (img.find("bufferView") && mime == "image/png") {
                        const json::Value& v = r.item("bufferViews", img.at("bufferView").number);
                        const double off = v.numberOr("byteOffset", 0), len = v.numberOr("byteLength", -1);
                        if (!inRange(off, 0, 1e15) || !inRange(len, 0, 1e15) || off + len > static_cast<double>(binSize)) r.fail("image out of range");
                        out.texture = toRgb(decodePng(std::vector<std::uint8_t>(bin + static_cast<std::size_t>(off), bin + static_cast<std::size_t>(off + len)), name));
                        textureTaken = true;
                    }
                }
            }
        }
        (void)hasCol;
        out.uvs.insert(out.uvs.end(), uv.begin(), uv.end());
        out.colors.insert(out.colors.end(), col.begin(), col.end());
        std::vector<std::uint32_t> idx;
        if (const json::Value* ix = prim.find("indices")) {
            std::size_t in = 0, ic = 0;
            const std::vector<double> x = r.read(ix->number, in, ic);
            if (ic != 1 || in % 3) r.fail("indices are not triangles");
            for (double v : x) {
                if (!(v >= 0 && v < static_cast<double>(n))) r.fail("index out of range");
                idx.push_back(first + static_cast<std::uint32_t>(v));
            }
        } else {
            if (n % 3) r.fail("a non-indexed primitive needs a multiple of 3 vertices");
            for (std::size_t i = 0; i < n; ++i) idx.push_back(first + static_cast<std::uint32_t>(i));
        }
        checkCount(out.indices.size() + idx.size(), "loadMesh: " + name);
        out.indices.insert(out.indices.end(), idx.begin(), idx.end());
    };

    // Nodes of the default scene (or every root node), depth first, transforms composed.
    std::size_t visits = 0;
    std::function<void(double, const Mat4&, int)> visit = [&](double index, const Mat4& parent, int depth) {
        if (depth > 64 || ++visits > 100000) r.fail("node hierarchy too deep or too large");
        const json::Value& node = r.item("nodes", index);
        Mat4 local = identity();
        if (const json::Value* mtx = node.find("matrix"); mtx && mtx->isArray() && mtx->array.size() == 16) {
            for (int k = 0; k < 16; ++k) local[k] = mtx->array[k].number;
        } else {
            Mat4 T = identity(), R = identity(), S = identity();
            if (const json::Value* t = node.find("translation"); t && t->isArray() && t->array.size() == 3)
                for (int k = 0; k < 3; ++k) T[12 + k] = t->array[k].number;
            if (const json::Value* s = node.find("scale"); s && s->isArray() && s->array.size() == 3)
                for (int k = 0; k < 3; ++k) S[k * 5] = s->array[k].number;
            if (const json::Value* q = node.find("rotation"); q && q->isArray() && q->array.size() == 4) {
                const double x = q->array[0].number, y = q->array[1].number, z = q->array[2].number, w = q->array[3].number;
                R = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0, 2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
                     2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0, 0, 0, 0, 1};
            }
            local = mul(T, mul(R, S));
        }
        const Mat4 M = mul(parent, local);
        const auto ni = static_cast<std::size_t>(index);
        nodeSeen[ni] = true;
        for (int k = 0; k < 16; ++k) nodeWorld[ni][static_cast<std::size_t>(k)] = M[static_cast<std::size_t>(k)];
        if (const json::Value* mesh = node.find("mesh"); mesh && mesh->isNumber()) {
            const json::Value& me = r.item("meshes", mesh->number);
            const bool skinnedNode = !jointNodes.empty() && node.numberOr("skin", -1) == 0;
            if (const json::Value* prims = me.find("primitives"); prims && prims->isArray())
                for (const json::Value& p : prims->array) addPrimitive(p, M, skinnedNode);
        }
        if (const json::Value* kids = node.find("children"); kids && kids->isArray())
            for (const json::Value& k : kids->array) visit(k.number, M, depth + 1);
    };
    std::vector<double> roots;
    if (const json::Value* scenes = doc.find("scenes"); scenes && scenes->isArray() && !scenes->array.empty()) {
        const json::Value& scene = r.item("scenes", doc.numberOr("scene", 0));
        if (const json::Value* ns = scene.find("nodes"); ns && ns->isArray())
            for (const json::Value& n : ns->array) roots.push_back(n.number);
    } else if (nodes && nodes->isArray()) {
        for (std::size_t i = 0; i < nodes->array.size(); ++i) roots.push_back(static_cast<double>(i));
    }
    if (roots.empty()) {  // no scene graph: every mesh as it is
        if (const json::Value* meshes = doc.find("meshes"); meshes && meshes->isArray())
            for (const json::Value& me : meshes->array)
                if (const json::Value* prims = me.find("primitives"); prims && prims->isArray())
                    for (const json::Value& p : prims->array) addPrimitive(p, identity(), false);
    }
    for (double root : roots) visit(root, identity(), 0);
    if (out.indices.empty()) fail("no triangles");
    if (anySkinned) readRig(out, r, doc, jointNodes, jointOfNode, parentOfNode, nodeWorld, nodeSeen);
    else out.rig = MeshRig{};
    // Colours that were all defaults are dropped; uvs without a texture too.
    if (std::all_of(out.colors.begin(), out.colors.end(), [](std::uint8_t c) { return c == 255; })) out.colors.clear();
    if (out.texture.pixels.empty()) out.uvs.clear();
    computeNormals(out);
    checkMesh(out, "loadMesh: " + name);
    return out;
}

// ---- Wavefront OBJ ----

Mesh parseObj(const std::string& text, const std::string& name) {
    std::vector<float> v, vt, vn;
    std::vector<std::uint8_t> vc;
    bool colours = false;
    Mesh out;
    std::map<std::array<long, 3>, std::uint32_t> seen;  // (v, vt, vn) -> output vertex
    std::istringstream in(text);
    std::string line;
    std::size_t lineNo = 0;
    auto fail = [&](const std::string& why) { throw std::runtime_error("loadMesh: " + name + " line " + std::to_string(lineNo) + ": " + why); };
    while (std::getline(in, line)) {
        ++lineNo;
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "v") {
            double x[6];
            int k = 0;
            while (k < 6 && ls >> x[k]) ++k;
            if (k < 3) fail("a vertex needs x y z");
            for (int i = 0; i < 3; ++i) v.push_back(static_cast<float>(x[i]));
            if (k == 6) colours = true;
            for (int i = 3; i < 6; ++i) vc.push_back(k == 6 ? toByte(x[i] * 255.0) : 255);
            checkCount(v.size() / 3, "loadMesh: " + name);
        } else if (tag == "vt") {
            double u = 0, w = 0;
            if (!(ls >> u)) fail("a texture coordinate needs u");
            ls >> w;
            vt.push_back(static_cast<float>(u));
            vt.push_back(static_cast<float>(1.0 - w));  // OBJ's v grows upwards
            checkCount(vt.size() / 2, "loadMesh: " + name);
        } else if (tag == "vn") {
            double x = 0, y = 0, z = 0;
            if (!(ls >> x >> y >> z)) fail("a normal needs x y z");
            vn.insert(vn.end(), {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
            checkCount(vn.size() / 3, "loadMesh: " + name);
        } else if (tag == "f") {
            std::vector<std::uint32_t> face;
            std::string tok;
            while (ls >> tok) {
                long idx[3] = {0, 0, 0};
                std::size_t start = 0;
                for (int part = 0; part < 3 && start <= tok.size(); ++part) {
                    const std::size_t slash = tok.find('/', start);
                    const std::string s = tok.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
                    if (!s.empty()) {
                        try {
                            idx[part] = std::stol(s);
                        } catch (const std::exception&) {
                            fail("bad face index " + s);
                        }
                    }
                    if (slash == std::string::npos) break;
                    start = slash + 1;
                }
                const long counts[3] = {static_cast<long>(v.size() / 3), static_cast<long>(vt.size() / 2), static_cast<long>(vn.size() / 3)};
                for (int k = 0; k < 3; ++k) {
                    if (idx[k] < 0) idx[k] += counts[k] + 1;
                    if (idx[k] < 0 || idx[k] > counts[k] || (k == 0 && idx[k] == 0)) fail("face index out of range");
                }
                const std::array<long, 3> key{idx[0], idx[1], idx[2]};
                auto it = seen.find(key);
                if (it == seen.end()) {
                    const std::uint32_t id = static_cast<std::uint32_t>(out.vertexCount());
                    checkCount(id + 1, "loadMesh: " + name);
                    const std::size_t p = static_cast<std::size_t>(idx[0] - 1);
                    out.positions.insert(out.positions.end(), v.begin() + static_cast<long>(3 * p), v.begin() + static_cast<long>(3 * p + 3));
                    out.colors.insert(out.colors.end(), vc.begin() + static_cast<long>(3 * p), vc.begin() + static_cast<long>(3 * p + 3));
                    if (idx[1]) out.uvs.insert(out.uvs.end(), vt.begin() + 2 * (idx[1] - 1), vt.begin() + 2 * idx[1]);
                    else out.uvs.insert(out.uvs.end(), {0.0f, 0.0f});
                    if (idx[2]) out.normals.insert(out.normals.end(), vn.begin() + 3 * (idx[2] - 1), vn.begin() + 3 * idx[2]);
                    else out.normals.insert(out.normals.end(), {0.0f, 0.0f, 0.0f});
                    it = seen.emplace(key, id).first;
                }
                face.push_back(it->second);
            }
            if (face.size() < 3) fail("a face needs three vertices");
            for (std::size_t k = 1; k + 1 < face.size(); ++k) out.indices.insert(out.indices.end(), {face[0], face[k], face[k + 1]});
            checkCount(out.indices.size() / 3, "loadMesh: " + name);
        }
    }
    if (out.indices.empty()) throw std::runtime_error("loadMesh: " + name + ": no faces");
    if (!colours) out.colors.clear();
    out.uvs.clear();  // (an OBJ's texture lives in a .mtl; only the geometry and colours are read)
    computeNormals(out);
    checkMesh(out, "loadMesh: " + name);
    return out;
}

// ---- PLY ----

Mesh decodePly(const std::vector<std::uint8_t>& d, const std::string& name) {
    auto fail = [&](const std::string& why) -> void { throw std::runtime_error("loadMesh: " + name + ": " + why); };
    std::size_t at = 0;
    auto line = [&]() {
        std::string s;
        while (at < d.size() && d[at] != '\n') s += static_cast<char>(d[at++]);
        if (at >= d.size()) fail("header not finished");
        ++at;
        if (!s.empty() && s.back() == '\r') s.pop_back();
        return s;
    };
    if (line() != "ply") fail("not a PLY file");
    struct Prop {
        std::string name, type, countType;  // countType set for lists
    };
    struct Element {
        std::string name;
        std::size_t count = 0;
        std::vector<Prop> props;
    };
    std::vector<Element> elements;
    std::string format;
    for (int guard = 0;; ++guard) {
        if (guard > 10000) fail("header too long");
        const std::string s = line();
        std::istringstream ls(s);
        std::string tag;
        ls >> tag;
        if (tag == "format") ls >> format;
        else if (tag == "element") {
            Element e;
            double count = -1;
            ls >> e.name >> count;
            if (!inRange(count, 0, static_cast<double>(gMaxElements))) fail("bad element count");
            e.count = static_cast<std::size_t>(count);
            elements.push_back(e);
        } else if (tag == "property") {
            if (elements.empty()) fail("property before element");
            Prop p;
            ls >> p.type;
            if (p.type == "list") ls >> p.countType >> p.type;
            ls >> p.name;
            elements.back().props.push_back(p);
        } else if (tag == "end_header") break;
    }
    const bool ascii = format == "ascii";
    if (!ascii && format != "binary_little_endian") fail("format " + format + " is not supported");
    auto typeSize = [&](const std::string& t) -> std::size_t {
        if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
        if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
        if (t == "int" || t == "uint" || t == "float" || t == "int32" || t == "uint32" || t == "float32") return 4;
        if (t == "double" || t == "float64") return 8;
        fail("unknown property type " + t);
        return 0;
    };
    std::istringstream asciiIn;
    if (ascii) asciiIn.str(std::string(d.begin() + static_cast<long>(at), d.end()));
    auto value = [&](const std::string& t) -> double {
        if (ascii) {
            double x;
            if (!(asciiIn >> x)) fail("truncated data");
            return x;
        }
        const std::size_t n = typeSize(t);
        if (at + n > d.size()) fail("truncated data");
        const std::uint8_t* p = d.data() + at;
        at += n;
        if (t == "float" || t == "float32") {
            float f;
            std::memcpy(&f, p, 4);
            return f;
        }
        if (t == "double" || t == "float64") {
            double f;
            std::memcpy(&f, p, 8);
            return f;
        }
        if (t == "char" || t == "int8") return static_cast<std::int8_t>(p[0]);
        if (t == "uchar" || t == "uint8") return p[0];
        if (t == "short" || t == "int16") return static_cast<std::int16_t>(p[0] | (p[1] << 8));
        if (t == "ushort" || t == "uint16") return p[0] | (p[1] << 8);
        if (t == "int" || t == "int32") return static_cast<std::int32_t>(get32(p));
        return get32(p);
    };
    Mesh out;
    bool colours = false, uvs = false;
    for (const Element& e : elements) {
        if (e.name == "vertex") {
            checkCount(e.count, "loadMesh: " + name);
            out.positions.reserve(3 * e.count);
            for (std::size_t i = 0; i < e.count; ++i) {
                float xyz[3] = {0, 0, 0}, uv[2] = {0, 0};
                std::uint8_t rgb[3] = {255, 255, 255};
                for (const Prop& p : e.props) {
                    if (!p.countType.empty()) {
                        const double n = value(p.countType);
                        if (!inRange(n, 0, 1e6)) fail("bad list length");
                        for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) value(p.type);
                        continue;
                    }
                    const double x = value(p.type);
                    if (p.name == "x") xyz[0] = static_cast<float>(x);
                    else if (p.name == "y") xyz[1] = static_cast<float>(x);
                    else if (p.name == "z") xyz[2] = static_cast<float>(x);
                    else if (p.name == "red" || p.name == "green" || p.name == "blue") {
                        const double scaled = p.type == "float" || p.type == "float32" || p.type == "double" ? x * 255.0 : x;
                        rgb[p.name == "red" ? 0 : p.name == "green" ? 1 : 2] = toByte(scaled);
                        colours = true;
                    } else if (p.name == "u" || p.name == "s" || p.name == "texture_u") {
                        uv[0] = static_cast<float>(x);
                        uvs = true;
                    } else if (p.name == "v" || p.name == "t" || p.name == "texture_v") uv[1] = static_cast<float>(1.0 - x);
                }
                out.positions.insert(out.positions.end(), xyz, xyz + 3);
                out.colors.insert(out.colors.end(), rgb, rgb + 3);
                out.uvs.insert(out.uvs.end(), uv, uv + 2);
            }
        } else if (e.name == "face") {
            checkCount(e.count, "loadMesh: " + name);
            for (std::size_t i = 0; i < e.count; ++i)
                for (const Prop& p : e.props) {
                    if (p.countType.empty()) {
                        value(p.type);
                        continue;
                    }
                    const double n = value(p.countType);
                    if (!inRange(n, 0, 1e6)) fail("bad face size");
                    std::vector<std::uint32_t> face;
                    for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) {
                        const double x = value(p.type);
                        if (!(x >= 0 && x < static_cast<double>(out.vertexCount()))) fail("face index out of range");
                        face.push_back(static_cast<std::uint32_t>(x));
                    }
                    if (p.name != "vertex_indices" && p.name != "vertex_index") continue;
                    for (std::size_t k = 1; k + 1 < face.size(); ++k) out.indices.insert(out.indices.end(), {face[0], face[k], face[k + 1]});
                    checkCount(out.indices.size() / 3, "loadMesh: " + name);
                }
        } else {
            for (std::size_t i = 0; i < e.count; ++i)
                for (const Prop& p : e.props) {
                    if (!p.countType.empty()) {
                        const double n = value(p.countType);
                        if (!inRange(n, 0, 1e6)) fail("bad list length");
                        for (std::size_t k = 0; k < static_cast<std::size_t>(n); ++k) value(p.type);
                    } else value(p.type);
                }
        }
    }
    if (out.indices.empty()) fail("no faces (point clouds are not supported)");
    if (!colours) out.colors.clear();
    out.uvs.clear();
    (void)uvs;
    computeNormals(out);
    checkMesh(out, "loadMesh: " + name);
    return out;
}

Mesh loadMesh(const std::string& path) {
    const std::string e = extension(path);
    const std::vector<std::uint8_t> data = readBytes(path);
    if (e == ".glb") {
        Mesh m = decodeGlb(data, path);
        m.glb = data;
        return m;
    }
    if (e == ".obj") return parseObj(std::string(data.begin(), data.end()), path);
    if (e == ".ply") return decodePly(data, path);
    throw std::runtime_error("loadMesh: unknown mesh file type " + path);
}

void saveMesh(const std::string& path, const Mesh& m) {
    checkMesh(m, "saveMesh");
    if (m.empty()) throw std::invalid_argument("saveMesh: the mesh has no triangles");
    const std::string e = extension(path);
    if (e == ".glb") {
        if (!m.glb.empty()) return writeBytes(path, m.glb.data(), m.glb.size());
        const std::vector<std::uint8_t> b = encodeGlb(m);
        writeBytes(path, b.data(), b.size());
    } else if (e == ".ply") {
        const std::vector<std::uint8_t> col = vertexColours(m);
        std::ostringstream h;
        h << "ply\nformat binary_little_endian 1.0\ncomment NeuraCraft\nelement vertex " << m.vertexCount()
          << "\nproperty float x\nproperty float y\nproperty float z\n";
        if (!m.normals.empty()) h << "property float nx\nproperty float ny\nproperty float nz\n";
        h << "property uchar red\nproperty uchar green\nproperty uchar blue\nelement face " << m.triangleCount()
          << "\nproperty list uchar int vertex_indices\nend_header\n";
        std::string out = h.str();
        auto raw = [&](const void* p, std::size_t n) { out.append(static_cast<const char*>(p), n); };
        for (std::size_t i = 0; i < m.vertexCount(); ++i) {
            raw(&m.positions[3 * i], 12);
            if (!m.normals.empty()) raw(&m.normals[3 * i], 12);
            raw(&col[3 * i], 3);
        }
        for (std::size_t t = 0; t < m.triangleCount(); ++t) {
            const std::uint8_t three = 3;
            raw(&three, 1);
            raw(&m.indices[3 * t], 12);
        }
        writeBytes(path, out.data(), out.size());
    } else if (e == ".obj") {
        const std::filesystem::path p(path);
        const bool textured = !m.texture.pixels.empty() && !m.uvs.empty();
        const std::string stem = p.stem().string();
        const std::vector<std::uint8_t> col = vertexColours(m);
        std::ostringstream o;
        o.precision(7);
        o << "# NeuraCraft\n";
        if (textured) o << "mtllib " << stem << ".mtl\nusemtl material0\n";
        for (std::size_t i = 0; i < m.vertexCount(); ++i)
            o << "v " << m.positions[3 * i] << ' ' << m.positions[3 * i + 1] << ' ' << m.positions[3 * i + 2] << ' ' << col[3 * i] / 255.0 << ' '
              << col[3 * i + 1] / 255.0 << ' ' << col[3 * i + 2] / 255.0 << '\n';
        if (textured)
            for (std::size_t i = 0; i < m.vertexCount(); ++i) o << "vt " << m.uvs[2 * i] << ' ' << 1.0f - m.uvs[2 * i + 1] << '\n';
        if (!m.normals.empty())
            for (std::size_t i = 0; i < m.vertexCount(); ++i) o << "vn " << m.normals[3 * i] << ' ' << m.normals[3 * i + 1] << ' ' << m.normals[3 * i + 2] << '\n';
        for (std::size_t t = 0; t < m.triangleCount(); ++t) {
            o << 'f';
            for (int k = 0; k < 3; ++k) {
                const std::uint32_t i = m.indices[3 * t + k] + 1;
                o << ' ' << i;
                if (textured || !m.normals.empty()) o << '/' << (textured ? std::to_string(i) : std::string()) << (m.normals.empty() ? std::string() : "/" + std::to_string(i));
            }
            o << '\n';
        }
        const std::string text = o.str();
        writeBytes(path, text.data(), text.size());
        if (textured) {
            const std::string mtl = "newmtl material0\nKd 1 1 1\nmap_Kd " + stem + "_albedo.png\n";
            writeBytes((p.parent_path() / (stem + ".mtl")).string(), mtl.data(), mtl.size());
            savePng((p.parent_path() / (stem + "_albedo.png")).string(), m.texture);
        }
    } else {
        throw std::invalid_argument("saveMesh: unknown mesh file type " + path + " (.glb, .obj or .ply)");
    }
}

// ---- Rendering ----

namespace {
struct Vec3 {
    double x, y, z;
};
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalize(Vec3 a) {
    const double l = std::sqrt(dot(a, a));
    return l > 0 ? Vec3{a.x / l, a.y / l, a.z / l} : Vec3{0, 0, 1};
}
} // namespace

Image renderMesh(const Mesh& m, const MeshRenderOptions& o) {
    checkMesh(m, "renderMesh");
    if (o.width == 0 || o.height == 0 || o.width > 8192 || o.height > 8192) throw std::invalid_argument("renderMesh: bad size");
    const std::size_t ss = std::clamp<std::size_t>(o.supersample, 1, 4), W = o.width * ss, H = o.height * ss;
    // The bounding sphere: centre of the box, radius to the farthest vertex.
    Vec3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
    for (std::size_t i = 0; i < m.vertexCount(); ++i) {
        const Vec3 p{m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]};
        lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    Vec3 centre{(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2};
    double radius = 1e-9;
    for (std::size_t i = 0; i < m.vertexCount(); ++i) {
        const Vec3 d = Vec3{m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]} - centre;
        radius = std::max(radius, std::sqrt(dot(d, d)));
    }
    if (o.fixedFrame) {
        centre = {o.frameCentre[0], o.frameCentre[1], o.frameCentre[2]};
        radius = std::max(1e-9, static_cast<double>(o.frameRadius));
    }
    constexpr double kDeg = 3.14159265358979323846 / 180.0;
    const double yaw = o.yaw * kDeg, pitch = std::clamp<double>(o.pitch, -89.0, 89.0) * kDeg;
    const double fovY = std::clamp<double>(o.fov, 5.0, 120.0) * kDeg, aspect = static_cast<double>(W) / static_cast<double>(H);
    const double fit = std::min(std::tan(fovY / 2), std::tan(fovY / 2) * aspect);
    const double dist = radius * std::max(1.0, static_cast<double>(o.margin)) / std::sin(std::atan(fit));
    // Camera at yaw 0, pitch 0 sits on +z looking at the centre.
    const Vec3 eye{centre.x + dist * std::cos(pitch) * std::sin(yaw), centre.y + dist * std::sin(pitch), centre.z + dist * std::cos(pitch) * std::cos(yaw)};
    const Vec3 forward = normalize(centre - eye), right = normalize(cross(forward, Vec3{0, 1, 0})), up = cross(right, forward);
    const double f = 1.0 / std::tan(fovY / 2), nearZ = std::max(dist - radius * 1.5, dist * 1e-3);
    // Light: from above-left of the camera.
    const Vec3 light = normalize(Vec3{-forward.x * 0.6 - right.x * 0.4 + up.x * 0.5, -forward.y * 0.6 - right.y * 0.4 + up.y * 0.5,
                                      -forward.z * 0.6 - right.z * 0.4 + up.z * 0.5});

    struct Projected {
        double sx, sy, invZ;  // screen position, 1 / view depth
        bool ok;
    };
    std::vector<Projected> pr(m.vertexCount());
    parallelFor(0, m.vertexCount(), 4096, [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            const Vec3 d = Vec3{m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]} - eye;
            const double z = dot(d, forward);
            if (z < nearZ) {
                pr[i] = {0, 0, 0, false};
                continue;
            }
            const double x = dot(d, right) * f / (z * aspect), y = dot(d, up) * f / z;
            pr[i] = {(x * 0.5 + 0.5) * static_cast<double>(W), (0.5 - y * 0.5) * static_cast<double>(H), 1.0 / z, true};
        }
    });
    const bool textured = !m.texture.pixels.empty() && !m.uvs.empty();
    const bool coloured = !m.colors.empty();
    std::vector<float> normals = m.normals;
    if (normals.empty()) {
        Mesh copy;
        copy.positions = m.positions;
        copy.indices = m.indices;
        computeNormals(copy);
        normals = std::move(copy.normals);
    }

    std::vector<float> rgb(W * H * 3);
    for (std::size_t i = 0; i < W * H; ++i)
        for (int c = 0; c < 3; ++c) rgb[3 * i + c] = o.background[c];
    std::vector<double> zbuf(W * H, 0.0);  // 1 / z: larger is nearer
    // Bands of rows in parallel; each rasterizes the triangles that touch it.
    const std::size_t band = std::max<std::size_t>(8, H / 64);
    parallelFor(0, (H + band - 1) / band, 1, [&](std::size_t b0, std::size_t b1) {
        for (std::size_t b = b0; b < b1; ++b) {
            const double yTop = static_cast<double>(b * band), yBottom = static_cast<double>(std::min(H, (b + 1) * band));
            for (std::size_t t = 0; t < m.triangleCount(); ++t) {
                const std::uint32_t ia = m.indices[3 * t], ib = m.indices[3 * t + 1], ic = m.indices[3 * t + 2];
                const Projected &A = pr[ia], &B = pr[ib], &C = pr[ic];
                if (!A.ok || !B.ok || !C.ok) continue;
                const double minY = std::min({A.sy, B.sy, C.sy}), maxY = std::max({A.sy, B.sy, C.sy});
                if (maxY < yTop || minY >= yBottom) continue;
                const double area = (B.sx - A.sx) * (C.sy - A.sy) - (B.sy - A.sy) * (C.sx - A.sx);
                if (std::abs(area) < 1e-12) continue;
                // (clamped as doubles first: far-off vertices project to huge coordinates)
                const long x0 = static_cast<long>(std::clamp(std::floor(std::min({A.sx, B.sx, C.sx})), 0.0, static_cast<double>(W)));
                const long x1 = static_cast<long>(std::clamp(std::ceil(std::max({A.sx, B.sx, C.sx})), -1.0, static_cast<double>(W) - 1));
                const long y0 = static_cast<long>(std::clamp(std::floor(minY), yTop, yBottom));
                const long y1 = static_cast<long>(std::clamp(std::ceil(maxY), yTop - 1, yBottom - 1));
                for (long y = y0; y <= y1; ++y)
                    for (long x = x0; x <= x1; ++x) {
                        const double px = static_cast<double>(x) + 0.5, py = static_cast<double>(y) + 0.5;
                        double w0 = ((B.sx - px) * (C.sy - py) - (B.sy - py) * (C.sx - px)) / area;
                        double w1 = ((C.sx - px) * (A.sy - py) - (C.sy - py) * (A.sx - px)) / area;
                        double w2 = 1.0 - w0 - w1;
                        if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                        const double invZ = w0 * A.invZ + w1 * B.invZ + w2 * C.invZ;
                        const std::size_t pix = static_cast<std::size_t>(y) * W + static_cast<std::size_t>(x);
                        if (invZ <= zbuf[pix]) continue;
                        zbuf[pix] = invZ;
                        // Perspective-correct weights.
                        const double p0 = w0 * A.invZ / invZ, p1 = w1 * B.invZ / invZ, p2 = w2 * C.invZ / invZ;
                        float col[3] = {200, 200, 200};
                        if (textured) {
                            const float u = static_cast<float>(p0 * m.uvs[2 * ia] + p1 * m.uvs[2 * ib] + p2 * m.uvs[2 * ic]);
                            const float v = static_cast<float>(p0 * m.uvs[2 * ia + 1] + p1 * m.uvs[2 * ib + 1] + p2 * m.uvs[2 * ic + 1]);
                            const auto s = sampleTexture(m.texture, u, v);
                            col[0] = s[0];
                            col[1] = s[1];
                            col[2] = s[2];
                        } else if (coloured) {
                            for (int c = 0; c < 3; ++c)
                                col[c] = static_cast<float>(p0 * m.colors[3 * ia + c] + p1 * m.colors[3 * ib + c] + p2 * m.colors[3 * ic + c]);
                        }
                        if (o.lighting) {
                            Vec3 n = normalize(Vec3{p0 * normals[3 * ia] + p1 * normals[3 * ib] + p2 * normals[3 * ic],
                                                    p0 * normals[3 * ia + 1] + p1 * normals[3 * ib + 1] + p2 * normals[3 * ic + 1],
                                                    p0 * normals[3 * ia + 2] + p1 * normals[3 * ib + 2] + p2 * normals[3 * ic + 2]});
                            if (dot(n, forward) > 0) n = {-n.x, -n.y, -n.z};  // back faces lit as front faces
                            const float shade = static_cast<float>(0.45 + 0.55 * std::max(0.0, dot(n, light)));
                            for (float& c : col) c *= shade;
                        }
                        for (int c = 0; c < 3; ++c) rgb[3 * pix + c] = col[c];
                    }
            }
        }
    });
    Image out;
    out.width = o.width;
    out.height = o.height;
    out.channels = 3;
    out.pixels.resize(o.width * o.height * 3);
    const float inv = 1.0f / static_cast<float>(ss * ss);
    for (std::size_t y = 0; y < o.height; ++y)
        for (std::size_t x = 0; x < o.width; ++x)
            for (int c = 0; c < 3; ++c) {
                float s = 0;
                for (std::size_t dy = 0; dy < ss; ++dy)
                    for (std::size_t dx = 0; dx < ss; ++dx) s += rgb[3 * ((y * ss + dy) * W + x * ss + dx) + c];
                out.pixels[3 * (y * o.width + x) + c] = toByte(s * inv);
            }
    return out;
}

std::vector<Image> renderTurntable(const Mesh& mesh, std::size_t frames, float degrees, const MeshRenderOptions& options) {
    std::vector<Image> out;
    MeshRenderOptions o = options;
    for (std::size_t i = 0; i < frames; ++i) {
        o.yaw = options.yaw + degrees * static_cast<float>(i) / static_cast<float>(std::max<std::size_t>(1, frames));
        out.push_back(renderMesh(mesh, o));
    }
    return out;
}

// ---- From depth ----

Mesh meshFromDepth(const Image& picture, const std::vector<float>& depth, const DepthMeshOptions& o) {
    const std::size_t W = picture.width, H = picture.height;
    if (W < 2 || H < 2 || depth.size() != W * H) throw std::invalid_argument("meshFromDepth: depth must have one value per pixel");
    const std::size_t grid = std::clamp<std::size_t>(o.grid, 2, 2048);
    const std::size_t step = std::max<std::size_t>(1, (std::max(W, H) - 1 + grid - 2) / (grid - 1));  // ceil((side - 1) / (grid - 1))
    const std::size_t gw = (W - 1 + step - 1) / step + 1, gh = (H - 1 + step - 1) / step + 1;
    float dmin = std::numeric_limits<float>::max(), dmax = -std::numeric_limits<float>::max();
    for (float d : depth) {
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
    }
    const float range = dmax - dmin > 1e-12f ? dmax - dmin : 1.0f;
    const float aspect = static_cast<float>(W) / static_cast<float>(H);
    Mesh m;
    m.texture = toRgb(picture);
    std::vector<float> dn(gw * gh);
    for (std::size_t gy = 0; gy < gh; ++gy)
        for (std::size_t gx = 0; gx < gw; ++gx) {
            const std::size_t px = std::min(W - 1, gx * step), py = std::min(H - 1, gy * step);
            const float d = (depth[py * W + px] - dmin) / range;
            dn[gy * gw + gx] = d;
            const float u = static_cast<float>(px) / static_cast<float>(W - 1), v = static_cast<float>(py) / static_cast<float>(H - 1);
            m.positions.insert(m.positions.end(), {(u - 0.5f) * aspect, 0.5f - v, d * o.depth});
            m.uvs.insert(m.uvs.end(), {(static_cast<float>(px) + 0.5f) / static_cast<float>(W), (static_cast<float>(py) + 0.5f) / static_cast<float>(H)});
        }
    auto keep = [&](std::size_t a, std::size_t b, std::size_t c) {
        if (o.edges <= 0) return true;
        const float lo = std::min({dn[a], dn[b], dn[c]}), hi = std::max({dn[a], dn[b], dn[c]});
        return hi - lo <= o.edges;
    };
    for (std::size_t gy = 0; gy + 1 < gh; ++gy)
        for (std::size_t gx = 0; gx + 1 < gw; ++gx) {
            const std::uint32_t a = static_cast<std::uint32_t>(gy * gw + gx), b = a + 1, c = a + static_cast<std::uint32_t>(gw), d = c + 1;
            // Counter-clockwise seen from the front (+z).
            if (keep(a, c, b)) m.indices.insert(m.indices.end(), {a, c, b});
            if (keep(b, c, d)) m.indices.insert(m.indices.end(), {b, c, d});
        }
    if (m.indices.empty()) throw std::invalid_argument("meshFromDepth: every triangle crossed a depth edge (raise \"edges\")");
    computeNormals(m);
    return m;
}

} // namespace nc
