#include "Animation.h"

#include "Json.h"
#include "Parallel.h"
#include "Transform.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
using xf::Mat4;
using xf::Quat;
using xf::Vec3;

Vec3 sub3(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 add3(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
double dot3(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Mat4 matOf(const float* m) {
    Mat4 r;
    for (int k = 0; k < 16; ++k) r[static_cast<std::size_t>(k)] = m[k];
    return r;
}

// World matrices of every joint for the given local transforms.
std::vector<Mat4> jointWorlds(const MeshRig& rig, const float* locals) {
    const std::size_t J = rig.jointCount();
    std::vector<Mat4> w(J);
    const Mat4 root = rig.rootParent.size() == 16 ? matOf(rig.rootParent.data()) : xf::identity();
    for (std::size_t j = 0; j < J; ++j) {
        const float* l = locals + 10 * j;
        const Mat4 local = xf::trs(l, l + 3, l + 7);
        w[j] = xf::mul(rig.parents[j] < 0 ? root : w[static_cast<std::size_t>(rig.parents[j])], local);
    }
    return w;
}

// The local transforms `seconds` into the animation (interpolated between frames).
std::vector<float> localsAt(const MeshRig& rig, double seconds) {
    const std::size_t J = rig.jointCount(), F = rig.frameCount();
    if (F == 0) return rig.rest;
    const double x = std::clamp(seconds * rig.fps, 0.0, static_cast<double>(F - 1));
    const std::size_t f0 = static_cast<std::size_t>(std::floor(x)), f1 = std::min(f0 + 1, F - 1);
    const double u = x - static_cast<double>(f0);
    std::vector<float> out(10 * J);
    const float* a = rig.frames.data() + f0 * J * 10;
    const float* b = rig.frames.data() + f1 * J * 10;
    for (std::size_t j = 0; j < J; ++j) {
        float* o = out.data() + 10 * j;
        const float* p = a + 10 * j;
        const float* q = b + 10 * j;
        for (int k : {0, 1, 2, 7, 8, 9}) o[k] = static_cast<float>(p[k] + u * (q[k] - p[k]));
        const Quat r = xf::slerp(xf::normalized({p[3], p[4], p[5], p[6]}), xf::normalized({q[3], q[4], q[5], q[6]}), u);
        for (int k = 0; k < 4; ++k) o[3 + k] = static_cast<float>(r[static_cast<std::size_t>(k)]);
    }
    return out;
}
} // namespace

std::vector<float> skinnedPositions(const MeshRig& rig, const float* locals) {
    const std::size_t J = rig.jointCount(), n = rig.bindPositions.size() / 3;
    if (rig.joints.size() != 4 * n || rig.weights.size() != 4 * n || rig.inverseBind.size() != 16 * J)
        throw std::invalid_argument("skinnedPositions: the rig's arrays disagree");
    const std::vector<Mat4> world = jointWorlds(rig, locals);
    std::vector<Mat4> skin(J);
    for (std::size_t j = 0; j < J; ++j) skin[j] = xf::mul(world[j], matOf(rig.inverseBind.data() + 16 * j));
    std::vector<float> out(rig.bindPositions);
    parallelFor(0, n, 8192, [&](std::size_t v0, std::size_t v1) {
        for (std::size_t v = v0; v < v1; ++v) {
            const float* w = rig.weights.data() + 4 * v;
            if (w[0] + w[1] + w[2] + w[3] <= 0) continue;
            const Vec3 p{rig.bindPositions[3 * v], rig.bindPositions[3 * v + 1], rig.bindPositions[3 * v + 2]};
            Vec3 acc{0, 0, 0};
            for (int k = 0; k < 4; ++k) {
                if (w[k] <= 0) continue;
                const std::uint16_t j = rig.joints[4 * v + static_cast<std::size_t>(k)];
                if (j >= J) continue;
                const Vec3 q = xf::apply(skin[j], p);
                for (int c = 0; c < 3; ++c) acc[static_cast<std::size_t>(c)] += w[k] * q[static_cast<std::size_t>(c)];
            }
            for (int c = 0; c < 3; ++c) out[3 * v + static_cast<std::size_t>(c)] = static_cast<float>(acc[static_cast<std::size_t>(c)]);
        }
    });
    return out;
}

Mesh poseMesh(const Mesh& mesh, double seconds) {
    Mesh out;
    out.positions = mesh.rig.empty() ? mesh.positions : skinnedPositions(mesh.rig, localsAt(mesh.rig, seconds).data());
    out.uvs = mesh.uvs;
    out.colors = mesh.colors;
    out.indices = mesh.indices;
    out.texture = mesh.texture;
    if (mesh.rig.empty()) out.normals = mesh.normals;
    else computeNormals(out);
    return out;
}

void animationBounds(const Mesh& mesh, float centre[3], float& radius) {
    std::vector<std::vector<float>> poses;
    const std::size_t F = mesh.rig.frameCount();
    if (F > 1)
        for (std::size_t s = 0; s < 24; ++s)
            poses.push_back(skinnedPositions(mesh.rig, localsAt(mesh.rig, mesh.rig.duration() * static_cast<double>(s) / 23.0).data()));
    else poses.push_back(mesh.positions);
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (const auto& p : poses)
        for (std::size_t i = 0; i + 2 < p.size(); i += 3)
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], static_cast<double>(p[i + static_cast<std::size_t>(k)]));
                hi[k] = std::max(hi[k], static_cast<double>(p[i + static_cast<std::size_t>(k)]));
            }
    double r = 1e-9;
    for (int k = 0; k < 3; ++k) centre[k] = static_cast<float>((lo[k] + hi[k]) / 2);
    for (const auto& p : poses)
        for (std::size_t i = 0; i + 2 < p.size(); i += 3) {
            double d = 0;
            for (int k = 0; k < 3; ++k) d += (p[i + static_cast<std::size_t>(k)] - centre[k]) * (p[i + static_cast<std::size_t>(k)] - centre[k]);
            r = std::max(r, std::sqrt(d));
        }
    radius = static_cast<float>(r);
}

// ---- BVH ----

bool isMotionFile(const std::string& path) {
    std::string e = std::filesystem::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e == ".bvh";
}

MotionCapture parseBvh(const std::string& text, const std::string& name) {
    auto fail = [&](const std::string& why) -> void { throw std::runtime_error("parseBvh: " + name + ": " + why); };
    std::istringstream in(text);
    std::string tok;
    auto next = [&]() -> std::string {
        if (!(in >> tok)) fail("unexpected end of file");
        return tok;
    };
    auto number = [&]() {
        const std::string t = next();
        char* end = nullptr;
        const double v = std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end || !std::isfinite(v) || std::fabs(v) > 1e9) fail("bad number \"" + t.substr(0, 20) + "\"");
        return v;
    };
    MotionCapture m;
    // channels[j]: codes 0..2 = X/Y/Z position, 3..5 = X/Y/Z rotation
    std::vector<std::vector<int>> channels;
    if (next() != "HIERARCHY") fail("no HIERARCHY");
    std::vector<int> stack;
    std::size_t totalChannels = 0;
    if (next() != "ROOT") fail("no ROOT");
    std::function<void(int, const std::string&, bool)> joint = [&](int parent, const std::string& jname, bool end) {
        if (m.parents.size() >= 1024) fail("too many joints");
        if (stack.size() > 128) fail("hierarchy too deep");
        const int self = static_cast<int>(m.parents.size());
        m.names.push_back(jname);
        m.parents.push_back(parent);
        m.offsets.insert(m.offsets.end(), {0, 0, 0});
        channels.emplace_back();
        if (next() != "{") fail("expected {");
        stack.push_back(self);
        for (;;) {
            const std::string t = next();
            if (t == "}") break;
            if (t == "OFFSET") {
                for (int k = 0; k < 3; ++k) m.offsets[3 * static_cast<std::size_t>(self) + static_cast<std::size_t>(k)] = static_cast<float>(number());
            } else if (t == "CHANNELS" && !end) {
                const double n = number();
                if (!(n >= 0 && n <= 6) || n != std::floor(n)) fail("bad channel count");
                for (int k = 0; k < static_cast<int>(n); ++k) {
                    const std::string c = next();
                    static const char* codes[] = {"Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"};
                    int code = -1;
                    for (int q = 0; q < 6; ++q)
                        if (c == codes[q]) code = q;
                    if (code < 0) fail("unknown channel " + c.substr(0, 20));
                    channels[static_cast<std::size_t>(self)].push_back(code);
                }
                totalChannels += static_cast<std::size_t>(n);
            } else if (t == "JOINT" && !end) {
                joint(self, next(), false);
            } else if (t == "End" && !end) {
                next();  // "Site"
                joint(self, jname + "_end", true);
            } else fail("unexpected \"" + t.substr(0, 20) + "\"");
        }
        stack.pop_back();
    };
    joint(-1, next(), false);
    if (next() != "MOTION") fail("no MOTION");
    if (next() != "Frames:") fail("no Frames:");
    const double framesD = number();
    if (!(framesD >= 1 && framesD <= 1e6) || framesD != std::floor(framesD)) fail("bad frame count");
    if (next() != "Frame" || next() != "Time:") fail("no Frame Time:");
    const double dt = number();
    if (!(dt > 1e-5 && dt < 10)) fail("bad frame time");
    m.fps = 1.0 / dt;
    const std::size_t F = static_cast<std::size_t>(framesD), J = m.parents.size();
    if (F * J * 4 > (std::size_t(1) << 26)) fail("too much motion");
    m.rotations.resize(F * J * 4);
    m.rootPositions.resize(F * 3);
    for (std::size_t f = 0; f < F; ++f) {
        for (std::size_t j = 0; j < J; ++j) {
            Vec3 pos{m.offsets[3 * j], m.offsets[3 * j + 1], m.offsets[3 * j + 2]};
            Quat q{0, 0, 0, 1};
            for (int code : channels[j]) {
                const double v = number();
                if (code < 3) pos[static_cast<std::size_t>(code)] = v;
                else {
                    Vec3 axis{0, 0, 0};
                    axis[static_cast<std::size_t>(code - 3)] = 1;
                    q = xf::qmul(q, xf::axisAngle(axis, v * 3.14159265358979323846 / 180.0));  // rotations compose in channel order
                }
            }
            q = xf::normalized(q);
            for (int k = 0; k < 4; ++k) m.rotations[(f * J + j) * 4 + static_cast<std::size_t>(k)] = static_cast<float>(q[static_cast<std::size_t>(k)]);
            if (j == 0)
                for (int k = 0; k < 3; ++k) m.rootPositions[3 * f + static_cast<std::size_t>(k)] = static_cast<float>(pos[static_cast<std::size_t>(k)]);
        }
    }
    (void)totalChannels;
    return m;
}

MotionCapture loadBvh(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("loadBvh: cannot open " + path);
    std::ostringstream s;
    s << f.rdbuf();
    if (s.str().size() > (std::size_t(1) << 28)) throw std::runtime_error("loadBvh: " + path + " is too large");
    return parseBvh(s.str(), path);
}

// ---- Retargeting ----

namespace {

// The limbs of a standing skeleton from its joints' rest positions (+Y up, facing +Z, left at +X).
struct Limbs {
    std::vector<int> spine;  // root .. head end
    std::vector<int> leg[2], arm[2];  // [0] left, [1] right: branch on the spine .. end
};

Limbs findLimbs(const std::vector<Vec3>& P, const std::vector<int>& parents, const std::string& what) {
    const std::size_t J = P.size();
    std::vector<int> kids(J, 0);
    for (int p : parents)
        if (p >= 0) ++kids[static_cast<std::size_t>(p)];
    double loY = 1e30, hiY = -1e30, loX = 1e30, hiX = -1e30;
    for (const Vec3& p : P) {
        loY = std::min(loY, p[1]);
        hiY = std::max(hiY, p[1]);
        loX = std::min(loX, p[0]);
        hiX = std::max(hiX, p[0]);
    }
    const double H = std::max(hiY - loY, 1e-9), cx = (loX + hiX) / 2;
    auto pathTo = [&](int leaf) {
        std::vector<int> path;
        for (int j = leaf; j >= 0 && path.size() <= J; j = parents[static_cast<std::size_t>(j)]) path.push_back(j);
        std::reverse(path.begin(), path.end());
        return path;
    };
    Limbs L;
    // The head: the highest end joint above the root (a raised hand can be higher, but not over it).
    const std::size_t root0 = [&] {
        std::size_t r = 0;
        while (r < J && parents[r] >= 0) ++r;
        return r < J ? r : 0;
    }();
    int head = -1;
    for (int pass = 0; pass < 2 && head < 0; ++pass)
        for (std::size_t j = 0; j < J; ++j)
            if (!kids[j] && (pass == 1 || std::fabs(P[j][0] - P[root0][0]) < 0.12 * H) &&
                (head < 0 || P[j][1] > P[static_cast<std::size_t>(head)][1]))
                head = static_cast<int>(j);
    if (head < 0) throw std::invalid_argument(what + ": no joints");
    L.spine = pathTo(head);
    if (L.spine.size() < 2) throw std::invalid_argument(what + ": no spine (the highest end joint hangs from the root)");
    std::vector<int> spineAt(J, -1);
    for (std::size_t i = 0; i < L.spine.size(); ++i) spineAt[static_cast<std::size_t>(L.spine[i])] = static_cast<int>(i);
    struct Candidate {
        std::vector<int> chain;
        int branch = -1;  // index on the spine
    };
    std::vector<Candidate> cands;
    for (std::size_t j = 0; j < J; ++j) {
        if (kids[j] || spineAt[j] >= 0) continue;
        Candidate c;
        int b = static_cast<int>(j);
        std::vector<int> chain;
        while (b >= 0 && spineAt[static_cast<std::size_t>(b)] < 0) {
            chain.push_back(b);
            b = parents[static_cast<std::size_t>(b)];
        }
        if (b < 0) continue;  // another root
        chain.push_back(b);
        std::reverse(chain.begin(), chain.end());
        if (chain.size() < 3) continue;  // at least two bones
        c.chain = chain;
        c.branch = spineAt[static_cast<std::size_t>(b)];
        cands.push_back(c);
    }
    // Legs: chains that drop well below their branch, the longest drop on each side.
    double best[2] = {0, 0};
    int legBranch = -1;
    for (const Candidate& c : cands) {
        const Vec3& top = P[static_cast<std::size_t>(c.chain.front())];
        const Vec3& end = P[static_cast<std::size_t>(c.chain.back())];
        const double drop = top[1] - end[1];
        if (drop < 0.25 * H || end[1] > loY + 0.3 * H) continue;
        const int side = end[0] >= cx ? 0 : 1;
        if (drop > best[side]) {
            best[side] = drop;
            L.leg[side] = c.chain;
        }
    }
    if (L.leg[0].empty() || L.leg[1].empty())
        throw std::invalid_argument(what + ": two legs not found (it needs a figure standing upright, +Y up, legs apart; a base or props "
                                           "around it can leave its rig without legs)");
    legBranch = std::max(spineAt[static_cast<std::size_t>(L.leg[0].front())], spineAt[static_cast<std::size_t>(L.leg[1].front())]);
    // Arms: chains from higher on the spine reaching furthest sideways.
    double reach[2] = {0.1 * H, 0.1 * H};
    for (const Candidate& c : cands) {
        if (c.branch <= legBranch || c.chain == L.leg[0] || c.chain == L.leg[1]) continue;
        const Vec3& top = P[static_cast<std::size_t>(c.chain.front())];
        const Vec3& end = P[static_cast<std::size_t>(c.chain.back())];
        const double side = end[0] - top[0];
        const int s = side >= 0 ? 0 : 1;
        if (std::fabs(side) > reach[s]) {
            reach[s] = std::fabs(side);
            L.arm[s] = c.chain;
        }
    }
    return L;
}

// Where each point of a chain lies along it: 0 at its start, 1 at its end.
std::vector<double> along(const std::vector<Vec3>& P, const std::vector<int>& chain) {
    const Vec3 a = P[static_cast<std::size_t>(chain.front())], axis = sub3(P[static_cast<std::size_t>(chain.back())], a);
    const double ll = std::max(dot3(axis, axis), 1e-18);
    std::vector<double> t;
    for (int j : chain) t.push_back(dot3(sub3(P[static_cast<std::size_t>(j)], a), axis) / ll);
    return t;
}

// For each bone of `target` (joint i -> i + 1), the bone of `source` at the same place along the limb.
void matchChain(const std::vector<Vec3>& Pt, const std::vector<int>& target, const std::vector<Vec3>& Ps, const std::vector<int>& source,
                std::map<int, std::pair<int, int>>& match, std::map<int, int>& childOf, bool skipFirst) {
    // (skipFirst: the chain's first joint is matched elsewhere)
    const std::vector<double> tt = along(Pt, target), ts = along(Ps, source);
    for (std::size_t i = skipFirst ? 1 : 0; i + 1 < target.size(); ++i) {
        const Vec3 d = sub3(Pt[static_cast<std::size_t>(target[i + 1])], Pt[static_cast<std::size_t>(target[i])]);
        if (dot3(d, d) < 1e-14) continue;
        const double mid = (tt[i] + tt[i + 1]) / 2;
        int bestK = -1;
        double bestGap = 1e30;
        for (std::size_t k = 0; k + 1 < source.size(); ++k) {
            const double a = ts[k], b = ts[k + 1];
            if (b - a < 1e-4) continue;  // degenerate or backwards bones
            const double gap = mid < a ? a - mid : mid > b ? mid - b : 0;
            if (gap < bestGap) {
                bestGap = gap;
                bestK = static_cast<int>(k);
            }
        }
        if (bestK < 0) continue;
        match[target[i]] = {source[static_cast<std::size_t>(bestK)], source[static_cast<std::size_t>(bestK) + 1]};
        childOf[target[i]] = target[i + 1];
    }
}

void setMember(json::Value& obj, const std::string& key, json::Value v) {
    for (auto& kv : obj.object)
        if (kv.first == key) {
            kv.second = std::move(v);
            return;
        }
    obj.object.emplace_back(key, std::move(v));
}

std::uint32_t get32(const std::uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<std::uint32_t>(p[3]) << 24); }
void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int k = 0; k < 4; ++k) b.push_back(static_cast<std::uint8_t>(v >> (8 * k)));
}

// The glTF binary with its animations replaced by one: rotation tracks for `rotNodes` (frames x 4 each)
// and a translation track for `transNode` (frames x 3), sampled at `fps`.
std::vector<std::uint8_t> withAnimation(const std::vector<std::uint8_t>& glb, double fps, std::size_t frames, const std::vector<int>& rotNodes,
                                        const std::vector<std::vector<float>>& rotations, int transNode, const std::vector<float>& translations) {
    if (glb.size() < 20 || get32(glb.data()) != 0x46546C67) throw std::invalid_argument("animate: the model's glTF is missing");
    const std::size_t jsonLen = get32(glb.data() + 12);
    if (20 + jsonLen > glb.size()) throw std::invalid_argument("animate: bad glTF");
    json::Value doc = json::parse(std::string(glb.begin() + 20, glb.begin() + 20 + static_cast<long>(jsonLen)));
    std::vector<std::uint8_t> bin;
    const std::size_t at = 20 + ((jsonLen + 3) & ~std::size_t(3));
    if (at + 8 <= glb.size() && get32(glb.data() + at + 4) == 0x004E4942) {
        const std::size_t n = get32(glb.data() + at);
        if (at + 8 + n > glb.size()) throw std::invalid_argument("animate: truncated glTF");
        bin.assign(glb.begin() + static_cast<long>(at + 8), glb.begin() + static_cast<long>(at + 8 + n));
    }
    auto list = [&](const char* key) -> json::Value& {
        for (auto& kv : doc.object)
            if (kv.first == key) {
                if (!kv.second.isArray()) kv.second = json::makeArray();
                return kv.second;
            }
        doc.object.emplace_back(key, json::makeArray());
        return doc.object.back().second;
    };
    auto addData = [&](const std::vector<float>& data, std::size_t count, const char* type, bool minmax) {
        while (bin.size() % 4) bin.push_back(0);
        const std::size_t offset = bin.size();
        bin.resize(offset + data.size() * 4);
        std::memcpy(bin.data() + offset, data.data(), data.size() * 4);
        json::Value& views = list("bufferViews");
        views.array.push_back(json::makeObject({{"buffer", json::makeNumber(0)}, {"byteOffset", json::makeNumber(static_cast<double>(offset))},
                                                {"byteLength", json::makeNumber(static_cast<double>(data.size() * 4))}}));
        json::Value acc = json::makeObject({{"bufferView", json::makeNumber(static_cast<double>(views.array.size() - 1))},
                                            {"componentType", json::makeNumber(5126)}, {"count", json::makeNumber(static_cast<double>(count))},
                                            {"type", json::makeString(type)}});
        if (minmax) {
            const auto [lo, hi] = std::minmax_element(data.begin(), data.end());
            setMember(acc, "min", json::makeArray({json::makeNumber(*lo)}));
            setMember(acc, "max", json::makeArray({json::makeNumber(*hi)}));
        }
        json::Value& accs = list("accessors");
        accs.array.push_back(acc);
        return static_cast<double>(accs.array.size() - 1);
    };
    std::vector<float> times(frames);
    for (std::size_t f = 0; f < frames; ++f) times[f] = static_cast<float>(static_cast<double>(f) / fps);
    const double input = addData(times, frames, "SCALAR", true);
    json::Value samplers = json::makeArray(), channels = json::makeArray();
    auto track = [&](int node, const char* path, const std::vector<float>& values, const char* type) {
        const double out = addData(values, frames, type, false);
        samplers.array.push_back(json::makeObject({{"input", json::makeNumber(input)}, {"output", json::makeNumber(out)}, {"interpolation", json::makeString("LINEAR")}}));
        channels.array.push_back(json::makeObject({{"sampler", json::makeNumber(static_cast<double>(samplers.array.size() - 1))},
                                                   {"target", json::makeObject({{"node", json::makeNumber(node)}, {"path", json::makeString(path)}})}}));
    };
    for (std::size_t i = 0; i < rotNodes.size(); ++i) track(rotNodes[i], "rotation", rotations[i], "VEC4");
    if (transNode >= 0) track(transNode, "translation", translations, "VEC3");
    json::Value anims = json::makeArray({json::makeObject({{"name", json::makeString("motion")}, {"samplers", samplers}, {"channels", channels}})});
    setMember(doc, "animations", anims);
    json::Value& buffers = list("buffers");
    if (buffers.array.empty()) buffers.array.push_back(json::makeObject());
    setMember(buffers.array[0], "byteLength", json::makeNumber(static_cast<double>(bin.size())));
    while (bin.size() % 4) bin.push_back(0);
    std::string js = json::dump(doc);
    while (js.size() % 4) js.push_back(' ');
    std::vector<std::uint8_t> out;
    put32(out, 0x46546C67);
    put32(out, 2);
    put32(out, static_cast<std::uint32_t>(12 + 8 + js.size() + 8 + bin.size()));
    put32(out, static_cast<std::uint32_t>(js.size()));
    put32(out, 0x4E4F534A);
    out.insert(out.end(), js.begin(), js.end());
    put32(out, static_cast<std::uint32_t>(bin.size()));
    put32(out, 0x004E4942);
    out.insert(out.end(), bin.begin(), bin.end());
    return out;
}

} // namespace

Mesh animateMesh(const Mesh& rigged, const MotionCapture& motion, const AnimateOptions& o, std::vector<std::string>* report) {
    const MeshRig& rig = rigged.rig;
    if (rig.empty()) throw std::invalid_argument("animate: the model has no skeleton (rig it first)");
    if (rigged.glb.empty()) throw std::invalid_argument("animate: the model's glTF is missing");
    if (motion.frameCount() == 0) throw std::invalid_argument("animate: the motion has no frames");
    if (!(o.fps > 0 && o.fps <= 120) || !(o.speed > 0.01 && o.speed <= 100) || !(o.start >= 0) || !(o.seconds >= 0))
        throw std::invalid_argument("animate: fps, speed, start or seconds out of range");
    const std::size_t J = rig.jointCount(), S = motion.jointCount();

    // Target rest: world positions and rotations.
    const std::vector<Mat4> restWorld = jointWorlds(rig, rig.rest.data());
    std::vector<Vec3> Pt(J);
    for (std::size_t j = 0; j < J; ++j) Pt[j] = {restWorld[j][12], restWorld[j][13], restWorld[j][14]};
    double rpT[3], rpS[3];
    Quat rpQ;
    xf::decompose(rig.rootParent.size() == 16 ? matOf(rig.rootParent.data()) : xf::identity(), rpT, rpQ, rpS);
    std::vector<Quat> restLocal(J), restGlobal(J);
    for (std::size_t j = 0; j < J; ++j) {
        const float* l = rig.rest.data() + 10 * j;
        restLocal[j] = xf::normalized({l[3], l[4], l[5], l[6]});
        restGlobal[j] = xf::qmul(rig.parents[j] < 0 ? rpQ : restGlobal[static_cast<std::size_t>(rig.parents[j])], restLocal[j]);
    }
    // Source rest (all rotations zero): positions from the offsets.
    std::vector<Vec3> Ps(S);
    for (std::size_t j = 0; j < S; ++j) {
        const Vec3 off{motion.offsets[3 * j], motion.offsets[3 * j + 1], motion.offsets[3 * j + 2]};
        Ps[j] = motion.parents[j] < 0 ? off : add3(Ps[static_cast<std::size_t>(motion.parents[j])], off);
    }
    const Limbs lt = findLimbs(Pt, rig.parents, "animate: the model"), ls = findLimbs(Ps, motion.parents, "animate: the motion");
    std::map<int, std::pair<int, int>> match;  // target joint -> source bone (a, b)
    std::map<int, int> childOf;                // target joint -> the next joint of its limb
    // The root moves as a whole (below); a limb's first joint is on the spine and turns with it, so
    // matching starts at each chain's second joint.
    matchChain(Pt, lt.spine, Ps, ls.spine, match, childOf, true);
    for (int s = 0; s < 2; ++s) {
        matchChain(Pt, lt.leg[s], Ps, ls.leg[s], match, childOf, true);
        if (!lt.arm[s].empty() && !ls.arm[s].empty()) matchChain(Pt, lt.arm[s], Ps, ls.arm[s], match, childOf, true);
    }
    const int rootT = lt.spine.front(), rootS = ls.spine.front();
    if (report) {
        auto names = [&](const std::vector<int>& c) {
            std::string s;
            for (int j : c) s += (s.empty() ? "" : " > ") + rig.names[static_cast<std::size_t>(j)];
            return s;
        };
        report->push_back("spine: " + names(lt.spine));
        for (int s = 0; s < 2; ++s) {
            report->push_back(std::string(s ? "right" : "left") + " leg: " + names(lt.leg[s]));
            report->push_back(std::string(s ? "right" : "left") + " arm: " + (lt.arm[s].empty() ? std::string("(none)") : names(lt.arm[s])));
        }
        report->push_back(std::to_string(match.size() + 1) + " of " + std::to_string(J) + " joints follow the motion");
    }

    // Source frames used: a leading frame that jumps to the next (CMU's added T-pose) is skipped.
    const std::size_t SF = motion.frameCount();
    auto change = [&](std::size_t a, std::size_t b) {  // mean rotation between two frames, radians
        double sum = 0;
        for (std::size_t j = 0; j < S; ++j) {
            const float* p = motion.rotations.data() + (a * S + j) * 4;
            const float* q = motion.rotations.data() + (b * S + j) * 4;
            sum += 2 * std::acos(std::min(1.0, std::fabs(static_cast<double>(p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]))));
        }
        return sum / static_cast<double>(S);
    };
    std::size_t first = 0;
    while (first + 2 < SF && first < 4 && change(first, first + 1) > std::max(0.05, 8 * change(first + 1, first + 2))) ++first;
    const double srcStart = static_cast<double>(first) / motion.fps + o.start;
    const double srcEnd = static_cast<double>(SF - 1) / motion.fps;
    if (srcStart >= srcEnd) throw std::invalid_argument("animate: start is past the end of the motion");
    const double srcSpan = o.seconds > 0 ? std::min(o.seconds * o.speed, srcEnd - srcStart) : srcEnd - srcStart;
    const std::size_t F = std::min<std::size_t>(static_cast<std::size_t>(std::floor(srcSpan / o.speed * o.fps + 1e-9)) + 1, 18000);

    // Source pose at a time: global rotations and positions.
    std::vector<Quat> Gs(S);
    std::vector<Vec3> Pf(S);
    auto sourcePose = [&](double t) {
        const double x = std::clamp(t * motion.fps, 0.0, static_cast<double>(SF - 1));
        const std::size_t f0 = static_cast<std::size_t>(std::floor(x)), f1 = std::min(f0 + 1, SF - 1);
        const double u = x - static_cast<double>(f0);
        for (std::size_t j = 0; j < S; ++j) {
            const float* a = motion.rotations.data() + (f0 * S + j) * 4;
            const float* b = motion.rotations.data() + (f1 * S + j) * 4;
            const Quat q = xf::slerp({a[0], a[1], a[2], a[3]}, {b[0], b[1], b[2], b[3]}, u);
            const int p = motion.parents[j];
            const Vec3 off{motion.offsets[3 * j], motion.offsets[3 * j + 1], motion.offsets[3 * j + 2]};
            if (p < 0) {
                Gs[j] = q;
                for (int k = 0; k < 3; ++k)
                    Pf[j][static_cast<std::size_t>(k)] = motion.rootPositions[3 * f0 + static_cast<std::size_t>(k)] +
                                                         u * (motion.rootPositions[3 * f1 + static_cast<std::size_t>(k)] - motion.rootPositions[3 * f0 + static_cast<std::size_t>(k)]);
            } else {
                Gs[j] = xf::qmul(Gs[static_cast<std::size_t>(p)], q);
                Pf[j] = add3(Pf[static_cast<std::size_t>(p)], xf::rotate(Gs[static_cast<std::size_t>(p)], off));
            }
        }
    };
    // Heights: the hips above the feet, at rest on both skeletons and over the motion for the ground.
    auto lowestFoot = [](const std::vector<Vec3>& P, const Limbs& l) {
        return std::min(P[static_cast<std::size_t>(l.leg[0].back())][1], P[static_cast<std::size_t>(l.leg[1].back())][1]);
    };
    const double groundT = lowestFoot(Pt, lt), hipT = Pt[static_cast<std::size_t>(rootT)][1] - groundT;
    const double hipS = Ps[static_cast<std::size_t>(rootS)][1] - lowestFoot(Ps, ls);
    double groundS = 1e30;
    for (std::size_t f = 0; f < F; ++f) {
        sourcePose(srcStart + static_cast<double>(f) * o.speed / o.fps);
        groundS = std::min(groundS, lowestFoot(Pf, ls));
    }
    if (!(hipS > 1e-9) || !(hipT > 1e-9)) throw std::invalid_argument("animate: the hips are not above the feet");
    const double scale = hipT / hipS;

    std::vector<int> animated;  // target joints with rotation tracks
    animated.push_back(rootT);
    for (const auto& kv : match) animated.push_back(kv.first);
    std::vector<std::vector<float>> rotTracks(animated.size(), std::vector<float>(F * 4));
    std::vector<float> rootTrack(F * 3);
    std::vector<Quat> G(J);
    Vec3 rootStart{0, 0, 0};
    for (std::size_t f = 0; f < F; ++f) {
        sourcePose(srcStart + static_cast<double>(f) * o.speed / o.fps);
        if (f == 0) rootStart = Pf[static_cast<std::size_t>(rootS)];
        for (std::size_t j = 0; j < J; ++j) {
            const int p = rig.parents[j];
            const Quat parentG = p < 0 ? rpQ : G[static_cast<std::size_t>(p)];
            if (static_cast<int>(j) == rootT) {
                G[j] = xf::normalized(xf::qmul(Gs[static_cast<std::size_t>(rootS)], restGlobal[j]));
            } else if (auto m = match.find(static_cast<int>(j)); m != match.end()) {
                const Quat delta = Gs[static_cast<std::size_t>(m->second.first)];
                const Vec3 restDir = sub3(Pt[static_cast<std::size_t>(childOf[static_cast<int>(j)])], Pt[j]);
                const Vec3 srcDir = sub3(Pf[static_cast<std::size_t>(m->second.second)], Pf[static_cast<std::size_t>(m->second.first)]);
                const Quat fix = xf::between(xf::rotate(delta, restDir), srcDir);
                G[j] = xf::normalized(xf::qmul(fix, xf::qmul(delta, restGlobal[j])));
            } else {
                G[j] = xf::normalized(xf::qmul(parentG, restLocal[j]));
            }
        }
        for (std::size_t i = 0; i < animated.size(); ++i) {
            const auto j = static_cast<std::size_t>(animated[i]);
            const int p = rig.parents[j];
            Quat local = xf::normalized(xf::qmul(xf::conj(p < 0 ? rpQ : G[static_cast<std::size_t>(p)]), G[j]));
            if (f > 0) {  // keep neighbouring frames on the same hemisphere (smooth interpolation)
                const float* prev = rotTracks[i].data() + (f - 1) * 4;
                if (prev[0] * local[0] + prev[1] * local[1] + prev[2] * local[2] + prev[3] * local[3] < 0)
                    for (double& x : local) x = -x;
            }
            for (int k = 0; k < 4; ++k) rotTracks[i][f * 4 + static_cast<std::size_t>(k)] = static_cast<float>(local[static_cast<std::size_t>(k)]);
        }
        // The hips: the motion's travel (unless in place) and height, scaled, in the root's parent's frame.
        const Vec3 src = Pf[static_cast<std::size_t>(rootS)];
        Vec3 world = Pt[static_cast<std::size_t>(rootT)];
        if (!o.inPlace) {
            world[0] += (src[0] - rootStart[0]) * scale;
            world[2] += (src[2] - rootStart[2]) * scale;
        }
        world[1] = groundT + (src[1] - groundS) * scale;
        Vec3 local;
        if (rig.parents[static_cast<std::size_t>(rootT)] < 0) {
            const Vec3 d = xf::rotate(xf::conj(rpQ), Vec3{world[0] - rpT[0], world[1] - rpT[1], world[2] - rpT[2]});
            for (int k = 0; k < 3; ++k) local[static_cast<std::size_t>(k)] = d[static_cast<std::size_t>(k)] / (std::fabs(rpS[k]) > 1e-12 ? rpS[k] : 1.0);
        } else {
            const float* l = rig.rest.data() + 10 * static_cast<std::size_t>(rootT);
            local = {l[0], l[1], l[2]};
        }
        for (int k = 0; k < 3; ++k) rootTrack[f * 3 + static_cast<std::size_t>(k)] = static_cast<float>(local[static_cast<std::size_t>(k)]);
    }
    std::vector<int> nodes;
    for (int j : animated) nodes.push_back(rig.nodes[static_cast<std::size_t>(j)]);
    const std::vector<std::uint8_t> glb = withAnimation(rigged.glb, o.fps, F, nodes, rotTracks, rig.nodes[static_cast<std::size_t>(rootT)], rootTrack);
    Mesh out = decodeGlb(glb, "the animated model");
    out.glb = glb;
    return out;
}

} // namespace nc
