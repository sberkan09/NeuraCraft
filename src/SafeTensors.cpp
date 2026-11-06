#include "SafeTensors.h"
#include "Checkpoint.h"
#include "Profile.h"
#include "Json.h"
#include "Parallel.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {

std::uint64_t readLe64(const unsigned char* b) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
    return v;
}

// fp16 -> float for a whole array, without branches, so the compiler vectorizes it (the per-value
// version normalizes subnormals in a loop). Normal values get their exponent rebased; infinities and
// NaNs keep an all-ones exponent; subnormals come out of a float subtraction, exactly:
// (2^-14 + m 2^-24) - 2^-14 = m 2^-24.
void halvesToFloats(const std::uint16_t* in, float* out, std::size_t n) {
    constexpr std::uint32_t kMagic = 113u << 23;  // 2^-14 as a float's bits
    float magic;
    std::memcpy(&magic, &kMagic, sizeof(magic));
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t h = in[i];
        const std::uint32_t em = (h & 0x7FFFu) << 13;  // exponent and mantissa in a float's place
        const std::uint32_t exponent = em & 0x0F800000u;
        const std::uint32_t normal = em + ((127u - 15u) << 23);
        const std::uint32_t special = normal + ((128u - 16u) << 23);
        const std::uint32_t subBits = em + kMagic;
        float sub;
        std::memcpy(&sub, &subBits, sizeof(sub));
        sub -= magic;
        std::uint32_t subnormal;
        std::memcpy(&subnormal, &sub, sizeof(subnormal));
        std::uint32_t bits = exponent == 0x0F800000u ? special : normal;
        bits = exponent == 0 ? subnormal : bits;
        bits |= (h & 0x8000u) << 16;
        std::memcpy(out + i, &bits, sizeof(float));
    }
}

float bf16ToFloat(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::size_t dtypeSize(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 1;
    return 0;
}

} // namespace

std::size_t SafeTensors::Tensor::elements() const {
    std::size_t n = 1;
    for (std::size_t d : shape) n *= d;
    return n;
}

SafeTensors::SafeTensors(const std::string& path) : path_(path) {
    if (const std::size_t hash = path.rfind('#'); hash != std::string::npos && !std::ifstream(path)) {
        const SafeTensors file(path.substr(0, hash));
        *this = file.renamed(checkpointRenames(file, path.substr(hash + 1)));
        return;
    }
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("SafeTensors: cannot open " + path);
    fileSize_ = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    unsigned char lenBytes[8];
    if (fileSize_ < 8 || !in.read(reinterpret_cast<char*>(lenBytes), 8))
        throw std::runtime_error("SafeTensors: " + path + " is too short");
    const std::uint64_t headerLen = readLe64(lenBytes);
    if (headerLen > fileSize_ - 8 || headerLen > (std::uint64_t(100) << 20))
        throw std::runtime_error("SafeTensors: bad header length in " + path);
    std::string header(static_cast<std::size_t>(headerLen), '\0');
    if (!in.read(&header[0], static_cast<std::streamsize>(headerLen)))
        throw std::runtime_error("SafeTensors: truncated header in " + path);
    dataStart_ = 8 + static_cast<std::size_t>(headerLen);

    const json::Value root = json::parse(header);
    if (!root.isObject()) throw std::runtime_error("SafeTensors: header is not a JSON object");
    for (const auto& [name, v] : root.object) {
        if (name == "__metadata__") {
            for (const auto& [k, mv] : v.object)
                if (mv.isString()) metadata_[k] = mv.string;
            continue;
        }
        // Every number must be a whole, non-negative count within the file (the header is untrusted).
        auto count = [&](const json::Value& x) {
            if (!x.isNumber() || !(x.number >= 0) || x.number > 9007199254740992.0 || x.number != std::floor(x.number))
                throw std::runtime_error("SafeTensors: bad number in the header for " + name);
            return static_cast<std::size_t>(x.number);
        };
        if (!v.isObject() || !v.find("dtype") || !v.at("dtype").isString() || !v.find("shape") || !v.at("shape").isArray() || !v.find("data_offsets"))
            throw std::runtime_error("SafeTensors: bad entry for " + name);
        Tensor t;
        t.dtype = v.at("dtype").string;
        std::size_t elements = 1;
        bool overflow = false;
        for (const auto& d : v.at("shape").array) {
            t.shape.push_back(count(d));
            if (t.shape.back() && elements > std::numeric_limits<std::size_t>::max() / t.shape.back()) overflow = true;
            elements *= t.shape.back();
        }
        const auto& offsets = v.at("data_offsets").array;
        if (offsets.size() != 2) throw std::runtime_error("SafeTensors: bad data_offsets for " + name);
        t.begin = count(offsets[0]);
        t.end = count(offsets[1]);
        const std::size_t size = dtypeSize(t.dtype);
        if (overflow || (size && elements > std::numeric_limits<std::size_t>::max() / size))
            throw std::runtime_error("SafeTensors: tensor " + name + " is too big");
        if (t.end < t.begin || t.end > fileSize_ - dataStart_ || (size && t.end - t.begin != elements * size))
            throw std::runtime_error("SafeTensors: inconsistent size for tensor " + name);
        tensors_.emplace(name, std::move(t));
    }
}

SafeTensors SafeTensors::renamed(const std::vector<Rename>& entries) const {
    SafeTensors out;
    out.path_ = path_;
    out.dataStart_ = dataStart_;
    out.fileSize_ = fileSize_;
    out.metadata_ = metadata_;
    for (const Rename& r : entries) {
        Tensor t = info(r.source);
        if (r.count) {
            const std::size_t n = t.elements(), width = n ? (t.end - t.begin) / n : 0;
            if (r.first + r.count > n)
                throw std::runtime_error("SafeTensors: elements " + std::to_string(r.first) + "+" + std::to_string(r.count) + " out of range for " + r.source);
            t.begin += r.first * width;
            t.end = t.begin + r.count * width;
            const std::size_t row = t.shape.size() >= 2 ? n / t.shape[0] : 0;
            if (row && r.count % row == 0 && r.first % row == 0) t.shape[0] = r.count / row;
            else t.shape = {r.count};
        }
        out.tensors_[r.name] = std::move(t);
    }
    return out;
}

const SafeTensors::Tensor& SafeTensors::info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("SafeTensors: no tensor named '" + name + "' in " + path_);
    return it->second;
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& kv : tensors_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

const SafeTensors::Tensor& SafeTensors::loadable(const std::string& name) const {
    const Tensor& t = info(name);
    if (t.dtype != "F32" && t.dtype != "F16" && t.dtype != "BF16")
        throw std::runtime_error("SafeTensors: tensor " + name + " has unsupported dtype " + t.dtype);
    return t;
}

void SafeTensors::readBytes(const Tensor& t, std::size_t from, std::size_t bytes, void* dst, const std::string& name) const {
    // Parallel chunks, each thread with its own stream: storage serves several reads at once.
    const std::size_t chunk = std::size_t(16) << 20;
    std::atomic<bool> truncated{false};
    parallelFor(0, (bytes + chunk - 1) / chunk, 1, [&](std::size_t c0, std::size_t c1) {
        std::ifstream in(path_, std::ios::binary);
        for (std::size_t c = c0; c < c1 && in; ++c) {
            const std::size_t b0 = c * chunk, count = std::min(chunk, bytes - b0);
            in.seekg(static_cast<std::streamoff>(dataStart_ + t.begin + from + b0));
            in.read(static_cast<char*>(dst) + b0, static_cast<std::streamsize>(count));
        }
        if (!in) truncated = true;
    });
    if (truncated) throw std::runtime_error("SafeTensors: truncated data for " + name + " in " + path_);
}

Matrix SafeTensors::load(const std::string& name) const {
    NC_PROFILE("load: read tensors");
    const Tensor& t = loadable(name);
    const std::size_t n = t.elements();
    const std::size_t cols = t.shape.empty() ? 1 : t.shape.back();
    const std::size_t rows = cols ? n / cols : 0;

    Matrix m(t.shape.size() <= 1 ? 1 : rows, t.shape.size() <= 1 ? n : cols);
    if (t.dtype == "F32") {
        // Little-endian float32: straight into the matrix (all supported hosts are little-endian).
        readBytes(t, 0, n * 4, m.data(), name);
        return m;
    }
    const bool half = t.dtype == "F16";
    // fp16 / bf16 in chunks, converted as they arrive (on every core).
    const std::size_t chunk = std::size_t(1) << 22;  // elements
    parallelFor(0, (n + chunk - 1) / chunk, 1, [&](std::size_t c0, std::size_t c1) {
        std::vector<std::uint16_t> raw;
        for (std::size_t c = c0; c < c1; ++c) {
            const std::size_t i0 = c * chunk, count = std::min(chunk, n - i0);
            raw.resize(count);
            readBytes(t, i0 * 2, count * 2, raw.data(), name);
            float* dst = m.data() + i0;
            if (half) halvesToFloats(raw.data(), dst, count);
            else
                for (std::size_t i = 0; i < count; ++i) dst[i] = bf16ToFloat(raw[i]);
        }
    });
    return m;
}

Matrix SafeTensors::loadTransposed(const std::string& name, std::size_t rows, std::size_t cols) const {
    NC_PROFILE("load: read tensors");
    const Tensor& t = loadable(name);
    const std::size_t n = t.elements();
    if (n != rows * cols)
        throw std::runtime_error("SafeTensors: " + name + " has " + std::to_string(n) + " values, not " + std::to_string(rows) + " x " +
                                 std::to_string(cols));
    const std::size_t width = t.dtype == "F32" ? 4 : 2;
    std::unique_ptr<unsigned char[]> raw(new unsigned char[n * width]);  // (not zeroed: every byte is read into)
    readBytes(t, 0, n * width, raw.get(), name);
    Matrix out(cols, rows);
    // Bands of 32 source rows: converted in bulk into a buffer that stays in cache, then written out
    // as 32 x 32 tiles.
    constexpr std::size_t B = 32;
    const bool f32 = width == 4, half = t.dtype == "F16";
    parallelFor(0, (rows + B - 1) / B, 1, [&](std::size_t b0, std::size_t b1) {
        std::vector<float> band(B * cols);
        for (std::size_t bi = b0; bi < b1; ++bi) {
            const std::size_t i0 = bi * B, i1 = std::min(rows, i0 + B), count = (i1 - i0) * cols;
            const unsigned char* src = raw.get() + i0 * cols * width;
            if (f32) std::memcpy(band.data(), src, count * 4);
            else {
                const auto* h = reinterpret_cast<const std::uint16_t*>(src);
                if (half) halvesToFloats(h, band.data(), count);
                else
                    for (std::size_t k = 0; k < count; ++k) band[k] = bf16ToFloat(h[k]);
            }
            for (std::size_t j0 = 0; j0 < cols; j0 += B) {
                const std::size_t j1 = std::min(cols, j0 + B);
                for (std::size_t j = j0; j < j1; ++j) {
                    float* o = out.data() + j * rows;
                    const float* in = band.data() + j;
                    for (std::size_t i = i0; i < i1; ++i) o[i] = in[(i - i0) * cols];
                }
            }
        }
    });
    return out;
}

void saveSafeTensors(const std::string& path, const std::vector<std::pair<std::string, const Matrix*>>& tensors) {
    std::ostringstream header;
    header << '{';
    std::size_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const Matrix& m = *tensors[i].second;
        const std::size_t bytes = m.size() * 4;
        if (i) header << ',';
        header << '"' << tensors[i].first << "\":{\"dtype\":\"F32\",\"shape\":[" << m.rows() << ',' << m.cols()
               << "],\"data_offsets\":[" << offset << ',' << offset + bytes << "]}";
        offset += bytes;
    }
    header << '}';
    std::string h = header.str();
    while (h.size() % 8) h += ' ';  // the format pads the header to 8 bytes

    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("saveSafeTensors: cannot open " + path);
    unsigned char len[8];
    std::uint64_t n = h.size();
    for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>((n >> (8 * i)) & 0xFF);
    out.write(reinterpret_cast<const char*>(len), 8);
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (const auto& [name, m] : tensors) out.write(reinterpret_cast<const char*>(m->data()), static_cast<std::streamsize>(m->size() * 4));
    if (!out) throw std::runtime_error("saveSafeTensors: write failed for " + path);
}

} // namespace nc
