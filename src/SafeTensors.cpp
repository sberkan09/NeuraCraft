#include "SafeTensors.h"
#include "Profile.h"
#include "Json.h"
#include "Parallel.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
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

float halfToFloat(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);  // inf / nan
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
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
        Tensor t;
        t.dtype = v.at("dtype").string;
        for (const auto& d : v.at("shape").array) t.shape.push_back(static_cast<std::size_t>(d.number));
        const auto& offsets = v.at("data_offsets").array;
        if (offsets.size() != 2) throw std::runtime_error("SafeTensors: bad data_offsets for " + name);
        t.begin = static_cast<std::size_t>(offsets[0].number);
        t.end = static_cast<std::size_t>(offsets[1].number);
        const std::size_t size = dtypeSize(t.dtype);
        if (t.end < t.begin || dataStart_ + t.end > fileSize_ || (size && t.end - t.begin != t.elements() * size))
            throw std::runtime_error("SafeTensors: inconsistent size for tensor " + name);
        tensors_.emplace(name, std::move(t));
    }
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

Matrix SafeTensors::load(const std::string& name) const {
    NC_PROFILE("load: read tensors");
    const Tensor& t = info(name);
    if (t.dtype != "F32" && t.dtype != "F16" && t.dtype != "BF16")
        throw std::runtime_error("SafeTensors: tensor " + name + " has unsupported dtype " + t.dtype);
    const std::size_t n = t.elements();
    const std::size_t cols = t.shape.empty() ? 1 : t.shape.back();
    const std::size_t rows = cols ? n / cols : 0;

    Matrix m(t.shape.size() <= 1 ? 1 : rows, t.shape.size() <= 1 ? n : cols);
    const bool f32 = t.dtype == "F32", half = t.dtype == "F16";
    const std::size_t width = f32 ? 4 : 2;
    // Large tensors are read in parallel chunks (each thread with its own stream), converting
    // fp16 / bf16 as they arrive: storage serves several reads at once, and the conversion
    // runs on every core.
    const std::size_t chunk = std::size_t(1) << 22;  // elements
    std::atomic<bool> truncated{false};
    parallelFor(0, (n + chunk - 1) / chunk, 1, [&](std::size_t c0, std::size_t c1) {
        std::ifstream in(path_, std::ios::binary);
        std::vector<std::uint16_t> raw;
        for (std::size_t c = c0; c < c1; ++c) {
            const std::size_t i0 = c * chunk, count = std::min(chunk, n - i0);
            if (!in) break;
            in.seekg(static_cast<std::streamoff>(dataStart_ + t.begin + i0 * width));
            if (f32) {
                // Little-endian float32: straight into the matrix (all supported hosts are little-endian).
                in.read(reinterpret_cast<char*>(m.data() + i0), static_cast<std::streamsize>(count * 4));
            } else {
                raw.resize(count);
                if (in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * 2))) {
                    float* dst = m.data() + i0;
                    if (half)
                        for (std::size_t i = 0; i < count; ++i) dst[i] = halfToFloat(raw[i]);
                    else
                        for (std::size_t i = 0; i < count; ++i) dst[i] = bf16ToFloat(raw[i]);
                }
            }
        }
        if (!in) truncated = true;
    });
    if (truncated) throw std::runtime_error("SafeTensors: truncated data for " + name + " in " + path_);
    return m;
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
