#include "Data.h"
#include "Random.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
constexpr float kPi = 3.14159265358979f;

std::uint32_t readBigEndian32(std::istream& in, const std::string& path) {
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("IDX: truncated header in " + path);
    return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) | (std::uint32_t(b[2]) << 8) | std::uint32_t(b[3]);
}

std::ifstream openIdx(const std::string& path, std::uint32_t expectedMagic) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("IDX: cannot open " + path);
    const std::uint32_t magic = readBigEndian32(in, path);
    if (magic != expectedMagic) {
        std::ostringstream msg;
        msg << "IDX: " << path << " has magic 0x" << std::hex << magic << ", expected 0x" << expectedMagic
            << " (is it still gzipped?)";
        throw std::runtime_error(msg.str());
    }
    return in;
}
} // namespace

Matrix oneHot(const std::vector<int>& labels, std::size_t numClasses) {
    Matrix m(labels.size(), numClasses);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < 0 || static_cast<std::size_t>(labels[i]) >= numClasses)
            throw std::out_of_range("oneHot: label " + std::to_string(labels[i]) + " out of range");
        m(i, static_cast<std::size_t>(labels[i])) = 1.0f;
    }
    return m;
}

std::vector<int> argmaxRows(const Matrix& m) {
    std::vector<int> out(m.rows());
    for (std::size_t i = 0; i < m.rows(); ++i) out[i] = static_cast<int>(m.argmaxRow(i));
    return out;
}

Split trainTestSplit(const Matrix& x, const Matrix& y, float testFraction, bool shuffle) {
    if (x.rows() != y.rows()) throw std::invalid_argument("trainTestSplit: x and y row counts differ");
    if (testFraction < 0.0f || testFraction > 1.0f) throw std::invalid_argument("trainTestSplit: fraction must be in [0, 1]");
    std::vector<std::size_t> idx(x.rows());
    std::iota(idx.begin(), idx.end(), 0);
    if (shuffle) std::shuffle(idx.begin(), idx.end(), rng());
    const auto testCount = static_cast<std::size_t>(std::round(testFraction * static_cast<float>(x.rows())));
    const std::vector<std::size_t> testIdx(idx.begin(), idx.begin() + testCount);
    const std::vector<std::size_t> trainIdx(idx.begin() + testCount, idx.end());
    return {{x.gatherRows(trainIdx), y.gatherRows(trainIdx)}, {x.gatherRows(testIdx), y.gatherRows(testIdx)}};
}

void Standardizer::fit(const Matrix& x) {
    if (x.rows() == 0) throw std::invalid_argument("Standardizer::fit: empty data");
    mean = x.sumRows() / static_cast<float>(x.rows());
    stddev = Matrix(1, x.cols());
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j) {
            const float d = x(i, j) - mean(0, j);
            stddev(0, j) += d * d;
        }
    for (auto& s : stddev.values()) {
        s = std::sqrt(s / static_cast<float>(x.rows()));
        if (s < 1e-8f) s = 1.0f;  // constant column: leave centered values as-is
    }
}

Matrix Standardizer::transform(const Matrix& x) const {
    if (mean.empty()) throw std::logic_error("Standardizer::transform called before fit");
    if (x.cols() != mean.cols()) throw std::invalid_argument("Standardizer::transform: column count differs");
    Matrix out(x);
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j) out(i, j) = (x(i, j) - mean(0, j)) / stddev(0, j);
    return out;
}

Matrix loadCSV(const std::string& path, bool hasHeader) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("loadCSV: cannot open " + path);
    std::vector<float> values;
    std::size_t cols = 0, rows = 0, lineNo = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (hasHeader && lineNo == 1) continue;
        if (line.find_first_not_of(" \t") == std::string::npos) continue;

        std::stringstream ss(line);
        std::string cell;
        std::size_t count = 0;
        while (std::getline(ss, cell, ',')) {
            try {
                std::size_t used = 0;
                values.push_back(std::stof(cell, &used));
                if (cell.find_first_not_of(" \t", used) != std::string::npos) throw std::invalid_argument(cell);
            } catch (const std::exception&) {
                throw std::runtime_error("loadCSV: non-numeric value '" + cell + "' at " + path + ":" +
                                         std::to_string(lineNo));
            }
            ++count;
        }
        if (rows == 0) cols = count;
        else if (count != cols)
            throw std::runtime_error("loadCSV: line " + std::to_string(lineNo) + " has " + std::to_string(count) +
                                     " columns, expected " + std::to_string(cols));
        ++rows;
    }
    return Matrix(rows, cols, std::move(values));
}

void saveCSV(const std::string& path, const Matrix& m) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("saveCSV: cannot open " + path);
    out.precision(9);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) out << m(i, j) << (j + 1 < m.cols() ? "," : "\n");
    }
}

Matrix loadIdxImages(const std::string& path, std::size_t limit) {
    auto in = openIdx(path, 0x00000803);
    std::size_t count = readBigEndian32(in, path);
    const std::size_t rows = readBigEndian32(in, path);
    const std::size_t cols = readBigEndian32(in, path);
    if (limit > 0) count = std::min(count, limit);

    const std::size_t pixels = rows * cols;
    std::vector<unsigned char> raw(count * pixels);
    if (!in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
        throw std::runtime_error("IDX: truncated image data in " + path);
    std::vector<float> data(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) data[i] = static_cast<float>(raw[i]) / 255.0f;
    return Matrix(count, pixels, std::move(data));
}

std::vector<int> loadIdxLabels(const std::string& path, std::size_t limit) {
    auto in = openIdx(path, 0x00000801);
    std::size_t count = readBigEndian32(in, path);
    if (limit > 0) count = std::min(count, limit);
    std::vector<unsigned char> raw(count);
    if (!in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
        throw std::runtime_error("IDX: truncated label data in " + path);
    return std::vector<int>(raw.begin(), raw.end());
}

Dataset makeXor() {
    return {Matrix{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, Matrix{{0}, {1}, {1}, {0}}};
}

Dataset makeSpiral(std::size_t pointsPerClass, std::size_t classes, float noise) {
    if (pointsPerClass == 0 || classes == 0) throw std::invalid_argument("makeSpiral: sizes must be > 0");
    Dataset d{Matrix(pointsPerClass * classes, 2), Matrix(pointsPerClass * classes, classes)};
    std::normal_distribution<float> jitter(0.0f, noise);
    for (std::size_t c = 0; c < classes; ++c) {
        for (std::size_t i = 0; i < pointsPerClass; ++i) {
            const std::size_t row = c * pointsPerClass + i;
            const float r = pointsPerClass > 1 ? static_cast<float>(i) / static_cast<float>(pointsPerClass - 1) : 1.0f;
            const float t = static_cast<float>(c) * 4.0f + r * 4.0f + (noise > 0 ? jitter(rng()) : 0.0f);
            d.x(row, 0) = r * std::sin(t);
            d.x(row, 1) = r * std::cos(t);
            d.y(row, c) = 1.0f;
        }
    }
    return d;
}

Dataset makeSine(std::size_t samples, float noise) {
    if (samples < 2) throw std::invalid_argument("makeSine: need at least 2 samples");
    Dataset d{Matrix(samples, 1), Matrix(samples, 1)};
    std::normal_distribution<float> jitter(0.0f, noise);
    for (std::size_t i = 0; i < samples; ++i) {
        const float x = -kPi + 2.0f * kPi * static_cast<float>(i) / static_cast<float>(samples - 1);
        d.x(i, 0) = x;
        d.y(i, 0) = std::sin(x) + (noise > 0 ? jitter(rng()) : 0.0f);
    }
    return d;
}

} // namespace nc
