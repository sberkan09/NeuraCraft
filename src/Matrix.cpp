#include "Matrix.h"
#include "Parallel.h"
#include "Backend.h"
#include "Random.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace nc {

namespace {
std::string shapeOf(std::size_t r, std::size_t c) {
    std::ostringstream os;
    os << "(" << r << "x" << c << ")";
    return os.str();
}
} // namespace

Matrix::Matrix(std::size_t rows, std::size_t cols, float fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> values) {
    rows_ = values.size();
    cols_ = rows_ ? values.begin()->size() : 0;
    data_.reserve(rows_ * cols_);
    for (const auto& r : values) {
        if (r.size() != cols_) {
            throw std::invalid_argument("Matrix: all rows must have the same length");
        }
        data_.insert(data_.end(), r.begin(), r.end());
    }
}

namespace detail {

std::size_t storageBytesFor(std::size_t bytes) {
    if (bytes < kPageBackedBytes) return bytes;
    return (bytes + kStoragePage - 1) / kStoragePage * kStoragePage;
}

void* allocateStorage(std::size_t bytes) {
    void* p = nullptr;
#ifdef _WIN32
    p = _aligned_malloc(std::max<std::size_t>(1, storageBytesFor(bytes)), bytes >= kPageBackedBytes ? kStoragePage : 64);
#else
    if (bytes >= kPageBackedBytes) {
        if (posix_memalign(&p, kStoragePage, storageBytesFor(bytes)) != 0) p = nullptr;
    } else {
        p = std::malloc(std::max<std::size_t>(1, bytes));
    }
#endif
    if (!p) throw std::bad_alloc();
    return p;
}

void freeStorage(void* p, std::size_t bytes) noexcept {
    // Large blocks may hold weights the GPU keeps a copy of.
    if (p && bytes >= kPageBackedBytes) gpuHostMemoryFreed(p, bytes);
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

} // namespace detail

Matrix::Matrix(std::size_t rows, std::size_t cols, const std::vector<float>& data)
    : Matrix(rows, cols, FloatVector(data.begin(), data.end())) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, FloatVector data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {
    if (data_.size() != rows_ * cols_) {
        throw std::invalid_argument("Matrix: data size does not match " + shapeOf(rows, cols));
    }
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) { return Matrix(rows, cols, 0.0f); }
Matrix Matrix::ones(std::size_t rows, std::size_t cols) { return Matrix(rows, cols, 1.0f); }

Matrix Matrix::uniform(std::size_t rows, std::size_t cols, float low, float high) {
    Matrix m(rows, cols);
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& v : m.data_) v = dist(rng());
    return m;
}

Matrix Matrix::normal(std::size_t rows, std::size_t cols, float mean, float stddev) {
    Matrix m(rows, cols);
    std::normal_distribution<float> dist(mean, stddev);
    for (auto& v : m.data_) v = dist(rng());
    return m;
}

float& Matrix::at(std::size_t r, std::size_t c) {
    if (r >= rows_ || c >= cols_) throw std::out_of_range("Matrix::at index out of bounds");
    return (*this)(r, c);
}

float Matrix::at(std::size_t r, std::size_t c) const {
    if (r >= rows_ || c >= cols_) throw std::out_of_range("Matrix::at index out of bounds");
    return (*this)(r, c);
}

void Matrix::setElement(int row, int col, float value) {
    if (row < 0 || col < 0) throw std::out_of_range("Matrix::setElement negative index");
    at(static_cast<std::size_t>(row), static_cast<std::size_t>(col)) = value;
}

float Matrix::getElement(int row, int col) const {
    if (row < 0 || col < 0) throw std::out_of_range("Matrix::getElement negative index");
    return at(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
}

void Matrix::print() const { print(std::cout); }

void Matrix::print(std::ostream& os) const {
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            os << std::setw(10) << std::setprecision(4) << (*this)(i, j) << (j + 1 < cols_ ? " " : "");
        }
        os << '\n';
    }
}

void Matrix::requireSameShape(const Matrix& other, const char* op) const {
    if (!sameShape(other)) {
        throw std::invalid_argument(std::string("Matrix ") + op + ": shape mismatch " +
                                    shapeOf(rows_, cols_) + " vs " + shapeOf(other.rows_, other.cols_));
    }
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    if (cols_ != rhs.rows_) {
        throw std::invalid_argument("Matrix multiply: shape mismatch " + shapeOf(rows_, cols_) +
                                    " * " + shapeOf(rhs.rows_, rhs.cols_));
    }
    Matrix result(rows_, rhs.cols_);
    nc::matmul(*this, false, rhs, false, result);
    return result;
}

Matrix Matrix::transposeMatmul(const Matrix& rhs) const {
    // (this^T)(rhs): this is (n x a), rhs is (n x b) -> (a x b)
    if (rows_ != rhs.rows_) {
        throw std::invalid_argument("Matrix transposeMatmul: shape mismatch " + shapeOf(rows_, cols_) +
                                    "^T * " + shapeOf(rhs.rows_, rhs.cols_));
    }
    Matrix result(cols_, rhs.cols_);
    nc::matmul(*this, true, rhs, false, result);
    return result;
}

Matrix Matrix::matmulTranspose(const Matrix& rhs) const {
    // this * rhs^T: this is (n x k), rhs is (m x k) -> (n x m)
    if (cols_ != rhs.cols_) {
        throw std::invalid_argument("Matrix matmulTranspose: shape mismatch " + shapeOf(rows_, cols_) +
                                    " * " + shapeOf(rhs.rows_, rhs.cols_) + "^T");
    }
    Matrix result(rows_, rhs.rows_);
    nc::matmul(*this, false, rhs, true, result);
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    // 32 x 32 tiles keep both the reads and the writes within a few cache lines;
    // bands of tile rows run in parallel.
    constexpr std::size_t B = 32;
    const float* src = data_.data();
    float* dst = result.data_.data();
    const std::size_t R = rows_, C = cols_;
    parallelFor(0, (R + B - 1) / B, std::max<std::size_t>(1, 4096 / std::max<std::size_t>(1, C)), [&](std::size_t b0, std::size_t b1) {
        for (std::size_t bi = b0; bi < b1; ++bi) {
            const std::size_t i0 = bi * B, i1 = std::min(R, i0 + B);
            for (std::size_t j0 = 0; j0 < C; j0 += B) {
                const std::size_t j1 = std::min(C, j0 + B);
                for (std::size_t i = i0; i < i1; ++i)
                    for (std::size_t j = j0; j < j1; ++j) dst[j * R + i] = src[i * C + j];
            }
        }
    });
    return result;
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    Matrix result(*this);
    result += rhs;
    return result;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    Matrix result(*this);
    result -= rhs;
    return result;
}

Matrix Matrix::hadamard(const Matrix& rhs) const {
    requireSameShape(rhs, "hadamard");
    Matrix result(*this);
    for (std::size_t i = 0; i < data_.size(); ++i) result.data_[i] *= rhs.data_[i];
    return result;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
    requireSameShape(rhs, "+");
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += rhs.data_[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
    requireSameShape(rhs, "-");
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] -= rhs.data_[i];
    return *this;
}

Matrix Matrix::operator*(float s) const {
    Matrix result(*this);
    result *= s;
    return result;
}

Matrix Matrix::operator/(float s) const { return *this * (1.0f / s); }

Matrix Matrix::operator+(float s) const {
    Matrix result(*this);
    for (auto& v : result.data_) v += s;
    return result;
}

Matrix Matrix::operator-(float s) const { return *this + (-s); }

Matrix& Matrix::operator*=(float s) {
    for (auto& v : data_) v *= s;
    return *this;
}

Matrix Matrix::operator-() const { return *this * -1.0f; }

Matrix Matrix::addRowVector(const Matrix& row) const {
    if (row.rows_ != 1 || row.cols_ != cols_) {
        throw std::invalid_argument("Matrix addRowVector: expected (1x" + std::to_string(cols_) +
                                    ") got " + shapeOf(row.rows_, row.cols_));
    }
    Matrix result(*this);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < cols_; ++j) result(i, j) += row.data_[j];
    return result;
}

Matrix& Matrix::addRowVectorInPlace(const Matrix& row) {
    if (row.rows_ != 1 || row.cols_ != cols_) {
        throw std::invalid_argument("Matrix addRowVectorInPlace: expected (1x" + std::to_string(cols_) +
                                    ") got " + shapeOf(row.rows_, row.cols_));
    }
    const float* r = row.data_.data();
    parallelFor(0, rows_, std::max<std::size_t>(1, 16384 / std::max<std::size_t>(1, cols_)), [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            float* d = data_.data() + i * cols_;
            for (std::size_t j = 0; j < cols_; ++j) d[j] += r[j];
        }
    });
    return *this;
}

Matrix Matrix::sumRows() const {
    Matrix result(1, cols_);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < cols_; ++j) result.data_[j] += (*this)(i, j);
    return result;
}

Matrix Matrix::sumCols() const {
    Matrix result(rows_, 1);
    for (std::size_t i = 0; i < rows_; ++i) {
        float acc = 0.0f;
        for (std::size_t j = 0; j < cols_; ++j) acc += (*this)(i, j);
        result.data_[i] = acc;
    }
    return result;
}

Matrix Matrix::apply(const std::function<float(float)>& fn) const {
    Matrix result(*this);
    for (auto& v : result.data_) v = fn(v);
    return result;
}

void Matrix::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

Matrix Matrix::row(std::size_t r) const { return sliceRows(r, r + 1); }

Matrix Matrix::gatherRows(const std::vector<std::size_t>& indices) const {
    Matrix result(indices.size(), cols_);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= rows_) throw std::out_of_range("Matrix gatherRows index out of bounds");
        std::copy_n(data_.begin() + indices[i] * cols_, cols_, result.data_.begin() + i * cols_);
    }
    return result;
}

Matrix Matrix::sliceRows(std::size_t begin, std::size_t end) const {
    if (begin > end || end > rows_) throw std::out_of_range("Matrix sliceRows range out of bounds");
    return Matrix(end - begin, cols_, FloatVector(data_.begin() + static_cast<long>(begin * cols_),
                                                  data_.begin() + static_cast<long>(end * cols_)));
}

float Matrix::sum() const {
    double acc = 0.0;
    for (float v : data_) acc += v;
    return static_cast<float>(acc);
}

float Matrix::mean() const { return data_.empty() ? 0.0f : sum() / static_cast<float>(data_.size()); }

float Matrix::max() const {
    if (data_.empty()) throw std::logic_error("Matrix::max on empty matrix");
    return *std::max_element(data_.begin(), data_.end());
}

float Matrix::min() const {
    if (data_.empty()) throw std::logic_error("Matrix::min on empty matrix");
    return *std::min_element(data_.begin(), data_.end());
}

std::size_t Matrix::argmaxRow(std::size_t r) const {
    if (r >= rows_ || cols_ == 0) throw std::out_of_range("Matrix::argmaxRow out of bounds");
    const float* begin = data_.data() + r * cols_;
    return static_cast<std::size_t>(std::max_element(begin, begin + cols_) - begin);
}

bool Matrix::approxEquals(const Matrix& other, float tolerance) const {
    if (!sameShape(other)) return false;
    for (std::size_t i = 0; i < data_.size(); ++i)
        if (std::fabs(data_[i] - other.data_[i]) > tolerance) return false;
    return true;
}

Matrix operator*(float s, const Matrix& m) { return m * s; }

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    m.print(os);
    return os;
}

} // namespace nc
