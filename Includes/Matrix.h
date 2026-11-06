#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iosfwd>
#include <vector>

namespace nc {

// Matrix storage. Buffers of at least kPageBackedBytes are page-aligned and
// padded to whole pages, so on Apple silicon the Metal backend can hand them
// to the GPU in place (unified memory, no copies). Smaller ones use malloc.
constexpr std::size_t kStoragePage = 16384;          // Apple silicon page size (a multiple of 4 KB)
constexpr std::size_t kPageBackedBytes = 64 * 1024;

namespace detail {
void* allocateStorage(std::size_t bytes);
void freeStorage(void* p, std::size_t bytes) noexcept;  // bytes: what was asked of allocateStorage
std::size_t storageBytesFor(std::size_t bytes);  // bytes actually reserved for a request
} // namespace detail

template <typename T>
struct StorageAllocator {
    using value_type = T;
    StorageAllocator() noexcept = default;
    template <typename U>
    StorageAllocator(const StorageAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) { return static_cast<T*>(detail::allocateStorage(n * sizeof(T))); }
    void deallocate(T* p, std::size_t n) noexcept { detail::freeStorage(p, n * sizeof(T)); }
};
template <typename T, typename U>
bool operator==(const StorageAllocator<T>&, const StorageAllocator<U>&) noexcept { return true; }
template <typename T, typename U>
bool operator!=(const StorageAllocator<T>&, const StorageAllocator<U>&) noexcept { return false; }

using FloatVector = std::vector<float, StorageAllocator<float>>;

// Dense, row-major float matrix. A batch of samples is stored one sample per row.
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, float fill = 0.0f);
    Matrix(std::initializer_list<std::initializer_list<float>> values);
    Matrix(std::size_t rows, std::size_t cols, const std::vector<float>& data);
    Matrix(std::size_t rows, std::size_t cols, FloatVector data);

    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix ones(std::size_t rows, std::size_t cols);
    static Matrix uniform(std::size_t rows, std::size_t cols, float low, float high);
    static Matrix normal(std::size_t rows, std::size_t cols, float mean, float stddev);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    FloatVector& values() { return data_; }
    const FloatVector& values() const { return data_; }
    // True when the storage is page-aligned and padded (see kPageBackedBytes);
    // storageBytes() is then the page-rounded size that may be mapped.
    bool pageBacked() const { return data_.capacity() * sizeof(float) >= kPageBackedBytes; }
    std::size_t storageBytes() const { return detail::storageBytesFor(data_.capacity() * sizeof(float)); }

    // Unchecked element access.
    float& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    float operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    // Bounds-checked element access (throws std::out_of_range).
    float& at(std::size_t r, std::size_t c);
    float at(std::size_t r, std::size_t c) const;

    // Original API, kept for compatibility.
    void setElement(int row, int col, float value);
    float getElement(int row, int col) const;
    int getRows() const { return static_cast<int>(rows_); }
    int getCols() const { return static_cast<int>(cols_); }
    void print() const;
    void print(std::ostream& os) const;

    // Matrix product (throws std::invalid_argument on shape mismatch).
    Matrix operator*(const Matrix& rhs) const;
    Matrix matmul(const Matrix& rhs) const { return *this * rhs; }
    // this^T * rhs and this * rhs^T without materializing the transpose.
    Matrix transposeMatmul(const Matrix& rhs) const;
    Matrix matmulTranspose(const Matrix& rhs) const;
    Matrix transpose() const;

    // Element-wise arithmetic (shapes must match exactly).
    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix hadamard(const Matrix& rhs) const;
    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);

    // Scalar arithmetic.
    Matrix operator*(float s) const;
    Matrix operator/(float s) const;
    Matrix operator+(float s) const;
    Matrix operator-(float s) const;
    Matrix& operator*=(float s);
    Matrix operator-() const;

    // Broadcasting helpers for (1 x cols) row vectors.
    Matrix addRowVector(const Matrix& row) const;
    Matrix& addRowVectorInPlace(const Matrix& row);  // multi-threaded, no copy
    Matrix sumRows() const;   // column sums -> (1 x cols)
    Matrix sumCols() const;   // row sums    -> (rows x 1)

    Matrix apply(const std::function<float(float)>& fn) const;
    void fill(float value);

    Matrix row(std::size_t r) const;
    Matrix gatherRows(const std::vector<std::size_t>& indices) const;
    Matrix sliceRows(std::size_t begin, std::size_t end) const;

    float sum() const;
    float mean() const;
    float max() const;
    float min() const;
    std::size_t argmaxRow(std::size_t r) const;

    bool sameShape(const Matrix& other) const { return rows_ == other.rows_ && cols_ == other.cols_; }
    bool approxEquals(const Matrix& other, float tolerance = 1e-5f) const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    FloatVector data_;

    void requireSameShape(const Matrix& other, const char* op) const;
};

Matrix operator*(float s, const Matrix& m);
std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace nc

#endif // MATRIX_H
