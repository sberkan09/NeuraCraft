#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iosfwd>
#include <vector>

namespace nc {

// Dense, row-major float matrix. A batch of samples is stored one sample per row.
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, float fill = 0.0f);
    Matrix(std::initializer_list<std::initializer_list<float>> values);
    Matrix(std::size_t rows, std::size_t cols, std::vector<float> data);

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
    std::vector<float>& values() { return data_; }
    const std::vector<float>& values() const { return data_; }

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
    std::vector<float> data_;

    void requireSameShape(const Matrix& other, const char* op) const;
};

Matrix operator*(float s, const Matrix& m);
std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace nc

#endif // MATRIX_H
