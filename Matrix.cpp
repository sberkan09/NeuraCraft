#include "Matrix.h"

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), mat(rows, std::vector<int>(cols, 0)) {}

void Matrix::setElement(int row, int col, int value) {
    if (row >= 0 && row < this->rows && col >= 0 && col < this->cols) {
        mat[row][col] = value;
    } else {
        std::cerr << "Error: Index out of bounds" << std::endl;
    }
}

int Matrix::getElement(int row, int col) const {
    if (row >= 0 && row < this->rows && col >= 0 && col < this->cols) {
        return mat[row][col];
    } else {
        std::cerr << "Error: Index out of bounds" << std::endl;
        return -1;
    }
}

void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    if (this->cols != rhs.rows) {
        std::cerr << "Error: Matrix dimensions do not match for multiplication" << std::endl;
        return Matrix(0, 0); // Returning an empty matrix as an error state
    }

    Matrix result(this->rows, rhs.cols);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < rhs.cols; j++) {
            int sum = 0;
            for (int k = 0; k < this->cols; k++) {
                sum += this->mat[i][k] * rhs.mat[k][j];
            }
            result.setElement(i, j, sum);
        }
    }
    return result;
}

int Matrix::getRows() const { return rows; }
int Matrix::getCols() const { return cols; }
