#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class Matrix {
private:
    std::vector<std::vector<int>> mat;
    int rows;
    int cols;

public:
    Matrix(int rows, int cols); // Constructor
    void setElement(int row, int col, int value);
    int getElement(int row, int col) const;
    void print() const;
    int getRows() const;
    int getCols() const;

    // Overload the * operator for matrix multiplication
    Matrix operator*(const Matrix& rhs) const;
};

#endif // MATRIX_H
