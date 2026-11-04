#ifndef DATA_H
#define DATA_H

#include "Matrix.h"

#include <string>
#include <vector>

namespace nc {

struct Dataset {
    Matrix x;
    Matrix y;
};

struct Split {
    Dataset train;
    Dataset test;
};

// ---- Label helpers ------------------------------------------------------------
Matrix oneHot(const std::vector<int>& labels, std::size_t numClasses);
std::vector<int> argmaxRows(const Matrix& m);

// ---- Preprocessing ------------------------------------------------------------
// Shuffles rows, then puts `testFraction` of them in the test set.
Split trainTestSplit(const Matrix& x, const Matrix& y, float testFraction, bool shuffle = true);

// Per-column standardization: fit on training data, reuse for test data.
struct Standardizer {
    Matrix mean;  // (1 x cols)
    Matrix stddev;  // (1 x cols)
    void fit(const Matrix& x);
    Matrix transform(const Matrix& x) const;
    Matrix fitTransform(const Matrix& x) { fit(x); return transform(x); }
};

// ---- Loading ------------------------------------------------------------------
// Numeric CSV (comma separated). Throws std::runtime_error on bad input.
Matrix loadCSV(const std::string& path, bool hasHeader = false);
void saveCSV(const std::string& path, const Matrix& m);

// MNIST / Fashion-MNIST IDX files (uncompressed). Images are flattened to rows
// and scaled to [0, 1]. `limit` = 0 loads everything.
Matrix loadIdxImages(const std::string& path, std::size_t limit = 0);
std::vector<int> loadIdxLabels(const std::string& path, std::size_t limit = 0);

// ---- Toy datasets -------------------------------------------------------------
Dataset makeXor();                                                   // 4 x 2 -> 4 x 1
Dataset makeSpiral(std::size_t pointsPerClass, std::size_t classes, float noise = 0.2f);  // y one-hot
Dataset makeSine(std::size_t samples, float noise = 0.0f);           // x in [-pi, pi], y = sin(x)

} // namespace nc

#endif // DATA_H
