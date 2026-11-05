#ifndef NEURACRAFT_VECTOR_MATH_H
#define NEURACRAFT_VECTOR_MATH_H

#include <cstddef>

namespace nc {

// Elementwise exp / erf over arrays, written as branch-free polynomial loops
// the compiler vectorizes (SSE / AVX / NEON). In place is fine (y == x).
//   expArray: Cephes-style range reduction + degree-6 polynomial, within
//             2 ulp of std::exp on [-87, 88]; below that 0, above it +inf.
//   erfArray: within 4e-7 of std::erf everywhere.
// Several times faster than calling std::exp / std::erf per element, which is
// what softmax and GELU spend their time on in the image models.
void expArray(const float* x, float* y, std::size_t n);
void erfArray(const float* x, float* y, std::size_t n);

// Row-wise softmax(scale * row) in place, using expArray.
void softmaxRow(float* row, std::size_t n, float scale = 1.0f);

} // namespace nc

#endif // NEURACRAFT_VECTOR_MATH_H
