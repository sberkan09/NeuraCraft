#ifndef NEURACRAFT_TRANSFORM_H
#define NEURACRAFT_TRANSFORM_H

// Small 3D transform helpers shared by the mesh reader and the animation code: column-major 4x4
// matrices (glTF's layout), unit quaternions (x, y, z, w) and translation-rotation-scale triples.

#include <array>
#include <cmath>

namespace nc {
namespace xf {

using Mat4 = std::array<double, 16>;
using Quat = std::array<double, 4>;  // x, y, z, w
using Vec3 = std::array<double, 3>;

inline Mat4 identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }

inline Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + rr] * b[c * 4 + k];
            r[c * 4 + rr] = s;
        }
    return r;
}

inline Vec3 apply(const Mat4& m, const Vec3& p) {
    return {m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
            m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]};
}

inline Quat normalized(Quat q) {
    const double l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(l > 1e-12)) return {0, 0, 0, 1};
    for (double& x : q) x /= l;
    return q;
}

inline Quat qmul(const Quat& a, const Quat& b) {
    return {a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]};
}

inline Quat conj(const Quat& q) { return {-q[0], -q[1], -q[2], q[3]}; }

inline Vec3 rotate(const Quat& q, const Vec3& v) {
    const Quat p = qmul(qmul(q, {v[0], v[1], v[2], 0}), conj(q));
    return {p[0], p[1], p[2]};
}

inline Quat axisAngle(const Vec3& axis, double angle) {
    const double l = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (!(l > 1e-12)) return {0, 0, 0, 1};
    const double s = std::sin(angle / 2) / l;
    return {axis[0] * s, axis[1] * s, axis[2] * s, std::cos(angle / 2)};
}

// The shortest rotation taking direction a to direction b (both non-zero).
inline Quat between(const Vec3& a, const Vec3& b) {
    const double la = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), lb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    if (!(la > 1e-12) || !(lb > 1e-12)) return {0, 0, 0, 1};
    const Vec3 u{a[0] / la, a[1] / la, a[2] / la}, v{b[0] / lb, b[1] / lb, b[2] / lb};
    const double d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    if (d < -0.999999) {  // opposite: any perpendicular axis
        Vec3 axis{0, -u[2], u[1]};
        if (std::fabs(u[0]) > 0.9) axis = {-u[2], 0, u[0]};
        return axisAngle(axis, 3.14159265358979323846);
    }
    return normalized({u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0], 1 + d});
}

inline Quat slerp(Quat a, Quat b, double t) {
    double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (d < 0) {
        for (double& x : b) x = -x;
        d = -d;
    }
    if (d > 0.9995) {
        Quat r;
        for (int k = 0; k < 4; ++k) r[k] = a[k] + t * (b[k] - a[k]);
        return normalized(r);
    }
    const double th = std::acos(d), s = std::sin(th), wa = std::sin((1 - t) * th) / s, wb = std::sin(t * th) / s;
    return {wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2], wa * a[3] + wb * b[3]};
}

inline Mat4 rotationMatrix(const Quat& q) {
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    return {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0, 2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
            2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0, 0, 0, 0, 1};
}

// translation (3), rotation (4), scale (3) -> T * R * S
inline Mat4 trs(const float* t, const float* r, const float* s) {
    Mat4 m = rotationMatrix(normalized({r[0], r[1], r[2], r[3]}));
    for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 3; ++k) m[c * 4 + k] *= s[c];
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    return m;
}

// A matrix without shear -> translation, rotation, scale.
inline void decompose(const Mat4& m, double* t, Quat& q, double* s) {
    t[0] = m[12];
    t[1] = m[13];
    t[2] = m[14];
    double r[9];
    for (int c = 0; c < 3; ++c) {
        s[c] = std::sqrt(m[c * 4] * m[c * 4] + m[c * 4 + 1] * m[c * 4 + 1] + m[c * 4 + 2] * m[c * 4 + 2]);
        for (int k = 0; k < 3; ++k) r[c * 3 + k] = s[c] > 1e-12 ? m[c * 4 + k] / s[c] : (c == k ? 1 : 0);
    }
    // r is column-major: r[c * 3 + k] = row k, column c
    const double m00 = r[0], m11 = r[4], m22 = r[8], trace = m00 + m11 + m22;
    if (trace > 0) {
        const double w = std::sqrt(1 + trace) * 2;
        q = {(r[5] - r[7]) / w, (r[6] - r[2]) / w, (r[1] - r[3]) / w, w / 4};
    } else if (m00 > m11 && m00 > m22) {
        const double w = std::sqrt(1 + m00 - m11 - m22) * 2;
        q = {w / 4, (r[3] + r[1]) / w, (r[6] + r[2]) / w, (r[5] - r[7]) / w};
    } else if (m11 > m22) {
        const double w = std::sqrt(1 + m11 - m00 - m22) * 2;
        q = {(r[3] + r[1]) / w, w / 4, (r[7] + r[5]) / w, (r[6] - r[2]) / w};
    } else {
        const double w = std::sqrt(1 + m22 - m00 - m11) * 2;
        q = {(r[6] + r[2]) / w, (r[7] + r[5]) / w, w / 4, (r[1] - r[3]) / w};
    }
    q = normalized(q);
}

} // namespace xf
} // namespace nc

#endif // NEURACRAFT_TRANSFORM_H
