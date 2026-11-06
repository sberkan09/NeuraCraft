#ifndef NEURACRAFT_GPU_GRAPH_H
#define NEURACRAFT_GPU_GRAPH_H

// Whole passes on the GPU: the activations of a network (a VAE half, a UNet step) stay in device
// memory from the first layer to the last, instead of every layer copying its input up and its
// output back. CUDA builds only; elsewhere enabled() is false and the callers keep their CPU path.
// NEURACRAFT_GPU_GRAPH=0 turns it off (every layer then runs on its own as before).

#include "Matrix.h"
#include "Network.h"

#include <cstddef>
#include <utility>

#ifdef NEURACRAFT_HAS_CUDA
#include "CudaBackend.h"
#endif

namespace nc::gpugraph {

// A CUDA build, the GPU device selected, a device present, and not switched off.
bool enabled();

// Runs `net` (B images, one per row of `input`) on the device and returns true, or returns false
// without doing anything when a layer has no device version. Supported: Conv2D (float weights),
// GroupNorm, SiLU, Upsample2D, Pad2D, ResnetBlock2D, SpatialSelfAttention.
bool forward(Network& net, const Matrix& input, Matrix& output);

#ifdef NEURACRAFT_HAS_CUDA
// A device buffer of floats (inside a cuda::dev::Session).
class DeviceTensor {
public:
    DeviceTensor() = default;
    explicit DeviceTensor(std::size_t n) : p_(cuda::dev::alloc(n)), n_(n) {}
    ~DeviceTensor() { cuda::dev::release(p_); }
    DeviceTensor(DeviceTensor&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    DeviceTensor& operator=(DeviceTensor&& o) noexcept {
        if (this != &o) {
            cuda::dev::release(p_);
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    float* get() const { return p_; }
    std::size_t size() const { return n_; }

    static DeviceTensor upload(const float* host, std::size_t n) {
        DeviceTensor t(n);
        cuda::dev::upload(t.p_, host, n);
        return t;
    }
    static DeviceTensor upload(const Matrix& m) { return upload(m.data(), m.size()); }
    Matrix download(std::size_t rows, std::size_t cols) const {
        Matrix m(rows, cols);
        cuda::dev::download(m.data(), p_, rows * cols);
        return m;
    }

private:
    float* p_ = nullptr;
    std::size_t n_ = 0;
};
#endif

} // namespace nc::gpugraph

#endif
