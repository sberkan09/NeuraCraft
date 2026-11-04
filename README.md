# NeuraCraft

A small, dependency-free neural network library written from scratch in C++17.
It covers everything needed to define, train, evaluate and save feed-forward
networks: a matrix type, layers with hand-written backpropagation, losses,
optimizers and data utilities — plus demos and a gradient-checked test suite.

## Build and run

Requires CMake ≥ 3.16 (3.20+ for `ctest --test-dir`) and a C++17 compiler
(GCC, Clang or Apple Clang). On macOS the Accelerate and Metal backends are
built automatically; no other dependencies.

```
$ ./build_and_run.sh            # XOR, spiral and sine demos
$ ./build_and_run.sh spiral     # a single demo
$ ./build_and_run.sh --test     # unit tests + demos via ctest
$ ./build_and_run.sh bench      # matrix-multiply speed of every backend
$ ./build_and_run.sh --gpu spiral   # run large products on the GPU
```

Or with CMake directly:

```
$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build --parallel
$ ctest --test-dir build --output-on-failure
$ ./bin/NeuraCraft help
```

### MNIST

```
$ scripts/download_mnist.sh data/mnist
$ ./build_and_run.sh mnist data/mnist 5     # 5 epochs
```

Trains a 784-128-64-10 ReLU network with dropout and Adam, reports validation
accuracy on the 10k test set every epoch and saves the weights to
`data/mnist/mnist.weights`. An epoch takes a few seconds on a laptop CPU.

## Performance and devices

All matrix products in the library go through one `gemm()` dispatcher
(`Backend.h`), which picks an implementation:

| Backend | When | Notes |
|---|---|---|
| NeuraCraft blocked kernel | default CPU path on Linux / Windows | packed, register-tiled SIMD kernel (SSE/AVX/NEON via vector extensions), multi-threaded; results don't depend on the thread count |
| Apple Accelerate | default CPU path on macOS | uses the AMX matrix units on Apple silicon |
| Metal | `setDevice(Device::GPU)` / `--gpu` on Apple silicon | simdgroup-matrix kernel on M1 and later, tiled fallback elsewhere; shaders compile at runtime (no full Xcode needed) |

```cpp
nc::setThreadCount(8);                 // CPU threads (default: all cores)
if (nc::setDevice(nc::Device::GPU))    // false -> stays on the CPU
    std::cout << nc::deviceDescription() << "\n";
nc::setGpuMinWork(1 << 22);            // smaller products stay on the CPU
```

The GPU only pays off for large products (big batches, wide layers): data
lives in host memory and is copied to the GPU per product, so small ones run
faster on the CPU. Accelerate is very strong on Apple silicon; run
`./build_and_run.sh bench` to see which backend wins on your machine. Setting
`NEURACRAFT_METAL_KERNEL=tiled` or `=simdgroup` overrides the Metal kernel choice.

Measured matrix-multiply throughput (GFLOPS, square n×n):

| n | old loop (v1.0) | blocked, 2-core x86, portable | blocked, same CPU, `-march=native` |
|---:|---:|---:|---:|
| 256 | 12.4 | 35 | 91 |
| 1024 | 8.4 | 45 | 108 |
| 2048 | – | 44 | 119 |

`build_and_run.sh` builds with `NEURACRAFT_NATIVE=ON` (all instructions the CPU
supports); plain CMake builds stay portable unless you pass
`-DNEURACRAFT_NATIVE=ON`. On the same 2-core machine an MNIST epoch went from
6.8 s to 2.3 s with identical results.

## Example

```cpp
#include "NeuraCraft.h"
using namespace nc;

int main() {
    setSeed(42);
    Dataset data = makeSpiral(100, 3);            // x: 300x2, y: 300x3 one-hot
    Split split = trainTestSplit(data.x, data.y, 0.2f);

    Network net;
    net.add<Dense>(2, 64, Init::He);
    net.add<ReLU>();
    net.add<Dense>(64, 3);                         // logits

    SoftmaxCrossEntropy loss;
    Adam optimizer(0.01f);
    FitOptions opts;
    opts.epochs = 300;
    opts.batchSize = 32;
    opts.trackAccuracy = true;
    opts.validationX = &split.test.x;
    opts.validationY = &split.test.y;
    net.fit(split.train.x, split.train.y, loss, optimizer, opts);

    float acc = net.accuracy(split.test.x, split.test.y);
    net.save("spiral.weights");                    // later: net.load(...)
}
```

Link your own program against the `neuracraft_lib` CMake target:

```cmake
add_subdirectory(neuracraft)
target_link_libraries(my_app PRIVATE neuracraft_lib)
```

## What's inside

| Component | Header | Contents |
|---|---|---|
| Matrix | `Matrix.h` | Row-major float matrix: products (incl. `AᵀB`, `ABᵀ`), element-wise ops, broadcasting, reductions, row gather/slice |
| Layers | `Dense.h`, `Activations.h` | `Dense` (Xavier / He / zero init), `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout` |
| Losses | `Loss.h` | `MSE`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SoftmaxCrossEntropy` (fused, on logits) |
| Optimizers | `Optimizer.h` | `SGD` (momentum, weight decay), `Adam` (optional decoupled weight decay) |
| Network | `Network.h` | `add<Layer>(...)`, `fit` (mini-batches, shuffling, validation, accuracy, callbacks), `predict`, `evaluate`, `accuracy`, `summary`, `save` / `load` |
| Data | `Data.h` | One-hot / argmax, train/test split, standardization, CSV load/save, MNIST IDX loader, XOR / spiral / sine generators |
| Backends | `Backend.h`, `Parallel.h` | `gemm()` dispatch, CPU / Accelerate / Metal, `setDevice`, `setThreadCount`, `parallelFor` |
| Random | `Random.h` | `setSeed()` makes initialization, shuffling and dropout reproducible |

Conventions: a batch is a `(samples × features)` matrix, one sample per row.
Every layer's `backward()` takes `dL/d(output)`, stores parameter gradients and
returns `dL/d(input)`. To add a layer, subclass `nc::Layer`.

For classification, end the network with a `Dense` layer and use
`SoftmaxCrossEntropy`; it applies softmax itself, so call `nc::softmax()` on
`predict()` output when you need probabilities. For binary classification use
`Sigmoid` + `BinaryCrossEntropy`.

`save()` writes parameter values only (a text "state dict"); `load()` expects a
network with the same architecture and leaves it unchanged if the file doesn't
match.

## Tests

`tests/tests.cpp` checks matrix operations, every layer's backward pass against
numerical gradients (central differences), loss values, optimizer convergence,
dropout statistics, XOR training, save/load, CSV and IDX loading, every GEMM
backend against a double-precision reference (all transpose combinations,
ragged sizes), CPU-vs-GPU training equivalence and the thread pool. CI runs
the tests, demos and benchmarks on Ubuntu and macOS (Apple silicon, both Metal
kernels).

## Layout

```
Includes/   public headers (NeuraCraft.h includes everything)
src/        library sources + Main.cpp (demo CLI -> bin/NeuraCraft)
tests/      unit tests
scripts/    download_mnist.sh
```
