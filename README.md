# NeuraCraft

A small, dependency-free neural network library written from scratch in C++17.
It covers everything needed to define, train, evaluate and save feed-forward
networks: a matrix type, layers with hand-written backpropagation, losses,
optimizers and data utilities — plus demos and a gradient-checked test suite.

## Build and run

Requires CMake ≥ 3.10 (3.20+ for `ctest --test-dir`) and a C++17 compiler
(GCC, Clang or Apple Clang).

```
$ ./build_and_run.sh            # XOR, spiral and sine demos
$ ./build_and_run.sh spiral     # a single demo
$ ./build_and_run.sh --test     # unit tests + demos via ctest
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
`data/mnist/mnist.weights`.

Reference run (10 epochs, seed 42, single CPU core, Release build): **97.97 %
test accuracy**, about 7 seconds per epoch.

| epoch | train loss | train acc | test loss | test acc |
|---:|---:|---:|---:|---:|
| 1 | 0.3162 | 90.75 % | 0.1399 | 95.94 % |
| 2 | 0.1429 | 95.71 % | 0.0950 | 96.98 % |
| 4 | 0.0894 | 97.22 % | 0.0704 | 97.74 % |
| 6 | 0.0702 | 97.75 % | 0.0712 | 97.81 % |
| 8 | 0.0536 | 98.26 % | 0.0631 | 98.04 % |
| 10 | 0.0469 | 98.39 % | 0.0696 | 97.97 % |

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
dropout statistics, XOR training, save/load, CSV and IDX loading. CI runs the
tests and demos on Ubuntu and macOS.

## Layout

```
Includes/   public headers (NeuraCraft.h includes everything)
src/        library sources + Main.cpp (demo CLI -> bin/NeuraCraft)
tests/      unit tests
scripts/    download_mnist.sh
```
