# NeuraCraft

A small, dependency-free neural network library written from scratch in C++17:
dense, convolutional and transformer layers with hand-written backpropagation,
losses, optimizers, data and text utilities, a multi-threaded SIMD CPU backend,
Apple Accelerate and Metal GPU support — plus demos (MNIST, a character-level
GPT) and a gradient-checked test suite.

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
`data/mnist/mnist.weights`.

Reference run (10 epochs, seed 42): **97.97 % test accuracy**. With the v1.0
single-threaded kernel an epoch took ~7 s; the current backends take ~2.3 s on
the same 2-core machine (see below).

| epoch | train loss | train acc | test loss | test acc |
|---:|---:|---:|---:|---:|
| 1 | 0.3162 | 90.75 % | 0.1399 | 95.94 % |
| 2 | 0.1429 | 95.71 % | 0.0950 | 96.98 % |
| 4 | 0.0894 | 97.22 % | 0.0704 | 97.74 % |
| 6 | 0.0702 | 97.75 % | 0.0712 | 97.81 % |
| 8 | 0.0536 | 98.26 % | 0.0631 | 98.04 % |
| 10 | 0.0469 | 98.39 % | 0.0696 | 97.97 % |

A convolutional network (2× Conv3×3 + MaxPool, then Dense) does better:

```
$ ./build_and_run.sh mnist data/mnist 3 cnn   # -> data/mnist/mnist_cnn.weights
$ ./build_and_run.sh eval data/mnist cnn      # reload, per-digit accuracy, show mistakes
```

| model | parameters | epochs | test accuracy | time / epoch (2-core x86) |
|---|---:|---:|---:|---:|
| MLP 784-128-64-10 | 109 k | 3 | 97.10 % | 2.3 s |
| CNN 16-32 conv + 128 dense | 207 k | 3 | 98.83 % | 46 s |

### Language model (GPT)

```
$ scripts/download_shakespeare.sh                     # data/shakespeare.txt (1.1 MB)
$ ./build_and_run.sh gpt data/shakespeare.txt 2       # BPE tokens (default)
$ ./build_and_run.sh gpt data/shakespeare.txt 2 char  # character tokens
$ ./build_and_run.sh gpt data/shakespeare.txt 3 --resume   # 3 more epochs from the checkpoint
$ ./build_and_run.sh gpt data/shakespeare.txt 6 --cosine --dim 192 --layers 6 --heads 6 --context 128
$ ./build_and_run.sh generate data/shakespeare.txt "ROMEO:" 200
```

`gpt` trains a decoder (default: 4 layers, 4 heads, dim 128, context 64
tokens, modern architecture) on any text file, prints a sample after every epoch and saves
`<file>.gpt.weights`, `.gpt.config`, `.gpt.tokenizer` and a
`.gpt.checkpoint` (weights + Adam state + step + random state, written after
every epoch); `generate` rebuilds the model from the config and samples with
temperature 0.8, top-k 20. Training clips the global gradient norm to 1;
`--cosine` adds linear warmup and cosine decay of the learning rate over the
whole run (`--lr` is the peak). For a short run a constant rate is better:
over 2 epochs it reaches 1.817 per character, cosine 1.894 (peak 1e-3) or 1.842
(peak 2e-3), because the rate decays before the model has learned enough.
Over 6 epochs the picture flips: cosine with peak 2e-3 ends at **1.589**,
the constant rate at 1.604.

Two architectures are available (`--arch`):

| | classic (GPT-2) | modern (Llama-style, default) |
|---|---|---|
| positions | learned table | rotary (RoPE) in attention |
| normalization | LayerNorm | RMSNorm |
| MLP | GELU, 4× wide | SwiGLU, same parameter count |
| output layer | separate | tied to the embedding |
| parameters (BPE 512) | 0.93 M | 0.86 M |
| val loss / char after 2 epochs | 1.817 | **1.713** |
| time / epoch | 86 s | 79 s |

With RoPE, attention scores depend only on relative distance, so the KV
cache can simply slide once the context is full (sliding-window attention, as
in Mistral). For a single layer that is identical to recomputing the last 64
tokens; deeper layers also carry information from beyond the window. Measured
on the trained model over 1136 tokens past the window, mean next-token loss is
3.1048 sliding vs 3.1046 recomputing (and 3.0973 when rebuilding from the
last half window), so the O(1) sliding cache costs nothing in quality. Models
with learned positions rebuild the cache from its last half instead. `--resume` continues from the checkpoint, and a resumed run
is bit-for-bit identical to one that never stopped; asking for more epochs
than originally planned stretches the cosine curve from that point. The default tokenizer is byte-level BPE with 512
tokens (learned from the file in under a second); `char` uses one token per
character. Validation loss is also reported per character, so the two are
comparable.

Tiny Shakespeare, 2 epochs, 2-core x86 sandbox:

| tokenizer | chars / token | parameters | time | val loss per character |
|---|---:|---:|---:|---:|
| char (65) | 1.00 | 0.82 M | 487 s | 1.911 |
| BPE (512) | 1.92 | 0.93 M | **259 s** (one epoch now takes 86 s, see Performance) | **1.817** |

BPE sees about twice as much text in the same 64-token context and needs half
the steps per epoch, so it is both better and faster here.

Generation uses a **KV cache**: every attention layer keeps the keys and
values of earlier positions, so each new token runs alone through the network
instead of re-processing the whole window. On the model above that is
**~1170 tokens/s instead of ~130** (9×), with identical output while the text
fits in the context window. When the window fills up, the cache is rebuilt
from its most recent half (positions are learned absolute embeddings, so the
window cannot simply slide); pass no context to `generate()` for the exact
sliding-window behaviour. Samples already
have the play's structure (speaker names, verse lines) with partly invented
words; more epochs or a wider model improve them:

```
JULIET:
O, I belure you'er frilince, beless, they are. Venount?

QUEEN MARGARET:
Whoppleack, Vposed, he is hered armporire.
```

Building one yourself:

```cpp
BpeTokenizer tok = BpeTokenizer::train(text, 512);    // or CharTokenizer tok(text);
Dataset d = makeNextTokenDataset(tok.encode(text), 64);  // x, y: (sequences x 64) ids

GPTConfig cfg;
cfg.vocabSize = tok.vocabSize();
cfg.contextLength = 64;
cfg.dim = 128; cfg.heads = 4; cfg.layers = 4;
Network net;
auto ctx = buildGPT(net, cfg);       // (batch x seq) ids -> (batch*seq x vocab) logits

SparseSoftmaxCrossEntropy loss;      // targets: next-token ids, any shape
Adam opt(1e-3f);
net.fit(d.x, d.y, loss, opt);

SamplingOptions s; s.temperature = 0.8f; s.topK = 20;
std::string out = tok.decode(generate(net, tok.encode("ROMEO:"), 200, cfg.contextLength, s, ctx));  // ctx: KV cache
```

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

The GPU only pays off for large products (big batches, wide layers), so small
ones stay on the CPU. Matrices of 64 KB and more are stored page-aligned, and
on Apple silicon's unified memory the Metal backend maps them into GPU
buffers in place: no per-product allocation or copy, and the result is
written straight into the output matrix (raw-pointer `gemmGpu` still copies).
On the macOS CI runner's virtual GPU this is 15–45 % faster than copying,
e.g. 150 → 212 GFLOPS at n = 1024. Accelerate is very strong on Apple silicon; run
`./build_and_run.sh bench` to see which backend wins on your machine. Setting
`NEURACRAFT_METAL_KERNEL=tiled` or `=simdgroup` overrides the Metal kernel choice.

Measured matrix-multiply throughput (GFLOPS, square n×n):

| n | old loop (v1.0) | blocked, 2-core x86, portable | blocked, same CPU, `-march=native` |
|---:|---:|---:|---:|
| 256 | 12.4 | 35 | 91 |
| 1024 | 8.4 | 45 | 108 |
| 2048 | – | 44 | 119 |

For the GPT demo (many products with a transposed operand), packing those
operands with sequential reads plus caching GELU's `tanh` cut a training epoch
from 129.5 s to 85.9 s (−34 %) with identical losses.

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
| Layers | `Dense.h`, `Activations.h`, `Conv.h` | `Dense` (Xavier / He / zero init), `Conv2D` (stride, padding; im2col + GEMM), `MaxPool2D`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout` |
| Losses | `Loss.h` | `MSE`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SoftmaxCrossEntropy` (fused, on logits), `SparseSoftmaxCrossEntropy` (integer targets) |
| Transformer | `Transformer.h` | `TokenEmbedding` (optional learned positions), `LayerNorm`, `RMSNorm`, `GELU`, `SwiGLUGate`, multi-head `CausalSelfAttention` (optional RoPE, KV cache, sliding window), `Residual` blocks, `TiedOutput`, `buildGPT()` with `GPTConfig::modern()` |
| Text | `Text.h` | `Tokenizer` interface, `CharTokenizer`, byte-level `BpeTokenizer` (train / save / load), `makeNextTokenDataset`, `generate()` (temperature, top-k) |
| Optimizers | `Optimizer.h` | `SGD` (momentum, weight decay), `Adam` (optional decoupled weight decay), state save/load, `CosineSchedule` (warmup + cosine), `clipGradientNorm` |
| Network | `Network.h` | `add<Layer>(...)`, `fit` (mini-batches, shuffling, validation, accuracy, LR schedule, gradient clipping, step limits, callbacks), `predict`, `evaluate`, `accuracy`, `summary`, `save` / `load`, `saveCheckpoint` / `loadCheckpoint` (exact resume) |
| Data | `Data.h` | One-hot / argmax, train/test split, standardization, CSV load/save, MNIST IDX loader, XOR / spiral / sine generators |
| Backends | `Backend.h`, `Parallel.h` | `gemm()` / `matmul()` dispatch, CPU / Accelerate / Metal (zero-copy on unified memory), `setDevice`, `setThreadCount`, `parallelFor` |
| Random | `Random.h` | `setSeed()` makes initialization, shuffling and dropout reproducible |

Conventions: a batch is a `(samples × features)` matrix, one sample per row.
Images are flattened channel-major (C, H, W) into a row; `Conv2D` and
`MaxPool2D` take the input `ImageShape` in their constructor. Language models
take `(batch × seqLen)` token ids, one sequence per row; `TokenEmbedding`
expands them to `(batch·seqLen) × dim` rows, and attention layers read the
sequence length from a `SequenceContext` shared with the embedding (`buildGPT`
wires this up).
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
ragged sizes), CPU-vs-GPU training equivalence, the thread pool, gradient
checks for convolution, pooling, LayerNorm, GELU, attention and a whole GPT,
attention causality, KV-cached inference against the full forward pass,
RMSNorm, SwiGLU, rotary attention and a whole modern GPT (gradient checks),
RoPE's relative-position invariance, exact sliding-cache generation far past
the window, exact checkpoint resume, and GPTs (both architectures) that
memorize and regenerate a sequence. CI runs
the tests, demos and benchmarks on Ubuntu and macOS (Apple silicon, both Metal
kernels).

## Layout

```
Includes/   public headers (NeuraCraft.h includes everything)
src/        library sources + Main.cpp (demo CLI -> bin/NeuraCraft)
tests/      unit tests
scripts/    download_mnist.sh, download_shakespeare.sh
```
