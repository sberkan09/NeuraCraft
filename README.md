# NeuraCraft

A small, dependency-free neural network library written from scratch in C++17:
dense, convolutional and transformer layers with hand-written backpropagation,
losses, optimizers, data and text utilities, a multi-threaded SIMD CPU backend,
Apple Accelerate and Metal GPU support — plus demos (MNIST, a GPT trained from
scratch, and OpenAI's pretrained GPT-2 running on the library's own layers)
and a gradient-checked test suite.

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

### Pretrained GPT-2

```
$ scripts/download_gpt2.sh data/gpt2      # 124M-parameter GPT-2 from Hugging Face (~550 MB)
$ ./build_and_run.sh gpt2 data/gpt2 "Once upon a time, in a small village by the sea," 120
$ ./build_and_run.sh gpt2 data/gpt2 "The meaning of life is" 20 --check
```

`gpt2` loads OpenAI's GPT-2 weights (`model.safetensors`) and tokenizer
(`vocab.json` + `merges.txt`) into the library's own layers and streams a
sampled continuation (temperature 0.8, top-k 40; `--greedy`,
`--temperature`, `--top-k` change that). Larger GPT-2 checkpoints
(gpt2-medium, -large, -xl) load the same way.

It matches the reference implementation exactly: `scripts/gpt2_reference.py`
is an independent NumPy GPT-2 (no shared code). For the prompt above both
print the same token ids, top-5 logits and greedy continuation (*"The meaning
of life is not the same as the meaning of death. …"*); over a 55-token mixed
English/Turkish/emoji/code prompt all 55 × 50257 logits agree within
1.1e-4 with the same argmax at every position. The tokenizer produces the
same ids as Hugging Face `tokenizers` on 20 tricky inputs (contractions,
whitespace runs, Turkish, CJK, emoji, code, symbols).

`--int8` converts the weights to 8-bit integers after loading (one float
scale per 64 inputs of every output, so a single outlier cannot coarsen a
whole row). Generation is memory-bandwidth bound — every token reads all the
weights — so a quarter of the bytes is much faster, at almost no cost:

| GPT-2 124M, 2-core x86 | float32 | int8 |
|---|---:|---:|
| generation | 15.5 tokens/s | **39 tokens/s** |
| perplexity (4096 Shakespeare tokens) | 106.8 | 107.3 (+0.5 %) |
| greedy continuation above | — | identical |

(With one scale per row instead of per 64 inputs, perplexity rose 4.2 %.)
The `gpt2` command keeps GPT-2's activations in float (`setInt8Inputs(false)`):
its activations have extreme outliers, and quantizing them too moved
perplexity by -3.7 % instead of +0.5 %.
Loading takes 1.5 s: layers are built without allocating weights or gradient
buffers (`SkipParameterInit`), and the checkpoint fills them directly.

### Llama family (Llama 3.x, TinyLlama, Llama 2, Mistral)

```
$ scripts/download_llama32.sh data/llama3        # Llama-3.2-1B-Instruct (~2.5 GB bf16)
$ ./build_and_run.sh llama data/llama3 "What is the capital of Turkey?" --chat
$ ./build_and_run.sh llama data/llama3 -i                     # multi-turn chat (/reset, /exit)
$ scripts/download_tinyllama.sh data/tinyllama   # TinyLlama-1.1B-Chat (Apache 2.0, ~2.2 GB bf16)
$ ./build_and_run.sh llama data/tinyllama "Write a haiku about Istanbul." 64 --chat --system "You are a poet."
$ ./build_and_run.sh llama data/llama3 "What is the capital of Turkey?" 20 --chat --check --fp32
```

`llama` loads a Hugging Face Llama-architecture checkpoint (`config.json`,
`model.safetensors` or a sharded `model.safetensors.index.json`,
`tokenizer.json`, and `generation_config.json` for the stop tokens) into the
same layers as the modern GPT: RMSNorm, rotary attention with grouped-query
heads (Llama 3.2 1B: 32 query heads sharing 8 key/value heads), SwiGLU and a
tied or untied output head. Llama 3.1+'s `"rope_type": "llama3"` frequency
scaling is applied (`RopeScaling`, `ropeFrequencies()`).

Sampling follows the model's `generation_config.json` (Llama 3.2: temperature
0.6, top-p 0.9) with top-k 40; `--temperature`, `--top-k`, `--top-p`,
`--repeat-penalty`, `--greedy` and `--seed` override it. `-i` keeps one
conversation going: a `Generator` holds the KV cache between turns, so each
turn only runs the new message through the model.

`--chat` picks the model's template: Llama 3's header/`<|eot_id|>` turns
(with the knowledge-cutoff/today system preamble Hugging Face renders) or
the Zephyr `<|system|>`/`<|user|>`/`<|assistant|>` turns of TinyLlama-Chat.
Generation stops at any of the model's end-of-turn ids.

`LlamaTokenizer` reads both kinds of `tokenizer.json`: SentencePiece style
(Llama 2, TinyLlama, Mistral: `▁` normalizer, byte fallback) and byte-level
BPE (Llama 3: the Llama 3 split pattern implemented by hand over full Unicode
letter/number tables, `ignore_merges`). Special tokens are split out first.
It gives the same ids as Hugging Face `tokenizers` on 34 tricky inputs for
Llama 3 (contractions in any case, digit runs, whitespace and line-break
runs, Turkish, CJK, Devanagari, Arabic, emoji sequences, code, the chat
template) and 51 for TinyLlama.

Weights are int8 by default (`--fp32` keeps float32), including the
embedding table, whose int8 copy also serves a tied output; each tensor is
quantized as it is read, so the float copy of the whole model never exists.
Products run on integers: activations are quantized per 32 values (as
llama.cpp's q8_0) and each group is an int8 dot product — AVX-VNNI
`vpdpbusd` (or AVX2 `vpmaddubsw`) on x86, `sdot` on Apple silicon and other
ARMv8.2+ CPUs, plain C++ elsewhere. Weight rows are prefetched 4 KB ahead,
which took the kernel from 11 to ~20 GB/s on a machine that streams memory at
~23 GB/s, and prompts process four weight rows per pass over the input.
`scripts/llama_reference.py` is an independent NumPy Llama: with `--fp32
--check` both print the same token ids, top-5 logits and greedy answer
(*"The capital of Turkey is Ankara."*), and over a mixed 42-token prompt all
42 × 128256 Llama 3.2 logits agree within 2.9e-5 (TinyLlama: 54 × 32000
within 3e-5).

| 2-core x86, int8 / float32 | Llama 3.2 1B | TinyLlama 1.1B |
|---|---:|---:|
| memory (peak RSS) | **1.6** / 5.9 GB | **1.5** / 4.8 GB |
| generation (160-token chat answer, incl. prompt) | **11.5** / 2.3 tokens/s | **13.9** / 2.3 tokens/s |
| one token / prompt processing | 13 / 33 tokens/s | — |
| loading (bf16 checkpoint) | 12 / 35 s | 12 / 35 s |
| perplexity, 1024 Shakespeare tokens | 35.131 / 35.088¹ | 13.269 / 13.264 |
| greedy answers above | identical | identical |

int8 generation was 3.6 and 4.5 tokens/s before the integer kernels (weights
expanded to float per row); with float activations perplexity was 35.090 and
13.277, so quantizing the activations costs 0.1 % on Llama 3.2.

¹ 256-token windows (512 for TinyLlama). The float32 value comes from the
NumPy reference, whose logits match NeuraCraft's float32 ones to 3e-5: a
float32 Llama 3.2 plus a 256-token pass does not fit in this machine's memory.

Not supported yet (the loader refuses them with a message): other RoPE
scaling types (linear, dynamic, YaRN), attention biases (Qwen 2; its
tokenizer pattern is supported), sliding-window attention (Mistral v0.1) and
tokenizer.json files with other pre-tokenizers (e.g. Metaspace).

### Text to image: Stable Diffusion (SD-Turbo)

```
$ scripts/download_sd_turbo.sh data/sd-turbo                  # SD-Turbo, fp16 (~2.6 GB)
$ ./build_and_run.sh txt2img data/sd-turbo "A cinematic photo of a red fox sitting in the snow, golden hour" fox.png --seed 42
$ ./build_and_run.sh txt2img data/sd-turbo "The Istanbul skyline at sunset, oil painting" ist.png --steps 2 --seed 7
$ ./build_and_run.sh vae data/sd-turbo photo.ppm out.png      # the autoencoder alone: encode + decode
```

<p><img src="docs/images/sd_turbo_fox.jpg" width="320"> <img src="docs/images/sd_turbo_istanbul.jpg" width="320"></p>

Both pictures above were generated by NeuraCraft on a 2-core CPU from the
prompts shown (1 and 2 steps). `textToImage()` runs the whole Stable Diffusion
pipeline on a diffusers directory, loading each model only for its stage
(peak memory 3.6 GB, the UNet's):

| SD-Turbo, 512 x 512, 2-core x86 | time |
|---|---:|
| CLIP text encoder (load + encode) | 15 s |
| UNet: load / one denoising step | 28 s / 14 s |
| VAE decode | 50 s |
| total for one image (1 step) | 107 s |

Pieces, each checked against an independent NumPy implementation (`scripts/`):

- **Prompt:** `ClipTokenizer` (CLIP's lowercased byte-level BPE with `</w>`
  word ends; same ids as transformers' `CLIPTokenizer` on 20 prompts, including
  SD 2's quirk of turning every "!" into the pad token) and
  `loadClipTextEncoder()` (Hugging Face `CLIPTextModel` on the GPT layers;
  `GELU` has erf and "quick" modes). SD-Turbo's 23-layer OpenCLIP ViT-H text
  model agrees with `clip_reference.py` within 7.9e-6.
- **Denoiser:** `loadUNet()` builds diffusers' `UNet2DConditionModel` (SD 1.x /
  2.x / Turbo layouts: time-conditioned residual blocks, spatial transformers
  with self- and cross-attention and GEGLU, skip connections); 866 M parameters,
  within 2.5e-6 of `unet_reference.py`. `multiHeadAttention()` processes
  queries in chunks, so 4096-pixel self-attention needs no 4096 x 4096 matrix
  per head. `EulerScheduler` follows diffusers' `EulerDiscreteScheduler`
  (scaled-linear betas; trailing, leading or linspace spacing).
  Classifier-free guidance (`--guidance`, `--negative`) is there for non-Turbo
  models.
- **Image:** `loadVae()` builds the KL autoencoder (`AutoencoderKL`) from the
  new image layers in `Image.h` (`GroupNorm`, `SiLU`, `Upsample2D`, `Pad2D`,
  `ResnetBlock2D`, `SpatialSelfAttention`) and `Conv2D`, whose inference path
  works in row strips written straight into channel-major output (memory
  stays bounded at 512 x 512; 1x1 convolutions are one GEMM). It agrees with
  `vae_reference.py` within 5.4e-6; an encode/decode round trip keeps a
  picture at 29-32 dB PSNR. PNG output and PPM input are built in.

The VAE decoder (~2.5 TFLOP at 512 x 512) is the slowest stage on a CPU; on
Apple silicon the products go to Accelerate. Next: GPU convolutions, int8
UNet weights, a tiny preview decoder, image-to-image.

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
| Transformer | `Transformer.h` | `TokenEmbedding` (optional learned positions), `LayerNorm`, `RMSNorm`, `GELU`, `SwiGLUGate`, multi-head `CausalSelfAttention` (optional RoPE with Llama 3 scaling, grouped-query attention, KV cache, sliding window), `Residual` blocks, `TiedOutput`, `buildGPT()` with `GPTConfig::modern()` |
| Quantization | `Quantized.h` | `QuantizedWeights` (int8, per-group scales; integer AVX-VNNI / AVX2 / NEON products), `Network::quantize()` for Dense, attention and embeddings (shared with tied outputs), `setInt8Inputs` |
| Image models | `Image.h`, `Diffusion.h` | `GroupNorm`, `SiLU`, `Upsample2D`, `Pad2D`, `ResnetBlock2D`, `SpatialSelfAttention`, PNG / PPM files, `loadVae()`, `loadClipTextEncoder()`, `ClipTokenizer`, `loadUNet()`, `multiHeadAttention()`, `EulerScheduler`, `textToImage()` |
| Pretrained | `Pretrained.h`, `SafeTensors.h`, `Json.h` | `loadGPT2()` (Hugging Face GPT-2 checkpoints), `loadLlama()` (Llama 3.x / TinyLlama / Llama 2 / Mistral, int8 or float32, sharded), `Gpt2Tokenizer` (byte-level BPE, identical ids), `LlamaTokenizer` (`tokenizer.json`: SentencePiece style and Llama 3 byte-level), safetensors reader (F32/F16/BF16) and writer, small JSON parser |
| Text | `Text.h` | `Tokenizer` interface, `CharTokenizer`, byte-level `BpeTokenizer` (train / save / load), `makeNextTokenDataset`, `generate()` (temperature, top-k, top-p, repetition penalty, streaming callback, stop tokens), `Generator` (multi-turn, KV cache kept between calls) |
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
the window, exact checkpoint resume, GPTs (both architectures) that memorize
and regenerate a sequence, the JSON parser, safetensors (incl. F16/BF16),
GPT-2 pre-tokenization (checked against Hugging Face), the GPT-2 tokenizer and
the GPT-2 loader on a synthetic checkpoint, int8 quantization error bounds,
grouped-query attention (gradients and KV cache), both Llama tokenizer kinds
and the Llama 3 pre-tokenizer (against Hugging Face splits), Llama 3 RoPE
scaling (against Hugging Face's formula), the Llama loader against an
independent reference forward pass (float32, int8, tied output, RoPE scaling,
sharded files, stop tokens), int8 embeddings, the integer int8 kernels
(error bound, single-row vs. batched paths, ragged sizes), multi-turn
generation against generating from the whole transcript, top-p and the
repetition penalty, that inference passes keep no backward state, the
convolution inference path against the training path, the image layers
(known values and gradient checks), spatial attention against a naive
implementation, PNG/PPM files (chunks, CRCs and pixels parsed back), the VAE
loader on a synthetic diffusers checkpoint, the GELU variants, the CLIP
tokenizer (cleanup, splitting, merges, added tokens, padding), the CLIP text
encoder against a naive reference on a synthetic checkpoint, chunked
multi-head attention against a naive one, the timestep embedding, the Euler
scheduler against diffusers' values, a synthetic UNet (text and time
conditioning), and the whole text-to-image pipeline on tiny synthetic models. CI runs
the tests, demos and benchmarks on Ubuntu and macOS (Apple silicon, both Metal
kernels).

## Layout

```
Includes/   public headers (NeuraCraft.h includes everything)
src/        library sources + Main.cpp (demo CLI -> bin/NeuraCraft)
tests/      unit tests
scripts/    download_{mnist,shakespeare,gpt2,tinyllama,llama32,sd_turbo}.sh, gpt2_reference.py, llama_reference.py,
            vae_reference.py, clip_reference.py, unet_reference.py,
            gen_unicode_tables.py (src/UnicodeTables.cpp)
```
