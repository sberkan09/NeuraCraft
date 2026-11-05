// NeuraCraft demo CLI.
//
//   NeuraCraft                 run the xor, spiral and sine demos
//   NeuraCraft xor|spiral|sine run one demo
//   NeuraCraft mnist <dir> [epochs] [mlp|cnn]
//                              train a digit classifier on MNIST IDX files in <dir>
//   NeuraCraft eval <dir> [mlp|cnn]
//                              evaluate the saved weights on the MNIST test set
//                              (see scripts/download_mnist.sh)
//   NeuraCraft gpt <text-file> [epochs] [bpe|char]
//                              train a GPT on a text file (scripts/download_shakespeare.sh)
//   NeuraCraft generate <text-file> [prompt] [length]
//                              sample from the trained GPT
//   NeuraCraft gpt2 <dir> [prompt] [tokens] [options]
//                              run pretrained GPT-2 weights (scripts/download_gpt2.sh)
//   NeuraCraft llama <dir> [prompt] [tokens] [options]
//                              run a Llama-family checkpoint (scripts/download_llama32.sh, download_tinyllama.sh)
//   NeuraCraft txt2img <dir> "prompt" [out.png] [--steps N] [--seed S] [--size N] [--width W --height H]
//                              text to image with Stable Diffusion / SD-Turbo (scripts/download_sd_turbo.sh)
//   NeuraCraft img2img <dir> in.png "prompt" [out.png] [--strength S] [--steps N] ...
//   NeuraCraft video <dir> morph "prompt 1" "prompt 2" ... [-o out.gif] [--seeds 1,2] [--frames N] [--fps F]
//   NeuraCraft video <dir> zoom "prompt" ... [-o out.gif] [--frames N] [--zoom Z] [--pan DX,DY] [--rotate DEG]
//   NeuraCraft video <dir> t2v "prompt" [-o out.gif] [--frames 16] [--steps 4]   (AnimateDiff text to video)
//                              image to image
//   NeuraCraft vae <dir> [in.ppm] [out.png] [--size N]
//                              Stable Diffusion VAE round trip (scripts/download_sd_turbo.sh)
//   NeuraCraft bench [--quick] matrix-multiply throughput of each backend
//
// Global options: --gpu (Metal on Apple silicon), --gpu-attention, --threads N, --profile

#include "NeuraCraft.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace nc;

namespace {

void banner(const std::string& title) {
    std::cout << "\n=== " << title << " " << std::string(title.size() < 60 ? 60 - title.size() : 0, '=') << '\n';
}

int runXor() {
    banner("XOR (2-8-1, tanh + sigmoid, binary cross-entropy)");
    const Dataset data = makeXor();

    Network net;
    net.add<Dense>(2, 8);
    net.add<Tanh>();
    net.add<Dense>(8, 1);
    net.add<Sigmoid>();
    net.summary();

    BinaryCrossEntropy loss;
    Adam optimizer(0.05f);
    FitOptions opts;
    opts.epochs = 1000;
    opts.batchSize = 4;
    opts.printEvery = 250;
    opts.trackAccuracy = true;
    net.fit(data.x, data.y, loss, optimizer, opts);

    const Matrix out = net.predict(data.x);
    for (std::size_t i = 0; i < data.x.rows(); ++i) {
        std::cout << "  " << data.x(i, 0) << " XOR " << data.x(i, 1) << " = " << std::fixed
                  << std::setprecision(3) << out(i, 0) << std::defaultfloat << "  (target " << data.y(i, 0) << ")\n";
    }
    const float acc = net.accuracy(data.x, data.y);
    std::cout << "accuracy: " << acc * 100.0f << "%\n";
    return acc == 1.0f ? 0 : 1;
}

int runSpiral() {
    banner("Spiral, 3 classes (2-64-64-3, ReLU, softmax cross-entropy)");
    const Dataset data = makeSpiral(100, 3, 0.2f);
    const Split split = trainTestSplit(data.x, data.y, 0.2f);

    Network net;
    net.add<Dense>(2, 64, Init::He);
    net.add<ReLU>();
    net.add<Dense>(64, 64, Init::He);
    net.add<ReLU>();
    net.add<Dense>(64, 3);
    net.summary();

    SoftmaxCrossEntropy loss;
    Adam optimizer(0.01f);
    FitOptions opts;
    opts.epochs = 300;
    opts.batchSize = 32;
    opts.printEvery = 50;
    opts.trackAccuracy = true;
    opts.validationX = &split.test.x;
    opts.validationY = &split.test.y;
    net.fit(split.train.x, split.train.y, loss, optimizer, opts);

    // ASCII map of the learned decision regions; training points shown in caps.
    const int w = 56, h = 24;
    const char region[] = {'.', '-', '~'};
    const char point[] = {'A', 'B', 'C'};
    std::vector<std::string> canvas(h, std::string(w, ' '));
    Matrix grid(static_cast<std::size_t>(w * h), 2);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            grid(static_cast<std::size_t>(r * w + c), 0) = -1.1f + 2.2f * static_cast<float>(c) / (w - 1);
            grid(static_cast<std::size_t>(r * w + c), 1) = 1.1f - 2.2f * static_cast<float>(r) / (h - 1);
        }
    const auto labels = argmaxRows(net.predict(grid));
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) canvas[r][c] = region[labels[static_cast<std::size_t>(r * w + c)]];
    for (std::size_t i = 0; i < data.x.rows(); i += 3) {
        const int c = static_cast<int>(std::lround((data.x(i, 0) + 1.1f) / 2.2f * (w - 1)));
        const int r = static_cast<int>(std::lround((1.1f - data.x(i, 1)) / 2.2f * (h - 1)));
        if (r >= 0 && r < h && c >= 0 && c < w) canvas[r][c] = point[data.y.argmaxRow(i)];
    }
    std::cout << "decision regions (. - ~) and samples (A B C):\n";
    for (const auto& line : canvas) std::cout << "  |" << line << "|\n";

    const float acc = net.accuracy(split.test.x, split.test.y);
    std::cout << "test accuracy: " << std::fixed << std::setprecision(1) << acc * 100.0f << std::defaultfloat << "%\n";
    return acc >= 0.9f ? 0 : 1;
}

int runSine() {
    banner("Sine regression (1-32-32-1, tanh, MSE)");
    const Dataset data = makeSine(200, 0.05f);

    Network net;
    net.add<Dense>(1, 32);
    net.add<Tanh>();
    net.add<Dense>(32, 32);
    net.add<Tanh>();
    net.add<Dense>(32, 1);
    net.summary();

    MSE loss;
    Adam optimizer(0.01f);
    FitOptions opts;
    opts.epochs = 400;
    opts.batchSize = 32;
    opts.printEvery = 100;
    net.fit(data.x, data.y, loss, optimizer, opts);

    const Dataset clean = makeSine(9);
    const Matrix out = net.predict(clean.x);
    std::cout << "        x    sin(x)   predicted\n";
    for (std::size_t i = 0; i < clean.x.rows(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << std::setw(9) << clean.x(i, 0) << std::setw(10)
                  << clean.y(i, 0) << std::setw(12) << out(i, 0) << '\n';
    }
    MSE eval;
    const float mse = net.evaluate(clean.x, clean.y, eval);
    std::cout << "MSE vs. noiseless sine: " << std::setprecision(5) << mse << std::defaultfloat << '\n';
    return mse < 0.01f ? 0 : 1;
}

struct Mnist {
    Matrix trainX, trainY, testX, testY;
    std::vector<int> testLabels;
};

bool loadMnist(const std::string& dir, bool withTraining, Mnist& m) {
    try {
        if (withTraining) {
            m.trainX = loadIdxImages(dir + "/train-images-idx3-ubyte");
            m.trainY = oneHot(loadIdxLabels(dir + "/train-labels-idx1-ubyte"), 10);
        }
        m.testX = loadIdxImages(dir + "/t10k-images-idx3-ubyte");
        m.testLabels = loadIdxLabels(dir + "/t10k-labels-idx1-ubyte");
        m.testY = oneHot(m.testLabels, 10);
    } catch (const std::exception& e) {
        std::cerr << "could not load MNIST: " << e.what() << "\n"
                  << "download it with: scripts/download_mnist.sh " << dir << '\n';
        return false;
    }
    return true;
}

// The architectures must match between training and `eval`, since weight files
// store parameter values only.
bool buildMnistModel(Network& net, const std::string& model) {
    if (model == "mlp") {
        net.add<Dense>(784, 128, Init::He);
        net.add<ReLU>();
        net.add<Dropout>(0.2f);
        net.add<Dense>(128, 64, Init::He);
        net.add<ReLU>();
        net.add<Dense>(64, 10);
        return true;
    }
    if (model == "cnn") {
        net.add<Conv2D>(ImageShape{1, 28, 28}, 16, 3, 1, 1);
        net.add<ReLU>();
        net.add<MaxPool2D>(ImageShape{16, 28, 28}, 2);
        net.add<Conv2D>(ImageShape{16, 14, 14}, 32, 3, 1, 1);
        net.add<ReLU>();
        net.add<MaxPool2D>(ImageShape{32, 14, 14}, 2);
        net.add<Dense>(32 * 7 * 7, 128, Init::He);
        net.add<ReLU>();
        net.add<Dropout>(0.3f);
        net.add<Dense>(128, 10);
        return true;
    }
    std::cerr << "unknown model '" << model << "' (use mlp or cnn)\n";
    return false;
}

std::string mnistWeightsPath(const std::string& dir, const std::string& model) {
    return dir + (model == "mlp" ? "/mnist.weights" : "/mnist_" + model + ".weights");
}

int runMnist(const std::string& dir, std::size_t epochs, const std::string& model) {
    banner("MNIST " + model + " (Adam, batch 64)");
    Network net;
    if (!buildMnistModel(net, model)) return 2;
    Mnist m;
    if (!loadMnist(dir, true, m)) return 2;
    std::cout << "train: " << m.trainX.rows() << " images, test: " << m.testX.rows() << " images\n";
    net.summary();

    SoftmaxCrossEntropy loss;
    Adam optimizer(0.001f);
    FitOptions opts;
    opts.epochs = epochs;
    opts.batchSize = 64;
    opts.trackAccuracy = true;
    opts.validationX = &m.testX;
    opts.validationY = &m.testY;
    const auto start = std::chrono::steady_clock::now();
    net.fit(m.trainX, m.trainY, loss, optimizer, opts);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const std::string weights = mnistWeightsPath(dir, model);
    net.save(weights);
    std::cout << "trained in " << std::fixed << std::setprecision(1) << seconds << " s ("
              << seconds / static_cast<double>(epochs) << " s/epoch)\n"
              << "weights saved to " << weights << '\n'
              << "test accuracy: " << std::setprecision(2) << net.accuracy(m.testX, m.testY) * 100.0f
              << std::defaultfloat << "%\n";
    return 0;
}

void printDigit(const Matrix& images, std::size_t row) {
    static const char shades[] = " .:-=+*#%@";
    for (std::size_t y = 0; y < 28; y += 2) {  // halve rows: terminal cells are tall
        std::cout << "    ";
        for (std::size_t x = 0; x < 28; ++x) {
            const float v = std::max(images(row, y * 28 + x), images(row, (y + 1) * 28 + x));
            std::cout << shades[static_cast<int>(v * 9.0f + 0.5f)];
        }
        std::cout << '\n';
    }
}

int runEval(const std::string& dir, const std::string& model) {
    banner("Evaluate saved MNIST " + model);
    Network net;
    if (!buildMnistModel(net, model)) return 2;
    Mnist m;
    if (!loadMnist(dir, false, m)) return 2;
    const std::string weights = mnistWeightsPath(dir, model);
    try {
        net.load(weights);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ntrain first with: NeuraCraft mnist " << dir << " 5 " << model << '\n';
        return 2;
    }
    std::cout << "loaded " << weights << " (" << net.parameterCount() << " parameters)\n";

    std::vector<int> predicted;
    for (std::size_t start = 0; start < m.testX.rows(); start += 500) {
        const std::size_t end = std::min(m.testX.rows(), start + 500);
        const auto batch = argmaxRows(net.predict(m.testX.sliceRows(start, end)));
        predicted.insert(predicted.end(), batch.begin(), batch.end());
    }

    std::size_t correct = 0, perTotal[10] = {}, perCorrect[10] = {};
    std::vector<std::size_t> mistakes;
    for (std::size_t i = 0; i < predicted.size(); ++i) {
        const int label = m.testLabels[i];
        ++perTotal[label];
        if (predicted[i] == label) {
            ++correct;
            ++perCorrect[label];
        } else {
            mistakes.push_back(i);
        }
    }
    std::cout << "test accuracy: " << std::fixed << std::setprecision(2)
              << 100.0 * static_cast<double>(correct) / static_cast<double>(predicted.size()) << "%  ("
              << mistakes.size() << " mistakes)\nper digit:";
    for (int d = 0; d < 10; ++d)
        std::cout << "  " << d << ": " << std::setprecision(1)
                  << 100.0 * static_cast<double>(perCorrect[d]) / static_cast<double>(std::max<std::size_t>(1, perTotal[d]))
                  << "%";
    std::cout << std::defaultfloat << "\n";

    for (std::size_t k = 0; k < std::min<std::size_t>(3, mistakes.size()); ++k) {
        const std::size_t i = mistakes[k];
        std::cout << "\n  test image #" << i << ": label " << m.testLabels[i] << ", predicted " << predicted[i] << '\n';
        printDigit(m.testX, i);
    }
    return 0;
}

GPTConfig textModelConfig(std::size_t vocab) {
    GPTConfig cfg;
    cfg.vocabSize = vocab;
    cfg.contextLength = 64;
    cfg.dim = 128;
    cfg.heads = 4;
    cfg.layers = 4;
    cfg.dropout = 0.1f;
    return cfg;
}

std::string gptWeightsPath(const std::string& textPath) { return textPath + ".gpt.weights"; }
std::string gptTokenizerPath(const std::string& textPath) { return textPath + ".gpt.tokenizer"; }
std::string gptConfigPath(const std::string& textPath) { return textPath + ".gpt.config"; }
std::string gptCheckpointPath(const std::string& textPath) { return textPath + ".gpt.checkpoint"; }

// Model built for generation/resume: the saved config wins over the defaults.
GPTConfig savedOrDefaultConfig(const std::string& textPath, std::size_t vocab) {
    std::ifstream probe(gptConfigPath(textPath));
    GPTConfig cfg = probe ? loadGPTConfig(gptConfigPath(textPath)) : textModelConfig(vocab);
    cfg.vocabSize = vocab;
    return cfg;
}

struct GptOptions {
    std::size_t epochs = 1;
    std::string tokenizer = "bpe";
    bool resume = false;
    bool cosine = false;  // warmup + cosine decay (helps long runs; constant is better for a few epochs)
    float learningRate = 1e-3f;
    std::size_t dim = 0, layers = 0, heads = 0, context = 0;  // 0 = default / saved
    std::string arch = "modern";  // modern (RoPE, RMSNorm, SwiGLU, tied) or classic (GPT-2 style)
};
constexpr std::size_t kBpeVocab = 512;

// The tokenizer file records which tokenizer the weights were trained with:
// BPE merges, or a marker meaning "character vocabulary of the text file".
void saveTokenizer(const std::string& textPath, const Tokenizer& tok) {
    if (const auto* bpe = dynamic_cast<const BpeTokenizer*>(&tok)) {
        bpe->save(gptTokenizerPath(textPath));
    } else {
        std::ofstream out(gptTokenizerPath(textPath));
        out << "NEURACRAFT-CHAR 1\n";
        if (!out) throw std::runtime_error("cannot write " + gptTokenizerPath(textPath));
    }
}

std::unique_ptr<Tokenizer> loadTokenizer(const std::string& textPath, const std::string& text) {
    std::ifstream in(gptTokenizerPath(textPath));
    std::string magic;
    if (!in || !(in >> magic) || magic == "NEURACRAFT-CHAR") return std::make_unique<CharTokenizer>(text);
    return std::make_unique<BpeTokenizer>(BpeTokenizer::load(gptTokenizerPath(textPath)));
}

void printSample(Network& net, const SequenceContextPtr& ctx, const Tokenizer& tok, const std::string& prompt,
                 std::size_t length, std::size_t context) {
    SamplingOptions sampling;
    sampling.temperature = 0.8f;
    sampling.topK = 20;
    std::cout << "----- sample -----\n"
              << prompt << tok.decode(generate(net, tok.encode(prompt), length, context, sampling, ctx))
              << "\n------------------" << std::endl;
}

int runGpt(const std::string& path, const GptOptions& o) {
    const std::string& tokenizerKind = o.tokenizer;
    banner("GPT language model (" + tokenizerKind + " tokens)");
    if (tokenizerKind != "char" && tokenizerKind != "bpe") {
        std::cerr << "unknown tokenizer '" << tokenizerKind << "' (use char or bpe)\n";
        return 2;
    }
    std::string text;
    try {
        text = loadText(path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ndownload a corpus with: scripts/download_shakespeare.sh\n";
        return 2;
    }
    std::unique_ptr<Tokenizer> tok;
    if (o.resume) {
        tok = loadTokenizer(path, text);  // must match the checkpoint
    } else if (tokenizerKind == "bpe") {
        const auto t0 = std::chrono::steady_clock::now();
        tok = std::make_unique<BpeTokenizer>(BpeTokenizer::train(text, kBpeVocab));
        std::cout << "trained BPE (" << tok->vocabSize() << " tokens) in " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n"
                  << std::defaultfloat;
    } else {
        tok = std::make_unique<CharTokenizer>(text);
    }
    const std::vector<int> ids = tok->encode(text);
    const std::size_t split = ids.size() * 9 / 10;
    const std::vector<int> trainIds(ids.begin(), ids.begin() + static_cast<long>(split));
    const std::vector<int> valIds(ids.begin() + static_cast<long>(split), ids.end());
    // Loss per character makes runs with different tokenizers comparable.
    const double tokensPerChar = static_cast<double>(valIds.size()) / static_cast<double>(tok->decode(valIds).size());

    GPTConfig cfg = o.resume ? savedOrDefaultConfig(path, tok->vocabSize()) : textModelConfig(tok->vocabSize());
    if (!o.resume) {
        if (o.arch == "modern") cfg.modern();
        if (o.dim) cfg.dim = o.dim;
        if (o.layers) cfg.layers = o.layers;
        if (o.heads) cfg.heads = o.heads;
        if (o.context) cfg.contextLength = o.context;
    }
    const Dataset train = makeNextTokenDataset(trainIds, cfg.contextLength);
    const Dataset val = makeNextTokenDataset(valIds, cfg.contextLength);
    std::cout << path << ": " << text.size() << " characters -> " << ids.size() << " tokens ("
              << std::setprecision(3) << 1.0 / tokensPerChar << std::defaultfloat << " chars/token), vocabulary "
              << tok->vocabSize() << ", " << train.x.rows() << " training / " << val.x.rows()
              << " validation sequences of " << cfg.contextLength << " tokens\n";

    Network net;
    const SequenceContextPtr ctx = buildGPT(net, cfg);
    net.summary();
    SparseSoftmaxCrossEntropy loss;
    Adam optimizer(o.learningRate);

    const std::size_t batchSize = 32;
    const std::size_t stepsPerEpoch = (train.x.rows() + batchSize - 1) / batchSize;
    std::size_t startStep = 0;
    if (o.resume) {
        try {
            startStep = loadCheckpoint(gptCheckpointPath(path), net, optimizer);
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n(start a new run without --resume)\n";
            return 2;
        }
        std::cout << "resumed from step " << startStep << " (epoch " << startStep / stepsPerEpoch << ")\n";
    } else {
        saveTokenizer(path, *tok);
        saveGPTConfig(gptConfigPath(path), cfg);
    }
    // The schedule spans all epochs of the run, counted from the first step, so a
    // resumed run continues the same curve (--resume with N epochs extends it).
    const std::size_t doneEpochs = startStep / stepsPerEpoch;
    CosineSchedule schedule;
    schedule.base = o.learningRate;
    schedule.totalSteps = (doneEpochs + o.epochs) * stepsPerEpoch;
    schedule.warmupSteps = std::min<std::size_t>(200, schedule.totalSteps / 10);
    schedule.minRatio = 0.1f;

    FitOptions opts;
    opts.epochs = o.epochs;
    opts.batchSize = batchSize;
    opts.printEveryBatches = 50;
    opts.validationX = &val.x;
    opts.validationY = &val.y;
    if (o.cosine) opts.learningRate = schedule;
    opts.clipGradNorm = 1.0f;
    opts.startStep = startStep;
    const std::string prompt(1, text[0]);
    opts.onEpochEnd = [&](const EpochStats& s) {
        std::cout << "validation loss per character: " << std::fixed << std::setprecision(4)
                  << s.validationLoss * tokensPerChar << "  (step " << s.step << ", lr " << std::scientific
                  << std::setprecision(2) << s.learningRate << ", grad norm " << std::fixed << std::setprecision(2)
                  << s.gradientNorm << ")" << std::defaultfloat << '\n';
        saveCheckpoint(gptCheckpointPath(path), net, optimizer, s.step);
        printSample(net, ctx, *tok, prompt, 300 / (tokenizerKind == "bpe" ? 2 : 1), cfg.contextLength);
    };
    const auto start = std::chrono::steady_clock::now();
    net.fit(train.x, train.y, loss, optimizer, opts);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    net.save(gptWeightsPath(path));
    std::cout << "trained in " << std::fixed << std::setprecision(1) << seconds << " s ("
              << seconds / static_cast<double>(o.epochs) << " s/epoch, incl. samples)\n" << std::defaultfloat
              << "saved " << gptWeightsPath(path) << ", " << gptConfigPath(path) << ", " << gptTokenizerPath(path)
              << " and " << gptCheckpointPath(path) << "\ncontinue training with: NeuraCraft gpt " << path
              << " N --resume\ngenerate text with: NeuraCraft generate " << path << " \"PROMPT\" 500\n";
    return 0;
}

int runGenerate(const std::string& path, const std::string& prompt, std::size_t length) {
    std::string text;
    std::unique_ptr<Tokenizer> tok;
    try {
        text = loadText(path);
        tok = loadTokenizer(path, text);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    const GPTConfig cfg = savedOrDefaultConfig(path, tok->vocabSize());
    Network net;
    const SequenceContextPtr ctx = buildGPT(net, cfg);
    try {
        net.load(gptWeightsPath(path));
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ntrain first with: NeuraCraft gpt " << path << '\n';
        return 2;
    }
    std::vector<int> promptIds;
    try {
        promptIds = tok->encode(prompt.empty() ? std::string(1, text[0]) : prompt);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    SamplingOptions sampling;
    sampling.temperature = 0.8f;
    sampling.topK = 20;
    // `length` counts tokens; with BPE a token is ~2 characters.
    const auto start = std::chrono::steady_clock::now();
    const std::vector<int> out = generate(net, promptIds, length, cfg.contextLength, sampling, ctx);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << prompt << tok->decode(out) << '\n';
    std::cerr << out.size() << " tokens in " << std::fixed << std::setprecision(2) << seconds << " s ("
              << std::setprecision(0) << static_cast<double>(out.size()) / seconds << " tokens/s, KV cache)\n";
    return 0;
}

std::string quoted(const std::string& s) {  // Python-repr-like, for --check output
    std::string out = "'";
    for (char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\'') out += "\\'";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out + "'";
}

int runGpt2(const std::string& dir, const std::string& prompt, std::size_t tokens, const SamplingOptions& base,
            bool check, bool int8) {
    const auto t0 = std::chrono::steady_clock::now();
    PretrainedModel model;
    std::unique_ptr<Gpt2Tokenizer> tok;
    try {
        tok = std::make_unique<Gpt2Tokenizer>(dir + "/vocab.json", dir + "/merges.txt");
        model = loadGPT2(dir);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ndownload GPT-2 with: scripts/download_gpt2.sh " << dir << '\n';
        return 2;
    }
    if (int8) {
        // GPT-2's activations have extreme outliers: keep them float, quantize only the weights.
        setInt8Inputs(false);
        model.net.quantize();
    }
    const double loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << (int8 ? "loaded GPT-2, int8 weights (" : "loaded GPT-2 (");
    std::cerr << "" << model.config.layers
              << " layers, dim " << model.config.dim << ") in " << std::fixed << std::setprecision(1) << loadSeconds
              << " s" << std::defaultfloat << '\n';

    std::vector<int> ids = tok->encode(prompt);
    if (ids.empty()) ids.push_back(tok->endOfText());
    if (check) {
        std::cout << "prompt ids:";
        for (int id : ids) std::cout << ' ' << id;
        std::cout << '\n';
        Matrix x(1, ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) x(0, i) = static_cast<float>(ids[i]);
        const Matrix logits = model.net.predict(x);
        const float* last = logits.data() + (logits.rows() - 1) * logits.cols();
        std::vector<int> order(logits.cols());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int a, int b) { return last[a] > last[b]; });
        std::cout << "top-5 next:" << std::fixed << std::setprecision(4);
        for (int k = 0; k < 5; ++k) std::cout << ' ' << order[static_cast<std::size_t>(k)] << ':' << last[order[static_cast<std::size_t>(k)]];
        std::cout << std::defaultfloat << '\n';
        SamplingOptions greedy;
        greedy.temperature = 0.0f;
        const auto more = generate(model.net, ids, tokens, model.config.contextLength, greedy, model.context);
        std::vector<int> all = ids;
        all.insert(all.end(), more.begin(), more.end());
        std::cout << "greedy: " << quoted(tok->decode(all)) << '\n';
        return 0;
    }

    SamplingOptions sampling = base;
    sampling.stopToken = tok->endOfText();
    sampling.onToken = [&](int id) {
        if (id != tok->endOfText()) std::cout << tok->decode({id}) << std::flush;
    };
    std::cout << prompt << std::flush;
    const auto start = std::chrono::steady_clock::now();
    const auto out = generate(model.net, ids, tokens, model.config.contextLength, sampling, model.context);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << '\n';
    std::cerr << out.size() << " tokens in " << std::fixed << std::setprecision(2) << seconds << " s ("
              << std::setprecision(1) << static_cast<double>(out.size()) / seconds << " tokens/s, prompt "
              << ids.size() << " tokens)" << std::defaultfloat << '\n';
    return 0;
}

struct LlamaOptions {
    std::string prompt = "What is the capital of Turkey?";
    bool promptGiven = false;
    std::string system;
    std::size_t tokens = 128;
    bool chat = false, check = false, int8 = true, interactive = false;
    bool temperatureGiven = false, topPGiven = false;
    SamplingOptions sampling;
};

// Turn layout of the model's chat format.
struct ChatTemplate {
    bool llama3 = false;
    std::string today;

    std::string firstTurn(const std::string& system, const std::string& user) const {
        if (llama3)  // as Hugging Face renders it: a system turn with the knowledge cutoff and today's date
            return "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: " + today +
                   "\n\n" + system + "<|eot_id|>" + nextTurn(user);
        // Zephyr-style turns used by TinyLlama-Chat, each ended by </s>.
        return (system.empty() ? std::string() : "<|system|>\n" + system + "</s>\n") + "<|user|>\n" + user +
               "</s>\n<|assistant|>\n";
    }
    std::string nextTurn(const std::string& user) const {
        if (llama3) return "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        return "<|user|>\n" + user + "</s>\n<|assistant|>\n";
    }
    // Between a finished reply and the next user turn; closes a reply that hit the token limit first.
    std::string between(bool replyClosed) const {
        if (llama3) return replyClosed ? "" : "<|eot_id|>";
        return replyClosed ? "\n" : "</s>\n";
    }
};

// A procedural test picture (sky, sun, hills, stripes, checkerboard) for the VAE demo.
Image testPattern(std::size_t size) {
    Image img;
    img.width = img.height = size;
    img.pixels.resize(size * size * 3);
    for (std::size_t y = 0; y < size; ++y)
        for (std::size_t x = 0; x < size; ++x) {
            const double u = static_cast<double>(x) / size, v = static_cast<double>(y) / size;
            double r = 0.35 + 0.4 * (1 - v), g = 0.55 + 0.3 * (1 - v), b = 0.95;
            if ((u - 0.72) * (u - 0.72) + (v - 0.25) * (v - 0.25) < 0.012) {
                r = 1.0;
                g = 0.85;
                b = 0.2;
            }
            const double hill = 0.62 + 0.08 * std::sin(u * 9.0);
            if (v > hill) {
                r = 0.15 + 0.1 * std::sin(u * 40);
                g = 0.55 - 0.3 * (v - hill);
                b = 0.2;
            }
            if (v > 0.85) r = g = b = ((x / (size / 16)) + (y / (size / 16))) % 2 ? 0.9 : 0.1;
            if (u < 0.12 && v < 0.8) {
                const bool s = (y / 6) % 2;
                r = s ? 0.9 : 0.6;
                g = 0.1;
                b = s ? 0.2 : 0.5;
            }
            const double px[3] = {r, g, b};
            for (std::size_t c = 0; c < 3; ++c)
                img.pixels[(y * size + x) * 3 + c] = static_cast<std::uint8_t>(std::lround(std::min(1.0, std::max(0.0, px[c])) * 255));
        }
    return img;
}

int runVae(const std::string& dir, const std::string& input, const std::string& output, std::size_t size) {
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
    Image img;
    try {
        img = input.empty() ? testPattern(size) : loadPpm(input);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
    if (img.channels != 3 || img.width % 8 || img.height % 8) {
        std::cerr << "the image must be RGB with width and height multiples of 8 (got " << img.width << "x" << img.height << ")\n";
        return 2;
    }
    auto t = Clock::now();
    Vae vae;
    try {
        vae = loadVae(dir, img.width, img.height);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ndownload the VAE with: scripts/download_sd_turbo.sh " << dir << '\n';
        return 2;
    }
    std::cout << "loaded the VAE for " << img.width << "x" << img.height << " in " << std::fixed << std::setprecision(1) << seconds(t)
              << " s; latents are " << vae.latent.channels << "x" << vae.latent.height << "x" << vae.latent.width << '\n';
    t = Clock::now();
    const Matrix z = vae.encode(imageToTensor(img));
    std::cout << "encode " << seconds(t) << " s";
    t = Clock::now();
    const Matrix back = vae.decode(z);
    std::cout << ", decode " << seconds(t) << " s\n";
    const Image rec = tensorToImage(back.data(), vae.image);
    double se = 0.0;
    for (std::size_t i = 0; i < img.pixels.size(); ++i) {
        const double d = static_cast<double>(img.pixels[i]) - rec.pixels[i];
        se += d * d;
    }
    const std::string original = output.substr(0, output.rfind('.')) + "_input.png";
    savePng(original, img);
    savePng(output, rec);
    std::cout << "reconstruction PSNR " << std::setprecision(2) << 10.0 * std::log10(255.0 * 255.0 / (se / img.pixels.size()))
              << " dB; wrote " << output << " (input: " << original << ")" << std::defaultfloat << '\n';
    return 0;
}

int runLlama(const std::string& dir, const LlamaOptions& o) {
    const auto t0 = std::chrono::steady_clock::now();
    PretrainedModel model;
    std::unique_ptr<LlamaTokenizer> tok;
    try {
        tok = std::make_unique<LlamaTokenizer>(dir + "/tokenizer.json");
        model = loadLlama(dir, o.int8);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\ndownload a model with: scripts/download_llama32.sh (or download_tinyllama.sh) " << dir << '\n';
        return 2;
    }
    const double loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "loaded " << model.config.layers << "-layer Llama (dim " << model.config.dim << ", " << model.config.heads
              << " heads, " << model.config.kvHeads << " kv heads, "
              << (o.int8 ? std::string(QuantizedWeights::kernelName()) : std::string("float32")) << ") in "
              << std::fixed << std::setprecision(1) << loadSeconds << " s" << std::defaultfloat << '\n';

    ChatTemplate chat;
    chat.llama3 = tok->tokenId("<|start_header_id|>") >= 0;
    char today[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(today, sizeof(today), "%d %b %Y", std::localtime(&now));
    chat.today = today;
    const bool useChat = o.chat || o.interactive;

    SamplingOptions sampling = o.sampling;
    if (!o.temperatureGiven && model.temperature >= 0.0f) sampling.temperature = model.temperature;
    if (!o.topPGiven && model.topP > 0.0f) sampling.topP = model.topP;
    sampling.stopToken = tok->eos();
    sampling.stopTokens = model.stopTokens;
    auto isStop = [&](int id) {
        return id == sampling.stopToken || std::find(sampling.stopTokens.begin(), sampling.stopTokens.end(), id) != sampling.stopTokens.end();
    };

    if (o.interactive) {
        std::cerr << "chat (temperature " << sampling.temperature << ", top-k " << sampling.topK << ", top-p " << sampling.topP
                  << "); /reset starts over, /exit or Ctrl-D quits\n";
        Generator session(model.net, model.context, model.config.contextLength);
        bool firstTurn = true, closed = true;
        std::string line = o.promptGiven ? o.prompt : std::string();
        bool havePending = o.promptGiven;
        for (;;) {
            if (!havePending) {
                std::cout << "\n> " << std::flush;
                if (!std::getline(std::cin, line)) break;
            } else {
                std::cout << "\n> " << line << '\n';
            }
            havePending = false;
            if (line == "/exit" || line == "/quit") break;
            if (line == "/reset") {
                session.reset();
                firstTurn = closed = true;
                std::cout << "(conversation cleared)\n";
                continue;
            }
            if (line.empty()) continue;
            const std::string text = firstTurn ? chat.firstTurn(o.system, line) : chat.between(closed) + chat.nextTurn(line);
            const std::vector<int> ids = tok->encode(text, firstTurn);
            SamplingOptions turn = sampling;
            bool startOfReply = true;
            turn.onToken = [&](int id) {
                std::string piece = tok->decodePiece(id);
                if (startOfReply && !piece.empty() && piece[0] == ' ') piece.erase(0, 1);
                startOfReply = false;
                std::cout << piece << std::flush;
            };
            const auto start = std::chrono::steady_clock::now();
            const auto reply = session.generate(ids, o.tokens, turn);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            closed = !reply.empty() && isStop(reply.back());
            firstTurn = false;
            std::cout << '\n';
            std::cerr << "[" << reply.size() << " tokens, " << std::fixed << std::setprecision(1)
                      << static_cast<double>(reply.size()) / seconds << " tokens/s incl. " << ids.size() << " new prompt tokens; "
                      << session.history().size() << " tokens in the conversation]" << std::defaultfloat << '\n';
        }
        return 0;
    }

    const std::string text = useChat ? chat.firstTurn(o.system, o.prompt) : o.prompt;
    const std::vector<int> ids = tok->encode(text, true);

    if (o.check) {
        std::cout << "prompt ids:";
        for (int id : ids) std::cout << ' ' << id;
        std::cout << '\n';
        Matrix x(1, ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) x(0, i) = static_cast<float>(ids[i]);
        const Matrix logits = model.net.predict(x);
        const float* last = logits.data() + (logits.rows() - 1) * logits.cols();
        std::vector<int> order(logits.cols());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int a, int b) { return last[a] > last[b]; });
        std::cout << "top-5 next:" << std::fixed << std::setprecision(4);
        for (int k = 0; k < 5; ++k) std::cout << ' ' << order[static_cast<std::size_t>(k)] << ':' << last[order[static_cast<std::size_t>(k)]];
        std::cout << std::defaultfloat << '\n';
        SamplingOptions greedy;
        greedy.temperature = 0.0f;
        greedy.stopToken = sampling.stopToken;
        greedy.stopTokens = sampling.stopTokens;
        const auto more = generate(model.net, ids, o.tokens, model.config.contextLength, greedy, model.context);
        std::cout << "greedy: " << quoted(tok->decode(more)) << '\n';
        return 0;
    }

    bool first = true;
    sampling.onToken = [&](int id) {
        std::string piece = tok->decodePiece(id);
        if (first && !piece.empty() && piece[0] == ' ') piece.erase(0, 1);
        first = false;
        std::cout << piece << std::flush;
    };
    if (!useChat) std::cout << o.prompt << std::flush;
    const auto start = std::chrono::steady_clock::now();
    const auto out = generate(model.net, ids, o.tokens, model.config.contextLength, sampling, model.context);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << '\n';
    std::cerr << out.size() << " tokens in " << std::fixed << std::setprecision(2) << seconds << " s ("
              << std::setprecision(1) << static_cast<double>(out.size()) / seconds << " tokens/s incl. a "
              << ids.size() << "-token prompt)" << std::defaultfloat << '\n';
    return 0;
}

// Times `run` (one n x n x n product); returns GFLOPS.
double timeGemm(std::size_t n, const std::function<void()>& run) {
    run();  // warm-up
    const double flops = 2.0 * static_cast<double>(n) * n * n;
    int reps = 0;
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    do {
        run();
        ++reps;
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    } while (elapsed < 0.25);
    return flops * reps / elapsed / 1e9;
}

int runBench(bool quick) {
    banner("Matrix multiply benchmark");
    std::cout << "CPU backend: " << cpuBackendName() << ", " << threadCount() << " threads\n"
              << "GPU: " << (gpuAvailable() ? gpuName() + " (Metal)" : std::string("not available")) << "\n\n";

    using RawGemm = void (*)(bool, bool, std::size_t, std::size_t, std::size_t, const float*, std::size_t,
                             const float*, std::size_t, float*, std::size_t, bool);
    struct Column {
        std::string name;
        RawGemm raw;        // raw-pointer implementation, or
        bool gpuMatmul;     // Matrix-level GPU path (page-backed operands mapped in place)
    };
    std::vector<Column> columns{{"blocked", gemmBlocked, false}};
    if (cpuBackendName() != "NeuraCraft blocked kernel") columns.push_back({"accelerate", gemmCpu, false});
    if (gpuAvailable()) {
        columns.push_back({"metal copy", gemmGpu, false});
        columns.push_back({"metal 0-copy", nullptr, true});
    }

    const std::vector<std::size_t> sizes = quick ? std::vector<std::size_t>{64, 129}
                                                 : std::vector<std::size_t>{128, 256, 512, 1024, 2048};
    std::cout << "     n";
    for (const auto& col : columns) std::cout << std::setw(14) << col.name;
    std::cout << "   (GFLOPS)\n";

    int mismatches = 0;
    for (std::size_t n : sizes) {
        const Matrix a = Matrix::uniform(n, n, -1, 1), b = Matrix::uniform(n, n, -1, 1);
        Matrix expected(n, n);
        gemmBlocked(false, false, n, n, n, a.data(), n, b.data(), n, expected.data(), n, false);
        std::cout << std::setw(6) << n;
        for (const auto& col : columns) {
            Matrix c(n, n);
            double gflops = 0.0;
            if (col.gpuMatmul) {
                const Device savedDevice = currentDevice();
                const std::size_t savedMin = gpuMinWork();
                setDevice(Device::GPU);
                setGpuMinWork(0);
                gflops = timeGemm(n, [&] { matmul(a, false, b, false, c); });
                setGpuMinWork(savedMin);
                setDevice(savedDevice);
            } else {
                gflops = timeGemm(n, [&] { col.raw(false, false, n, n, n, a.data(), n, b.data(), n, c.data(), n, false); });
            }
            const bool ok = c.approxEquals(expected, 1e-3f * static_cast<float>(n));
            mismatches += !ok;
            std::cout << std::setw(13) << std::fixed << std::setprecision(1) << gflops << (ok ? ' ' : '!');
        }
        std::cout << std::defaultfloat << '\n';
    }
    if (mismatches) std::cout << "\n" << mismatches << " result(s) disagreed with the reference ('!')\n";

    // Int8 weights (LLM inference): one token reads every weight once (bandwidth bound), a prompt reuses them.
    std::cout << "\nint8 weights (" << QuantizedWeights::kernelName() << "), 2048 -> 8192 layer:\n";
    const QuantizedWeights q = QuantizedWeights::fromOutIn(Matrix::uniform(8192, 2048, -1, 1));
    for (std::size_t rows : {std::size_t(1), std::size_t(64)}) {
        const Matrix x = Matrix::uniform(rows, 2048, -1, 1);
        q.multiply(x);
        int reps = 0;
        const auto start = std::chrono::steady_clock::now();
        double elapsed = 0.0;
        do {
            q.multiply(x);
            ++reps;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } while (elapsed < (quick ? 0.2 : 1.0));
        const double seconds = elapsed / reps;
        std::cout << std::setw(6) << rows << " row" << (rows == 1 ? " " : "s") << std::fixed << std::setprecision(1) << std::setw(9)
                  << 2.0 * static_cast<double>(rows) * 2048 * 8192 / seconds / 1e9 << " GOPS" << std::setw(8)
                  << static_cast<double>(q.bytes()) / seconds / 1e9 << " GB/s of weights" << std::defaultfloat << '\n';
    }

    // Image models: thousands of pixels through each int8 layer (a 64x64 latent through a 1280-wide
    // projection). CPU expands the weights for the float GEMM; the GPU int8 kernel reads them as int8.
    const std::size_t pixels = quick ? 512 : 4096, width = quick ? 320 : 1280;
    std::cout << "\nint8 image layer, " << pixels << " pixels x " << width << " -> " << width << ":\n";
    const QuantizedWeights qi = QuantizedWeights::fromOutIn(Matrix::uniform(width, width, -1, 1));
    const Matrix px = Matrix::uniform(pixels, width, -1, 1);
    const Matrix reference = qi.multiplyFloat(px);
    auto timeInt8 = [&](const char* label, Device device) {
        const Device savedDevice = currentDevice();
        setDevice(device);
        Matrix y = qi.multiply(px);
        int reps = 0;
        const auto start = std::chrono::steady_clock::now();
        double elapsed = 0.0;
        do {
            y = qi.multiply(px);
            ++reps;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } while (elapsed < (quick ? 0.2 : 1.0));
        const bool gpuInt8 = qi.gpuProducts(pixels);
        setDevice(savedDevice);
        const bool ok = y.approxEquals(reference, 1e-3f * static_cast<float>(width));
        mismatches += !ok;
        std::cout << std::setw(22) << label << std::fixed << std::setprecision(1) << std::setw(9)
                  << 2.0 * static_cast<double>(pixels) * width * width / (elapsed / reps) / 1e9 << " GFLOPS"
                  << (ok ? "" : "  (disagrees with the reference!)") << (device == Device::GPU && !gpuInt8 ? "  (expanded to float)" : "")
                  << std::defaultfloat << '\n';
    };
    timeInt8("CPU", Device::CPU);
    if (gpuAvailable()) timeInt8("GPU", Device::GPU);

    // UNet self-attention at the 64x64 latent's top level: 4096 tokens, 320 wide, 8 heads.
    const std::size_t tokens = quick ? 512 : 4096;
    std::cout << "\nattention, " << tokens << " tokens x 320, 8 heads:\n";
    const Matrix aq = Matrix::uniform(tokens, 320, -1, 1), ak = Matrix::uniform(tokens, 320, -1, 1),
                 av = Matrix::uniform(tokens, 320, -1, 1);
    Matrix attnRef;
    auto timeAttention = [&](const char* label, Device device) {
        const Device savedDevice = currentDevice();
        setDevice(device);
        Matrix y = multiHeadAttention(aq, ak, av, 8);
        int reps = 0;
        const auto start = std::chrono::steady_clock::now();
        double elapsed = 0.0;
        do {
            y = multiHeadAttention(aq, ak, av, 8);
            ++reps;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } while (elapsed < (quick ? 0.2 : 1.0));
        Matrix probe(tokens, 320);
        const bool fused = gpuAttention(aq, ak, av, 8, probe);
        setDevice(savedDevice);
        if (attnRef.empty()) attnRef = y;
        const bool ok = y.approxEquals(attnRef, 1e-3f);
        mismatches += !ok;
        std::cout << std::setw(22) << label << std::fixed << std::setprecision(2) << std::setw(9) << 1e3 * elapsed / reps << " ms"
                  << (device == Device::GPU ? (fused ? "  (fused kernel)" : "  (GEMMs + CPU softmax)") : "")
                  << (ok ? "" : "  (disagrees with the CPU!)") << std::defaultfloat << '\n';
    };
    timeAttention("CPU", Device::CPU);
    if (gpuAvailable()) timeAttention("GPU", Device::GPU);

    // 3x3 convolutions as in the UNet: im2col + one GEMM, or 9 shifted GEMMs.
    std::cout << "\n3x3 convolution, CPU (ms):        im2col   shifted\n";
    struct ConvCase { std::size_t channels, size; };
    const std::vector<ConvCase> convs = quick ? std::vector<ConvCase>{{64, 32}}
                                              : std::vector<ConvCase>{{320, 64}, {640, 32}, {1280, 16}, {128, 256}};
    const Device benchDevice = currentDevice();
    setDevice(Device::CPU);
    const bool savedShifted = shiftedConvolutions();
    for (const ConvCase& cc : convs) {
        Conv2D conv({cc.channels, cc.size, cc.size}, cc.channels, 3, 1, 1);
        const Matrix x = Matrix::uniform(1, cc.channels * cc.size * cc.size, -1, 1);
        double ms[2];
        Matrix y[2];
        for (int sh = 0; sh < 2; ++sh) {
            setShiftedConvolutions(sh == 1);
            y[sh] = conv.forward(x, false);
            int reps = 0;
            const auto start = std::chrono::steady_clock::now();
            double elapsed = 0.0;
            do {
                conv.forward(x, false);
                ++reps;
                elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            } while (elapsed < (quick ? 0.2 : 1.0));
            ms[sh] = 1e3 * elapsed / reps;
        }
        const bool ok = y[1].approxEquals(y[0], 1e-3f);
        mismatches += !ok;
        std::cout << std::setw(6) << cc.channels << " ch " << std::setw(3) << cc.size << "x" << std::left << std::setw(13) << cc.size
                  << std::right << std::fixed << std::setprecision(1) << std::setw(10) << ms[0] << std::setw(10) << ms[1]
                  << (ok ? "" : "  (disagree!)") << std::defaultfloat << '\n';
    }
    setShiftedConvolutions(savedShifted);
    setDevice(benchDevice);
    return mismatches == 0 ? 0 : 1;
}

void usage() {
    std::cout << "NeuraCraft " << NEURACRAFT_VERSION << " - a small neural network library in C++17\n\n"
              << "usage:\n"
              << "  NeuraCraft                   run the xor, spiral and sine demos\n"
              << "  NeuraCraft xor|spiral|sine   run one demo\n"
              << "  NeuraCraft mnist <dir> [epochs] [mlp|cnn]\n"
              << "                               train on MNIST IDX files in <dir>, save weights\n"
              << "  NeuraCraft eval <dir> [mlp|cnn]\n"
              << "                               evaluate saved MNIST weights on the test set\n"
              << "  NeuraCraft gpt <text-file> [epochs] [bpe|char] [--resume] [--lr X] [--cosine]\n"
              << "                 [--arch modern|classic] [--dim N] [--layers N] [--heads N] [--context N]\n"
              << "                               train a GPT on a text file (BPE or character tokens)\n"
              << "  NeuraCraft generate <text-file> [prompt] [length]\n"
              << "                               sample text from the trained GPT\n"
              << "  NeuraCraft gpt2 <dir> [prompt] [tokens] [--temperature T] [--top-k K] [--greedy] [--int8] [--check]\n"
              << "                               run pretrained GPT-2 (scripts/download_gpt2.sh)\n"
              << "  NeuraCraft llama <dir> [prompt] [tokens] [--chat | -i] [--system S] [--fp32] [--check]\n"
              << "                 [--greedy | --temperature T] [--top-k K] [--top-p P] [--repeat-penalty R] [--seed N]\n"
              << "                               Llama 3.x / TinyLlama / Mistral, int8 by default (scripts/download_llama32.sh)\n"
              << "  NeuraCraft txt2img <dir> \"prompt\" [out.png] [--steps N] [--seed S] [--size N] [--guidance G --negative N]\n"
              << "                 [--tiny-vae <taesd dir>] [--fp32]\n"
              << "  NeuraCraft img2img <dir> in.png \"prompt\" [out.png] [--strength S] [--steps N] [--seed S] [--size N] ...\n"
              << "        image commands also take [--lora file.safetensors[:scale]] ... [--sampler euler|lcm]\n"
              << "        and [--controlnet DIR --control image.png [--canny] [--control-scale S]] (t2v: --control-dir frames/)\n"
              << "        and [--ip-adapter ip-adapter_sd15.safetensors --image-prompt pic.png [--ip-scale 0.6]]\n"
              << "  NeuraCraft video <dir> morph \"prompt\" \"prompt\" ... [-o out.gif] [--seeds A,B,..] [--frames N] [--fps F] [--no-loop]\n"
              << "  NeuraCraft video <dir> zoom \"prompt\" ... [-o out.gif] [--frames N] [--zoom Z] [--pan DX,DY] [--rotate DEG]\n"
              << "  NeuraCraft video <dir> t2v \"prompt\" [-o out.gif] [--frames 16] [--steps 4] [--seed S] [--motion file] [--guidance G]\n"
              << "        [--context 16 --overlap 4] [--init image.png | --init-dir frames/] [--strength 0.75]\n"
              << "        [--prompt-at FRAME:prompt ...]   (prompt travel)\n"
              << "  NeuraCraft video <dir> outpaint \"prompt\" ... [--keyframes 6] [--shrink 0.75] [--frames 12] [--zoom-in] [--start in.png]\n"
              << "  NeuraCraft inpaint <dir> in.png mask.png \"prompt\" [out.png] [--steps 25] [--strength 1] [--feather 4] [--guidance G]\n"
              << "  NeuraCraft upscale model.safetensors in.png out.png      (Real-ESRGAN compact; video takes --upscale model)\n"
              << "  NeuraCraft interpolate rife.safetensors frames/ out.gif [--passes N] [--fps F] [--loop]   (video: --interpolate rife)\n"
              << "  NeuraCraft depth <depth-anything dir> in.png depth.png [--size 518]   (control images: --depth-model dir)\n"
              << "  NeuraCraft parallax <depth-anything dir> in.png out.gif [--frames 48] [--amplitude 0.03] [--zoom 0.06] [--fps 24]\n"
              << "  NeuraCraft workflow steps.json                         (image / video steps in JSON, see Workflow.h)\n"
              << "  video output: .gif (256 colours), .png (animated PNG, full colour) or .mov (QuickTime Motion-JPEG)\n"
              << "        [--strength S] [--steps N] [--start in.png] [--seed S] [--size N] [--frames-dir DIR] [--full-vae]\n"
              << "        [--sharpen A] [--noise N] [--no-color-match]\n"
              << "                               image to image (input: PNG or PPM, cropped to the size)\n"
              << "                               text to image with Stable Diffusion (SD-Turbo: 1-4 steps)\n"
              << "  NeuraCraft vae <dir> [in.ppm] [out.png] [--size N]\n"
              << "                               encode and decode an image with Stable Diffusion's VAE\n"
              << "  NeuraCraft bench [--quick]   matrix-multiply speed of each backend\n\n"
              << "options:\n"
              << "  --gpu          run large matrix products on the GPU (Metal)\n"
              << "  --threads N    number of CPU threads (default: all cores)\n"
              << "  --gpu-attention  fused attention on the GPU, everything else on the CPU (best on Apple silicon)\n"
              << "  --profile      print where the time went (GEMMs, int8, convolutions, attention, ...)\n";
}

} // namespace

// FILE or FILE:SCALE (the last ':' followed by a number).
static Lora parseLora(const std::string& s) {
    const std::size_t colon = s.rfind(':');
    if (colon != std::string::npos && colon + 1 < s.size()) {
        try {
            std::size_t used = 0;
            const float scale = std::stof(s.substr(colon + 1), &used);
            if (used == s.size() - colon - 1) return {s.substr(0, colon), scale};
        } catch (const std::exception&) {
        }
    }
    return {s, 1.0f};
}

int main(int argc, char** argv) {
    setSeed(42);
    std::vector<std::string> args;
    bool wantGpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gpu") {
            wantGpu = true;
        } else if (a == "--gpu-attention") {
            setGpuAttention(true);
            if (!gpuAttentionEnabled()) std::cerr << "warning: no GPU available, --gpu-attention ignored\n";
        } else if (a == "--profile") {
            setProfiling(true);
        } else if (a == "--threads") {
            if (i + 1 >= argc) {
                std::cerr << "--threads needs a number\n";
                return 2;
            }
            setThreadCount(static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10)));
        } else {
            args.push_back(a);
        }
    }
    if (wantGpu && !setDevice(Device::GPU)) {
        std::cerr << "warning: no GPU available in this build/machine, using the CPU\n";
    }
    std::cout << "device: " << deviceDescription() << (gpuAttentionEnabled() && currentDevice() != Device::GPU ? " + GPU attention" : "") << '\n';

    const std::string cmd = args.empty() ? "all" : args[0];
    // --profile: where the time went, printed however the command ends.
    struct ProfileReport {
        ~ProfileReport() {
            if (profiling()) printProfile(std::cout);
        }
    } profileReport;
    try {
        if (cmd == "all") {
            int failed = 0;
            failed += runXor() != 0;
            failed += runSpiral() != 0;
            failed += runSine() != 0;
            return failed == 0 ? 0 : 1;
        }
        if (cmd == "xor") return runXor();
        if (cmd == "spiral") return runSpiral();
        if (cmd == "sine") return runSine();
        if (cmd == "bench") return runBench(args.size() > 1 && args[1] == "--quick");
        if (cmd == "mnist") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            const std::size_t epochs = args.size() > 2 ? static_cast<std::size_t>(std::strtoul(args[2].c_str(), nullptr, 10)) : 5;
            return runMnist(args[1], epochs == 0 ? 5 : epochs, args.size() > 3 ? args[3] : "mlp");
        }
        if (cmd == "eval") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            return runEval(args[1], args.size() > 2 ? args[2] : "mlp");
        }
        if (cmd == "gpt") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            GptOptions o;
            std::size_t positional = 0;
            for (std::size_t i = 2; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "--resume") o.resume = true;
                else if (a == "--cosine") o.cosine = true;
                else if (a == "--arch") {
                    o.arch = value();
                    if (o.arch != "classic" && o.arch != "modern") {
                        std::cerr << "--arch must be classic or modern\n";
                        return 2;
                    }
                }
                else if (a == "--lr") o.learningRate = std::stof(value());
                else if (a == "--dim") o.dim = std::stoul(value());
                else if (a == "--layers") o.layers = std::stoul(value());
                else if (a == "--heads") o.heads = std::stoul(value());
                else if (a == "--context") o.context = std::stoul(value());
                else if (positional == 0 && std::isdigit(static_cast<unsigned char>(a[0]))) {
                    o.epochs = std::max<std::size_t>(1, std::stoul(a));
                    ++positional;
                } else if (a == "bpe" || a == "char") {
                    o.tokenizer = a;
                } else {
                    std::cerr << "unknown gpt option: " << a << "\n";
                    return 2;
                }
            }
            return runGpt(args[1], o);
        }
        if (cmd == "gpt2") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            std::string prompt = "The meaning of life is";
            std::size_t tokens = 60;
            bool check = false, int8 = false;
            SamplingOptions sampling;
            sampling.temperature = 0.8f;
            sampling.topK = 40;
            std::size_t positional = 0;
            for (std::size_t i = 2; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "--temperature") sampling.temperature = std::stof(value());
                else if (a == "--top-k") sampling.topK = std::stoul(value());
                else if (a == "--greedy") sampling.temperature = 0.0f;
                else if (a == "--check") check = true;
                else if (a == "--int8") int8 = true;
                else if (positional == 0) { prompt = a; ++positional; }
                else if (positional == 1) { tokens = std::stoul(a); ++positional; }
                else {
                    std::cerr << "unknown gpt2 option: " << a << "\n";
                    return 2;
                }
            }
            return runGpt2(args[1], prompt, tokens, sampling, check, int8);
        }
        if (cmd == "txt2img" || cmd == "img2img") {
            const bool img2img = cmd == "img2img";
            if (args.size() < (img2img ? 4u : 3u)) {
                usage();
                return 2;
            }
            TextToImageOptions o;
            bool canny = false;
            std::string depthModel;
            if (img2img) o.steps = 2;  // SD-Turbo's image-to-image setting: 2 steps at strength 0.5
            std::string output = img2img ? "img2img.png" : "txt2img.png";
            const std::string prompt = args[img2img ? 3 : 2];
            for (std::size_t i = img2img ? 4 : 3; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "--steps") o.steps = std::stoul(value());
                else if (a == "--seed") o.seed = static_cast<unsigned>(std::stoul(value()));
                else if (a == "--size") o.width = o.height = std::stoul(value());
                else if (a == "--width") o.width = std::stoul(value());
                else if (a == "--height") o.height = std::stoul(value());
                else if (a == "--guidance") o.guidance = std::stof(value());
                else if (a == "--negative") o.negativePrompt = value();
                else if (a == "--tiny-vae") o.tinyVae = value();
                else if (a == "--int8") o.int8 = true;
                else if (a == "--strength") o.strength = std::stof(value());
                else if (a == "--fp32") o.int8 = false;
                else if (a == "--lora") o.loras.push_back(parseLora(value()));
                else if (a == "--sampler") o.sampler = value();
                else if (a == "--controlnet") o.controlNet = value();
                else if (a == "--control") o.controlImage = loadImage(value());
                else if (a == "--control-scale") o.controlScale = std::stof(value());
                else if (a == "--canny") canny = true;
                else if (a == "--depth-model") depthModel = value();
                else if (a == "--ip-adapter") o.ipAdapter = value();
                else if (a == "--image-prompt") o.imagePrompt = loadImage(value());
                else if (a == "--ip-scale") o.ipScale = std::stof(value());
                else if (a == "--image-encoder") o.imageEncoder = value();
                else if (a.rfind("--", 0) != 0) output = a;
                else {
                    std::cerr << "unknown txt2img option: " << a << "\n";
                    return 2;
                }
            }
            if (canny && !o.controlImage.pixels.empty()) o.controlImage = cannyEdges(o.controlImage);
            if (!depthModel.empty() && !o.controlImage.pixels.empty()) o.controlImage = DepthEstimator(depthModel).depthImage(o.controlImage);
            o.progress = [](const std::string& stage, double seconds) {
                std::cerr << "  " << stage << ": " << std::fixed << std::setprecision(1) << seconds << " s" << std::defaultfloat << '\n';
            };
            const auto start = std::chrono::steady_clock::now();
            try {
                const Image img = img2img ? imageToImage(args[1], loadImage(args[2]), prompt, o) : textToImage(args[1], prompt, o);
                if (output.size() > 4 && (output.compare(output.size() - 4, 4, ".jpg") == 0 || output.compare(output.size() - 4, 4, ".JPG") == 0))
                    saveJpeg(output, img, 92);
                else savePng(output, img);
            } catch (const std::exception& e) {
                std::cerr << e.what() << "\ndownload SD-Turbo with: scripts/download_sd_turbo.sh " << args[1] << '\n';
                return 2;
            }
            std::cout << "wrote " << output << " (" << o.width << "x" << o.height << ", " << o.steps << " step" << (o.steps == 1 ? "" : "s")
                      << ", seed " << o.seed << ") in " << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s" << std::defaultfloat << '\n';
            return 0;
        }
        if (cmd == "video") {
            if (args.size() < 4 || (args[2] != "morph" && args[2] != "zoom" && args[2] != "t2v" && args[2] != "outpaint")) {
                usage();
                return 2;
            }
            const bool morph = args[2] == "morph", t2v = args[2] == "t2v", outpaint = args[2] == "outpaint";
            OutpaintZoomOptions oz;
            TextToImageOptions o;
            TextToVideoOptions vo;
            MorphOptions mo;
            ZoomOptions zo;
            std::vector<std::string> prompts;
            std::vector<unsigned> seeds;
            std::string output = morph ? "morph.gif" : t2v ? "t2v.gif" : outpaint ? "outpaint.gif" : "zoom.gif", framesDir, start;
            float fps = morph ? 10.0f : 8.0f;
            bool fullVae = false, framesSet = false, strengthSet = false, cannyVideo = false;
            std::string upscaleModel, interpolateModel, depthModelVideo;
            std::size_t interpPasses = 1;
            std::size_t frames = 0;
            auto split = [](const std::string& s) {
                std::vector<std::string> parts;
                std::stringstream ss(s);
                for (std::string p; std::getline(ss, p, ',');) parts.push_back(p);
                return parts;
            };
            for (std::size_t i = 3; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "-o" || a == "--output") output = value();
                else if (a == "--frames") frames = std::stoul(value()), framesSet = true;
                else if (a == "--fps") fps = std::stof(value());
                else if (a == "--seed") zo.seed = static_cast<unsigned>(std::stoul(value())), seeds = {zo.seed};
                else if (a == "--seeds") {
                    seeds.clear();
                    for (const std::string& s : split(value())) seeds.push_back(static_cast<unsigned>(std::stoul(s)));
                } else if (a == "--no-loop") mo.loop = false;
                else if (a == "--steps") mo.steps = zo.steps = vo.steps = std::stoul(value());
                else if (a == "--guidance") vo.guidance = o.guidance = std::stof(value());
                else if (a == "--negative") vo.negativePrompt = o.negativePrompt = value();
                else if (a == "--motion") vo.motion = value();
                else if (a == "--schedule") vo.betaSchedule = value();
                else if (a == "--zoom") zo.zoom = std::stof(value());
                else if (a == "--pan") {
                    const auto parts = split(value());
                    if (parts.size() != 2) throw std::invalid_argument("--pan takes DX,DY");
                    zo.dx = std::stof(parts[0]);
                    zo.dy = std::stof(parts[1]);
                } else if (a == "--rotate") zo.rotate = std::stof(value());
                else if (a == "--strength") zo.strength = vo.strength = std::stof(value()), strengthSet = true;
                else if (a == "--prompt-at") {  // FRAME:prompt
                    const std::string s = value();
                    const std::size_t colon = s.find(':');
                    if (colon == std::string::npos) throw std::invalid_argument("--prompt-at takes FRAME:prompt");
                    vo.promptAt.emplace_back(std::stoul(s.substr(0, colon)), s.substr(colon + 1));
                }
                else if (a == "--keyframes") oz.keyframes = std::stoul(value());
                else if (a == "--shrink") oz.shrink = std::stof(value());
                else if (a == "--zoom-in") oz.zoomIn = true;
                else if (a == "--controlnet") vo.controlNet = value();
                else if (a == "--control") vo.controlImages.push_back(loadImage(value()));
                else if (a == "--control-dir") {
                    std::vector<std::string> files;
                    for (const auto& e : std::filesystem::directory_iterator(value())) {
                        const std::string ext = e.path().extension().string();
                        if (ext == ".png" || ext == ".ppm") files.push_back(e.path().string());
                    }
                    std::sort(files.begin(), files.end());
                    for (const std::string& f : files) vo.controlImages.push_back(loadImage(f));
                } else if (a == "--control-scale") vo.controlScale = std::stof(value());
                else if (a == "--canny") cannyVideo = true;
                else if (a == "--depth-model") depthModelVideo = value();
                else if (a == "--ip-adapter") vo.ipAdapter = o.ipAdapter = value();
                else if (a == "--image-prompt") vo.imagePrompt = o.imagePrompt = loadImage(value());
                else if (a == "--ip-scale") vo.ipScale = o.ipScale = std::stof(value());
                else if (a == "--image-encoder") vo.imageEncoder = o.imageEncoder = value();
                else if (a == "--upscale") upscaleModel = value();
                else if (a == "--interpolate") interpolateModel = value();
                else if (a == "--interp-passes") interpPasses = std::stoul(value());
                else if (a == "--context") vo.context = std::stoul(value());
                else if (a == "--lora") o.loras.push_back(parseLora(value()));
                else if (a == "--sampler") o.sampler = vo.sampler = value();
                else if (a == "--overlap") vo.overlap = std::stoul(value());
                else if (a == "--init") vo.init.push_back(loadImage(value()));
                else if (a == "--init-dir") {
                    std::vector<std::string> files;
                    for (const auto& e : std::filesystem::directory_iterator(value())) {
                        const std::string ext = e.path().extension().string();
                        if (ext == ".png" || ext == ".ppm") files.push_back(e.path().string());
                    }
                    std::sort(files.begin(), files.end());
                    for (const std::string& f : files) vo.init.push_back(loadImage(f));
                    if (!framesSet) frames = files.size(), framesSet = true;
                }
                else if (a == "--no-color-match") zo.colorCoherence = false;
                else if (a == "--sharpen") zo.sharpen = std::stof(value());
                else if (a == "--noise") zo.noise = std::stof(value());
                else if (a == "--start") start = value();
                else if (a == "--size") o.width = o.height = std::stoul(value());
                else if (a == "--width") o.width = std::stoul(value());
                else if (a == "--height") o.height = std::stoul(value());
                else if (a == "--tiny-vae") o.tinyVae = value();
                else if (a == "--full-vae") fullVae = true;
                else if (a == "--frames-dir") framesDir = value();
                else if (a == "--int8") o.int8 = true;
                else if (a == "--fp32") o.int8 = false;
                else if (a.rfind("--", 0) != 0) prompts.push_back(a);
                else {
                    std::cerr << "unknown video option: " << a << "\n";
                    return 2;
                }
            }
            if (prompts.empty()) throw std::invalid_argument("video: give at least one prompt");
            if (seeds.empty()) seeds = {0};
            // TAESD makes decoding ~10x cheaper, which matters for dozens of frames.
            if (o.tinyVae.empty() && !fullVae && std::ifstream(args[1] + "/taesd/config.json")) o.tinyVae = args[1] + "/taesd";
            if (t2v) {
                if (prompts.size() != 1) throw std::invalid_argument("video t2v: give one prompt");
                vo.width = o.width;
                vo.height = o.height;
                vo.seed = seeds[0];
                vo.int8 = o.int8;
                vo.tinyVae = o.tinyVae;
                vo.loras = o.loras;
                if (framesSet) vo.frames = frames;
                // --canny: edges of the control frames, or of the init frames when no control frames are given.
                if (!vo.controlNet.empty() && vo.controlImages.empty()) vo.controlImages = vo.init;
                if (cannyVideo)
                    for (Image& c : vo.controlImages) c = cannyEdges(c);
                if (!depthModelVideo.empty()) {
                    DepthEstimator de(depthModelVideo);
                    for (Image& c : vo.controlImages) c = de.depthImage(c);
                }
            } else if (outpaint) {
                oz.seed = seeds[0];
                if (framesSet) oz.framesPerKey = frames;
                oz.steps = mo.steps == 1 ? 25 : mo.steps;  // --steps (default 25; 4-6 with --sampler lcm and LCM-LoRA)
                if (strengthSet) oz.strength = zo.strength;
            } else if (morph) {
                if (seeds.size() == 1 && prompts.size() == 1) throw std::invalid_argument("video morph: give two prompts or several --seeds");
                if (seeds.size() > 1 && prompts.size() == 1) prompts.resize(seeds.size(), prompts[0]);
                if (framesSet) mo.framesPerSegment = frames;
            } else if (framesSet) zo.frames = frames;
            const auto begin = std::chrono::steady_clock::now();
            o.progress = [](const std::string& stage, double seconds) {
                std::cerr << "  " << stage << ": " << std::fixed << std::setprecision(1) << seconds << " s" << std::defaultfloat << '\n';
            };
            vo.progress = o.progress;
            std::vector<Image> video;
            if (t2v) {
                try {
                    video = textToVideo(args[1], prompts[0], vo);
                } catch (const std::exception& e) {
                    std::cerr << e.what() << "\ndownload a model and motion module with: scripts/download_animatediff.sh " << args[1] << '\n';
                    return 2;
                }
            } else try {
                StableDiffusion sd(args[1], o, !morph);
                if (outpaint) {
                    Image first;
                    if (!start.empty()) first = loadImage(start);
                    video = outpaintZoom(sd, prompts, oz, start.empty() ? nullptr : &first);
                } else if (morph) video = morphAnimation(sd, prompts, seeds, mo);
                else {
                    Image first;
                    if (!start.empty()) first = loadImage(start);
                    video = zoomAnimation(sd, prompts, zo, start.empty() ? nullptr : &first);
                }
            } catch (const std::exception& e) {
                std::cerr << e.what() << "\ndownload SD-Turbo with: scripts/download_sd_turbo.sh " << args[1] << '\n';
                return 2;
            }
            if (!interpolateModel.empty()) {  // RIFE: 2^passes - 1 frames between each pair, same duration
                const auto ti = std::chrono::steady_clock::now();
                const FrameInterpolator fi(interpolateModel);
                const bool loops = morph && mo.loop;
                video = fi.interpolate(video, interpPasses, loops);
                fps *= static_cast<float>(std::size_t(1) << interpPasses);
                std::cerr << "  interpolate x" << (std::size_t(1) << interpPasses) << ": " << std::fixed << std::setprecision(1)
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count() << " s" << std::defaultfloat << '\n';
            }
            if (!upscaleModel.empty()) {  // super-resolution of every frame before writing
                const auto tu = std::chrono::steady_clock::now();
                const Upscaler up(upscaleModel);
                for (Image& f : video) f = up.upscale(f);
                std::cerr << "  upscale x" << up.scale() << ": " << std::fixed << std::setprecision(1)
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - tu).count() << " s" << std::defaultfloat << '\n';
            }
            auto endsWith = [&](const char* ext) {
                const std::size_t n = std::strlen(ext);
                return output.size() >= n && output.compare(output.size() - n, n, ext) == 0;
            };
            if (endsWith(".png")) saveApng(output, video, fps);           // full colour, lossless
            else if (endsWith(".mov")) saveMjpegMov(output, video, fps);  // QuickTime, Motion-JPEG
            else saveGif(output, video, fps);
            if (!framesDir.empty()) {
                std::filesystem::create_directories(framesDir);
                for (std::size_t i = 0; i < video.size(); ++i) {
                    char name[32];
                    std::snprintf(name, sizeof name, "/frame_%04zu.png", i);
                    savePng(framesDir + name, video[i]);
                }
                std::cout << "frames in " << framesDir << "; for an MP4: ffmpeg -framerate " << fps << " -i " << framesDir
                          << "/frame_%04d.png -pix_fmt yuv420p video.mp4\n";
            }
            std::cout << "wrote " << output << " (" << video.size() << " frames, " << o.width << "x" << o.height << ", " << fps
                      << " fps) in " << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() << " s" << std::defaultfloat << '\n';
            return 0;
        }
        if (cmd == "workflow") {  // file.json
            if (args.size() < 2) {
                usage();
                return 2;
            }
            std::ifstream f(args[1]);
            if (!f) throw std::runtime_error("cannot open " + args[1]);
            std::stringstream text;
            text << f.rdbuf();
            const auto start = std::chrono::steady_clock::now();
            const auto clips = runWorkflow(text.str(), ".", [](const std::string& line) { std::cerr << "  " << line << '\n'; });
            std::cout << "workflow done (" << clips.size() << " clips) in " << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s" << std::defaultfloat << '\n';
            return 0;
        }
        if (cmd == "depth" || cmd == "parallax") {  // depth-model in.png out [options]
            if (args.size() < 4) {
                usage();
                return 2;
            }
            std::size_t size = 518;
            ParallaxOptions po;
            float fps = 24.0f;
            for (std::size_t i = 4; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "--size") size = std::stoul(value());
                else if (a == "--frames") po.frames = std::stoul(value());
                else if (a == "--amplitude") po.amplitude = std::stof(value());
                else if (a == "--zoom") po.zoom = std::stof(value());
                else if (a == "--fps") fps = std::stof(value());
                else if (a == "--no-loop") po.loop = false;
                else {
                    std::cerr << "unknown " << cmd << " option: " << a << "\n";
                    return 2;
                }
            }
            const auto start = std::chrono::steady_clock::now();
            DepthEstimator de(args[1]);
            const Image in = toRgb(loadImage(args[2]));
            const std::string& o = args[3];
            auto ends = [&](const char* e) { return o.size() > 4 && o.compare(o.size() - 4, 4, e) == 0; };
            if (cmd == "depth") {
                const Image d = de.depthImage(in, size);
                if (ends(".jpg")) saveJpeg(o, d, 92);
                else savePng(o, d);
            } else {
                const std::vector<Image> frames = parallaxAnimation(in, de.estimate(in, size), po);
                if (ends(".png")) saveApng(o, frames, fps);
                else if (ends(".mov")) saveMjpegMov(o, frames, fps);
                else saveGif(o, frames, fps);
            }
            std::cout << "wrote " << o << " in " << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s" << std::defaultfloat << '\n';
            return 0;
        }
        if (cmd == "interpolate") {  // model frames-dir out [--passes N] [--fps F] [--loop]
            if (args.size() < 4) {
                usage();
                return 2;
            }
            std::size_t passes = 1;
            float fps = 8.0f;
            bool loop = false;
            for (std::size_t i = 4; i < args.size(); ++i) {
                if (args[i] == "--passes" && i + 1 < args.size()) passes = std::stoul(args[++i]);
                else if (args[i] == "--fps" && i + 1 < args.size()) fps = std::stof(args[++i]);
                else if (args[i] == "--loop") loop = true;
                else {
                    std::cerr << "unknown interpolate option: " << args[i] << "\n";
                    return 2;
                }
            }
            std::vector<std::string> files;
            for (const auto& e : std::filesystem::directory_iterator(args[2])) {
                const std::string ext = e.path().extension().string();
                if (ext == ".png" || ext == ".ppm") files.push_back(e.path().string());
            }
            std::sort(files.begin(), files.end());
            std::vector<Image> frames;
            for (const std::string& f : files) frames.push_back(toRgb(loadImage(f)));
            const auto start = std::chrono::steady_clock::now();
            const std::vector<Image> out = FrameInterpolator(args[1]).interpolate(frames, passes, loop);
            const std::string& o = args[3];
            const float outFps = fps * static_cast<float>(std::size_t(1) << passes);
            if (o.size() > 4 && o.compare(o.size() - 4, 4, ".png") == 0) saveApng(o, out, outFps);
            else if (o.size() > 4 && o.compare(o.size() - 4, 4, ".mov") == 0) saveMjpegMov(o, out, outFps);
            else saveGif(o, out, outFps);
            std::cout << "wrote " << o << " (" << frames.size() << " -> " << out.size() << " frames, " << outFps << " fps) in " << std::fixed
                      << std::setprecision(1) << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s"
                      << std::defaultfloat << '\n';
            return 0;
        }
        if (cmd == "upscale") {
            if (args.size() < 4) {
                usage();
                return 2;
            }
            try {
                const Upscaler up(args[1]);
                const auto start = std::chrono::steady_clock::now();
                const Image out = up.upscale(loadImage(args[2]));
                if (args[3].size() > 4 && args[3].compare(args[3].size() - 4, 4, ".jpg") == 0) saveJpeg(args[3], out, 92);
                else savePng(args[3], out);
                std::cout << "wrote " << args[3] << " (" << out.width << "x" << out.height << ") in " << std::fixed << std::setprecision(1)
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s" << std::defaultfloat << '\n';
            } catch (const std::exception& e) {
                std::cerr << e.what() << "\ndownload Real-ESRGAN's compact model with: scripts/download_upscaler.sh\n";
                return 2;
            }
            return 0;
        }
        if (cmd == "inpaint") {
            if (args.size() < 5) {
                usage();
                return 2;
            }
            TextToImageOptions o;
            o.steps = 25;
            o.strength = 1.0f;
            float feather = 4.0f;
            std::string output = "inpaint.png";
            for (std::size_t i = 5; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "--steps") o.steps = std::stoul(value());
                else if (a == "--seed") o.seed = static_cast<unsigned>(std::stoul(value()));
                else if (a == "--size") o.width = o.height = std::stoul(value());
                else if (a == "--width") o.width = std::stoul(value());
                else if (a == "--height") o.height = std::stoul(value());
                else if (a == "--guidance") o.guidance = std::stof(value());
                else if (a == "--negative") o.negativePrompt = value();
                else if (a == "--tiny-vae") o.tinyVae = value();
                else if (a == "--strength") o.strength = std::stof(value());
                else if (a == "--feather") feather = std::stof(value());
                else if (a == "--lora") o.loras.push_back(parseLora(value()));
                else if (a == "--sampler") o.sampler = value();
                else if (a == "--fp32") o.int8 = false;
                else if (a.rfind("--", 0) != 0) output = a;
                else {
                    std::cerr << "unknown inpaint option: " << a << "\n";
                    return 2;
                }
            }
            if (o.tinyVae.empty() && std::ifstream(args[1] + "/taesd/config.json")) o.tinyVae = args[1] + "/taesd";
            o.progress = [](const std::string& stage, double seconds) {
                std::cerr << "  " << stage << ": " << std::fixed << std::setprecision(1) << seconds << " s" << std::defaultfloat << '\n';
            };
            const auto start = std::chrono::steady_clock::now();
            try {
                StableDiffusion sd(args[1], o, true);
                const Image img = sd.inpaint(loadImage(args[2]), loadImage(args[3]), args[4], o.seed, o.steps, o.strength, feather);
                if (output.size() > 4 && output.compare(output.size() - 4, 4, ".jpg") == 0) saveJpeg(output, img, 92);
                else savePng(output, img);
            } catch (const std::exception& e) {
                std::cerr << e.what() << "\ndownload an inpainting model with: scripts/download_sd_inpaint.sh " << args[1] << '\n';
                return 2;
            }
            std::cout << "wrote " << output << " in " << std::fixed << std::setprecision(1)
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s" << std::defaultfloat << '\n';
            return 0;
        }
        if (cmd == "vae") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            std::string input, output = "vae_reconstruction.png";
            std::size_t size = 512, positional = 0;
            for (std::size_t i = 2; i < args.size(); ++i) {
                if (args[i] == "--size" && i + 1 < args.size()) size = std::stoul(args[++i]);
                else if (positional == 0 && args[i].size() > 4 && args[i].substr(args[i].size() - 4) == ".png") { output = args[i]; positional = 2; }
                else if (positional == 0) { input = args[i]; ++positional; }
                else if (positional == 1) { output = args[i]; ++positional; }
                else {
                    std::cerr << "unknown vae option: " << args[i] << "\n";
                    return 2;
                }
            }
            return runVae(args[1], input, output, size);
        }
        if (cmd == "llama") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            LlamaOptions o;
            o.sampling.temperature = 0.7f;
            o.sampling.topK = 40;
            std::size_t positional = 0;
            for (std::size_t i = 2; i < args.size(); ++i) {
                const std::string& a = args[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                    return args[++i];
                };
                if (a == "--chat") o.chat = true;
                else if (a == "--interactive" || a == "-i") o.interactive = true;
                else if (a == "--system") { o.system = value(); o.chat = true; }
                else if (a == "--fp32") o.int8 = false;
                else if (a == "--int8") o.int8 = true;
                else if (a == "--check") o.check = true;
                else if (a == "--greedy") { o.sampling.temperature = 0.0f; o.temperatureGiven = true; }
                else if (a == "--temperature") { o.sampling.temperature = std::stof(value()); o.temperatureGiven = true; }
                else if (a == "--top-k") o.sampling.topK = std::stoul(value());
                else if (a == "--top-p") { o.sampling.topP = std::stof(value()); o.topPGiven = true; }
                else if (a == "--repeat-penalty") o.sampling.repetitionPenalty = std::stof(value());
                else if (a == "--seed") setSeed(static_cast<unsigned>(std::stoul(value())));
                else if (positional == 0) { o.prompt = a; o.promptGiven = !a.empty(); ++positional; }
                else if (positional == 1) { o.tokens = std::stoul(a); ++positional; }
                else {
                    std::cerr << "unknown llama option: " << a << "\n";
                    return 2;
                }
            }
            return runLlama(args[1], o);
        }
        if (cmd == "generate") {
            if (args.size() < 2) {
                usage();
                return 2;
            }
            const std::size_t length = args.size() > 3 ? static_cast<std::size_t>(std::strtoul(args[3].c_str(), nullptr, 10)) : 500;
            return runGenerate(args[1], args.size() > 2 ? args[2] : "", length == 0 ? 500 : length);
        }
        if (cmd == "-h" || cmd == "--help" || cmd == "help") {
            usage();
            return 0;
        }
        std::cerr << "unknown command: " << cmd << "\n\n";
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
