// NeuraCraft demo CLI.
//
//   NeuraCraft                 run the xor, spiral and sine demos
//   NeuraCraft xor|spiral|sine run one demo
//   NeuraCraft mnist <dir> [epochs]
//                              train a digit classifier on MNIST IDX files in <dir>
//                              (see scripts/download_mnist.sh)
//   NeuraCraft bench [--quick] matrix-multiply throughput of each backend
//
// Global options: --gpu (Metal on Apple silicon), --threads N

#include "NeuraCraft.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

int runMnist(const std::string& dir, std::size_t epochs) {
    banner("MNIST (784-128-64-10, ReLU + dropout, Adam)");
    Matrix trainX, testX;
    std::vector<int> trainLabels, testLabels;
    try {
        trainX = loadIdxImages(dir + "/train-images-idx3-ubyte");
        trainLabels = loadIdxLabels(dir + "/train-labels-idx1-ubyte");
        testX = loadIdxImages(dir + "/t10k-images-idx3-ubyte");
        testLabels = loadIdxLabels(dir + "/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::cerr << "could not load MNIST: " << e.what() << "\n"
                  << "download it with: scripts/download_mnist.sh " << dir << '\n';
        return 2;
    }
    const Matrix trainY = oneHot(trainLabels, 10);
    const Matrix testY = oneHot(testLabels, 10);
    std::cout << "train: " << trainX.rows() << " images, test: " << testX.rows() << " images\n";

    Network net;
    net.add<Dense>(784, 128, Init::He);
    net.add<ReLU>();
    net.add<Dropout>(0.2f);
    net.add<Dense>(128, 64, Init::He);
    net.add<ReLU>();
    net.add<Dense>(64, 10);
    net.summary();

    SoftmaxCrossEntropy loss;
    Adam optimizer(0.001f);
    FitOptions opts;
    opts.epochs = epochs;
    opts.batchSize = 64;
    opts.trackAccuracy = true;
    opts.validationX = &testX;
    opts.validationY = &testY;
    net.fit(trainX, trainY, loss, optimizer, opts);

    const std::string weights = dir + "/mnist.weights";
    net.save(weights);
    std::cout << "weights saved to " << weights << '\n';
    std::cout << "test accuracy: " << std::fixed << std::setprecision(2) << net.accuracy(testX, testY) * 100.0f
              << std::defaultfloat << "%\n";
    return 0;
}

// Times one backend on an n x n x n product; returns GFLOPS.
double timeGemm(void (*fn)(bool, bool, std::size_t, std::size_t, std::size_t, const float*, std::size_t,
                           const float*, std::size_t, float*, std::size_t, bool),
                const Matrix& a, const Matrix& b, Matrix& c) {
    const std::size_t n = a.rows();
    fn(false, false, n, n, n, a.data(), n, b.data(), n, c.data(), n, false);  // warm-up
    const double flops = 2.0 * static_cast<double>(n) * n * n;
    int reps = 0;
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    do {
        fn(false, false, n, n, n, a.data(), n, b.data(), n, c.data(), n, false);
        ++reps;
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    } while (elapsed < 0.25);
    return flops * reps / elapsed / 1e9;
}

int runBench(bool quick) {
    banner("Matrix multiply benchmark");
    std::cout << "CPU backend: " << cpuBackendName() << ", " << threadCount() << " threads\n"
              << "GPU: " << (gpuAvailable() ? gpuName() + " (Metal)" : std::string("not available")) << "\n\n";

    struct Backend {
        const char* name;
        void (*fn)(bool, bool, std::size_t, std::size_t, std::size_t, const float*, std::size_t, const float*,
                   std::size_t, float*, std::size_t, bool);
    };
    std::vector<Backend> backends{{"blocked", gemmBlocked}};
    if (cpuBackendName() != "NeuraCraft blocked kernel") backends.push_back({"accelerate", gemmCpu});
    if (gpuAvailable()) backends.push_back({"metal", gemmGpu});

    const std::vector<std::size_t> sizes = quick ? std::vector<std::size_t>{64, 129}
                                                 : std::vector<std::size_t>{128, 256, 512, 1024, 2048};
    std::cout << "     n";
    for (const auto& be : backends) std::cout << std::setw(14) << be.name;
    std::cout << "   (GFLOPS)\n";

    int mismatches = 0;
    for (std::size_t n : sizes) {
        const Matrix a = Matrix::uniform(n, n, -1, 1), b = Matrix::uniform(n, n, -1, 1);
        Matrix expected(n, n);
        gemmBlocked(false, false, n, n, n, a.data(), n, b.data(), n, expected.data(), n, false);
        std::cout << std::setw(6) << n;
        for (const auto& be : backends) {
            Matrix c(n, n);
            const double gflops = timeGemm(be.fn, a, b, c);
            const bool ok = c.approxEquals(expected, 1e-3f * static_cast<float>(n));
            mismatches += !ok;
            std::cout << std::setw(13) << std::fixed << std::setprecision(1) << gflops << (ok ? ' ' : '!');
        }
        std::cout << std::defaultfloat << '\n';
    }
    if (mismatches) std::cout << "\n" << mismatches << " result(s) disagreed with the reference ('!')\n";
    return mismatches == 0 ? 0 : 1;
}

void usage() {
    std::cout << "NeuraCraft " << NEURACRAFT_VERSION << " - a small neural network library in C++17\n\n"
              << "usage:\n"
              << "  NeuraCraft                   run the xor, spiral and sine demos\n"
              << "  NeuraCraft xor|spiral|sine   run one demo\n"
              << "  NeuraCraft mnist <dir> [epochs]\n"
              << "                               train on MNIST IDX files in <dir>\n"
              << "  NeuraCraft bench [--quick]   matrix-multiply speed of each backend\n\n"
              << "options:\n"
              << "  --gpu          run large matrix products on the GPU (Metal)\n"
              << "  --threads N    number of CPU threads (default: all cores)\n";
}

} // namespace

int main(int argc, char** argv) {
    setSeed(42);
    std::vector<std::string> args;
    bool wantGpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gpu") {
            wantGpu = true;
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
    std::cout << "device: " << deviceDescription() << '\n';

    const std::string cmd = args.empty() ? "all" : args[0];
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
            return runMnist(args[1], epochs == 0 ? 5 : epochs);
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
