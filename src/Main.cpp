// NeuraCraft demo CLI.
//
//   NeuraCraft                 run the xor, spiral and sine demos
//   NeuraCraft xor|spiral|sine run one demo
//   NeuraCraft mnist <dir> [epochs]
//                              train a digit classifier on MNIST IDX files in <dir>
//                              (see scripts/download_mnist.sh)

#include "NeuraCraft.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

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

void usage() {
    std::cout << "NeuraCraft " << NEURACRAFT_VERSION << " - a small neural network library in C++17\n\n"
              << "usage:\n"
              << "  NeuraCraft                   run the xor, spiral and sine demos\n"
              << "  NeuraCraft xor|spiral|sine   run one demo\n"
              << "  NeuraCraft mnist <dir> [epochs]\n"
              << "                               train on MNIST IDX files in <dir>\n";
}

} // namespace

int main(int argc, char** argv) {
    setSeed(42);
    const std::string cmd = argc > 1 ? argv[1] : "all";
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
        if (cmd == "mnist") {
            if (argc < 3) {
                usage();
                return 2;
            }
            const std::size_t epochs = argc > 3 ? static_cast<std::size_t>(std::strtoul(argv[3], nullptr, 10)) : 5;
            return runMnist(argv[2], epochs == 0 ? 5 : epochs);
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
