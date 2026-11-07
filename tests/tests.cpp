// NeuraCraft unit tests. Dependency-free: a tiny TEST/CHECK harness below.
#include "NeuraCraft.h"
#include "Checkpoint.h"
#include "Onnx.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace nc;

// ---- Harness -------------------------------------------------------------------

namespace {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string where(const char* file, int line) {
    return std::string(file) + ":" + std::to_string(line) + ": ";
}

} // namespace

#define TEST(name)                                     \
    static void name();                                \
    static const Registrar registrar_##name(#name, name); \
    static void name()

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) throw Failure(where(__FILE__, __LINE__) + "CHECK(" #cond ")"); \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                    \
    do {                                                                                         \
        const double va_ = (a), vb_ = (b);                                                       \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                                  \
            std::ostringstream os_;                                                              \
            os_ << where(__FILE__, __LINE__) << #a " = " << va_ << ", " #b " = " << vb_          \
                << " (tolerance " << (tol) << ")";                                               \
            throw Failure(os_.str());                                                            \
        }                                                                                        \
    } while (0)

#define CHECK_THROWS(expr, ExceptionType)                                                  \
    do {                                                                                   \
        bool thrown_ = false;                                                              \
        try {                                                                              \
            (void)(expr);                                                                  \
        } catch (const ExceptionType&) {                                                   \
            thrown_ = true;                                                                \
        }                                                                                  \
        if (!thrown_) throw Failure(where(__FILE__, __LINE__) + #expr " did not throw " #ExceptionType); \
    } while (0)

namespace {

std::string tempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() / ("neuracraft_test_" + name)).string();
}

// Compares analytic parameter and input gradients with central differences.
void gradientCheck(Network& net, Loss& loss, const Matrix& x, const Matrix& y) {
    const Matrix out = net.forward(x, true);  // training mode keeps what backward needs (no dropout here)
    loss.compute(out, y);
    const Matrix inputGrad = net.backward(loss.gradient());

    auto lossAt = [&]() { return static_cast<double>(loss.compute(net.forward(x, false), y)); };
    auto compare = [](double analytic, double numeric, const std::string& what) {
        const double denom = std::max(1.0, std::fabs(analytic) + std::fabs(numeric));
        if (std::fabs(analytic - numeric) / denom > 2e-2 && std::fabs(analytic - numeric) > 2e-3) {
            std::ostringstream os;
            os << "gradient mismatch for " << what << ": analytic " << analytic << " numeric " << numeric;
            throw Failure(os.str());
        }
    };

    const float eps = 1e-2f;
    auto params = net.parameters();
    for (std::size_t p = 0; p < params.size(); ++p) {
        const Matrix analytic = *params[p].grad;  // copy before further forward passes
        for (std::size_t i = 0; i < params[p].value->size(); ++i) {
            float& w = params[p].value->data()[i];
            const float original = w;
            w = original + eps;
            const double plus = lossAt();
            w = original - eps;
            const double minus = lossAt();
            w = original;
            compare(analytic.data()[i], (plus - minus) / (2.0 * eps),
                    "param " + std::to_string(p) + "[" + std::to_string(i) + "]");
        }
    }

    Matrix xp = x;
    for (std::size_t i = 0; i < xp.size(); ++i) {
        const float original = xp.data()[i];
        xp.data()[i] = original + eps;
        const double plus = loss.compute(net.forward(xp, false), y);
        xp.data()[i] = original - eps;
        const double minus = loss.compute(net.forward(xp, false), y);
        xp.data()[i] = original;
        compare(inputGrad.data()[i], (plus - minus) / (2.0 * eps), "input[" + std::to_string(i) + "]");
    }
}

} // namespace

// ---- Matrix ----------------------------------------------------------------------

TEST(matrix_construction_and_access) {
    Matrix m(2, 3, 1.5f);
    CHECK(m.rows() == 2 && m.cols() == 3 && m.size() == 6);
    CHECK(m(1, 2) == 1.5f);
    m.setElement(0, 1, 7.0f);
    CHECK(m.getElement(0, 1) == 7.0f);
    CHECK(m.getRows() == 2 && m.getCols() == 3);
    CHECK_THROWS(m.at(2, 0), std::out_of_range);
    CHECK_THROWS(m.getElement(-1, 0), std::out_of_range);
    CHECK_THROWS((Matrix{{1, 2}, {3}}), std::invalid_argument);
    CHECK_THROWS(Matrix(2, 2, std::vector<float>{1, 2, 3}), std::invalid_argument);
}

TEST(matrix_multiply) {
    const Matrix a{{1, 2, 3}, {4, 5, 6}};
    const Matrix b{{7, 8}, {9, 10}, {11, 12}};
    CHECK((a * b).approxEquals(Matrix{{58, 64}, {139, 154}}));
    CHECK_THROWS(a * a, std::invalid_argument);
}

TEST(matrix_transpose_products) {
    setSeed(1);
    const Matrix a = Matrix::uniform(5, 3, -1, 1);
    const Matrix b = Matrix::uniform(5, 4, -1, 1);
    const Matrix c = Matrix::uniform(6, 3, -1, 1);
    CHECK(a.transpose().rows() == 3 && a.transpose().cols() == 5);
    CHECK(a.transposeMatmul(b).approxEquals(a.transpose() * b));
    CHECK(a.matmulTranspose(c).approxEquals(a * c.transpose()));
    CHECK_THROWS(a.transposeMatmul(c), std::invalid_argument);
}

TEST(matrix_elementwise_and_reductions) {
    const Matrix a{{1, 2}, {3, 4}};
    const Matrix b{{5, 6}, {7, 8}};
    CHECK((a + b).approxEquals(Matrix{{6, 8}, {10, 12}}));
    CHECK((b - a).approxEquals(Matrix{{4, 4}, {4, 4}}));
    CHECK(a.hadamard(b).approxEquals(Matrix{{5, 12}, {21, 32}}));
    CHECK((a * 2.0f).approxEquals(2.0f * a));
    CHECK((a / 2.0f).approxEquals(Matrix{{0.5f, 1}, {1.5f, 2}}));
    CHECK((-a).approxEquals(Matrix{{-1, -2}, {-3, -4}}));
    CHECK(a.addRowVector(Matrix{{10, 20}}).approxEquals(Matrix{{11, 22}, {13, 24}}));
    CHECK(a.sumRows().approxEquals(Matrix{{4, 6}}));
    CHECK(a.sumCols().approxEquals(Matrix{{3}, {7}}));
    CHECK_NEAR(a.sum(), 10.0, 1e-6);
    CHECK_NEAR(a.mean(), 2.5, 1e-6);
    CHECK(a.max() == 4.0f && a.min() == 1.0f);
    CHECK(Matrix({{0.1f, 0.7f, 0.2f}}).argmaxRow(0) == 1);
    CHECK(a.gatherRows({1, 0, 1}).approxEquals(Matrix{{3, 4}, {1, 2}, {3, 4}}));
    CHECK(a.sliceRows(1, 2).approxEquals(Matrix{{3, 4}}));
    CHECK_THROWS(a + Matrix(3, 2), std::invalid_argument);
    CHECK_THROWS(a.addRowVector(Matrix(1, 3)), std::invalid_argument);
}

// ---- Layers & gradients ------------------------------------------------------------

TEST(dense_forward_known_values) {
    Dense d(2, 2);
    d.weights() = Matrix{{1, 2}, {3, 4}};
    d.bias() = Matrix{{0.5f, -1}};
    const Matrix out = d.forward(Matrix{{1, 1}, {2, 0}}, false);
    CHECK(out.approxEquals(Matrix{{4.5f, 5}, {2.5f, 3}}));
    CHECK_THROWS(d.forward(Matrix(1, 3), false), std::invalid_argument);
    CHECK(d.parameters().size() == 2);
}

TEST(gradcheck_dense_tanh_mse) {
    setSeed(2);
    Network net;
    net.add<Dense>(3, 4);
    net.add<Tanh>();
    net.add<Dense>(4, 2);
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(5, 3, -1, 1), Matrix::uniform(5, 2, -1, 1));
}

TEST(gradcheck_relu_sigmoid_bce) {
    setSeed(3);
    Network net;
    net.add<Dense>(3, 6, Init::He);
    net.add<ReLU>();
    net.add<Dense>(6, 1);
    net.add<Sigmoid>();
    BinaryCrossEntropy loss;
    gradientCheck(net, loss, Matrix::uniform(6, 3, -1, 1), Matrix{{0}, {1}, {1}, {0}, {1}, {0}});
}

TEST(gradcheck_leaky_relu_softmax_layer_cce) {
    setSeed(4);
    Network net;
    net.add<Dense>(4, 5);
    net.add<LeakyReLU>(0.1f);
    net.add<Dense>(5, 3);
    net.add<Softmax>();
    CategoricalCrossEntropy loss;
    gradientCheck(net, loss, Matrix::uniform(4, 4, -1, 1), oneHot({0, 2, 1, 2}, 3));
}

TEST(gradcheck_softmax_cross_entropy) {
    setSeed(5);
    Network net;
    net.add<Dense>(2, 8);
    net.add<Tanh>();
    net.add<Dense>(8, 4);
    SoftmaxCrossEntropy loss;
    gradientCheck(net, loss, Matrix::uniform(6, 2, -1, 1), oneHot({0, 1, 2, 3, 1, 0}, 4));
}

TEST(softmax_rows_sum_to_one_and_stable) {
    const Matrix p = softmax(Matrix{{1000, 1001, 1002}, {-5, 0, 5}});
    for (std::size_t i = 0; i < p.rows(); ++i) CHECK_NEAR(p.row(i).sum(), 1.0, 1e-5);
    CHECK(std::isfinite(p(0, 0)));
    CHECK(p.argmaxRow(0) == 2);
    CHECK_NEAR(sigmoid(0.0f), 0.5, 1e-7);
    CHECK(std::isfinite(sigmoid(-1000.0f)) && std::isfinite(sigmoid(1000.0f)));
}

TEST(fused_softmax_ce_matches_softmax_plus_cce) {
    const Matrix logits{{2.0f, -1.0f, 0.5f}, {0.1f, 0.2f, 3.0f}};
    const Matrix y = oneHot({0, 2}, 3);
    SoftmaxCrossEntropy fused;
    CategoricalCrossEntropy cce;
    Softmax sm;
    const float a = fused.compute(logits, y);
    const float b = cce.compute(sm.forward(logits, false), y);
    CHECK_NEAR(a, b, 1e-5);
    CHECK(fused.gradient().approxEquals(sm.backward(cce.gradient()), 1e-5f));
}

TEST(loss_known_values) {
    MSE mse;
    CHECK_NEAR(mse.compute(Matrix{{1, 2}}, Matrix{{3, 2}}), 2.0, 1e-6);
    CHECK(mse.gradient().approxEquals(Matrix{{-2, 0}}));
    BinaryCrossEntropy bce;
    CHECK_NEAR(bce.compute(Matrix{{0.5f}}, Matrix{{1}}), std::log(2.0), 1e-6);
    CHECK_THROWS(mse.compute(Matrix(2, 1), Matrix(1, 2)), std::invalid_argument);
}

TEST(dropout_train_and_eval) {
    setSeed(6);
    Dropout drop(0.5f);
    const Matrix x = Matrix::ones(200, 50);
    CHECK(drop.forward(x, false).approxEquals(x));
    const Matrix y = drop.forward(x, true);
    std::size_t zeros = 0;
    for (float v : y.values()) zeros += v == 0.0f;
    const double zeroFraction = static_cast<double>(zeros) / static_cast<double>(y.size());
    CHECK_NEAR(zeroFraction, 0.5, 0.05);
    CHECK_NEAR(y.mean(), 1.0, 0.05);  // inverted dropout keeps the expectation
    const Matrix g = drop.backward(Matrix::ones(200, 50));
    CHECK(g.approxEquals(y));
    CHECK_THROWS(Dropout(1.0f), std::invalid_argument);
}

// ---- Optimizers ----------------------------------------------------------------------

namespace {
// Minimizes f(w) = sum((w - 3)^2) with the given optimizer.
float minimizeQuadratic(Optimizer& opt, int steps) {
    Matrix w(1, 3, 0.0f), g(1, 3);
    std::vector<Param> params{{&w, &g}};
    for (int s = 0; s < steps; ++s) {
        for (std::size_t i = 0; i < w.size(); ++i) g.data()[i] = 2.0f * (w.data()[i] - 3.0f);
        opt.step(params);
    }
    return (w - 3.0f).apply([](float v) { return v * v; }).sum();
}
} // namespace

TEST(optimizers_minimize_quadratic) {
    SGD sgd(0.1f);
    CHECK(minimizeQuadratic(sgd, 100) < 1e-6f);
    SGD momentum(0.05f, 0.9f);
    CHECK(minimizeQuadratic(momentum, 300) < 1e-4f);
    Adam adam(0.1f);
    CHECK(minimizeQuadratic(adam, 500) < 1e-4f);
    CHECK_THROWS(SGD(0.0f), std::invalid_argument);
    CHECK_THROWS(SGD(0.1f, 1.0f), std::invalid_argument);
}

// ---- Network ------------------------------------------------------------------------

TEST(network_learns_xor) {
    setSeed(7);
    const Dataset d = makeXor();
    Network net;
    net.add<Dense>(2, 8);
    net.add<Tanh>();
    net.add<Dense>(8, 1);
    net.add<Sigmoid>();
    BinaryCrossEntropy loss;
    Adam opt(0.05f);
    FitOptions opts;
    opts.epochs = 800;
    opts.batchSize = 4;
    opts.verbose = false;
    const History h = net.fit(d.x, d.y, loss, opt, opts);
    CHECK(h.size() == 800);
    CHECK(h.back().loss < h.front().loss);
    CHECK(h.back().loss < 0.05f);
    CHECK_NEAR(net.accuracy(d.x, d.y), 1.0, 1e-6);
}

TEST(network_fit_reports_validation_and_is_deterministic) {
    auto train = []() {
        setSeed(8);
        const Dataset d = makeSpiral(40, 2, 0.1f);
        Network net;
        net.add<Dense>(2, 16, Init::He);
        net.add<ReLU>();
        net.add<Dense>(16, 2);
        SoftmaxCrossEntropy loss;
        SGD opt(0.1f, 0.9f);
        FitOptions opts;
        opts.epochs = 20;
        opts.batchSize = 16;
        opts.verbose = false;
        opts.trackAccuracy = true;
        opts.validationX = &d.x;
        opts.validationY = &d.y;
        std::size_t callbacks = 0;
        opts.onEpochEnd = [&callbacks](const EpochStats&) { ++callbacks; };
        History h = net.fit(d.x, d.y, loss, opt, opts);
        CHECK(callbacks == 20);
        return h;
    };
    const History a = train();
    const History b = train();
    CHECK(a.back().accuracy >= 0.0f && a.back().validationLoss >= 0.0f && a.back().validationAccuracy >= 0.0f);
    CHECK(a.back().loss == b.back().loss);
}

TEST(network_fit_rejects_bad_input) {
    Network net;
    net.add<Dense>(2, 1);
    MSE loss;
    SGD opt(0.1f);
    CHECK_THROWS(net.fit(Matrix(3, 2), Matrix(2, 1), loss, opt), std::invalid_argument);
    FitOptions opts;
    opts.batchSize = 0;
    CHECK_THROWS(net.fit(Matrix(3, 2), Matrix(3, 1), loss, opt, opts), std::invalid_argument);
    Network empty;
    CHECK_THROWS(empty.predict(Matrix(1, 2)), std::logic_error);
}

TEST(network_save_and_load_roundtrip) {
    setSeed(9);
    auto build = []() {
        Network net;
        net.add<Dense>(3, 5);
        net.add<ReLU>();
        net.add<Dense>(5, 2);
        return net;
    };
    Network a = build();
    Network b = build();
    const Matrix x = Matrix::uniform(4, 3, -1, 1);
    CHECK(!a.predict(x).approxEquals(b.predict(x)));
    CHECK(a.parameterCount() == 3 * 5 + 5 + 5 * 2 + 2);

    const std::string path = tempPath("weights.txt");
    a.save(path);
    b.load(path);
    CHECK(a.predict(x).approxEquals(b.predict(x), 1e-6f));

    Network other;
    other.add<Dense>(3, 4);
    CHECK_THROWS(other.load(path), std::runtime_error);
    CHECK_THROWS(b.load(tempPath("does_not_exist")), std::runtime_error);
    std::remove(path.c_str());
}

TEST(network_summary_lists_layers) {
    Network net;
    net.add<Dense>(4, 3);
    net.add<Dropout>(0.25f);
    std::ostringstream os;
    net.summary(os);
    CHECK(os.str().find("Dense(4 -> 3)") != std::string::npos);
    CHECK(os.str().find("Dropout(0.25)") != std::string::npos);
    CHECK(os.str().find("15") != std::string::npos);
}

// ---- Data -----------------------------------------------------------------------------

TEST(one_hot_and_argmax) {
    const Matrix m = oneHot({2, 0, 1}, 3);
    CHECK(m.approxEquals(Matrix{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}));
    CHECK(argmaxRows(m) == std::vector<int>({2, 0, 1}));
    CHECK_THROWS(oneHot({3}, 3), std::out_of_range);
    CHECK_NEAR(accuracyScore(Matrix{{0.9f}, {0.2f}}, Matrix{{1}, {1}}), 0.5, 1e-6);
}

TEST(train_test_split_and_standardizer) {
    setSeed(10);
    const Dataset d = makeSpiral(50, 2);
    const Split s = trainTestSplit(d.x, d.y, 0.2f);
    CHECK(s.train.x.rows() == 80 && s.test.x.rows() == 20);
    CHECK(s.train.y.rows() == 80 && s.test.y.cols() == 2);

    Standardizer st;
    const Matrix z = st.fitTransform(Matrix{{1, 10}, {3, 10}, {5, 10}});
    CHECK_NEAR(z.sumRows()(0, 0), 0.0, 1e-5);
    CHECK_NEAR(z(2, 0), std::sqrt(1.5), 1e-4);
    CHECK_NEAR(z(0, 1), 0.0, 1e-6);  // constant column stays finite
}

TEST(csv_roundtrip_and_errors) {
    const std::string path = tempPath("data.csv");
    const Matrix m{{1.5f, -2}, {3, 4.25f}};
    saveCSV(path, m);
    CHECK(loadCSV(path).approxEquals(m));
    {
        std::ofstream out(path);
        out << "a,b\n1,2\r\n3,4\n\n";
    }
    CHECK(loadCSV(path, true).approxEquals(Matrix{{1, 2}, {3, 4}}));
    {
        std::ofstream out(path);
        out << "1,2\n3\n";
    }
    CHECK_THROWS(loadCSV(path), std::runtime_error);
    {
        std::ofstream out(path);
        out << "1,x\n";
    }
    CHECK_THROWS(loadCSV(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(idx_loader_reads_mnist_format) {
    const std::string images = tempPath("images.idx");
    const std::string labels = tempPath("labels.idx");
    auto be32 = [](std::ofstream& out, std::uint32_t v) {
        const unsigned char b[4] = {static_cast<unsigned char>(v >> 24), static_cast<unsigned char>(v >> 16),
                                    static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v)};
        out.write(reinterpret_cast<const char*>(b), 4);
    };
    {
        std::ofstream out(images, std::ios::binary);
        be32(out, 0x803);
        be32(out, 3);
        be32(out, 2);
        be32(out, 2);
        const unsigned char px[12] = {0, 255, 51, 102, 1, 2, 3, 4, 255, 255, 0, 0};
        out.write(reinterpret_cast<const char*>(px), 12);
    }
    {
        std::ofstream out(labels, std::ios::binary);
        be32(out, 0x801);
        be32(out, 3);
        const unsigned char lb[3] = {7, 0, 9};
        out.write(reinterpret_cast<const char*>(lb), 3);
    }
    const Matrix x = loadIdxImages(images);
    CHECK(x.rows() == 3 && x.cols() == 4);
    CHECK_NEAR(x(0, 1), 1.0, 1e-6);
    CHECK_NEAR(x(0, 2), 0.2, 1e-6);
    CHECK(loadIdxImages(images, 2).rows() == 2);
    CHECK(loadIdxLabels(labels) == std::vector<int>({7, 0, 9}));
    CHECK_THROWS(loadIdxImages(labels), std::runtime_error);  // wrong magic
    std::remove(images.c_str());
    std::remove(labels.c_str());
}

TEST(toy_datasets_have_expected_shapes) {
    const Dataset sp = makeSpiral(10, 4);
    CHECK(sp.x.rows() == 40 && sp.x.cols() == 2 && sp.y.cols() == 4);
    CHECK_NEAR(sp.y.sum(), 40.0, 1e-6);
    const Dataset sn = makeSine(5);
    CHECK_NEAR(sn.x(0, 0), -3.14159265, 1e-5);
    CHECK_NEAR(sn.y(2, 0), 0.0, 1e-5);
}

// ---- Convolution ----------------------------------------------------------------------

TEST(conv2d_forward_known_values) {
    // 1x3x3 image, 2x2 kernel of ones, bias 1 -> sums of 2x2 windows + 1.
    Conv2D conv({1, 3, 3}, 1, 2);
    conv.weights().fill(1.0f);
    conv.bias().fill(1.0f);
    const Matrix out = conv.forward(Matrix{{1, 2, 3, 4, 5, 6, 7, 8, 9}}, false);
    CHECK(out.approxEquals(Matrix{{13, 17, 25, 29}}));
    CHECK(conv.outputShape().height == 2 && conv.outputShape().width == 2);

    Conv2D padded({1, 3, 3}, 2, 3, 1, 1);
    CHECK(padded.outputSize(9) == 2 * 3 * 3);
    Conv2D strided({1, 5, 5}, 1, 3, 2, 0);
    CHECK(strided.outputShape().height == 2 && strided.outputShape().width == 2);
    CHECK_THROWS(conv.forward(Matrix(1, 8), false), std::invalid_argument);
    CHECK_THROWS(Conv2D({1, 2, 2}, 1, 3), std::invalid_argument);
}

TEST(gradcheck_conv2d_padding) {
    setSeed(20);
    Network net;
    net.add<Conv2D>(ImageShape{2, 5, 4}, 3, 3, 1, 1);
    net.add<Tanh>();
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(2, 2 * 5 * 4, -1, 1), Matrix::uniform(2, 3 * 5 * 4, -1, 1));
}

TEST(gradcheck_conv2d_stride) {
    setSeed(21);
    Network net;
    net.add<Conv2D>(ImageShape{1, 7, 6}, 2, 3, 2, 0, Init::Xavier);
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(3, 42, -1, 1), Matrix::uniform(3, 2 * 3 * 2, -1, 1));
}

TEST(maxpool_forward_and_gradcheck) {
    MaxPool2D pool({1, 4, 4}, 2);
    const Matrix x{{1, 2, 5, 6, 3, 4, 7, 8, -1, -2, 0, 9, -3, -4, 1, 2}};
    CHECK(pool.forward(x, false).approxEquals(Matrix{{4, 8, -1, 9}}));
    const Matrix dx = pool.backward(Matrix{{1, 2, 3, 4}});
    CHECK(dx.approxEquals(Matrix{{0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 4, 0, 0, 0, 0}}));

    // Pool directly on well-separated inputs (gaps of 0.1 >> the 0.01 probe),
    // so the numerical gradient never flips which element is the maximum.
    setSeed(22);
    Network net;
    net.add<MaxPool2D>(ImageShape{2, 6, 6}, 2);
    net.add<Dense>(2 * 3 * 3, 3);
    net.add<Tanh>();
    Matrix input(2, 72);
    for (std::size_t i = 0; i < input.size(); ++i) input.data()[i] = static_cast<float>((i * 37) % 144) / 10.0f - 7.2f;
    MSE loss;
    gradientCheck(net, loss, input, Matrix::uniform(2, 3, -1, 1));
}

TEST(cnn_learns_simple_patterns) {
    // Vertical vs horizontal bars on 6x6 images.
    setSeed(23);
    Matrix x(40, 36);
    Matrix y(40, 2);
    std::uniform_int_distribution<int> pos(0, 5);
    for (std::size_t i = 0; i < 40; ++i) {
        const bool vertical = i % 2 == 0;
        const int p = pos(rng());
        for (int k = 0; k < 6; ++k) x(i, vertical ? static_cast<std::size_t>(k * 6 + p) : static_cast<std::size_t>(p * 6 + k)) = 1.0f;
        y(i, vertical ? 0 : 1) = 1.0f;
    }
    Network net;
    net.add<Conv2D>(ImageShape{1, 6, 6}, 4, 3, 1, 1);
    net.add<ReLU>();
    net.add<MaxPool2D>(ImageShape{4, 6, 6}, 2);
    net.add<Dense>(4 * 3 * 3, 2);
    SoftmaxCrossEntropy loss;
    Adam opt(0.01f);
    FitOptions opts;
    opts.epochs = 60;
    opts.batchSize = 8;
    opts.verbose = false;
    net.fit(x, y, loss, opt, opts);
    CHECK_NEAR(net.accuracy(x, y), 1.0, 1e-6);
}

// ---- Transformer / language model -------------------------------------------------------

TEST(layernorm_and_gelu_gradcheck) {
    setSeed(30);
    Network net;
    net.add<Dense>(4, 6);
    net.add<LayerNorm>(6);
    net.add<GELU>();
    net.add<Dense>(6, 2);
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(5, 4, -1.5f, 1.5f), Matrix::uniform(5, 2, -1, 1));

    LayerNorm ln(3);
    const Matrix y = ln.forward(Matrix{{1, 2, 3}, {10, 10, 13}}, false);
    for (std::size_t r = 0; r < 2; ++r) CHECK_NEAR(y.row(r).sum(), 0.0, 1e-5);
    CHECK_NEAR(GELU().forward(Matrix{{0.0f}}, false)(0, 0), 0.0, 1e-7);
}

TEST(attention_gradcheck_and_residual) {
    setSeed(31);
    auto ctx = std::make_shared<SequenceContext>();
    ctx->length = 3;  // input: 2 sequences of 3 positions
    Network net;
    auto& block = net.add<Residual>();
    block.add<LayerNorm>(4);
    block.add<CausalSelfAttention>(4, 2, ctx);
    net.add<Dense>(4, 3);
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(6, 4, -1, 1), Matrix::uniform(6, 3, -1, 1));
}

TEST(attention_is_causal) {
    setSeed(32);
    auto ctx = std::make_shared<SequenceContext>();
    ctx->length = 5;
    CausalSelfAttention att(8, 2, ctx);
    Matrix x = Matrix::uniform(5, 8, -1, 1);
    const Matrix before = att.forward(x, false);
    for (std::size_t j = 0; j < 8; ++j) x(4, j) += 3.0f;  // change only the last position
    const Matrix after = att.forward(x, false);
    CHECK(before.sliceRows(0, 4).approxEquals(after.sliceRows(0, 4), 1e-6f));
    CHECK(!before.row(4).approxEquals(after.row(4), 1e-3f));
    ctx->length = 4;
    CHECK_THROWS(att.forward(x, false), std::invalid_argument);  // 5 rows, length 4
}

TEST(sparse_cross_entropy_matches_dense) {
    const Matrix logits{{2.0f, -1.0f, 0.5f}, {0.1f, 0.2f, 3.0f}, {1, 1, 1}};
    SparseSoftmaxCrossEntropy sparse;
    SoftmaxCrossEntropy dense;
    CHECK_NEAR(sparse.compute(logits, Matrix{{0, 2, 1}}), dense.compute(logits, oneHot({0, 2, 1}, 3)), 1e-5);
    CHECK(sparse.gradient().approxEquals(dense.gradient(), 1e-6f));
    CHECK_THROWS(sparse.compute(logits, Matrix{{0, 3, 1}}), std::out_of_range);
    CHECK_THROWS(sparse.compute(logits, Matrix{{0, 1}}), std::invalid_argument);
}

TEST(gpt_gradcheck_end_to_end) {
    setSeed(33);
    GPTConfig cfg;
    cfg.vocabSize = 5;
    cfg.contextLength = 4;
    cfg.dim = 8;
    cfg.heads = 2;
    cfg.layers = 1;
    cfg.mlpRatio = 2;
    Network net;
    buildGPT(net, cfg);
    // Embeddings start at std 0.02; widen them so the 0.01 finite-difference
    // probe is small relative to the values LayerNorm normalizes.
    auto& emb = dynamic_cast<TokenEmbedding&>(net.layer(0));
    emb.tokenTable() = Matrix::uniform(cfg.vocabSize, cfg.dim, -1, 1);
    emb.positionTable() = Matrix::uniform(cfg.contextLength, cfg.dim, -1, 1);
    SparseSoftmaxCrossEntropy loss;
    // Token-id inputs: the numerical probe never changes the rounded ids, and
    // TokenEmbedding returns a zero input gradient, so the check still holds.
    gradientCheck(net, loss, Matrix{{0, 3, 1, 4}, {2, 2, 0, 1}}, Matrix{{3, 1, 4, 0}, {2, 0, 1, 3}});
}

TEST(tokenizer_and_dataset) {
    const CharTokenizer tok("hello world");
    CHECK(tok.vocabSize() == 8);  // ' ' d e h l o r w
    const auto ids = tok.encode("hold");
    CHECK(tok.decode(ids) == "hold");
    CHECK(tok.contains('w') && !tok.contains('z'));
    CHECK_THROWS(tok.encode("z"), std::invalid_argument);

    const Dataset d = makeNextTokenDataset({0, 1, 2, 3, 4, 5, 6}, 3, 2);
    CHECK(d.x.rows() == 2 && d.x.cols() == 3);
    CHECK(d.x.approxEquals(Matrix{{0, 1, 2}, {2, 3, 4}}));
    CHECK(d.y.approxEquals(Matrix{{1, 2, 3}, {3, 4, 5}}));
    CHECK_THROWS(makeNextTokenDataset({1, 2}, 3), std::invalid_argument);
}

TEST(bpe_split_train_encode_decode) {
    const std::string text = "the cat sat on the mat. the cat ate 42 rats!\n\nthe end";
    std::string joined;
    for (const auto& w : splitWords(text)) joined += w;
    CHECK(joined == text);
    CHECK(splitWords("hi  there, 12x")[1] == " ");  // "hi", " ", " there", ",", " 12", "x"
    CHECK(splitWords("hi  there, 12x").size() == 6);

    const BpeTokenizer tok = BpeTokenizer::train(text, 280);
    CHECK(tok.vocabSize() > 256 && tok.vocabSize() <= 280);
    CHECK(tok.piece(256).size() == 2);
    const auto ids = tok.encode(text);
    CHECK(ids.size() < text.size());
    CHECK(tok.decode(ids) == text);
    // Unseen bytes still round-trip (byte fallback).
    const std::string other = "Zebra \xc3\xa7ok g\xc3\xbczel~\t";
    CHECK(tok.decode(tok.encode(other)) == other);
    // " the" occurs often enough to become one token.
    bool hasThe = false;
    for (std::size_t id = 256; id < tok.vocabSize(); ++id) hasThe |= tok.piece(static_cast<int>(id)) == " the";
    CHECK(hasThe);
    CHECK_THROWS(BpeTokenizer::train(text, 100), std::invalid_argument);
}

TEST(bpe_save_load_and_determinism) {
    const std::string text = "low lower lowest newer newest wider widest low low lower";
    const BpeTokenizer a = BpeTokenizer::train(text, 270);
    const BpeTokenizer b = BpeTokenizer::train(text, 270);
    CHECK(a.encode(text) == b.encode(text));
    const std::string path = tempPath("bpe.txt");
    a.save(path);
    const BpeTokenizer c = BpeTokenizer::load(path);
    CHECK(c.vocabSize() == a.vocabSize());
    CHECK(c.encode(text) == a.encode(text));
    {
        std::ofstream out(path);
        out << "NEURACRAFT-BPE 1\n1\n300 2\n";
    }
    CHECK_THROWS(BpeTokenizer::load(path), std::runtime_error);
    std::remove(path.c_str());
}

namespace {
Network smallGpt(SequenceContextPtr& ctx, std::size_t context = 16) {
    GPTConfig cfg;
    cfg.vocabSize = 11;
    cfg.contextLength = context;
    cfg.dim = 16;
    cfg.heads = 4;
    cfg.layers = 2;
    Network net;
    ctx = buildGPT(net, cfg);
    return net;
}
} // namespace

TEST(kv_cache_matches_full_forward) {
    setSeed(40);
    SequenceContextPtr ctx;
    Network net = smallGpt(ctx);
    const std::vector<float> ids{3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    const Matrix full = net.predict(Matrix(1, ids.size(), ids));  // 10 x vocab

    ctx->incremental = true;
    ctx->offset = 0;
    Matrix cached(0, full.cols());
    std::vector<float> rows;
    std::size_t pos = 0;
    for (std::size_t chunk : {4, 1, 1, 3, 1}) {  // prompt, then single tokens / a small chunk
        const Matrix out = net.predict(Matrix(1, chunk, std::vector<float>(ids.begin() + pos, ids.begin() + pos + chunk)));
        rows.insert(rows.end(), out.values().begin(), out.values().end());
        ctx->offset += chunk;
        pos += chunk;
    }
    ctx->incremental = false;
    ctx->offset = 0;
    CHECK(Matrix(ids.size(), full.cols(), rows).approxEquals(full, 1e-4f));

    // A wrong offset must be detected, not silently produce garbage.
    ctx->incremental = true;
    ctx->offset = 3;
    CHECK_THROWS(net.predict(Matrix(1, 1, std::vector<float>{2})), std::logic_error);
    ctx->incremental = false;
    ctx->offset = 0;
}

TEST(kv_cached_generation_matches_uncached) {
    setSeed(41);
    SequenceContextPtr ctx;
    Network net = smallGpt(ctx);
    SamplingOptions greedy;
    greedy.temperature = 0.0f;
    const std::vector<int> prompt{1, 2, 3};
    // Within the context window both paths see exactly the same tokens.
    CHECK(generate(net, prompt, 13, 16, greedy) == generate(net, prompt, 13, 16, greedy, ctx));
    CHECK(!ctx->incremental && ctx->offset == 0);  // state restored

    // Beyond the window the cache is rebuilt; it must keep working.
    const auto longRun = generate(net, prompt, 60, 16, greedy, ctx);
    CHECK(longRun.size() == 60);
    // Training-mode forward still works afterwards (incremental mode is off).
    Matrix batch(2, 16);
    CHECK(net.forward(batch, true).rows() == 32);
    // A prompt longer than the window is trimmed to its last tokens: the first
    // step sees the same window on both paths. (Later steps differ by design:
    // the full cache is rebuilt from half a window instead of sliding.)
    std::vector<int> longPrompt(40, 4);
    CHECK(generate(net, longPrompt, 1, 16, greedy, ctx) == generate(net, longPrompt, 1, 16, greedy));
    CHECK(generate(net, longPrompt, 5, 16, greedy, ctx).size() == 5);
}

TEST(gpt_memorizes_a_sequence_and_generates_it) {
    setSeed(34);
    const std::string text = "abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh ";
    const CharTokenizer tok(text);
    const Dataset d = makeNextTokenDataset(tok.encode(text), 8, 1);
    GPTConfig cfg;
    cfg.vocabSize = tok.vocabSize();
    cfg.contextLength = 8;
    cfg.dim = 32;
    cfg.heads = 4;
    cfg.layers = 2;
    Network net;
    buildGPT(net, cfg);
    SparseSoftmaxCrossEntropy loss;
    Adam opt(0.01f);
    FitOptions opts;
    opts.epochs = 40;
    opts.batchSize = 16;
    opts.verbose = false;
    const History h = net.fit(d.x, d.y, loss, opt, opts);
    CHECK(h.back().loss < 0.1f);
    SamplingOptions greedy;
    greedy.temperature = 0.0f;
    CHECK(tok.decode(generate(net, tok.encode("abc"), 13, cfg.contextLength, greedy)) == "defgh abcdefg");
    SamplingOptions sampled;
    sampled.temperature = 0.8f;
    sampled.topK = 3;
    CHECK(generate(net, tok.encode("a"), 20, cfg.contextLength, sampled).size() == 20);
}

// ---- Modern transformer parts ----------------------------------------------------------------

TEST(rmsnorm_and_swiglu_gradcheck) {
    setSeed(60);
    Network net;
    net.add<Dense>(4, 8);
    net.add<RMSNorm>(8);
    net.add<SwiGLUGate>();
    net.add<Dense>(4, 2);
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(5, 4, -1.5f, 1.5f), Matrix::uniform(5, 2, -1, 1));

    RMSNorm rms(2);
    CHECK(rms.forward(Matrix{{3, 4}}, false).approxEquals(Matrix{{3 / std::sqrt(12.5f), 4 / std::sqrt(12.5f)}}, 1e-5f));
    CHECK(SwiGLUGate().forward(Matrix{{0, 5}}, false).approxEquals(Matrix{{0}}));
}

TEST(rotary_attention_gradcheck) {
    setSeed(61);
    auto ctx = std::make_shared<SequenceContext>();
    ctx->length = 3;
    Network net;
    auto& block = net.add<Residual>();
    block.add<RMSNorm>(8);
    block.add<CausalSelfAttention>(8, 2, ctx, true);
    net.add<Dense>(8, 3);
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(6, 8, -1, 1), Matrix::uniform(6, 3, -1, 1));
    CHECK_THROWS(CausalSelfAttention(6, 2, ctx, true), std::invalid_argument);  // odd head size
}

TEST(rotary_scores_depend_on_relative_position_only) {
    // The same tokens at shifted absolute positions give the same outputs.
    setSeed(62);
    auto ctx = std::make_shared<SequenceContext>();
    CausalSelfAttention att(8, 2, ctx, true);
    const Matrix x = Matrix::uniform(4, 8, -1, 1);
    ctx->incremental = true;
    ctx->offset = 0;
    const Matrix at0 = att.forward(x, false);
    // Prime the cache with 7 unrelated positions, then drop them with a window.
    ctx->offset = 0;
    att.forward(Matrix::uniform(7, 8, -1, 1), false);
    ctx->offset = 7;
    ctx->window = 4;  // keeps only the 4 new rows
    const Matrix at7 = att.forward(x, false);
    ctx->incremental = false;
    ctx->offset = 0;
    ctx->window = 0;
    CHECK(at0.approxEquals(at7, 1e-4f));
}

namespace {
Network modernGpt(SequenceContextPtr& ctx, std::size_t context = 16) {
    GPTConfig cfg;
    cfg.vocabSize = 11;
    cfg.contextLength = context;
    cfg.dim = 16;
    cfg.heads = 4;
    cfg.layers = 2;
    cfg.modern();
    Network net;
    ctx = buildGPT(net, cfg);
    return net;
}
} // namespace

TEST(modern_gpt_gradcheck_and_tied_parameters) {
    setSeed(63);
    GPTConfig cfg;
    cfg.vocabSize = 5;
    cfg.contextLength = 4;
    cfg.dim = 8;
    cfg.heads = 2;
    cfg.layers = 1;
    cfg.mlpRatio = 2;
    cfg.modern();
    Network net;
    buildGPT(net, cfg);
    auto& emb = dynamic_cast<TokenEmbedding&>(net.layer(0));
    emb.tokenTable() = Matrix::uniform(cfg.vocabSize, cfg.dim, -1, 1);
    SparseSoftmaxCrossEntropy loss;
    gradientCheck(net, loss, Matrix{{0, 3, 1, 4}, {2, 2, 0, 1}}, Matrix{{3, 1, 4, 0}, {2, 0, 1, 3}});

    // Tied output adds no parameters; there is no position table.
    GPTConfig untied = cfg;
    untied.tieEmbeddings = false;
    Network other;
    buildGPT(other, untied);
    CHECK(other.parameterCount() == net.parameterCount() + cfg.dim * cfg.vocabSize + cfg.vocabSize);
    CHECK(emb.parameters().size() == 1);
    CHECK(cfg.mlpHidden() % 8 == 0);
}

namespace {
// Largest logit difference between the incremental (sliding KV cache) path
// and the uncached forward pass over the last `window` tokens, feeding `steps`
// tokens one at a time after a 3-token prompt.
float slidingCacheError(Network& net, SequenceContextPtr& ctx, std::size_t window, int steps) {
    auto row = [](const std::vector<int>& toks, std::size_t from) {
        Matrix x(1, toks.size() - from);
        for (std::size_t i = from; i < toks.size(); ++i) x(0, i - from) = static_cast<float>(toks[i]);
        return x;
    };
    std::vector<int> toks{1, 2, 3};
    ctx->incremental = true;
    ctx->offset = 0;
    ctx->window = window;
    net.predict(row(toks, 0));
    ctx->offset = toks.size();
    float worst = 0.0f;
    for (int step = 0; step < steps; ++step) {
        toks.push_back((step * 7 + 3) % 11);
        const Matrix cached = net.predict(row(toks, toks.size() - 1));
        ctx->offset += 1;
        ctx->incremental = false;  // reference pass leaves the KV caches untouched
        const Matrix full = net.predict(row(toks, toks.size() - std::min(window, toks.size())));
        ctx->incremental = true;
        const float* last = full.data() + (full.rows() - 1) * full.cols();
        for (std::size_t j = 0; j < full.cols(); ++j) worst = std::max(worst, std::fabs(last[j] - cached.data()[j]));
    }
    ctx->incremental = false;
    ctx->offset = 0;
    ctx->window = 0;
    return worst;
}
} // namespace

TEST(modern_kv_cache_slides) {
    // One layer: rotary scores depend only on relative distance, so the sliding
    // cache equals recomputing the window, even 40 tokens past an 8-token window.
    setSeed(64);
    GPTConfig one;
    one.vocabSize = 11;
    one.contextLength = 8;
    one.dim = 16;
    one.heads = 4;
    one.layers = 1;
    one.modern();
    Network shallow;
    SequenceContextPtr ctx1 = buildGPT(shallow, one);
    CHECK(slidingCacheError(shallow, ctx1, 8, 40) < 1e-3f);

    // Deeper models: exact while the text fits the window. Past it, later layers
    // keep information from beyond the window (sliding-window attention), so
    // they legitimately differ from a fresh recomputation.
    SequenceContextPtr ctx;
    Network net = modernGpt(ctx, 8);
    CHECK(slidingCacheError(net, ctx, 8, 5) < 1e-3f);

    SamplingOptions greedy;
    greedy.temperature = 0.0f;
    CHECK(generate(net, {1, 2, 3}, 30, 8, greedy, ctx).size() == 30);
    CHECK(ctx->window == 0 && !ctx->incremental);
}

TEST(generator_keeps_the_cache_across_turns) {
    // Greedy multi-turn generation equals generating from the whole transcript at once.
    SamplingOptions greedy;
    greedy.temperature = 0.0f;
    for (bool rotary : {false, true}) {
        setSeed(42);
        SequenceContextPtr ctx;
        Network net = rotary ? modernGpt(ctx, 64) : smallGpt(ctx, 64);
        Generator chat(net, ctx, 64);
        const std::vector<int> turn1{1, 2, 3}, turn2{4, 5};
        const auto reply1 = chat.generate(turn1, 6, greedy);
        CHECK(reply1 == generate(net, turn1, 6, 64, greedy));
        CHECK(chat.cachedTokens() == turn1.size() + reply1.size() - 1);  // the last reply token is fed next turn
        std::vector<int> transcript = turn1;
        transcript.insert(transcript.end(), reply1.begin(), reply1.end());
        transcript.insert(transcript.end(), turn2.begin(), turn2.end());
        const auto reply2 = chat.generate(turn2, 6, greedy);
        CHECK(reply2 == generate(net, transcript, 6, 64, greedy));
        CHECK(chat.history().size() == transcript.size() + reply2.size());
        CHECK(!ctx->incremental);
        // Stop tokens end a turn early; reset() starts over.
        SamplingOptions stop = greedy;
        stop.stopTokens = {reply2[0]};
        chat.reset();
        CHECK(chat.history().empty() && chat.cachedTokens() == 0);
        CHECK(chat.generate(transcript, 6, stop).size() == 1);
    }
    // Learned positions: a conversation longer than the context rebuilds the cache and keeps going.
    setSeed(43);
    SequenceContextPtr ctx;
    Network net = smallGpt(ctx, 16);
    Generator chat(net, ctx, 16);
    for (int turn = 0; turn < 6; ++turn) CHECK(chat.generate({1, 2, 3}, 5, greedy).size() == 5);
    CHECK(chat.cachedTokens() <= 16);
}

TEST(top_p_and_repetition_penalty) {
    setSeed(44);
    const float logits[4] = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
    SamplingOptions o;
    o.temperature = 1.0f;
    o.topP = 0.7f;  // keeps 0.5 + 0.3
    int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4000; ++i) ++counts[sampleToken(logits, 4, o)];
    CHECK(counts[2] == 0 && counts[3] == 0);
    CHECK_NEAR(counts[0] / 4000.0, 0.625, 0.03);  // 0.5 / 0.8
    o.topP = 0.4f;  // the most likely token alone covers it
    for (int i = 0; i < 100; ++i) CHECK(sampleToken(logits, 4, o) == 0);
    o.topP = 1.0f;
    for (int i = 0; i < 4000; ++i) ++counts[sampleToken(logits, 4, o)];
    CHECK(counts[3] > 100);  // off: every token can come up

    // A strong penalty steers greedy decoding away from recent tokens, within the window only.
    const float l2[3] = {2.0f, 1.5f, -1.0f};
    SamplingOptions g;
    g.temperature = 0.0f;
    CHECK(sampleToken(l2, 3, g, {0}) == 0);
    g.repetitionPenalty = 2.0f;
    CHECK(sampleToken(l2, 3, g, {0}) == 1);
    CHECK(sampleToken(l2, 3, g, {0, 1}) == 0);  // both penalized: 1.0 vs 0.75
    g.repetitionWindow = 1;
    CHECK(sampleToken(l2, 3, g, {0, 2}) == 0);  // token 0 is outside the window
}

TEST(modern_gpt_learns_and_config_roundtrip) {
    setSeed(65);
    const std::string text = "abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh ";
    const CharTokenizer tok(text);
    const Dataset d = makeNextTokenDataset(tok.encode(text), 8, 1);
    GPTConfig cfg;
    cfg.vocabSize = tok.vocabSize();
    cfg.contextLength = 8;
    cfg.dim = 32;
    cfg.heads = 4;
    cfg.layers = 2;
    cfg.modern();
    Network net;
    auto ctx = buildGPT(net, cfg);
    SparseSoftmaxCrossEntropy loss;
    Adam opt(0.01f);
    FitOptions opts;
    opts.epochs = 40;
    opts.batchSize = 16;
    opts.verbose = false;
    CHECK(net.fit(d.x, d.y, loss, opt, opts).back().loss < 0.1f);
    SamplingOptions greedy;
    greedy.temperature = 0.0f;
    CHECK(tok.decode(generate(net, tok.encode("abc"), 13, 8, greedy, ctx)) == "defgh abcdefg");

    const std::string path = tempPath("modern.config");
    saveGPTConfig(path, cfg);
    const GPTConfig r = loadGPTConfig(path);
    CHECK(r.position == PositionEncoding::Rotary && r.norm == Normalization::RMSNorm &&
          r.mlp == FeedForward::SwiGLU && r.tieEmbeddings);
    std::remove(path.c_str());

    // Weights saved from a tied model load back into the same architecture.
    const std::string weights = tempPath("modern.weights");
    net.save(weights);
    Network again;
    buildGPT(again, r);
    again.load(weights);
    const Matrix x = d.x.sliceRows(0, 2);
    CHECK(net.predict(x).approxEquals(again.predict(x), 1e-6f));
    std::remove(weights.c_str());
}

// ---- Quantization and grouped-query attention -------------------------------------------------

TEST(int8_weights_error_bounds_and_outliers) {
    setSeed(80);
    Matrix w = Matrix::uniform(200, 70, -1, 1);  // in x out
    w(3, 5) = 50.0f;                              // an outlier in output 5
    const QuantizedWeights q = QuantizedWeights::fromInOut(w);
    CHECK(q.inputs() == 200 && q.outputs() == 70 && q.group() == 64);
    CHECK(q.bytes() < w.size() * sizeof(float) / 3);
    const Matrix back = q.dequantize();
    // Per-group scales: only the outlier's own group (inputs 0..63 of output 5) is coarse.
    float worstOther = 0.0f, worstOutlierGroup = 0.0f;
    for (std::size_t k = 0; k < 200; ++k)
        for (std::size_t j = 0; j < 70; ++j) {
            const float e = std::fabs(back(k, j) - w(k, j));
            if (j == 5 && k < 64) worstOutlierGroup = std::max(worstOutlierGroup, e);
            else worstOther = std::max(worstOther, e);
        }
    CHECK(worstOther <= 1.0f / 127.0f * 0.5f + 1e-6f);
    CHECK(worstOutlierGroup <= 50.0f / 127.0f * 0.5f + 1e-5f);

    const Matrix x = Matrix::uniform(3, 200, -1, 1);
    const Matrix exact = x * back;  // the product with the dequantized weights
    CHECK(q.multiplyFloat(x).approxEquals(exact, 1e-3f));
    // Integer products round each input to 1/127 of its 32-column block's
    // largest magnitude (here <= 1), so the error is at most sum_k |w_kj| / 254.
    const Matrix fast = q.multiply(x);
    for (std::size_t j = 0; j < 70; ++j) {
        double bound = 1e-4;
        for (std::size_t k = 0; k < 200; ++k) bound += std::fabs(back(k, j)) / 254.0;
        for (std::size_t i = 0; i < 3; ++i) CHECK(std::fabs(fast(i, j) - exact(i, j)) <= bound);
    }
    setInt8Inputs(false);
    CHECK(q.multiply(x).approxEquals(q.multiplyFloat(x), 0.0f));
    setInt8Inputs(true);
    const QuantizedWeights fromRows = QuantizedWeights::fromOutIn(w.transpose());
    CHECK(fromRows.dequantize().approxEquals(back, 0.0f));
}

TEST(int8_kernel_paths_agree) {
    // Single rows (token-by-token) and batches (prompts, four weight rows at a
    // time) give the same numbers, for ragged sizes and block layouts.
    setSeed(82);
    for (auto [out, in] : {std::pair<std::size_t, std::size_t>{37, 200}, {9, 70}, {6, 40}, {130, 64}, {5, 1000}}) {
        const QuantizedWeights q = QuantizedWeights::fromOutIn(Matrix::uniform(out, in, -1, 1));
        const Matrix x = Matrix::uniform(5, in, -3, 3);
        const Matrix all = q.multiply(x), ref = q.multiplyFloat(x);
        double scale = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) scale = std::max(scale, static_cast<double>(std::fabs(ref.data()[i])));
        for (std::size_t i = 0; i < 5; ++i) {
            const Matrix one = q.multiply(x.sliceRows(i, i + 1));
            for (std::size_t j = 0; j < out; ++j) {
                CHECK_NEAR(one(0, j), all(i, j), 1e-4 * scale);
                CHECK_NEAR(all(i, j), ref(i, j), 0.02 * scale);
            }
        }
    }
    CHECK(std::string(QuantizedWeights::kernelName()).rfind("int8", 0) == 0);
}

TEST(dense_and_network_quantization) {
    setSeed(81);
    Dense d(64, 32);
    const Matrix x = Matrix::uniform(4, 64, -1, 1);
    const Matrix before = d.forward(x, false);
    CHECK(d.quantize() == 1 && d.quantized() && d.parameters().empty());
    CHECK(d.forward(x, false).approxEquals(before, 0.02f));
    CHECK_THROWS(d.backward(Matrix(4, 32)), std::logic_error);
    CHECK(d.describe().find("int8") != std::string::npos);

    SequenceContextPtr ctx;
    Network net = modernGpt(ctx, 16);
    const Matrix ids{{1, 2, 3, 4, 5, 6}};
    const Matrix fp32 = net.predict(ids);
    CHECK(net.quantize() > 0);
    const Matrix int8 = net.predict(ids);
    float maxDiff = 0.0f;
    for (std::size_t i = 0; i < fp32.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(fp32.data()[i] - int8.data()[i]));
    CHECK(maxDiff < 0.05f);
    SamplingOptions greedy;
    greedy.temperature = 0.0f;
    CHECK(generate(net, {1, 2}, 10, 16, greedy, ctx).size() == 10);  // KV cache works on int8 layers
}

TEST(grouped_query_attention_gradcheck) {
    for (std::size_t kv : {2, 1}) {  // grouped-query and multi-query attention
        setSeed(82);
        auto ctx = std::make_shared<SequenceContext>();
        ctx->length = 3;
        Network net;
        auto& block = net.add<Residual>();
        block.add<RMSNorm>(8);
        block.add<CausalSelfAttention>(8, 4, ctx, true, 10000.0f, kv);
        net.add<Dense>(8, 3);
        MSE loss;
        gradientCheck(net, loss, Matrix::uniform(6, 8, -1, 1), Matrix::uniform(6, 3, -1, 1));
        auto& att = dynamic_cast<CausalSelfAttention&>(block.inner().layer(1));
        CHECK(att.qkvWeight().cols() == 8 + 2 * kv * 2);
    }
    auto ctx = std::make_shared<SequenceContext>();
    CHECK_THROWS(CausalSelfAttention(8, 4, ctx, false, 10000.0f, 3), std::invalid_argument);
}

TEST(grouped_query_kv_cache_matches_full_forward) {
    setSeed(83);
    GPTConfig cfg;
    cfg.vocabSize = 11;
    cfg.contextLength = 16;
    cfg.dim = 16;
    cfg.heads = 4;
    cfg.kvHeads = 2;
    cfg.layers = 2;
    cfg.modern();
    Network net;
    SequenceContextPtr ctx = buildGPT(net, cfg);
    const std::vector<float> ids{3, 1, 4, 1, 5, 9, 2, 6};
    const Matrix full = net.predict(Matrix(1, ids.size(), ids));
    ctx->incremental = true;
    ctx->offset = 0;
    std::vector<float> rows;
    for (std::size_t pos = 0; pos < ids.size();) {
        const std::size_t chunk = pos == 0 ? 3 : 1;
        const Matrix out = net.predict(Matrix(1, chunk, std::vector<float>(ids.begin() + pos, ids.begin() + pos + chunk)));
        rows.insert(rows.end(), out.values().begin(), out.values().end());
        ctx->offset += chunk;
        pos += chunk;
    }
    ctx->incremental = false;
    ctx->offset = 0;
    CHECK(Matrix(ids.size(), full.cols(), rows).approxEquals(full, 1e-4f));
}

TEST(skip_parameter_init_builds_empty_layers) {
    {
        SkipParameterInit skip;
        CHECK(SkipParameterInit::active());
        Dense d(1000, 1000);
        CHECK(d.weights().empty() && d.parameters()[0].grad->empty());
    }
    CHECK(!SkipParameterInit::active());
    Dense normal(4, 4);
    CHECK(!normal.weights().empty());
}

// ---- Pretrained models ----------------------------------------------------------------------

TEST(json_parser) {
    const auto v = json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": "x\"y\\nç🚀"}, "t": true, "n": null})");
    CHECK(v.isObject() && v.at("a").array.size() == 3);
    CHECK_NEAR(v.at("a").array[2].number, -300.0, 1e-9);
    CHECK(v.at("b").at("c").string == "x\"y\\n\xc3\xa7\xf0\x9f\x9a\x80");
    CHECK(v.at("t").boolean && v.at("n").type == json::Value::Type::Null);
    CHECK_NEAR(v.numberOr("missing", 7.0), 7.0, 0);
    CHECK_THROWS(json::parse("{\"a\": }"), std::runtime_error);
    CHECK_THROWS(json::parse("[1, 2"), std::runtime_error);
    CHECK_THROWS(v.at("zzz"), std::runtime_error);
}

TEST(safetensors_roundtrip_and_half_precision) {
    const std::string path = tempPath("t.safetensors");
    const Matrix a{{1, 2, 3}, {4, 5, 6}}, b{{-0.5f}};
    saveSafeTensors(path, {{"a", &a}, {"b", &b}});
    const SafeTensors st(path);
    CHECK(st.has("a") && st.has("b") && !st.has("c"));
    CHECK(st.load("a").approxEquals(a) && st.load("b").approxEquals(b));
    CHECK(st.info("a").shape == std::vector<std::size_t>({2, 3}));
    CHECK_THROWS(st.load("c"), std::runtime_error);

    // Hand-written F16 / BF16 tensors: 1.0, -2.0, 0.5 and a subnormal half.
    const std::string header = R"({"h":{"dtype":"F16","shape":[4],"data_offsets":[0,8]},"g":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}})";
    {
        std::ofstream out(path, std::ios::binary);
        const std::uint64_t n = header.size();
        for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xFF));
        out << header;
        const std::uint16_t halves[] = {0x3C00, 0xC000, 0x3800, 0x0001, 0x3F80, 0x4049};  // ..., bf16 1.0, 3.140625
        out.write(reinterpret_cast<const char*>(halves), sizeof(halves));
    }
    const SafeTensors hs(path);
    const Matrix h = hs.load("h"), g = hs.load("g");
    CHECK(h.rows() == 1 && h.cols() == 4);
    CHECK_NEAR(h(0, 0), 1.0, 0);
    CHECK_NEAR(h(0, 1), -2.0, 0);
    CHECK_NEAR(h(0, 2), 0.5, 0);
    CHECK_NEAR(h(0, 3), 5.9604645e-8, 1e-12);
    CHECK_NEAR(g(0, 0), 1.0, 0);
    CHECK_NEAR(g(0, 1), 3.140625, 0);

    // Every half there is: the bulk conversion must agree with the definition bit for bit.
    {
        const std::string all = R"({"all":{"dtype":"F16","shape":[65536],"data_offsets":[0,131072]}})";
        std::ofstream out(path, std::ios::binary);
        const std::uint64_t n = all.size();
        for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xFF));
        out << all;
        for (std::uint32_t v = 0; v < 65536; ++v) {
            const std::uint16_t h = static_cast<std::uint16_t>(v);
            out.write(reinterpret_cast<const char*>(&h), 2);
        }
    }
    const Matrix every = SafeTensors(path).load("all");
    std::size_t wrong = 0;
    for (std::uint32_t v = 0; v < 65536; ++v) {
        const int e = static_cast<int>((v >> 10) & 0x1F), m = static_cast<int>(v & 0x3FF);
        const double sign = (v & 0x8000) ? -1.0 : 1.0;
        const float got = every(0, v);
        if (e == 0x1F) {
            if (m == 0 ? !(std::isinf(got) && (got < 0) == (sign < 0)) : !std::isnan(got)) ++wrong;
            continue;
        }
        const double want = sign * (e == 0 ? std::ldexp(m, -24) : std::ldexp(1024 + m, e - 25));
        if (static_cast<double>(got) != want || std::signbit(got) != (sign < 0)) ++wrong;
    }
    CHECK(wrong == 0);
    std::remove(path.c_str());
}

TEST(gpt2_pretokenizer_matches_regex_semantics) {
    using V = std::vector<std::string>;
    CHECK(Gpt2Tokenizer::splitWords("Hello world") == V({"Hello", " world"}));
    CHECK(Gpt2Tokenizer::splitWords("don't I'LL") == V({"don", "'t", " I", "'", "LL"}));
    CHECK(Gpt2Tokenizer::splitWords("a  b") == V({"a", " ", " b"}));
    CHECK(Gpt2Tokenizer::splitWords("x\n\ny") == V({"x", "\n", "\n", "y"}));
    CHECK(Gpt2Tokenizer::splitWords("end   ") == V({"end", "   "}));
    CHECK(Gpt2Tokenizer::splitWords("v2.0!!") == V({"v", "2", ".", "0", "!!"}));
    CHECK(Gpt2Tokenizer::splitWords(" g\xc3\xbczel \xf0\x9f\x9a\x80") == V({" g\xc3\xbczel", " \xf0\x9f\x9a\x80"}));
    std::string joined;
    for (const auto& w : Gpt2Tokenizer::splitWords("Mixed\t text\r\n with 123 and \xe2\x80\x9cquotes\xe2\x80\x9d")) joined += w;
    CHECK(joined == "Mixed\t text\r\n with 123 and \xe2\x80\x9cquotes\xe2\x80\x9d");
}

namespace {
// GPT-2's bytes_to_unicode, written independently of the library.
std::string gpt2ByteSymbol(int b) {
    auto printable = [](int x) { return (x >= '!' && x <= '~') || (x >= 0xA1 && x <= 0xAC) || (x >= 0xAE && x <= 0xFF); };
    unsigned cp = static_cast<unsigned>(b);
    if (!printable(b)) {
        unsigned n = 0;
        for (int x = 0; x < b; ++x) n += !printable(x);
        cp = 256 + n;
    }
    std::string s;
    if (cp < 0x80) s += static_cast<char>(cp);
    else {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

std::string tinyGpt2Tokenizer() {
    const std::string dir = tempPath("gpt2tok");
    std::filesystem::create_directories(dir);
    std::ofstream vocab(dir + "/vocab.json");
    vocab << '{';
    for (int b = 0; b < 256; ++b) {
        std::string sym = gpt2ByteSymbol(b);
        if (sym == "\"" || sym == "\\") sym = "\\" + sym;
        vocab << '"' << sym << "\":" << b << ',';
    }
    const std::string G = gpt2ByteSymbol(' ');  // "Ġ"
    vocab << '"' << G << "t\":256,\"he\":257,\"" << G << "the\":258,\"<|endoftext|>\":259}";
    std::ofstream merges(dir + "/merges.txt");
    merges << "#version: 0.2\n" << G << " t\nh e\n" << G << "t he\n";
    return dir;
}
} // namespace

TEST(gpt2_tokenizer_merges_and_bytes) {
    const std::string dir = tinyGpt2Tokenizer();
    const Gpt2Tokenizer tok(dir + "/vocab.json", dir + "/merges.txt");
    CHECK(tok.vocabSize() == 260 && tok.endOfText() == 259);
    CHECK(tok.encode(" the") == std::vector<int>({258}));
    CHECK(tok.encode("the") == std::vector<int>({'t', 257}));
    CHECK(tok.encode(" the the") == std::vector<int>({258, 258}));
    const std::string binary("\x00\x01\x7f\xff sp\xc3\xa7", 9);
    CHECK(tok.decode(tok.encode(binary)) == binary);
    CHECK(tok.decode({258, 259}) == " the<|endoftext|>");
    std::filesystem::remove_all(dir);
}

TEST(gpt2_loader_maps_huggingface_tensors) {
    setSeed(70);
    const std::string dir = tempPath("gpt2tiny");
    std::filesystem::create_directories(dir);
    {
        std::ofstream cfg(dir + "/config.json");
        cfg << R"({"model_type":"gpt2","vocab_size":20,"n_positions":16,"n_embd":8,"n_head":2,"n_layer":1,)"
            << R"("layer_norm_epsilon":1e-5,"activation_function":"gelu_new"})";
    }
    std::vector<std::pair<std::string, Matrix>> owned;
    auto add = [&](const std::string& name, std::size_t r, std::size_t c) { owned.emplace_back(name, Matrix::uniform(r, c, -1, 1)); };
    add("wte.weight", 20, 8);
    add("wpe.weight", 16, 8);
    for (const char* n : {"h.0.ln_1.weight", "h.0.ln_1.bias", "h.0.ln_2.weight", "h.0.ln_2.bias", "h.0.attn.c_proj.bias",
                          "h.0.mlp.c_proj.bias", "ln_f.weight", "ln_f.bias"})
        add(n, 1, 8);
    add("h.0.attn.c_attn.weight", 8, 24);
    add("h.0.attn.c_attn.bias", 1, 24);
    add("h.0.attn.c_proj.weight", 8, 8);
    add("h.0.mlp.c_fc.weight", 8, 32);
    add("h.0.mlp.c_fc.bias", 1, 32);
    add("h.0.mlp.c_proj.weight", 32, 8);
    std::vector<std::pair<std::string, const Matrix*>> refs;
    for (const auto& [n, m] : owned) refs.emplace_back(n, &m);
    saveSafeTensors(dir + "/model.safetensors", refs);

    PretrainedModel model = loadGPT2(dir);
    CHECK(model.config.layers == 1 && model.config.dim == 8 && model.config.tieEmbeddings);
    auto& emb = dynamic_cast<TokenEmbedding&>(model.net.layer(0));
    CHECK(emb.tokenTable().approxEquals(owned[0].second, 0));
    auto& att = dynamic_cast<CausalSelfAttention&>(dynamic_cast<Residual&>(model.net.layer(1)).inner().layer(1));
    CHECK(att.qkvWeight().approxEquals(owned[10].second, 0));
    // Tied output: 20 logits per position; the model still trains after releasing gradients.
    const Matrix x{{1, 5, 19}};
    CHECK(model.net.predict(x).cols() == 20);
    SparseSoftmaxCrossEntropy loss;
    Adam opt(1e-3f);
    CHECK(std::isfinite(model.net.trainBatch(x, Matrix{{5, 19, 2}}, loss, opt)));

    // Wrong shape / missing tensor are reported.
    owned[10].second = Matrix(8, 23);
    refs.clear();
    for (const auto& [n, m] : owned) refs.emplace_back(n, &m);
    saveSafeTensors(dir + "/model.safetensors", refs);
    CHECK_THROWS(loadGPT2(dir), std::runtime_error);
    std::filesystem::remove_all(dir);
}

// ---- Llama ---------------------------------------------------------------------------------

namespace {
const char* kSp = "\xe2\x96\x81";  // "▁"

// A small tokenizer.json in the Llama / TinyLlama format: specials, 256 byte
// tokens, a few merges and the Prepend + Replace normalizer.
std::string tinyLlamaTokenizer(bool withPreTokenizer = false) {
    const std::string dir = tempPath(withPreTokenizer ? "llamatok_pre" : "llamatok");
    std::filesystem::create_directories(dir);
    const std::string S = kSp;
    std::ofstream f(dir + "/tokenizer.json");
    f << R"({"added_tokens":[{"id":0,"content":"<unk>"},{"id":1,"content":"<s>"},{"id":2,"content":"</s>"},)"
      << R"({"id":264,"content":"<|user|>"}],)"
      << R"("normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":")" << S
      << R"("},{"type":"Replace","pattern":{"String":" "},"content":")" << S << R"("}]},)"
      << R"("pre_tokenizer":)" << (withPreTokenizer ? R"({"type":"Metaspace","replacement":"_"})" : "null") << ','
      << R"("model":{"type":"BPE","vocab":{"<unk>":0,"<s>":1,"</s>":2,)";
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        f << '"' << name << "\":" << 3 + b << ',';
    }
    f << '"' << S << "\":259,\"h\":260,\"i\":261,\"" << S << "h\":262,\"" << S << "hi\":263},"
      << R"("merges":[")" << S << R"( h",[")" << S << R"(h","i"]]}})";
    return dir;
}

std::size_t llamaVocab = 12, llamaDim = 16, llamaHeads = 4, llamaKv = 2, llamaHidden = 12, llamaLayers = 2;

std::map<std::string, Matrix> tinyLlamaWeights(bool tied) {
    const std::size_t D = llamaDim, kv = llamaKv * (llamaDim / llamaHeads), H = llamaHidden;
    std::map<std::string, Matrix> w;
    w["model.embed_tokens.weight"] = Matrix::uniform(llamaVocab, D, -1, 1);
    for (std::size_t i = 0; i < llamaLayers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        w[p + "input_layernorm.weight"] = Matrix::uniform(1, D, 0.5f, 1.5f);
        w[p + "post_attention_layernorm.weight"] = Matrix::uniform(1, D, 0.5f, 1.5f);
        w[p + "self_attn.q_proj.weight"] = Matrix::uniform(D, D, -0.5f, 0.5f);
        w[p + "self_attn.k_proj.weight"] = Matrix::uniform(kv, D, -0.5f, 0.5f);
        w[p + "self_attn.v_proj.weight"] = Matrix::uniform(kv, D, -0.5f, 0.5f);
        w[p + "self_attn.o_proj.weight"] = Matrix::uniform(D, D, -0.5f, 0.5f);
        w[p + "mlp.gate_proj.weight"] = Matrix::uniform(H, D, -0.5f, 0.5f);
        w[p + "mlp.up_proj.weight"] = Matrix::uniform(H, D, -0.5f, 0.5f);
        w[p + "mlp.down_proj.weight"] = Matrix::uniform(D, H, -0.5f, 0.5f);
    }
    w["model.norm.weight"] = Matrix::uniform(1, D, 0.5f, 1.5f);
    if (!tied) w["lm_head.weight"] = Matrix::uniform(llamaVocab, D, -1, 1);
    return w;
}

// Llama 3-style scaling small enough to matter here: with head size 4 the two
// rotation pairs fall in the interpolated band and the slowed-down band.
const RopeScaling kTestScaling{8.0f, 1.0f, 4.0f, 16};

void writeLlamaConfig(const std::string& dir, bool tied, bool scaled = false) {
    std::ofstream cfg(dir + "/config.json");
    cfg << R"({"model_type":"llama","vocab_size":)" << llamaVocab << R"(,"hidden_size":)" << llamaDim
        << R"(,"num_attention_heads":)" << llamaHeads << R"(,"num_key_value_heads":)" << llamaKv
        << R"(,"num_hidden_layers":)" << llamaLayers << R"(,"intermediate_size":)" << llamaHidden
        << R"(,"max_position_embeddings":32,"rms_norm_eps":1e-5,"rope_theta":500.0,"hidden_act":"silu",)"
        << R"("attention_bias":false,"eos_token_id":2,"tie_word_embeddings":)" << (tied ? "true" : "false")
        << R"(,"rope_scaling":)"
        << (scaled ? R"({"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":16})"
                   : "null")
        << '}';
}

// Straightforward Llama forward pass (Hugging Face layout, double precision).
std::vector<std::vector<double>> llamaReference(std::map<std::string, Matrix>& w, const std::vector<int>& ids, bool tied,
                                                bool scaled = false) {
    const std::size_t T = ids.size(), D = llamaDim, Hh = llamaHeads, hd = D / Hh, group = Hh / llamaKv;
    const double eps = 1e-5, theta = 500.0;
    using Rows = std::vector<std::vector<double>>;
    auto linear = [](const Rows& x, const Matrix& W) {  // W is out x in
        Rows y(x.size(), std::vector<double>(W.rows(), 0.0));
        for (std::size_t t = 0; t < x.size(); ++t)
            for (std::size_t o = 0; o < W.rows(); ++o)
                for (std::size_t i = 0; i < W.cols(); ++i) y[t][o] += x[t][i] * W(o, i);
        return y;
    };
    auto rms = [&](const Rows& x, const Matrix& g) {
        Rows y = x;
        for (auto& r : y) {
            double ss = 0;
            for (double v : r) ss += v * v;
            const double inv = 1.0 / std::sqrt(ss / static_cast<double>(r.size()) + eps);
            for (std::size_t i = 0; i < r.size(); ++i) r[i] *= inv * g(0, i);
        }
        return y;
    };
    // Hugging Face's _compute_llama3_parameters, written out separately from the library.
    auto frequency = [&](std::size_t j) {
        const double inv = std::pow(theta, -2.0 * static_cast<double>(j) / static_cast<double>(hd));
        if (!scaled) return inv;
        const double factor = kTestScaling.factor, low = kTestScaling.lowFreqFactor, high = kTestScaling.highFreqFactor;
        const double old = static_cast<double>(kTestScaling.originalContext), wavelen = 2 * 3.14159265358979323846 / inv;
        const double llama = wavelen > old / low ? inv / factor : inv;
        const double smooth = (old / wavelen - low) / (high - low);
        const bool medium = !(wavelen < old / high) && !(wavelen > old / low);
        return medium ? (1 - smooth) * llama / factor + smooth * llama : llama;
    };
    auto rope = [&](Rows& x, std::size_t heads) {
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t h = 0; h < heads; ++h)
                for (std::size_t j = 0; j < hd / 2; ++j) {
                    const double a = static_cast<double>(t) * frequency(j);
                    double& x1 = x[t][h * hd + j];
                    double& x2 = x[t][h * hd + j + hd / 2];
                    const double r1 = x1 * std::cos(a) - x2 * std::sin(a), r2 = x1 * std::sin(a) + x2 * std::cos(a);
                    x1 = r1;
                    x2 = r2;
                }
    };
    Rows x(T, std::vector<double>(D));
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t i = 0; i < D; ++i) x[t][i] = w["model.embed_tokens.weight"](static_cast<std::size_t>(ids[t]), i);
    for (std::size_t l = 0; l < llamaLayers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const Rows h = rms(x, w[p + "input_layernorm.weight"]);
        Rows q = linear(h, w[p + "self_attn.q_proj.weight"]), k = linear(h, w[p + "self_attn.k_proj.weight"]);
        const Rows v = linear(h, w[p + "self_attn.v_proj.weight"]);
        rope(q, Hh);
        rope(k, llamaKv);
        Rows merged(T, std::vector<double>(D, 0.0));
        for (std::size_t hh = 0; hh < Hh; ++hh) {
            const std::size_t g = hh / group;
            for (std::size_t t = 0; t < T; ++t) {
                std::vector<double> s(t + 1);
                double mx = -1e300, sum = 0;
                for (std::size_t u = 0; u <= t; ++u) {
                    for (std::size_t j = 0; j < hd; ++j) s[u] += q[t][hh * hd + j] * k[u][g * hd + j];
                    s[u] /= std::sqrt(static_cast<double>(hd));
                    mx = std::max(mx, s[u]);
                }
                for (double& e : s) sum += (e = std::exp(e - mx));
                for (std::size_t u = 0; u <= t; ++u)
                    for (std::size_t j = 0; j < hd; ++j) merged[t][hh * hd + j] += s[u] / sum * v[u][g * hd + j];
            }
        }
        const Rows o = linear(merged, w[p + "self_attn.o_proj.weight"]);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < D; ++i) x[t][i] += o[t][i];
        const Rows h2 = rms(x, w[p + "post_attention_layernorm.weight"]);
        Rows gate = linear(h2, w[p + "mlp.gate_proj.weight"]);
        const Rows up = linear(h2, w[p + "mlp.up_proj.weight"]);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < gate[t].size(); ++i) gate[t][i] = gate[t][i] / (1.0 + std::exp(-gate[t][i])) * up[t][i];
        const Rows down = linear(gate, w[p + "mlp.down_proj.weight"]);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < D; ++i) x[t][i] += down[t][i];
    }
    return linear(rms(x, w["model.norm.weight"]), tied ? w["model.embed_tokens.weight"] : w["lm_head.weight"]);
}

double maxLogitError(PretrainedModel& model, const std::vector<std::vector<double>>& ref, const std::vector<int>& ids) {
    Matrix x(1, ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) x(0, i) = static_cast<float>(ids[i]);
    const Matrix logits = model.net.predict(x);
    double err = 0;
    for (std::size_t t = 0; t < ids.size(); ++t)
        for (std::size_t j = 0; j < llamaVocab; ++j) err = std::max(err, std::fabs(logits(t, j) - ref[t][j]));
    return err;
}
} // namespace

TEST(llama_tokenizer_specials_merges_and_byte_fallback) {
    const std::string dir = tinyLlamaTokenizer();
    const LlamaTokenizer tok(dir + "/tokenizer.json");
    CHECK(tok.vocabSize() == 265 && tok.bos() == 1 && tok.eos() == 2);
    CHECK(tok.encode("hi", true) == std::vector<int>({1, 263}));
    CHECK(tok.encode("hi hi") == std::vector<int>({263, 263}));
    CHECK(tok.encode("ih") == std::vector<int>({259, 261, 260}));  // no merge for "▁ i"
    // "é" is not in the vocabulary: its two UTF-8 bytes become <0xC3> <0xA9>.
    CHECK(tok.encode("h\xc3\xa9") == std::vector<int>({262, 3 + 0xC3, 3 + 0xA9}));
    // Special tokens in the text are matched verbatim; text after them gets its own "▁".
    CHECK(tok.encode("<|user|>\nhi</s>") == std::vector<int>({264, 259, 3 + '\n', 260, 261, 2}));
    CHECK(tok.decode({1, 263, 263, 2}) == "hi hi");
    CHECK(tok.decode(tok.encode("h\xc3\xa9 hi")) == "h\xc3\xa9 hi");
    CHECK(tok.decodePiece(263) == " hi" && tok.decodePiece(2).empty() && tok.isSpecial(264) && !tok.isSpecial(263));
    CHECK_THROWS(tok.decodePiece(999), std::out_of_range);
    const std::string pre = tinyLlamaTokenizer(true);
    CHECK_THROWS(LlamaTokenizer(pre + "/tokenizer.json"), std::runtime_error);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(pre);
}

// A byte-level tokenizer.json in the Llama 3 format.
std::string tinyByteLevelTokenizer(bool ignoreMerges) {
    const std::string dir = tempPath(ignoreMerges ? "bltok" : "bltok_merges");
    std::filesystem::create_directories(dir);
    const std::string pattern =
        R"PAT((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)PAT";
    std::string escaped;
    for (char c : pattern) escaped += c == '\\' ? std::string("\\\\") : std::string(1, c);
    const std::string G = gpt2ByteSymbol(' ');
    std::ofstream f(dir + "/tokenizer.json");
    f << R"({"added_tokens":[{"id":260,"content":"<|begin_of_text|>"},{"id":261,"content":"<|eot_id|>"}],"normalizer":null,)"
      << R"("pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":")" << escaped
      << R"("},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},)"
      << R"("post_processor":{"type":"Sequence","processors":[{"type":"ByteLevel"},{"type":"TemplateProcessing",)"
      << R"("single":[{"SpecialToken":{"id":"<|begin_of_text|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}]}]},)"
      << R"("model":{"type":"BPE","ignore_merges":)" << (ignoreMerges ? "true" : "false") << R"(,"vocab":{)";
    for (int b = 0; b < 256; ++b) {
        std::string sym = gpt2ByteSymbol(b);
        if (sym == "\"" || sym == "\\") sym = "\\" + sym;
        f << '"' << sym << "\":" << b << ',';
    }
    f << R"("he":256,")" << G << R"(he":257,"llo":258,")" << G << R"(w":259},)"
      << R"("merges":[["h","e"],[")" << G << R"(","he"],[")" << G << R"(","w"]]}})";
    return dir;
}

TEST(llama3_pretokenizer_matches_huggingface) {
    using V = std::vector<std::string>;
    // Expected splits produced by Hugging Face tokenizers with the Llama 3.2 pattern.
    CHECK(LlamaTokenizer::splitLlama3("Hello world") == V({"Hello", " world"}));
    CHECK(LlamaTokenizer::splitLlama3("DON'T we'Re") == V({"DON", "'T", " we", "'Re"}));
    CHECK(LlamaTokenizer::splitLlama3("1234567 x") == V({"123", "456", "7", " x"}));
    CHECK(LlamaTokenizer::splitLlama3(" 123") == V({" ", "123"}));
    CHECK(LlamaTokenizer::splitLlama3("a  b") == V({"a", " ", " b"}));
    CHECK(LlamaTokenizer::splitLlama3("x\n\ny") == V({"x", "\n\n", "y"}));
    CHECK(LlamaTokenizer::splitLlama3("  \n  z") == V({"  \n", " ", " z"}));
    CHECK(LlamaTokenizer::splitLlama3("!hello ...") == V({"!hello", " ..."}));
    CHECK(LlamaTokenizer::splitLlama3("end   ") == V({"end", "   "}));
    CHECK(LlamaTokenizer::splitLlama3("v2.0!!\n") == V({"v", "2", ".", "0", "!!\n"}));
    CHECK(LlamaTokenizer::splitLlama3("\tq") == V({"\tq"}));
    CHECK(LlamaTokenizer::splitLlama3(" g\xc3\xbc" "zel \xf0\x9f\x9a\x80") == V({" g\xc3\xbc" "zel", " \xf0\x9f\x9a\x80"}));
    CHECK(LlamaTokenizer::splitLlama3("e\xcc\x81") == V({"e", "\xcc\x81"}));  // a combining mark is not a letter
    CHECK(LlamaTokenizer::splitLlama3("12345", 1) == V({"1", "2", "3", "4", "5"}));  // Qwen 2 variant
    std::string joined;
    for (const auto& w : LlamaTokenizer::splitLlama3("Mixed\t text\r\n with 123 and \xe2\x80\x9cquotes\xe2\x80\x9d  \n")) joined += w;
    CHECK(joined == "Mixed\t text\r\n with 123 and \xe2\x80\x9cquotes\xe2\x80\x9d  \n");
}

TEST(byte_level_llama_tokenizer) {
    const std::string dir = tinyByteLevelTokenizer(true);
    const LlamaTokenizer tok(dir + "/tokenizer.json");
    CHECK(tok.byteLevel() && tok.bos() == 260 && tok.eos() == 261 && tok.vocabSize() == 262);
    CHECK(tok.encode(" he", true) == std::vector<int>({260, 257}));
    CHECK(tok.encode("he w") == std::vector<int>({256, 259}));
    CHECK(tok.encode("llo") == std::vector<int>({258}));  // whole word in the vocabulary: merges skipped
    CHECK(tok.encode("he<|eot_id|>") == std::vector<int>({256, 261}));
    CHECK(tok.encode("123456") == std::vector<int>({'1', '2', '3', '4', '5', '6'}));
    const std::string text = "caf\xc3\xa9 \xf0\x9f\x9a\x80 x\n\ty";
    CHECK(tok.decode(tok.encode(text)) == text);
    CHECK(tok.decode({260, 257, 261}) == " he");  // specials are skipped, no space stripping
    CHECK(tok.decodePiece(257) == " he" && tok.isSpecial(261));
    const std::string plain = tinyByteLevelTokenizer(false);
    CHECK(LlamaTokenizer(plain + "/tokenizer.json").encode("llo") == std::vector<int>({'l', 'l', 'o'}));
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(plain);
}

TEST(rope_scaling_matches_llama3_formula) {
    const auto plain = ropeFrequencies(64, 500000.0);
    CHECK(plain.size() == 32);
    for (std::size_t i = 0; i < plain.size(); ++i)
        CHECK_NEAR(plain[i], std::pow(500000.0, -2.0 * static_cast<double>(i) / 64.0), 1e-15);
    // Llama 3.2's configuration; expected values from Hugging Face's formula.
    const RopeScaling llama32{32.0f, 1.0f, 4.0f, 8192};
    const auto f = ropeFrequencies(64, 500000.0, llama32);
    const std::pair<std::size_t, double> expected[] = {{0, 1.0},
                                                       {14, 0.003211445994752591},     // fast: unchanged
                                                       {15, 0.001290547928209264},     // interpolated
                                                       {16, 0.00042955679655936815},
                                                       {17, 9.70828780262767e-05},
                                                       {18, 1.9461638184831125e-05},   // slow: divided by 32
                                                       {31, 9.41830672543491e-08}};
    for (const auto& [i, v] : expected) CHECK_NEAR(f[i] / v, 1.0, 1e-9);
    auto ctx = std::make_shared<SequenceContext>();
    CausalSelfAttention att(8, 2, ctx, true);
    CHECK_THROWS(att.setRopeScaling(RopeScaling{8.0f, 4.0f, 1.0f, 16}), std::invalid_argument);
    // The scaling survives a config round trip.
    GPTConfig c;
    c.modern();
    c.vocabSize = 10;
    c.ropeScaling = llama32;
    const std::string path = tempPath("scaled.config");
    saveGPTConfig(path, c);
    const GPTConfig back = loadGPTConfig(path);
    CHECK(back.ropeScaling.factor == 32.0f && back.ropeScaling.highFreqFactor == 4.0f && back.ropeScaling.originalContext == 8192);
    std::filesystem::remove(path);
}

TEST(embedding_quantization_serves_lookup_and_tied_output) {
    setSeed(73);
    GPTConfig c;
    c.modern();
    c.vocabSize = 50;
    c.contextLength = 8;
    c.dim = 64;
    c.heads = 2;
    c.layers = 1;
    Network net;
    buildGPT(net, c);
    const Matrix x{{3, 14, 15, 9, 2}};
    const Matrix before = net.predict(x);
    auto& emb = dynamic_cast<TokenEmbedding&>(net.layer(0));
    const Matrix table = emb.tokenTable();
    CHECK(net.quantize() >= 4);  // embedding (shared with the output), attention, two MLP matrices
    CHECK(emb.quantized() && emb.tokenTable().empty() && net.parameterCount() < table.size());
    std::vector<float> row(64);
    emb.quantizedTable().expandRow(14, row.data());
    for (std::size_t j = 0; j < 64; ++j) CHECK_NEAR(row[j], table(14, j), 0.01);
    const Matrix after = net.predict(x);
    CHECK(after.approxEquals(before, 0.05f));
    net.forward(x, true);
    CHECK_THROWS(net.backward(Matrix(5, 50, 1.0f)), std::logic_error);
}

TEST(llama_loader_matches_reference_forward) {
    setSeed(71);
    const std::vector<int> ids{1, 5, 11, 0, 7, 7, 3};
    for (int variant = 0; variant < 3; ++variant) {
        const bool tied = variant >= 1, scaled = variant == 2;  // 2: Llama 3 style (tied, RoPE scaling)
        const std::string dir = tempPath("llama_tiny" + std::to_string(variant));
        std::filesystem::create_directories(dir);
        writeLlamaConfig(dir, tied, scaled);
        if (scaled) std::ofstream(dir + "/generation_config.json") << R"({"eos_token_id":[5,7]})";
        auto w = tinyLlamaWeights(tied);
        std::vector<std::pair<std::string, const Matrix*>> refs;
        for (const auto& [n, m] : w) refs.emplace_back(n, &m);
        saveSafeTensors(dir + "/model.safetensors", refs);
        const auto ref = llamaReference(w, ids, tied, scaled);
        double scale = 0;
        for (const auto& r : ref)
            for (double v : r) scale = std::max(scale, std::fabs(v));

        PretrainedModel fp = loadLlama(dir, false);
        CHECK(fp.config.kvHeads == llamaKv && fp.config.layers == llamaLayers && fp.config.tieEmbeddings == tied);
        CHECK(fp.config.ropeScaling.enabled() == scaled);
        CHECK(fp.stopTokens == (scaled ? std::vector<int>({5, 7}) : std::vector<int>({2})));
        CHECK(maxLogitError(fp, ref, ids) < 1e-4 * std::max(1.0, scale));
        PretrainedModel q = loadLlama(dir, true);
        // Random untrained weights amplify rounding more than a real model (TinyLlama: +0.1% perplexity).
        CHECK(maxLogitError(q, ref, ids) < 0.1 * scale);

        // The KV cache gives the same last-position logits as the full pass.
        SamplingOptions greedy;
        greedy.temperature = 0.0f;
        const auto withCache = generate(fp.net, ids, 4, fp.config.contextLength, greedy, fp.context);
        const auto without = generate(fp.net, ids, 4, fp.config.contextLength, greedy);
        CHECK(withCache == without);
        // Any listed stop token ends generation.
        greedy.stopTokens = {withCache[0]};
        CHECK(generate(fp.net, ids, 4, fp.config.contextLength, greedy, fp.context).size() == 1);

        // Sharded checkpoint: the same tensors split over two files plus an index.
        if (!tied) {
            std::filesystem::remove(dir + "/model.safetensors");
            std::vector<std::pair<std::string, const Matrix*>> a, b;
            std::ofstream index(dir + "/model.safetensors.index.json");
            index << R"({"metadata":{},"weight_map":{)";
            bool first = true;
            for (const auto& [n, m] : w) {
                const bool second = n.find("layers.1.") != std::string::npos || n == "lm_head.weight";
                (second ? b : a).emplace_back(n, &m);
                index << (first ? "" : ",") << '"' << n << "\":\"" << (second ? "part-2.safetensors" : "part-1.safetensors") << '"';
                first = false;
            }
            index << "}}";
            index.close();
            saveSafeTensors(dir + "/part-1.safetensors", a);
            saveSafeTensors(dir + "/part-2.safetensors", b);
            PretrainedModel sharded = loadLlama(dir, false);
            CHECK(maxLogitError(sharded, ref, ids) < 1e-4 * std::max(1.0, scale));
        }

        // Mismatched shapes and unsupported configurations are reported.
        w["model.layers.0.self_attn.k_proj.weight"] = Matrix(llamaDim, llamaDim);
        refs.clear();
        for (const auto& [n, m] : w) refs.emplace_back(n, &m);
        std::filesystem::remove(dir + "/model.safetensors.index.json");
        saveSafeTensors(dir + "/model.safetensors", refs);
        CHECK_THROWS(loadLlama(dir, false), std::runtime_error);
        {
            std::ofstream cfg(dir + "/config.json");
            cfg << R"({"model_type":"llama","vocab_size":12,"hidden_size":8,"num_attention_heads":4,)"
                << R"("num_hidden_layers":1,"intermediate_size":12,"rope_scaling":{"type":"linear","factor":2.0}})";
        }
        CHECK_THROWS(loadLlama(dir, false), std::runtime_error);
        std::filesystem::remove_all(dir);
    }
}

TEST(inference_forward_keeps_no_backward_state) {
    setSeed(72);
    Network net;
    net.add<Dense>(4, 6);
    net.add<GELU>();
    net.add<LayerNorm>(6);
    net.add<Dense>(6, 3);
    const Matrix x = Matrix::uniform(5, 4, -1, 1);
    const Matrix inference = net.forward(x, false);
    CHECK_THROWS(net.backward(Matrix(5, 3, 1.0f)), std::logic_error);  // nothing cached for backward
    CHECK(net.forward(x, true).approxEquals(inference, 1e-6f));        // same result in training mode
    CHECK(net.backward(Matrix(5, 3, 1.0f)).rows() == 5);
    // A later inference pass drops what the training pass cached.
    net.forward(x, false);
    CHECK_THROWS(net.backward(Matrix(5, 3, 1.0f)), std::logic_error);

    auto ctx = std::make_shared<SequenceContext>();
    ctx->length = 3;
    CausalSelfAttention att(8, 2, ctx, true, 10000.0f, 1);
    const Matrix seq = Matrix::uniform(6, 8, -1, 1);  // 2 sequences x 3 positions
    const Matrix y = att.forward(seq, false);
    CHECK_THROWS(att.backward(Matrix(6, 8, 1.0f)), std::logic_error);
    CHECK(att.forward(seq, true).approxEquals(y, 1e-5f));
}

// ---- Image models ----------------------------------------------------------------------------

TEST(conv2d_inference_path_matches_training_path) {
    // Inference works in row strips straight into channel-major output (1x1: one GEMM).
    setSeed(90);
    struct Case { ImageShape in; std::size_t out, k, stride, pad; };
    for (const Case& c : {Case{{3, 9, 7}, 5, 3, 1, 1}, Case{{4, 8, 8}, 6, 3, 2, 0}, Case{{6, 5, 4}, 3, 1, 1, 0},
                          Case{{2, 11, 13}, 4, 5, 2, 2}, Case{{3, 4, 4}, 2, 1, 2, 0}}) {
        Conv2D conv(c.in, c.out, c.k, c.stride, c.pad);
        conv.bias() = Matrix::uniform(1, c.out, -1, 1);
        const Matrix x = Matrix::uniform(2, c.in.size(), -1, 1);
        CHECK(conv.forward(x, false).approxEquals(conv.forward(x, true), 1e-4f));
    }
}

TEST(image_layers_known_values_and_gradcheck) {
    setSeed(91);
    // GroupNorm: every group has mean 0 and variance 1 before gamma / beta.
    GroupNorm gn({4, 3, 2}, 2, 1e-6f);
    const Matrix x = Matrix::uniform(2, 24, -2, 3);
    const Matrix y = gn.forward(x, false);
    for (std::size_t b = 0; b < 2; ++b)
        for (std::size_t g = 0; g < 2; ++g) {
            double s = 0, ss = 0;
            for (std::size_t i = 0; i < 12; ++i) {
                s += y(b, g * 12 + i);
                ss += y(b, g * 12 + i) * y(b, g * 12 + i);
            }
            CHECK_NEAR(s / 12, 0.0, 1e-5);
            CHECK_NEAR(ss / 12, 1.0, 1e-4);
        }
    // Upsample (nearest) and Pad2D (asymmetric) values.
    Upsample2D up({1, 2, 2}, 2);
    CHECK(up.forward(Matrix{{1, 2, 3, 4}}, false).approxEquals(
        Matrix{{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}}, 0.0f));
    Pad2D pad({1, 2, 2}, 0, 1, 0, 1);
    CHECK(pad.forward(Matrix{{1, 2, 3, 4}}, false).approxEquals(Matrix{{1, 2, 0, 3, 4, 0, 0, 0, 0}}, 0.0f));
    CHECK(std::fabs(SiLU().forward(Matrix{{1.0f}}, false)(0, 0) - 1.0f / (1.0f + std::exp(-1.0f))) < 1e-6f);

    // Gradients through GroupNorm, SiLU, Pad2D, Upsample2D and residual blocks (both shortcut kinds).
    Network net;
    auto& g = net.add<GroupNorm>(ImageShape{4, 3, 3}, 2, 1e-5f);
    g.gamma() = Matrix::uniform(1, 4, 0.5f, 1.5f);
    g.beta() = Matrix::uniform(1, 4, -0.5f, 0.5f);
    net.add<SiLU>();
    net.add<Pad2D>(ImageShape{4, 3, 3}, 1, 0, 0, 1);
    net.add<Upsample2D>(ImageShape{4, 4, 4}, 2);
    net.add<ResnetBlock2D>(ImageShape{4, 8, 8}, 4, 2, 1e-5f);
    net.add<ResnetBlock2D>(ImageShape{4, 8, 8}, 2, 2, 1e-5f);
    MSE loss;
    gradientCheck(net, loss, Matrix::uniform(2, 36, -1, 1), Matrix::uniform(2, 2 * 64, -1, 1));
}

TEST(spatial_attention_matches_naive_and_chunks) {
    // 64 x 64 pixels = 4096 queries, processed in four chunks; two heads.
    setSeed(92);
    const ImageShape s{4, 64, 64};
    const std::size_t C = 4, HW = 4096, heads = 2, dh = 2;
    SpatialSelfAttention att(s, heads, 2, 1e-6f);
    att.queryBias() = Matrix::uniform(1, C, -0.3f, 0.3f);
    att.outBias() = Matrix::uniform(1, C, -0.3f, 0.3f);
    const Matrix x = Matrix::uniform(1, s.size(), -1, 1);
    const Matrix got = att.forward(x, false);
    const Matrix n = att.norm().forward(x, false);
    auto proj = [&](Matrix& w, Matrix& b, std::size_t c, std::size_t p, const std::vector<double>& v) {
        double acc = b(0, c);
        for (std::size_t k = 0; k < C; ++k) acc += w(c, k) * v[k * HW + p];
        return acc;
    };
    std::vector<double> nv(n.data(), n.data() + n.size()), q(C * HW), k(C * HW), v(C * HW), o(C * HW, 0.0);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t p = 0; p < HW; ++p) {
            q[c * HW + p] = proj(att.query(), att.queryBias(), c, p, nv);
            k[c * HW + p] = proj(att.key(), att.keyBias(), c, p, nv);
            v[c * HW + p] = proj(att.value(), att.valueBias(), c, p, nv);
        }
    std::vector<double> sc(HW);
    for (std::size_t h = 0; h < heads; ++h)
        for (std::size_t i = 0; i < HW; i += 97) {  // a spread of query pixels
            double m = -1e300, sum = 0;
            for (std::size_t j = 0; j < HW; ++j) {
                double d = 0;
                for (std::size_t e = 0; e < dh; ++e) d += q[(h * dh + e) * HW + i] * k[(h * dh + e) * HW + j];
                m = std::max(m, sc[j] = d / std::sqrt(double(dh)));
            }
            for (std::size_t j = 0; j < HW; ++j) sum += sc[j] = std::exp(sc[j] - m);
            for (std::size_t e = 0; e < dh; ++e) {
                double acc = 0;
                for (std::size_t j = 0; j < HW; ++j) acc += sc[j] / sum * v[(h * dh + e) * HW + j];
                o[(h * dh + e) * HW + i] = acc;
            }
        }
    for (std::size_t i = 0; i < HW; i += 97)
        for (std::size_t c = 0; c < C; ++c) CHECK_NEAR(got(0, c * HW + i), x(0, c * HW + i) + proj(att.out(), att.outBias(), c, i, o), 1e-4);
    CHECK_THROWS(att.backward(got), std::logic_error);
}

TEST(png_and_ppm_files) {
    Image img;
    img.width = 300;  // a scanline of 900 bytes
    img.height = 240;
    img.pixels.resize(300 * 240 * 3);
    for (std::size_t i = 0; i < img.pixels.size(); ++i) img.pixels[i] = static_cast<std::uint8_t>(i * 7 % 251);
    const std::string png = tempPath("t.png"), ppm = tempPath("t.ppm");
    savePng(png, img);
    // Parse it back: chunks with valid CRCs, IHDR fields, a zlib stream that decodes to the pixels.
    std::ifstream f(png, std::ios::binary);
    std::vector<std::uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(d.size() > 8 && d[1] == 'P' && d[2] == 'N' && d[3] == 'G');
    auto be = [&](std::size_t at) { return (std::uint32_t(d[at]) << 24) | (d[at + 1] << 16) | (d[at + 2] << 8) | d[at + 3]; };
    std::vector<std::uint8_t> zlib;
    for (std::size_t at = 8; at < d.size();) {
        const std::uint32_t len = be(at);
        const std::string type(d.begin() + static_cast<long>(at + 4), d.begin() + static_cast<long>(at + 8));
        std::uint32_t crc = 0xFFFFFFFFu;
        for (std::size_t i = at + 4; i < at + 8 + len; ++i) {
            crc ^= d[i];
            for (int b = 0; b < 8; ++b) crc = crc & 1 ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
        }
        CHECK((~crc) == be(at + 8 + len));
        if (type == "IHDR") CHECK(be(at + 8) == 300 && be(at + 12) == 240 && d[at + 16] == 8 && d[at + 17] == 2);
        if (type == "IDAT") zlib.insert(zlib.end(), d.begin() + static_cast<long>(at + 8), d.begin() + static_cast<long>(at + 8 + len));
        at += 12 + len;
    }
    CHECK(zlib.size() > 2 && zlib[0] == 0x78 && (zlib[0] * 256 + zlib[1]) % 31 == 0);  // zlib header
    CHECK(loadPng(png).pixels == img.pixels);
    savePpm(ppm, img);
    const Image back = loadPpm(ppm);
    CHECK(back.width == 300 && back.height == 240 && back.pixels == img.pixels);
    // Tensor conversion round trip in [-1, 1].
    const Matrix t = imageToTensor(img);
    CHECK(t.cols() == 3 * 300 * 240 && t(0, 0) == -1.0f + 2.0f * img.pixels[0] / 255.0f);
    CHECK(tensorToImage(t.data(), {3, 240, 300}).pixels == img.pixels);
    std::filesystem::remove(png);
    std::filesystem::remove(ppm);
}

namespace {
// Tensor names and shapes of a diffusers AutoencoderKL with the given layout.
std::map<std::string, std::vector<std::size_t>> vaeTensorShapes(const std::vector<std::size_t>& ch, std::size_t layers,
                                                                 std::size_t latent) {
    std::map<std::string, std::vector<std::size_t>> t;
    auto conv = [&](const std::string& n, std::size_t o, std::size_t i, std::size_t k) {
        t[n + ".weight"] = {o, i, k, k};
        t[n + ".bias"] = {o};
    };
    auto norm = [&](const std::string& n, std::size_t c) {
        t[n + ".weight"] = {c};
        t[n + ".bias"] = {c};
    };
    auto resnet = [&](const std::string& p, std::size_t in, std::size_t out) {
        norm(p + ".norm1", in);
        conv(p + ".conv1", out, in, 3);
        norm(p + ".norm2", out);
        conv(p + ".conv2", out, out, 3);
        if (in != out) conv(p + ".conv_shortcut", out, in, 1);
    };
    auto attention = [&](const std::string& p, std::size_t c) {
        norm(p + ".group_norm", c);
        for (const char* n : {".to_q", ".to_k", ".to_v", ".to_out.0"}) {
            t[p + n + ".weight"] = {c, c};
            t[p + n + ".bias"] = {c};
        }
    };
    const std::size_t top = ch.back();
    conv("encoder.conv_in", ch[0], 3, 3);
    std::size_t c = ch[0];
    for (std::size_t i = 0; i < ch.size(); ++i) {
        for (std::size_t j = 0; j < layers; ++j, c = ch[i])
            resnet("encoder.down_blocks." + std::to_string(i) + ".resnets." + std::to_string(j), c, ch[i]);
        if (i + 1 < ch.size()) conv("encoder.down_blocks." + std::to_string(i) + ".downsamplers.0.conv", c, c, 3);
    }
    resnet("encoder.mid_block.resnets.0", top, top);
    attention("encoder.mid_block.attentions.0", top);
    resnet("encoder.mid_block.resnets.1", top, top);
    norm("encoder.conv_norm_out", top);
    conv("encoder.conv_out", 2 * latent, top, 3);
    conv("quant_conv", 2 * latent, 2 * latent, 1);
    conv("post_quant_conv", latent, latent, 1);
    conv("decoder.conv_in", top, latent, 3);
    resnet("decoder.mid_block.resnets.0", top, top);
    attention("decoder.mid_block.attentions.0", top);
    resnet("decoder.mid_block.resnets.1", top, top);
    c = top;
    for (std::size_t i = 0; i < ch.size(); ++i) {
        const std::size_t out = ch[ch.size() - 1 - i];
        for (std::size_t j = 0; j <= layers; ++j, c = out)
            resnet("decoder.up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j), c, out);
        if (i + 1 < ch.size()) conv("decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.conv", out, out, 3);
    }
    norm("decoder.conv_norm_out", ch[0]);
    conv("decoder.conv_out", 3, ch[0], 3);
    return t;
}
} // namespace

TEST(vae_loader_builds_and_runs_a_diffusers_checkpoint) {
    setSeed(93);
    const std::string dir = tempPath("vae_tiny");
    std::filesystem::create_directories(dir + "/vae");
    std::ofstream(dir + "/vae/config.json") << R"({"_class_name":"AutoencoderKL","act_fn":"silu","block_out_channels":[8,16],)"
                                            << R"("in_channels":3,"out_channels":3,"latent_channels":2,"layers_per_block":1,)"
                                            << R"("norm_num_groups":4,"scaling_factor":0.5})";
    std::map<std::string, Matrix> owned;
    for (const auto& [name, shape] : vaeTensorShapes({8, 16}, 1, 2)) {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        const bool scale = name.find("norm") != std::string::npos && name.size() > 7 && name.substr(name.size() - 7) == ".weight";
        owned[name] = scale ? Matrix::uniform(1, n, 0.8f, 1.2f) : Matrix::uniform(1, n, -0.2f, 0.2f);
    }
    std::vector<std::pair<std::string, const Matrix*>> refs;
    for (const auto& [n, m] : owned) refs.emplace_back(n, &m);
    saveSafeTensors(dir + "/vae/diffusion_pytorch_model.safetensors", refs);

    Vae vae = loadVae(dir, 16, 8);
    CHECK(vae.latent.channels == 2 && vae.latent.height == 4 && vae.latent.width == 8 && vae.scalingFactor == 0.5f);
    const Matrix img = Matrix::uniform(1, 3 * 16 * 8, -1, 1);
    const Matrix z = vae.encode(img);
    CHECK(z.cols() == 2 * 4 * 8);
    const Matrix moments = vae.encoder.predict(img);
    CHECK_NEAR(z(0, 5), moments(0, 5) * 0.5, 1e-6);  // the scaled mean: first half of the moments
    const Matrix back = vae.decode(z);
    CHECK(back.cols() == 3 * 16 * 8);
    CHECK(vae.decode(z).approxEquals(back, 0.0f));  // deterministic
    // The first decoder layer (post_quant_conv) got the checkpoint's weights, transposed to (in*k*k) x out.
    auto& pq = dynamic_cast<Conv2D&>(vae.decoder.layer(0));
    CHECK(pq.weights()(1, 0) == owned["post_quant_conv.weight"](0, 1));

    CHECK_THROWS(loadVae(dir, 15, 8), std::invalid_argument);  // not a multiple of 2^(blocks - 1) = 2
    owned["decoder.conv_in.weight"] = Matrix(1, 5);
    refs.clear();
    for (const auto& [n, m] : owned) refs.emplace_back(n, &m);
    saveSafeTensors(dir + "/vae/diffusion_pytorch_model.safetensors", refs);
    CHECK_THROWS(loadVae(dir, 16, 8), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST(gelu_variants_values_and_gradcheck) {
    CHECK_NEAR(GELU(GELU::Approximation::Erf).forward(Matrix{{1.0f}}, false)(0, 0), 0.8413447, 1e-6);
    CHECK_NEAR(GELU(GELU::Approximation::Sigmoid).forward(Matrix{{1.0f}}, false)(0, 0), 1.0 / (1.0 + std::exp(-1.702)), 1e-6);
    CHECK_NEAR(GELU().forward(Matrix{{1.0f}}, false)(0, 0), 0.841192, 1e-5);  // tanh approximation
    for (auto a : {GELU::Approximation::Erf, GELU::Approximation::Sigmoid}) {
        setSeed(94);
        Network net;
        net.add<Dense>(3, 5);
        net.add<GELU>(a);
        MSE loss;
        gradientCheck(net, loss, Matrix::uniform(4, 3, -2, 2), Matrix::uniform(4, 5, -1, 1));
    }
}

namespace {
std::string tinyClipTokenizer() {
    const std::string dir = tempPath("cliptok");
    std::filesystem::create_directories(dir);
    std::ofstream vocab(dir + "/vocab.json");
    vocab << '{';
    int id = 0;
    for (int pass = 0; pass < 2; ++pass)
        for (int b = 0; b < 256; ++b) {
            std::string sym = gpt2ByteSymbol(b);
            if (sym == "\"" || sym == "\\") sym = "\\" + sym;
            vocab << '"' << sym << (pass ? "</w>" : "") << "\":" << id++ << ',';
        }
    vocab << R"("he":512,"he</w>":513,"hel":514,"lo</w>":515,"<|startoftext|>":516,"<|endoftext|>":517})";
    std::ofstream(dir + "/merges.txt") << "#version: 0.2\nh e</w>\nh e\nhe l\nl o</w>\n";
    std::ofstream(dir + "/tokenizer_config.json")
        << R"({"added_tokens_decoder":{"0":{"content":"!","special":true},"516":{"content":"<|startoftext|>"},)"
        << R"("517":{"content":"<|endoftext|>"}}})";
    return dir;
}
} // namespace

TEST(clip_tokenizer_cleanup_split_and_padding) {
    using V = std::vector<std::string>;
    CHECK(ClipTokenizer::clean("  Hello\tWORLD\n\n Çağ  ") == "hello world çağ");
    CHECK(ClipTokenizer::clean("東京tower") == "東 京 tower");       // spaces around CJK ideographs
    CHECK(ClipTokenizer::clean("\xc4\xb0stanbul") == "i\xcc\x87stanbul");  // İ -> i + combining dot, as Python
    CHECK(ClipTokenizer::splitWords("it's a 4k photo,by <|endoftext|>x") ==
          V({"it", "'s", "a", "4", "k", "photo", ",", "by", "<|endoftext|>", "x"}));
    CHECK(ClipTokenizer::splitWords("12 ...!! don't") == V({"1", "2", "...!!", "don", "'t"}));

    const std::string dir = tinyClipTokenizer();
    const ClipTokenizer tok = ClipTokenizer::fromDirectory(dir);
    CHECK(tok.startOfText() == 516 && tok.endOfText() == 517);
    CHECK(tok.encode("he") == std::vector<int>({513}));                   // "h e</w>" merge
    CHECK(tok.encode("HELLO") == std::vector<int>({514, 515}));           // lowercased: "hel" + "lo</w>"
    CHECK(tok.encode("ab") == std::vector<int>({'a', 256 + 'b'}));        // bytes; the last carries </w>
    CHECK(tok.encode("he!he") == std::vector<int>({513, 0, 513}));        // the added "!" splits the raw text
    const auto ids = tok.encodeForModel("hello he", 8, 0);
    CHECK(ids == std::vector<int>({516, 514, 515, 513, 517, 0, 0, 0}));
    CHECK(tok.encodeForModel("he he he he he he", 5, 0) == std::vector<int>({516, 513, 513, 513, 517}));  // truncated, end kept
    CHECK(tok.decode({516, 514, 515, 513, 517}) == "hello he");  // end-of-word markers become spaces
    std::filesystem::remove_all(dir);
}

TEST(clip_text_encoder_matches_reference_forward) {
    setSeed(95);
    const std::size_t V = 20, D = 8, H = 2, F = 16, L = 2, T = 6, hd = D / H;
    const std::string dir = tempPath("clip_tiny");
    std::filesystem::create_directories(dir + "/text_encoder");
    std::ofstream(dir + "/text_encoder/config.json")
        << R"({"model_type":"clip_text_model","hidden_act":"gelu","vocab_size":20,"hidden_size":8,"num_attention_heads":2,)"
        << R"("intermediate_size":16,"num_hidden_layers":2,"max_position_embeddings":6,"layer_norm_eps":1e-5})";
    std::map<std::string, Matrix> w;
    const std::string p = "text_model.";
    w[p + "embeddings.token_embedding.weight"] = Matrix::uniform(V, D, -1, 1);
    w[p + "embeddings.position_embedding.weight"] = Matrix::uniform(T, D, -1, 1);
    for (std::size_t i = 0; i < L; ++i) {
        const std::string l = p + "encoder.layers." + std::to_string(i) + ".";
        for (const char* n : {"layer_norm1", "layer_norm2"}) {
            w[l + n + ".weight"] = Matrix::uniform(1, D, 0.5f, 1.5f);
            w[l + n + ".bias"] = Matrix::uniform(1, D, -0.3f, 0.3f);
        }
        for (const char* n : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            w[l + "self_attn." + n + ".weight"] = Matrix::uniform(D, D, -0.5f, 0.5f);
            w[l + "self_attn." + n + ".bias"] = Matrix::uniform(1, D, -0.3f, 0.3f);
        }
        w[l + "mlp.fc1.weight"] = Matrix::uniform(F, D, -0.5f, 0.5f);
        w[l + "mlp.fc1.bias"] = Matrix::uniform(1, F, -0.3f, 0.3f);
        w[l + "mlp.fc2.weight"] = Matrix::uniform(D, F, -0.5f, 0.5f);
        w[l + "mlp.fc2.bias"] = Matrix::uniform(1, D, -0.3f, 0.3f);
    }
    w[p + "final_layer_norm.weight"] = Matrix::uniform(1, D, 0.5f, 1.5f);
    w[p + "final_layer_norm.bias"] = Matrix::uniform(1, D, -0.3f, 0.3f);
    std::vector<std::pair<std::string, const Matrix*>> refs;
    for (const auto& [n, m] : w) refs.emplace_back(n, &m);
    saveSafeTensors(dir + "/text_encoder/model.safetensors", refs);

    // Reference: Hugging Face's CLIPTextModel written out naively.
    const std::vector<int> ids{3, 17, 5, 0, 0, 9};
    using Rows = std::vector<std::vector<double>>;
    auto ln = [&](const Rows& x, const std::string& n) {
        Rows y = x;
        for (auto& r : y) {
            double m = 0, v = 0;
            for (double e : r) m += e;
            m /= D;
            for (double e : r) v += (e - m) * (e - m);
            v /= D;
            for (std::size_t j = 0; j < D; ++j) r[j] = (r[j] - m) / std::sqrt(v + 1e-5) * w[n + ".weight"](0, j) + w[n + ".bias"](0, j);
        }
        return y;
    };
    auto lin = [&](const Rows& x, const std::string& n) {
        const Matrix& W = w[n + ".weight"];
        Rows y(x.size(), std::vector<double>(W.rows()));
        for (std::size_t t = 0; t < x.size(); ++t)
            for (std::size_t o = 0; o < W.rows(); ++o) {
                double acc = w[n + ".bias"](0, o);
                for (std::size_t k = 0; k < W.cols(); ++k) acc += W(o, k) * x[t][k];
                y[t][o] = acc;
            }
        return y;
    };
    Rows x(T, std::vector<double>(D));
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t j = 0; j < D; ++j)
            x[t][j] = w[p + "embeddings.token_embedding.weight"](static_cast<std::size_t>(ids[t]), j) +
                      w[p + "embeddings.position_embedding.weight"](t, j);
    for (std::size_t i = 0; i < L; ++i) {
        const std::string l = p + "encoder.layers." + std::to_string(i) + ".";
        const Rows h = ln(x, l + "layer_norm1");
        const Rows q = lin(h, l + "self_attn.q_proj"), k = lin(h, l + "self_attn.k_proj"), v = lin(h, l + "self_attn.v_proj");
        Rows merged(T, std::vector<double>(D, 0.0));
        for (std::size_t hh = 0; hh < H; ++hh)
            for (std::size_t a = 0; a < T; ++a) {
                std::vector<double> s(a + 1);
                double m = -1e300, sum = 0;
                for (std::size_t b = 0; b <= a; ++b) {
                    for (std::size_t e = 0; e < hd; ++e) s[b] += q[a][hh * hd + e] * k[b][hh * hd + e];
                    m = std::max(m, s[b] /= std::sqrt(double(hd)));
                }
                for (double& e : s) sum += e = std::exp(e - m);
                for (std::size_t b = 0; b <= a; ++b)
                    for (std::size_t e = 0; e < hd; ++e) merged[a][hh * hd + e] += s[b] / sum * v[b][hh * hd + e];
            }
        const Rows o = lin(merged, l + "self_attn.out_proj");
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t j = 0; j < D; ++j) x[t][j] += o[t][j];
        Rows f = lin(ln(x, l + "layer_norm2"), l + "mlp.fc1");
        for (auto& r : f)
            for (double& e : r) e = 0.5 * e * (1 + std::erf(e / std::sqrt(2.0)));
        const Rows f2 = lin(f, l + "mlp.fc2");
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t j = 0; j < D; ++j) x[t][j] += f2[t][j];
    }
    const Rows ref = ln(x, p + "final_layer_norm");

    ClipTextEncoder enc = loadClipTextEncoder(dir);
    CHECK(enc.length == T && enc.dim == D && enc.layers == L);
    const Matrix got = enc.encode(ids);
    CHECK(got.rows() == T && got.cols() == D);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t j = 0; j < D; ++j) CHECK_NEAR(got(t, j), ref[t][j], 1e-4);
    ClipTextEncoder q = loadClipTextEncoder(dir, true);  // int8: close to the reference (final states are layer-normed, ~unit scale)
    const Matrix gq = q.encode(ids);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t j = 0; j < D; ++j) CHECK_NEAR(gq(t, j), ref[t][j], 0.15);
    CHECK_THROWS(enc.encode({1, 2, 3}), std::invalid_argument);
    std::filesystem::remove_all(dir);
}

TEST(multi_head_attention_matches_naive_and_chunks) {
    setSeed(96);
    // 2100 queries x 2100 keys exceeds one 4M-float chunk of scores.
    for (auto [N, M] : {std::pair<std::size_t, std::size_t>{5, 7}, {2100, 2100}}) {
        const std::size_t C = 4, heads = 2, dh = 2;
        const Matrix q = Matrix::uniform(N, C, -1, 1), k = Matrix::uniform(M, C, -1, 1), v = Matrix::uniform(M, C, -1, 1);
        const Matrix got = multiHeadAttention(q, k, v, heads);
        for (std::size_t i = 0; i < N; i += (N > 10 ? 211 : 1))
            for (std::size_t h = 0; h < heads; ++h) {
                std::vector<double> s(M);
                double mx = -1e300, sum = 0;
                for (std::size_t j = 0; j < M; ++j) {
                    double d = 0;
                    for (std::size_t e = 0; e < dh; ++e) d += q(i, h * dh + e) * k(j, h * dh + e);
                    mx = std::max(mx, s[j] = d / std::sqrt(double(dh)));
                }
                for (double& e : s) sum += e = std::exp(e - mx);
                for (std::size_t e = 0; e < dh; ++e) {
                    double acc = 0;
                    for (std::size_t j = 0; j < M; ++j) acc += s[j] / sum * v(j, h * dh + e);
                    CHECK_NEAR(got(i, h * dh + e), acc, 1e-5);
                }
            }
    }
    CHECK_THROWS(multiHeadAttention(Matrix(2, 4), Matrix(3, 4), Matrix(3, 4), 3), std::invalid_argument);
}

TEST(timestep_embedding_and_euler_scheduler) {
    // dim 4: frequencies 1 and 1e-2; flip_sin_to_cos puts the cosines first.
    const Matrix e = timestepEmbedding(999.0f, 4);
    CHECK_NEAR(e(0, 0), std::cos(999.0), 1e-5);
    CHECK_NEAR(e(0, 1), std::cos(9.99), 1e-5);
    CHECK_NEAR(e(0, 2), std::sin(999.0), 1e-5);
    CHECK_NEAR(e(0, 3), std::sin(9.99), 1e-5);
    CHECK_NEAR(timestepEmbedding(5.0f, 4, false)(0, 0), std::sin(5.0), 1e-5);

    // Reference values from diffusers' formulas (scaled_linear betas, trailing spacing).
    const EulerScheduler one = EulerScheduler::standard(1);
    CHECK(one.timesteps == std::vector<double>({999.0}) && one.sigmas.size() == 2 && one.sigmas[1] == 0.0);
    CHECK_NEAR(one.sigmas[0], 14.614641229333639, 1e-6);
    CHECK_NEAR(one.initNoiseSigma, 14.614641229333639, 1e-6);  // trailing: the largest sigma itself
    const EulerScheduler four = EulerScheduler::standard(4);
    CHECK(four.timesteps == std::vector<double>({999, 749, 499, 249}));
    const double expected[4] = {14.614641229333639, 4.081729363724308, 1.612886194303876, 0.6932048996386885};
    for (int i = 0; i < 4; ++i) CHECK_NEAR(four.sigmas[static_cast<std::size_t>(i)], expected[i], 1e-6);
    CHECK_NEAR(four.inputScale(1), 1.0 / std::sqrt(expected[1] * expected[1] + 1.0), 1e-9);
    // Leading spacing with steps_offset 1 (SD 1.5-style config), init sigma sqrt(max^2 + 1).
    const std::string cfg = tempPath("scheduler.json");
    std::ofstream(cfg) << R"({"beta_start":0.00085,"beta_end":0.012,"beta_schedule":"scaled_linear","num_train_timesteps":1000,)"
                       << R"("timestep_spacing":"leading","steps_offset":1,"prediction_type":"epsilon"})";
    const EulerScheduler lead = EulerScheduler::fromConfig(cfg, 4);
    CHECK(lead.timesteps == std::vector<double>({751, 501, 251, 1}));
    CHECK_NEAR(lead.sigmas[0], 4.116696479276065, 1e-6);
    CHECK_NEAR(lead.sigmas[3], 0.04131441199678309, 1e-8);
    CHECK_NEAR(lead.initNoiseSigma, std::sqrt(4.116696479276065 * 4.116696479276065 + 1.0), 1e-6);
    std::ofstream(cfg) << R"({"prediction_type":"v_prediction"})";
    CHECK_THROWS(EulerScheduler::fromConfig(cfg, 4), std::runtime_error);
    std::filesystem::remove(cfg);
    // One step from sigma to 0 returns x - sigma * eps (the predicted clean latent).
    Matrix x{{1.0f, -2.0f}};
    one.step(0, Matrix{{0.5f, 0.25f}}, x);
    CHECK_NEAR(x(0, 0), 1.0 - 14.614641229333639 * 0.5, 1e-4);
    CHECK_NEAR(x(0, 1), -2.0 - 14.614641229333639 * 0.25, 1e-4);
}

namespace {
void addTensor(std::map<std::string, Matrix>& w, const std::string& name, std::size_t n, float lo, float hi) {
    w[name] = Matrix::uniform(1, n, lo, hi);
}

// A tiny diffusers UNet2DConditionModel: two levels, cross-attention in the first.
std::map<std::string, Matrix> tinyUNetWeights(const std::vector<std::size_t>& ch, std::size_t L, std::size_t ctx, std::size_t inChannels = 4) {
    std::map<std::string, Matrix> w;
    const std::size_t temb = 4 * ch[0];
    auto conv = [&](const std::string& n, std::size_t o, std::size_t i, std::size_t k) {
        addTensor(w, n + ".weight", o * i * k * k, -0.3f, 0.3f);
        addTensor(w, n + ".bias", o, -0.1f, 0.1f);
    };
    auto lin = [&](const std::string& n, std::size_t o, std::size_t i, bool bias) {
        addTensor(w, n + ".weight", o * i, -0.3f, 0.3f);
        if (bias) addTensor(w, n + ".bias", o, -0.1f, 0.1f);
    };
    auto norm = [&](const std::string& n, std::size_t c) {
        addTensor(w, n + ".weight", c, 0.8f, 1.2f);
        addTensor(w, n + ".bias", c, -0.1f, 0.1f);
    };
    auto resnet = [&](const std::string& p, std::size_t in, std::size_t out) {
        norm(p + ".norm1", in);
        conv(p + ".conv1", out, in, 3);
        lin(p + ".time_emb_proj", out, temb, true);
        norm(p + ".norm2", out);
        conv(p + ".conv2", out, out, 3);
        if (in != out) conv(p + ".conv_shortcut", out, in, 1);
    };
    auto attention = [&](const std::string& p, std::size_t C) {
        norm(p + ".norm", C);
        lin(p + ".proj_in", C, C, true);
        lin(p + ".proj_out", C, C, true);
        const std::string b = p + ".transformer_blocks.0";
        for (const char* n : {".norm1", ".norm2", ".norm3"}) norm(b + n, C);
        for (const char* n : {".attn1.to_q", ".attn1.to_k", ".attn1.to_v", ".attn2.to_q"}) lin(b + n, C, C, false);
        lin(b + ".attn2.to_k", C, ctx, false);
        lin(b + ".attn2.to_v", C, ctx, false);
        lin(b + ".attn1.to_out.0", C, C, true);
        lin(b + ".attn2.to_out.0", C, C, true);
        lin(b + ".ff.net.0.proj", 8 * C, C, true);
        lin(b + ".ff.net.2", C, 4 * C, true);
    };
    lin("time_embedding.linear_1", temb, ch[0], true);
    lin("time_embedding.linear_2", temb, temb, true);
    conv("conv_in", ch[0], inChannels, 3);
    std::vector<std::size_t> skips{ch[0]};
    std::size_t c = ch[0];
    for (std::size_t i = 0; i < ch.size(); ++i) {
        for (std::size_t j = 0; j < L; ++j) {
            resnet("down_blocks." + std::to_string(i) + ".resnets." + std::to_string(j), c, ch[i]);
            c = ch[i];
            if (i == 0) attention("down_blocks.0.attentions." + std::to_string(j), c);
            skips.push_back(c);
        }
        if (i + 1 < ch.size()) {
            conv("down_blocks." + std::to_string(i) + ".downsamplers.0.conv", c, c, 3);
            skips.push_back(c);
        }
    }
    resnet("mid_block.resnets.0", c, c);
    attention("mid_block.attentions.0", c);
    resnet("mid_block.resnets.1", c, c);
    for (std::size_t i = 0; i < ch.size(); ++i) {
        const std::size_t out = ch[ch.size() - 1 - i];
        for (std::size_t j = 0; j <= L; ++j) {
            resnet("up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j), c + skips.back(), out);
            skips.pop_back();
            c = out;
            if (i + 1 == ch.size()) attention("up_blocks." + std::to_string(i) + ".attentions." + std::to_string(j), c);
        }
        if (i + 1 < ch.size()) conv("up_blocks." + std::to_string(i) + ".upsamplers.0.conv", c, c, 3);
    }
    norm("conv_norm_out", ch[0]);
    conv("conv_out", 4, ch[0], 3);
    return w;
}

void saveAll(const std::string& file, const std::map<std::string, Matrix>& w) {
    std::vector<std::pair<std::string, const Matrix*>> refs;
    for (const auto& [n, m] : w) refs.emplace_back(n, &m);
    saveSafeTensors(file, refs);
}

void writeTinyUNet(const std::string& dir, std::size_t ctx, std::size_t inChannels = 4) {
    std::filesystem::create_directories(dir + "/unet");
    std::ofstream(dir + "/unet/config.json")
        << R"({"_class_name":"UNet2DConditionModel","act_fn":"silu","attention_head_dim":[2,2],"block_out_channels":[8,16],)"
        << R"("cross_attention_dim":)" << ctx << R"(,"down_block_types":["CrossAttnDownBlock2D","DownBlock2D"],"flip_sin_to_cos":true,)"
        << R"("freq_shift":0,"in_channels":)" << inChannels << R"(,"layers_per_block":1,"mid_block_type":"UNetMidBlock2DCrossAttn","norm_eps":1e-5,)"
        << R"("norm_num_groups":4,"out_channels":4,"up_block_types":["UpBlock2D","CrossAttnUpBlock2D"],"use_linear_projection":true,)"
        << R"("time_embedding_type":"positional","resnet_time_scale_shift":"default","transformer_layers_per_block":1})";
    saveAll(dir + "/unet/diffusion_pytorch_model.safetensors", tinyUNetWeights({8, 16}, 1, ctx, inChannels));
}
} // namespace

TEST(unet_loader_builds_and_conditions_on_text_and_time) {
    setSeed(97);
    const std::string dir = tempPath("unet_tiny");
    writeTinyUNet(dir, 6);
    UNet2DCondition unet = loadUNet(dir, 4, 6);
    CHECK(unet.latentShape().channels == 4 && unet.latentShape().height == 6 && unet.contextDim() == 6 && unet.parameterCount() > 1000);
    const Matrix z = Matrix::uniform(1, 4 * 6 * 4, -1, 1), ctxA = Matrix::uniform(5, 6, -1, 1), ctxB = Matrix::uniform(5, 6, -1, 1);
    const Matrix a = unet.forward(z, 999.0f, ctxA);
    CHECK(a.cols() == z.cols());
    CHECK(unet.forward(z, 999.0f, ctxA).approxEquals(a, 0.0f));      // deterministic
    CHECK(!unet.forward(z, 999.0f, ctxB).approxEquals(a, 1e-4f));    // the text matters (cross-attention)
    CHECK(!unet.forward(z, 10.0f, ctxA).approxEquals(a, 1e-4f));     // the timestep matters
    {
        std::vector<UNetCondition> conds(2);  // a batch is one forward() per condition
        conds[0].context = &ctxA;
        conds[1].context = &ctxB;
        const std::vector<Matrix> both = unet.forwardBatch(z, 999.0f, conds);
        CHECK(both.size() == 2 && both[0].approxEquals(a, 1e-6f) && both[1].approxEquals(unet.forward(z, 999.0f, ctxB), 1e-6f));
    }
    UNet2DCondition q = loadUNet(dir, 4, 6, true);                    // int8 weights stay close to float
    double scale = 0;
    for (std::size_t i = 0; i < a.size(); ++i) scale = std::max(scale, static_cast<double>(std::fabs(a.data()[i])));
    CHECK(q.forward(z, 999.0f, ctxA).approxEquals(a, static_cast<float>(0.05 * scale)));
    CHECK_THROWS(unet.forward(Matrix(1, 5), 1.0f, ctxA), std::invalid_argument);
    CHECK_THROWS(unet.forward(z, 1.0f, Matrix(5, 7)), std::invalid_argument);
    CHECK_THROWS(loadUNet(dir, 3, 6), std::invalid_argument);  // latent sides must be multiples of 2
    std::string cfg = loadText(dir + "/unet/config.json");
    cfg.insert(1, R"("addition_embed_type":"text_image",)");
    std::ofstream(dir + "/unet/config.json") << cfg;
    CHECK_THROWS(loadUNet(dir, 4, 6), std::runtime_error);  // SDXL-style conditioning is refused
    std::filesystem::remove_all(dir);
}

namespace {
// A TAESD (AutoencoderTiny) with 4 channels and one block per level.
void writeTinyTaesd(const std::string& dir, bool zeroButOutputBias) {
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/config.json") << R"({"_class_name":"AutoencoderTiny","act_fn":"relu","decoder_block_out_channels":[4,4,4,4],)"
                                        << R"("encoder_block_out_channels":[4,4,4,4],"num_decoder_blocks":[1,1,1,1],)"
                                        << R"("num_encoder_blocks":[1,1,1,1],"latent_channels":4,"upsampling_scaling_factor":2})";
    std::map<std::string, Matrix> w;
    auto conv = [&](const std::string& n, std::size_t o, std::size_t i, bool bias) {
        w[n + ".weight"] = zeroButOutputBias ? Matrix(1, o * i * 9) : Matrix::uniform(1, o * i * 9, -0.3f, 0.3f);
        if (bias) w[n + ".bias"] = zeroButOutputBias ? Matrix(1, o) : Matrix::uniform(1, o, -0.1f, 0.1f);
    };
    auto block = [&](const std::string& p) {
        for (const char* c : {".conv.0", ".conv.2", ".conv.4"}) conv(p + c, 4, 4, true);
    };
    // encoder: conv, block, (conv s2, block) x 3, conv out
    conv("encoder.layers.0", 4, 3, true);
    block("encoder.layers.1");
    for (int i = 0; i < 3; ++i) {
        conv("encoder.layers." + std::to_string(2 + 2 * i), 4, 4, false);
        block("encoder.layers." + std::to_string(3 + 2 * i));
    }
    conv("encoder.layers.8", 4, 4, true);
    // decoder: conv, relu, (block, upsample, conv) x 3, block, conv out
    conv("decoder.layers.0", 4, 4, true);
    for (int i = 0; i < 3; ++i) {
        block("decoder.layers." + std::to_string(2 + 3 * i));
        conv("decoder.layers." + std::to_string(4 + 3 * i), 4, 4, false);
    }
    block("decoder.layers.11");
    conv("decoder.layers.12", 3, 4, true);
    if (zeroButOutputBias) {
        w["encoder.layers.8.bias"] = Matrix{{0.5f, -1.0f, 2.0f, 0.25f}};
        w["decoder.layers.12.bias"] = Matrix{{0.25f, 0.5f, 1.0f}};
    }
    saveAll(dir + "/diffusion_pytorch_model.safetensors", w);
}
} // namespace

TEST(shifted_convolutions_match_im2col) {
    setSeed(55);
    struct Case { ImageShape in; std::size_t out, k, pad; };
    for (const Case& c : {Case{{5, 17, 13}, 7, 3, 1}, Case{{3, 20, 20}, 4, 3, 0}, Case{{4, 18, 21}, 6, 5, 2}, Case{{16, 20, 20}, 8, 3, 1}})
        for (bool int8 : {false, true}) {
            Conv2D conv(c.in, c.out, c.k, 1, c.pad);
            conv.bias() = Matrix::uniform(1, c.out, -1, 1);
            if (int8) conv.quantize();
            const Matrix x = Matrix::uniform(2, c.in.size(), -1, 1);
            setShiftedConvolutions(false);
            const Matrix ref = conv.forward(x, false);
            setShiftedConvolutions(true);
            const Matrix fast = conv.forward(x, false);
            CHECK(fast.approxEquals(ref, 1e-4f));
        }
}

TEST(int8_convolutions_and_large_products) {
    setSeed(100);
    // Small maps use the integer kernel, large ones expand weights for the float GEMM; both stay close
    // to the float layer, and the 1x1 / strided cases too.
    struct Case { ImageShape in; std::size_t out, k, stride, pad; };
    for (const Case& c : {Case{{8, 6, 5}, 12, 3, 1, 1}, Case{{16, 20, 20}, 8, 3, 1, 1}, Case{{8, 24, 24}, 6, 1, 1, 0},
                          Case{{4, 34, 34}, 5, 3, 2, 1}}) {
        Conv2D conv(c.in, c.out, c.k, c.stride, c.pad);
        conv.bias() = Matrix::uniform(1, c.out, -1, 1);
        const Matrix x = Matrix::uniform(1, c.in.size(), -1, 1);
        const Matrix ref = conv.forward(x, false);
        CHECK(conv.quantize() == 1 && conv.quantized() && conv.parameters().empty());
        const Matrix q = conv.forward(x, false);
        double scale = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) scale = std::max(scale, static_cast<double>(std::fabs(ref.data()[i])));
        CHECK(q.approxEquals(ref, static_cast<float>(0.03 * scale)));
        CHECK(conv.describe().find("int8") != std::string::npos);
        CHECK_THROWS(conv.backward(q), std::logic_error);
    }
    // QuantizedWeights: from kGemmRows rows on, the product uses expanded weights with exact inputs.
    const QuantizedWeights w = QuantizedWeights::fromOutIn(Matrix::uniform(37, 70, -1, 1));
    const Matrix many = Matrix::uniform(QuantizedWeights::kGemmRows + 3, 70, -1, 1);
    CHECK(w.multiply(many).approxEquals(w.multiplyFloat(many), 1e-4f));
    // multiplyLeft (W * b, the convolution layout) with a row stride wider than the product.
    const std::size_t n = 300;
    const Matrix b = Matrix::uniform(70, n + 5, -1, 1);
    Matrix c(37, n + 2, 9.0f);
    w.multiplyLeft(n, b.data(), n + 5, c.data(), n + 2);
    const Matrix ref = w.multiplyFloat(b.transpose()).transpose();
    float worst = 0;
    for (std::size_t j = 0; j < 37; ++j) {
        for (std::size_t i = 0; i < n; ++i) worst = std::max(worst, std::fabs(c(j, i) - ref(j, i)));
        CHECK(c(j, n) == 9.0f && c(j, n + 1) == 9.0f);
    }
    CHECK(worst < 1e-4f);
}

TEST(tiny_vae_loader_layers_and_value_ranges) {
    const std::string dir = tempPath("taesd_tiny");
    // With every weight zero, the outputs are the last biases, through the range conversions:
    // the decoder maps [0, 1] to [-1, 1], the encoder's latent is the bias itself.
    writeTinyTaesd(dir, true);
    TinyVae vae = loadTinyVae(dir, 16, 24);
    CHECK(vae.latent.channels == 4 && vae.latent.height == 3 && vae.latent.width == 2);
    const Matrix img = vae.decode(Matrix::uniform(1, 4 * 3 * 2, -5, 5));
    CHECK(img.cols() == 3 * 24 * 16);
    CHECK_NEAR(img(0, 0), 2 * 0.25 - 1, 1e-6);
    CHECK_NEAR(img(0, 24 * 16 + 7), 2 * 0.5 - 1, 1e-6);
    CHECK_NEAR(img(0, 2 * 24 * 16 + 100), 2 * 1.0 - 1, 1e-6);
    const Matrix z = vae.encode(Matrix::uniform(1, 3 * 24 * 16, -1, 1));
    CHECK(z.cols() == 4 * 3 * 2 && z(0, 0) == 0.5f && z(0, 6) == -1.0f && z(0, 12) == 2.0f);
    // Random weights: runs, deterministic, refuses sizes that are not multiples of 8.
    setSeed(99);
    writeTinyTaesd(dir, false);
    TinyVae r = loadTinyVae(dir, 16, 16);
    const Matrix lat = Matrix::uniform(1, 4 * 2 * 2, -2, 2);
    CHECK(r.decode(lat).approxEquals(r.decode(lat), 0.0f) && r.decode(lat).cols() == 3 * 256);
    CHECK_THROWS(loadTinyVae(dir, 12, 16), std::invalid_argument);
    std::filesystem::remove_all(dir);
}

namespace {
// Small PNGs written by Pillow / zlib (dynamic and fixed Huffman, adaptive and Paeth filters,
// a 4-bit palette with transparency, 16-bit grey) and their decoded pixels.
const std::vector<std::uint8_t> kPngRgb = {137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 9, 0, 0, 0, 7, 8, 2, 0, 0, 0, 85, 249, 240, 55, 0, 0, 0, 130, 73, 68, 65, 84, 120, 218, 37, 201, 43, 14, 195, 48, 16, 69, 209, 231, 220, 137, 10, 134, 15, 179, 84, 171, 170, 82, 96, 30, 238, 85, 116, 255, 219, 240, 175, 160, 135, 158, 36, 83, 78, 202, 93, 249, 161, 60, 148, 167, 242, 161, 188, 149, 77, 198, 71, 12, 7, 167, 59, 135, 51, 29, 28, 156, 229, 198, 83, 152, 51, 130, 29, 164, 224, 12, 122, 160, 96, 133, 241, 18, 56, 10, 118, 65, 133, 163, 144, 10, 171, 160, 98, 92, 34, 57, 4, 171, 48, 42, 103, 37, 85, 70, 69, 95, 227, 18, 211, 177, 96, 255, 239, 102, 52, 184, 73, 183, 241, 22, 56, 43, 152, 133, 179, 162, 70, 111, 172, 134, 181, 31, 51, 151, 33, 127, 151, 45, 48, 117, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130};
const std::vector<std::uint8_t> kPngRgbPixels = {0, 4, 0, 29, 5, 0, 58, 0, 0, 87, 7, 0, 116, 3, 0, 145, 0, 0, 174, 2, 0, 203, 1, 0, 232, 5, 0, 3, 44, 0, 32, 40, 11, 61, 43, 22, 90, 38, 33, 119, 40, 44, 148, 37, 55, 177, 40, 66, 206, 43, 77, 235, 41, 88, 6, 76, 0, 35, 80, 22, 64, 76, 44, 93, 75, 66, 122, 76, 88, 151, 81, 110, 180, 76, 132, 209, 76, 154, 238, 74, 176, 9, 111, 0, 38, 114, 33, 67, 114, 66, 96, 113, 99, 125, 113, 132, 154, 115, 165, 183, 116, 198, 212, 114, 231, 241, 114, 8, 12, 150, 0, 41, 151, 44, 70, 154, 88, 99, 152, 132, 128, 148, 176, 157, 153, 220, 186, 154, 8, 215, 150, 52, 244, 150, 96, 15, 189, 0, 44, 186, 55, 73, 190, 110, 102, 189, 165, 131, 185, 220, 160, 190, 19, 189, 186, 74, 218, 189, 129, 247, 190, 184, 18, 226, 0, 47, 229, 66, 76, 227, 132, 105, 224, 198, 134, 229, 8, 163, 229, 74, 192, 224, 140, 221, 222, 206, 250, 226, 16};
const std::vector<std::uint8_t> kPngPalette = {137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 6, 0, 0, 0, 5, 4, 3, 0, 0, 0, 148, 118, 128, 213, 0, 0, 0, 48, 80, 76, 84, 69, 0, 37, 74, 111, 148, 185, 222, 3, 40, 77, 114, 151, 188, 225, 6, 43, 80, 117, 154, 191, 228, 9, 46, 83, 120, 157, 194, 231, 12, 49, 86, 123, 160, 197, 234, 15, 52, 89, 126, 163, 200, 237, 18, 55, 92, 129, 166, 203, 118, 58, 233, 232, 0, 0, 0, 4, 116, 82, 78, 83, 255, 255, 255, 0, 64, 42, 169, 244, 0, 0, 0, 15, 73, 68, 65, 84, 120, 94, 99, 100, 84, 82, 98, 82, 66, 197, 0, 17, 241, 1, 231, 122, 177, 125, 227, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130};
const std::vector<std::uint8_t> kPngPalettePixels = {0, 37, 74, 255, 111, 148, 185, 255, 222, 3, 40, 255, 77, 114, 151, 0, 188, 225, 6, 255, 43, 80, 117, 255, 222, 3, 40, 255, 77, 114, 151, 0, 188, 225, 6, 255, 43, 80, 117, 255, 154, 191, 228, 255, 9, 46, 83, 255, 188, 225, 6, 255, 43, 80, 117, 255, 154, 191, 228, 255, 9, 46, 83, 255, 120, 157, 194, 255, 231, 12, 49, 255, 154, 191, 228, 255, 9, 46, 83, 255, 120, 157, 194, 255, 231, 12, 49, 255, 86, 123, 160, 255, 197, 234, 15, 255, 120, 157, 194, 255, 231, 12, 49, 255, 86, 123, 160, 255, 197, 234, 15, 255, 52, 89, 126, 255, 163, 200, 237, 255};
const std::vector<std::uint8_t> kPngGrey16 = {137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 5, 0, 0, 0, 4, 16, 0, 0, 0, 0, 51, 200, 118, 223, 0, 0, 0, 41, 73, 68, 65, 84, 120, 94, 99, 100, 96, 48, 58, 97, 12, 134, 44, 76, 123, 152, 247, 48, 159, 96, 62, 193, 116, 130, 5, 198, 96, 222, 195, 194, 188, 135, 233, 4, 211, 9, 144, 0, 0, 76, 118, 15, 145, 189, 200, 133, 255, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130};
const std::vector<std::uint8_t> kPngGrey16Pixels = {0, 50, 101, 152, 203, 2, 53, 104, 155, 205, 5, 56, 107, 157, 208, 8, 58, 109, 160, 211};
const std::vector<std::uint8_t> kPngFixedPaeth = {137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 8, 0, 0, 0, 6, 8, 2, 0, 0, 0, 113, 103, 72, 172, 0, 0, 0, 42, 73, 68, 65, 84, 120, 1, 99, 97, 96, 248, 47, 207, 248, 70, 158, 249, 141, 60, 235, 27, 121, 246, 55, 242, 156, 111, 228, 185, 223, 200, 243, 190, 97, 97, 103, 252, 142, 21, 209, 67, 2, 0, 37, 114, 48, 110, 150, 138, 15, 224, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130};
const std::vector<std::uint8_t> kPngFixedPaethPixels = {0, 0, 255, 31, 1, 235, 62, 4, 215, 93, 9, 195, 124, 16, 175, 155, 25, 155, 186, 36, 135, 217, 49, 115, 7, 1, 246, 38, 2, 226, 69, 5, 206, 100, 10, 186, 131, 17, 166, 162, 26, 146, 193, 37, 126, 224, 50, 106, 14, 2, 237, 45, 3, 217, 76, 6, 197, 107, 11, 177, 138, 18, 157, 169, 27, 137, 200, 38, 117, 231, 51, 97, 21, 3, 228, 52, 4, 208, 83, 7, 188, 114, 12, 168, 145, 19, 148, 176, 28, 128, 207, 39, 108, 238, 52, 88, 28, 4, 219, 59, 5, 199, 90, 8, 179, 121, 13, 159, 152, 20, 139, 183, 29, 119, 214, 40, 99, 245, 53, 79, 35, 5, 210, 66, 6, 190, 97, 9, 170, 128, 14, 150, 159, 21, 130, 190, 30, 110, 221, 41, 90, 252, 54, 70};


Image pngFromBytes(const std::vector<std::uint8_t>& bytes) {
    const std::string path = tempPath("fixture.png");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    Image img = loadImage(path);
    std::filesystem::remove(path);
    return img;
}
} // namespace

// A minimal GIF reader for the tests: global palette, delay, loop flag and
// LZW-decoded frames (no local palettes, no interlacing: what saveGif writes).
struct GifFile {
    std::size_t width = 0, height = 0;
    bool loops = false;
    std::vector<int> delays;
    std::vector<Image> frames;
};

GifFile readGif(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<std::uint8_t> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    GifFile g;
    if (d.size() < 13 || std::string(d.begin(), d.begin() + 6) != "GIF89a") throw std::runtime_error("not a GIF89a");
    g.width = d[6] | (d[7] << 8);
    g.height = d[8] | (d[9] << 8);
    std::size_t pos = 13;
    std::vector<std::uint8_t> palette;
    if (d[10] & 0x80) {
        const std::size_t n = std::size_t{2} << (d[10] & 7);
        palette.assign(d.begin() + 13, d.begin() + 13 + static_cast<long>(3 * n));
        pos += 3 * n;
    }
    auto subBlocks = [&](std::vector<std::uint8_t>* out) {
        while (d.at(pos) != 0) {
            const std::size_t len = d[pos++];
            if (out) out->insert(out->end(), d.begin() + static_cast<long>(pos), d.begin() + static_cast<long>(pos + len));
            pos += len;
        }
        ++pos;
    };
    while (pos < d.size()) {
        const std::uint8_t kind = d[pos++];
        if (kind == 0x3B) break;
        if (kind == 0x21) {
            const std::uint8_t label = d[pos++];
            std::vector<std::uint8_t> data;
            subBlocks(&data);
            if (label == 0xF9) g.delays.push_back(data.at(1) | (data.at(2) << 8));
            if (label == 0xFF && std::string(data.begin(), data.begin() + 11) == "NETSCAPE2.0") g.loops = true;
            continue;
        }
        if (kind != 0x2C) throw std::runtime_error("bad GIF block");
        const std::size_t w = d[pos + 4] | (d[pos + 5] << 8), h = d[pos + 6] | (d[pos + 7] << 8);
        pos += 9;
        const int minCode = d[pos++];
        std::vector<std::uint8_t> data;
        subBlocks(&data);
        // LZW
        const int clear = 1 << minCode, end = clear + 1;
        std::vector<std::vector<std::uint8_t>> table;
        auto reset = [&] {
            table.assign(static_cast<std::size_t>(end + 1), {});
            for (int i = 0; i < clear; ++i) table[static_cast<std::size_t>(i)] = {static_cast<std::uint8_t>(i)};
        };
        reset();
        int width = minCode + 1, prev = -1;
        std::size_t bitPos = 0;
        std::vector<std::uint8_t> indices;
        while (bitPos + static_cast<std::size_t>(width) <= data.size() * 8) {
            int code = 0;
            for (int b = 0; b < width; ++b, ++bitPos) code |= ((data[bitPos / 8] >> (bitPos % 8)) & 1) << b;
            if (code == clear) {
                reset();
                width = minCode + 1;
                prev = -1;
                continue;
            }
            if (code == end) break;
            std::vector<std::uint8_t> entry;
            if (code < static_cast<int>(table.size())) entry = table[static_cast<std::size_t>(code)];
            else if (prev >= 0 && code == static_cast<int>(table.size())) {
                entry = table[static_cast<std::size_t>(prev)];
                entry.push_back(entry[0]);
            } else throw std::runtime_error("bad LZW code");
            if (prev >= 0 && table.size() < 4096) {
                std::vector<std::uint8_t> added = table[static_cast<std::size_t>(prev)];
                added.push_back(entry[0]);
                table.push_back(added);
                if (static_cast<int>(table.size()) == (1 << width) && width < 12) ++width;
            }
            indices.insert(indices.end(), entry.begin(), entry.end());
            prev = code;
        }
        if (indices.size() != w * h) throw std::runtime_error("GIF frame has the wrong pixel count");
        Image f;
        f.width = w;
        f.height = h;
        for (std::uint8_t i : indices)
            for (int c = 0; c < 3; ++c) f.pixels.push_back(palette.at(3u * i + static_cast<unsigned>(c)));
        g.frames.push_back(std::move(f));
    }
    return g;
}

TEST(transform_image_and_animated_gif) {
    Image img;
    img.width = 40;
    img.height = 30;
    for (std::size_t y = 0; y < 30; ++y)
        for (std::size_t x = 0; x < 40; ++x)
            for (int c = 0; c < 3; ++c) img.pixels.push_back(static_cast<std::uint8_t>(c == 0 ? x * 6 : c == 1 ? y * 8 : 128));
    CHECK(transformImage(img, 1.0f).pixels == img.pixels);  // identity
    // A shift by whole pixels moves the content (edges clamp).
    const Image moved = transformImage(img, 1.0f, 3.0f, -2.0f);
    for (std::size_t y = 0; y < 27; ++y)
        for (std::size_t x = 3; x < 40; ++x)
            for (std::size_t c = 0; c < 3; ++c)
                CHECK(moved.pixels[(y * 40 + x) * 3 + c] == img.pixels[((y + 2) * 40 + x - 3) * 3 + c]);
    // Zooming in by 2 keeps the centre and stretches the gradient to half the slope.
    const Image zoomed = transformImage(img, 2.0f);
    const auto red = [](const Image& im, std::size_t x, std::size_t y) { return static_cast<int>(im.pixels[(y * im.width + x) * 3]); };
    CHECK(std::abs(2 * (red(zoomed, 30, 15) - red(zoomed, 10, 15)) - (red(img, 30, 15) - red(img, 10, 15))) <= 4);
    CHECK(std::abs(red(zoomed, 19, 15) - red(img, 19, 15)) <= 3);
    // A 180 degree rotation mirrors both axes.
    const Image turned = transformImage(img, 1.0f, 0.0f, 0.0f, 180.0f);
    CHECK(std::abs(red(turned, 0, 0) - red(img, 39, 29)) <= 1 && std::abs(red(turned, 39, 5) - red(img, 0, 24)) <= 1);
    CHECK_THROWS(transformImage(img, 0.0f), std::invalid_argument);

    // GIF: frames of a few flat colours come back exactly; a gradient comes back
    // close (one shared palette, dithered).
    std::vector<Image> frames;
    for (int k = 0; k < 3; ++k) {
        Image f;
        f.width = 40;
        f.height = 30;
        for (std::size_t y = 0; y < 30; ++y)
            for (std::size_t x = 0; x < 40; ++x) {
                const bool a = (x / 5 + y / 5 + static_cast<std::size_t>(k)) % 2 == 0;
                const std::uint8_t px[3] = {static_cast<std::uint8_t>(a ? 250 : 10), static_cast<std::uint8_t>(a ? 40 : 200),
                                            static_cast<std::uint8_t>(k * 100)};
                f.pixels.insert(f.pixels.end(), px, px + 3);
            }
        frames.push_back(f);
    }
    frames.push_back(img);
    const std::string path = tempPath("anim.gif");
    saveGif(path, frames, 4.0f);
    const GifFile g = readGif(path);
    CHECK(g.width == 40 && g.height == 30 && g.loops);
    CHECK(g.frames.size() == 4 && g.delays == std::vector<int>({25, 25, 25, 25}));
    for (int k = 0; k < 3; ++k) CHECK(g.frames[static_cast<std::size_t>(k)].pixels == frames[static_cast<std::size_t>(k)].pixels);
    double err = 0;
    for (std::size_t i = 0; i < img.pixels.size(); ++i) err += std::abs(g.frames[3].pixels[i] - img.pixels[i]);
    CHECK(err / static_cast<double>(img.pixels.size()) < 6.0);
    // Larger, noisy frames exercise the LZW table reset (more than 4096 codes).
    Image noise;
    noise.width = noise.height = 200;
    std::mt19937 rng(5);
    for (std::size_t i = 0; i < 200 * 200 * 3; ++i) noise.pixels.push_back(static_cast<std::uint8_t>(rng() % 256));
    saveGif(path, {noise, noise}, 10.0f, false);
    const GifFile n = readGif(path);
    CHECK(n.frames.size() == 2 && !n.loops && n.delays[0] == 10 && n.frames[0].pixels == n.frames[1].pixels);
    CHECK_THROWS(saveGif(path, {}, 8.0f), std::invalid_argument);
    CHECK_THROWS(saveGif(path, {img, noise}, 8.0f), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST(compressed_png_and_animated_png) {
    // savePng compresses (filters + own deflate) and loadPng (own inflate) reads it back exactly.
    std::mt19937 rng(11);
    const std::string path = tempPath("c.png");
    auto fileSize = [&] { return static_cast<std::size_t>(std::filesystem::file_size(path)); };
    for (std::size_t channels : {1u, 3u, 4u}) {
        Image noise;
        noise.width = 97;
        noise.height = 61;
        noise.channels = channels;
        for (std::size_t i = 0; i < 97 * 61 * channels; ++i) noise.pixels.push_back(static_cast<std::uint8_t>(rng() % 256));
        savePng(path, noise);
        CHECK(loadPng(path).pixels == noise.pixels);
        CHECK(fileSize() < noise.pixels.size() + 200);  // incompressible data falls back to stored blocks
    }
    Image smooth;
    smooth.width = 256;
    smooth.height = 200;
    for (std::size_t y = 0; y < 200; ++y)
        for (std::size_t x = 0; x < 256; ++x)
            for (int c = 0; c < 3; ++c) smooth.pixels.push_back(static_cast<std::uint8_t>((x + 2 * y + 40 * c) % 256));
    savePng(path, smooth);
    CHECK(loadPng(path).pixels == smooth.pixels);
    CHECK(fileSize() < smooth.pixels.size() / 20);  // gradients filter to near-constant rows
    Image flat = smooth;
    std::fill(flat.pixels.begin(), flat.pixels.end(), static_cast<std::uint8_t>(77));
    savePng(path, flat);
    CHECK(loadPng(path).pixels == flat.pixels && fileSize() < 1000);
    // Long repeats and every match length / distance code.
    std::vector<std::uint8_t> data;
    for (int i = 0; i < 200000; ++i) data.push_back(static_cast<std::uint8_t>((i % 7 == 0 ? rng() % 4 : 0) + (i / 1000) % 3));
    const auto z = deflateCompress(data);
    CHECK(z.size() < data.size() / 4 && z[0] == 0x78);
    // A short input that ends inside a match running to its last byte (the match search must stop there).
    const std::vector<std::uint8_t> tail{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    Image tiny;
    tiny.width = 4;
    tiny.height = 4;
    tiny.channels = 1;
    tiny.pixels = tail;
    savePng(path, tiny);
    CHECK(loadPng(path).pixels == tail);
    CHECK(deflateCompress(tail).size() > 2);

    // APNG: acTL with the frame count, one fcTL per frame, fdAT after the first,
    // sequence numbers in order; the first frame is the still image.
    std::vector<Image> frames{smooth, flat, smooth};
    saveApng(path, frames, 10.0f);
    CHECK(loadPng(path).pixels == smooth.pixels);
    std::ifstream in(path, std::ios::binary);
    const std::vector<std::uint8_t> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto be32 = [&](std::size_t p) {
        return (std::uint32_t{d[p]} << 24) | (std::uint32_t{d[p + 1]} << 16) | (std::uint32_t{d[p + 2]} << 8) | d[p + 3];
    };
    std::vector<std::string> types;
    std::vector<std::uint32_t> sequence;
    std::uint32_t frameCount = 0, delay = 0;
    for (std::size_t p = 8; p + 8 <= d.size();) {
        const std::uint32_t len = be32(p);
        const std::string type(d.begin() + static_cast<long>(p + 4), d.begin() + static_cast<long>(p + 8));
        types.push_back(type);
        if (type == "acTL") frameCount = be32(p + 8);
        if (type == "fcTL" || type == "fdAT") sequence.push_back(be32(p + 8));
        if (type == "fcTL") delay = (std::uint32_t{d[p + 8 + 20]} << 8) | d[p + 8 + 21];
        p += 12 + len;
    }
    CHECK(frameCount == 3 && delay == 100);
    CHECK(types == std::vector<std::string>({"IHDR", "acTL", "fcTL", "IDAT", "fcTL", "fdAT", "fcTL", "fdAT", "IEND"}));
    CHECK(sequence == std::vector<std::uint32_t>({0, 1, 2, 3, 4}));
    CHECK_THROWS(saveApng(path, {}, 8.0f), std::invalid_argument);
    CHECK_THROWS(saveApng(path, {smooth, Image{}}, 8.0f), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST(jpeg_encoder_and_quicktime_movie) {
    Image img;
    img.width = 37;  // not a multiple of the 16x16 MCU
    img.height = 21;
    for (std::size_t y = 0; y < 21; ++y)
        for (std::size_t x = 0; x < 37; ++x)
            for (int c = 0; c < 3; ++c) img.pixels.push_back(static_cast<std::uint8_t>(c == 0 ? x * 7 : c == 1 ? y * 12 : (x * y) % 256));
    const auto hi = encodeJpeg(img, 95), lo = encodeJpeg(img, 30);
    CHECK(lo.size() < hi.size());
    // Markers: SOI, APP0, 2 DQT, SOF0 with the size and 4:2:0 luma sampling, 4 DHT, SOS ... EOI.
    std::vector<int> markers;
    std::size_t p = 2, sos = 0;
    CHECK(hi[0] == 0xFF && hi[1] == 0xD8);
    while (p + 4 <= hi.size() && !sos) {
        CHECK(hi[p] == 0xFF);
        const int m = hi[p + 1], len = (hi[p + 2] << 8) | hi[p + 3];
        markers.push_back(m);
        if (m == 0xC0) CHECK(((hi[p + 5] << 8) | hi[p + 6]) == 21 && ((hi[p + 7] << 8) | hi[p + 8]) == 37 && hi[p + 9] == 3 && hi[p + 11] == 0x22);
        if (m == 0xDA) sos = p + 2 + static_cast<std::size_t>(len);
        p += 2 + static_cast<std::size_t>(len);
    }
    CHECK(markers == std::vector<int>({0xE0, 0xDB, 0xDB, 0xC0, 0xC4, 0xC4, 0xC4, 0xC4, 0xDA}));
    CHECK(hi[hi.size() - 2] == 0xFF && hi.back() == 0xD9);
    for (std::size_t i = sos; i + 2 < hi.size(); ++i)
        if (hi[i] == 0xFF) CHECK(hi[i + 1] == 0x00);  // every 0xFF in the entropy data is stuffed
    CHECK_THROWS(encodeJpeg(img, 0), std::invalid_argument);

    // QuickTime: ftyp, mdat with the JPEGs, moov whose sample table points at them.
    const std::string path = tempPath("m.mov");
    Image other = img;
    for (auto& v : other.pixels) v = static_cast<std::uint8_t>(255 - v);
    saveMjpegMov(path, {img, other, img}, 12.0f, 80);
    std::ifstream in(path, std::ios::binary);
    const std::vector<std::uint8_t> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto be32 = [&](std::size_t q) {
        return (std::uint32_t{d[q]} << 24) | (std::uint32_t{d[q + 1]} << 16) | (std::uint32_t{d[q + 2]} << 8) | d[q + 3];
    };
    auto find = [&](const char* tag) {
        for (std::size_t q = 4; q + 4 <= d.size(); ++q)
            if (std::equal(tag, tag + 4, d.begin() + static_cast<long>(q))) return q - 4;
        return d.size();
    };
    std::vector<std::string> top;
    for (std::size_t q = 0; q + 8 <= d.size(); q += be32(q)) top.emplace_back(d.begin() + static_cast<long>(q + 4), d.begin() + static_cast<long>(q + 8));
    CHECK(top == std::vector<std::string>({"ftyp", "mdat", "moov"}));
    const std::size_t stsz = find("stsz"), stco = find("stco"), stts = find("stts"), mdhd = find("mdhd");
    CHECK(stsz < d.size() && stco < d.size() && be32(stsz + 16) == 3 && be32(stco + 12) == 3);
    CHECK(be32(stts + 16) == 3 && be32(mdhd + 20) == 12000 && be32(stts + 20) == 1000);  // 12 fps
    for (std::size_t k = 0; k < 3; ++k) {
        const std::size_t off = be32(stco + 16 + 4 * k), size = be32(stsz + 20 + 4 * k);
        CHECK(d[off] == 0xFF && d[off + 1] == 0xD8 && d[off + size - 2] == 0xFF && d[off + size - 1] == 0xD9);
    }
    CHECK(find("jpeg") < d.size());
    CHECK_THROWS(saveMjpegMov(path, {img, Image{}}, 8.0f), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST(png_reader_and_image_resizing) {
    const Image rgb = pngFromBytes(kPngRgb);
    CHECK(rgb.width == 9 && rgb.height == 7 && rgb.channels == 3 && rgb.pixels == kPngRgbPixels);
    const Image pal = pngFromBytes(kPngPalette);
    CHECK(pal.width == 6 && pal.height == 5 && pal.channels == 4 && pal.pixels == kPngPalettePixels);
    const Image g16 = pngFromBytes(kPngGrey16);
    CHECK(g16.channels == 1 && g16.pixels == kPngGrey16Pixels);
    const Image fixed = pngFromBytes(kPngFixedPaeth);
    CHECK(fixed.width == 8 && fixed.height == 6 && fixed.pixels == kPngFixedPaethPixels);
    // Our own writer round-trips through the reader; PPM goes through loadImage too.
    const std::string path = tempPath("rt.png");
    savePng(path, rgb);
    CHECK(loadImage(path).pixels == rgb.pixels);
    std::ofstream(path) << "not an image";
    CHECK_THROWS(loadImage(path), std::runtime_error);
    std::filesystem::remove(path);

    // toRgb composites alpha over white and expands grey.
    Image ga;
    ga.width = 2;
    ga.height = 1;
    ga.channels = 2;
    ga.pixels = {0, 255, 0, 0};  // black opaque, black transparent
    CHECK(toRgb(ga).pixels == std::vector<std::uint8_t>({0, 0, 0, 255, 255, 255}));
    // resizeImage: bilinear upsampling of a 2x1 ramp, identity at the same size, box filter when shrinking a lot.
    Image ramp;
    ramp.width = 2;
    ramp.height = 1;
    ramp.channels = 1;
    ramp.pixels = {0, 200};
    CHECK(resizeImage(ramp, 4, 1).pixels == std::vector<std::uint8_t>({0, 50, 150, 200}));
    CHECK(resizeImage(rgb, 9, 7).pixels == rgb.pixels);
    Image checker;
    checker.width = checker.height = 16;
    checker.channels = 1;
    for (std::size_t i = 0; i < 256; ++i) checker.pixels.push_back((i % 16 + i / 16) % 2 ? 255 : 0);
    for (std::uint8_t v : resizeImage(checker, 4, 4).pixels) CHECK(v >= 126 && v <= 129);  // averages, no aliasing
    // fitImage: centred crop to the aspect ratio, then resize.
    Image wide;
    wide.width = 30;
    wide.height = 10;
    wide.channels = 1;
    for (std::size_t y = 0; y < 10; ++y)
        for (std::size_t x = 0; x < 30; ++x) wide.pixels.push_back(x < 10 || x >= 20 ? 0 : 255);  // white middle third
    const Image square = fitImage(wide, 10, 10);
    CHECK(square.width == 10 && square.height == 10);
    for (std::uint8_t v : square.pixels) CHECK(v == 255);
}

// Tokenizer, text encoder, UNet, VAE and scheduler, all tiny, in one diffusers
// directory; the tokenizer's own scratch directory is returned in `tok`.
std::string writeTinyStableDiffusion(const std::string& name, std::string& tok) {
    setSeed(98);
    const std::string dir = tempPath(name);
    std::filesystem::create_directories(dir + "/tokenizer");
    tok = tinyClipTokenizer();
    for (const char* f : {"vocab.json", "merges.txt", "tokenizer_config.json"})
        std::filesystem::copy_file(tok + "/" + f, dir + "/tokenizer/" + f, std::filesystem::copy_options::overwrite_existing);
    std::ofstream(dir + "/tokenizer/tokenizer_config.json") << R"({"pad_token":"!","added_tokens_decoder":{"0":{"content":"!"}}})";
    std::filesystem::create_directories(dir + "/text_encoder");
    std::ofstream(dir + "/text_encoder/config.json")
        << R"({"hidden_act":"gelu","vocab_size":518,"hidden_size":6,"num_attention_heads":2,"intermediate_size":12,)"
        << R"("num_hidden_layers":1,"max_position_embeddings":8})";
    {
        std::map<std::string, Matrix> w;
        const std::string p = "text_model.", l = p + "encoder.layers.0.";
        addTensor(w, p + "embeddings.token_embedding.weight", 518 * 6, -1, 1);
        addTensor(w, p + "embeddings.position_embedding.weight", 8 * 6, -1, 1);
        for (const char* n : {"layer_norm1", "layer_norm2"}) {
            addTensor(w, l + n + ".weight", 6, 0.8f, 1.2f);
            addTensor(w, l + n + ".bias", 6, -0.1f, 0.1f);
        }
        for (const char* n : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            addTensor(w, l + "self_attn." + n + ".weight", 36, -0.4f, 0.4f);
            addTensor(w, l + "self_attn." + n + ".bias", 6, -0.1f, 0.1f);
        }
        addTensor(w, l + "mlp.fc1.weight", 72, -0.4f, 0.4f);
        addTensor(w, l + "mlp.fc1.bias", 12, -0.1f, 0.1f);
        addTensor(w, l + "mlp.fc2.weight", 72, -0.4f, 0.4f);
        addTensor(w, l + "mlp.fc2.bias", 6, -0.1f, 0.1f);
        addTensor(w, p + "final_layer_norm.weight", 6, 0.8f, 1.2f);
        addTensor(w, p + "final_layer_norm.bias", 6, -0.1f, 0.1f);
        saveAll(dir + "/text_encoder/model.safetensors", w);
    }
    writeTinyUNet(dir, 6);
    std::filesystem::create_directories(dir + "/vae");
    std::ofstream(dir + "/vae/config.json") << R"({"act_fn":"silu","block_out_channels":[4,4,8,8],"latent_channels":4,)"
                                            << R"("layers_per_block":1,"norm_num_groups":4,"scaling_factor":0.18215})";
    {
        std::map<std::string, Matrix> w;
        for (const auto& [name, shape] : vaeTensorShapes({4, 4, 8, 8}, 1, 4)) {
            std::size_t n = 1;
            for (std::size_t d : shape) n *= d;
            addTensor(w, name, n, -0.2f, 0.2f);
        }
        saveAll(dir + "/vae/diffusion_pytorch_model.safetensors", w);
    }
    std::filesystem::create_directories(dir + "/scheduler");
    std::ofstream(dir + "/scheduler/scheduler_config.json") << R"({"beta_schedule":"scaled_linear","timestep_spacing":"trailing"})";
    return dir;
}

namespace {
// diffusers name -> original single-file name (the inverse of Checkpoint.cpp's renames), to build test checkpoints.
std::string ldmResnet(const std::string& r) {
    const std::vector<std::pair<std::string, std::string>> map = {{"norm1.", "in_layers.0."}, {"conv1.", "in_layers.2."},
        {"time_emb_proj.", "emb_layers.1."}, {"norm2.", "out_layers.0."}, {"conv2.", "out_layers.3."}, {"conv_shortcut.", "skip_connection."}};
    for (const auto& [a, b] : map)
        if (r.rfind(a, 0) == 0) return b + r.substr(a.size());
    return r;
}
// "<prefix>N.<rest>" -> N and rest
bool splitIndex(const std::string& s, const std::string& prefix, std::size_t& n, std::string& rest) {
    if (s.rfind(prefix, 0) != 0) return false;
    const std::size_t dot = s.find('.', prefix.size());
    n = std::stoul(s.substr(prefix.size(), dot - prefix.size()));
    rest = s.substr(dot + 1);
    return true;
}
std::string ldmUnetName(const std::string& n, std::size_t L, const std::vector<bool>& upAttention) {
    std::size_t b, j;
    std::string r, r2;
    const std::string P = "model.diffusion_model.";
    if (n.rfind("time_embedding.linear_1.", 0) == 0) return P + "time_embed.0." + n.substr(24);
    if (n.rfind("time_embedding.linear_2.", 0) == 0) return P + "time_embed.2." + n.substr(24);
    if (n.rfind("conv_in.", 0) == 0) return P + "input_blocks.0.0." + n.substr(8);
    if (n.rfind("conv_norm_out.", 0) == 0) return P + "out.0." + n.substr(14);
    if (n.rfind("conv_out.", 0) == 0) return P + "out.2." + n.substr(9);
    if (splitIndex(n, "down_blocks.", b, r)) {
        if (splitIndex(r, "resnets.", j, r2)) return P + "input_blocks." + std::to_string(1 + b * (L + 1) + j) + ".0." + ldmResnet(r2);
        if (splitIndex(r, "attentions.", j, r2)) return P + "input_blocks." + std::to_string(1 + b * (L + 1) + j) + ".1." + r2;
        if (r.rfind("downsamplers.0.conv.", 0) == 0) return P + "input_blocks." + std::to_string(1 + b * (L + 1) + L) + ".0.op." + r.substr(20);
    }
    if (n.rfind("mid_block.", 0) == 0) {
        const std::string m = n.substr(10);
        if (splitIndex(m, "resnets.", j, r2)) return P + "middle_block." + std::to_string(2 * j) + "." + ldmResnet(r2);
        if (splitIndex(m, "attentions.", j, r2)) return P + "middle_block.1." + r2;
    }
    if (splitIndex(n, "up_blocks.", b, r)) {
        if (splitIndex(r, "resnets.", j, r2)) return P + "output_blocks." + std::to_string(b * (L + 1) + j) + ".0." + ldmResnet(r2);
        if (splitIndex(r, "attentions.", j, r2)) return P + "output_blocks." + std::to_string(b * (L + 1) + j) + ".1." + r2;
        if (r.rfind("upsamplers.0.conv.", 0) == 0)
            return P + "output_blocks." + std::to_string(b * (L + 1) + L) + (upAttention[b] ? ".2.conv." : ".1.conv.") + r.substr(18);
    }
    throw std::runtime_error("ldmUnetName: " + n);
}
std::string ldmVaeName(const std::string& n, std::size_t levels) {
    const std::string P = "first_stage_model.";
    auto res = [](std::string r) { return r.rfind("conv_shortcut.", 0) == 0 ? "nin_shortcut." + r.substr(14) : r; };
    for (const std::string half : {"encoder.", "decoder."}) {
        if (n.rfind(half, 0) != 0) continue;
        const std::string r = n.substr(half.size());
        std::size_t b, j;
        std::string r1, r2;
        if (r.rfind("conv_norm_out.", 0) == 0) return P + half + "norm_out." + r.substr(14);
        if (r.rfind("mid_block.resnets.0.", 0) == 0) return P + half + "mid.block_1." + res(r.substr(20));
        if (r.rfind("mid_block.resnets.1.", 0) == 0) return P + half + "mid.block_2." + res(r.substr(20));
        if (r.rfind("mid_block.attentions.0.", 0) == 0) {
            const std::string a = r.substr(23);
            const std::vector<std::pair<std::string, std::string>> map = {{"group_norm.", "norm."}, {"to_q.", "q."}, {"to_k.", "k."}, {"to_v.", "v."}, {"to_out.0.", "proj_out."}};
            for (const auto& [x, y] : map)
                if (a.rfind(x, 0) == 0) return P + half + "mid.attn_1." + y + a.substr(x.size());
        }
        if (splitIndex(r, "down_blocks.", b, r1)) {
            if (splitIndex(r1, "resnets.", j, r2)) return P + half + "down." + std::to_string(b) + ".block." + std::to_string(j) + "." + res(r2);
            if (r1.rfind("downsamplers.0.conv.", 0) == 0) return P + half + "down." + std::to_string(b) + ".downsample.conv." + r1.substr(20);
        }
        if (splitIndex(r, "up_blocks.", b, r1)) {
            const std::string u = std::to_string(levels - 1 - b);
            if (splitIndex(r1, "resnets.", j, r2)) return P + half + "up." + u + ".block." + std::to_string(j) + "." + res(r2);
            if (r1.rfind("upsamplers.0.conv.", 0) == 0) return P + half + "up." + u + ".upsample.conv." + r1.substr(18);
        }
        return P + n;  // conv_in, conv_out
    }
    return P + n;  // quant_conv, post_quant_conv
}

std::map<std::string, Matrix> loadAllTensors(const std::string& file) {
    std::map<std::string, Matrix> out;
    const SafeTensors st(file);
    for (const std::string& n : st.names()) out[n] = st.load(n);
    return out;
}
} // namespace

TEST(single_file_checkpoints_load_like_diffusers_folders) {
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_single", tok);
    const std::string ckdir = tempPath("sd_single_ck");
    std::filesystem::create_directories(ckdir + "/tokenizer");
    for (const char* f : {"vocab.json", "merges.txt"})
        std::filesystem::copy_file(dir + "/tokenizer/" + f, ckdir + "/tokenizer/" + f, std::filesystem::copy_options::overwrite_existing);
    // The same weights in the original layout: SD 1.x text encoder naming first.
    std::map<std::string, Matrix> all;
    for (auto& [n, m] : loadAllTensors(dir + "/unet/diffusion_pytorch_model.safetensors")) all[ldmUnetName(n, 1, {false, true})] = m;
    for (auto& [n, m] : loadAllTensors(dir + "/vae/diffusion_pytorch_model.safetensors")) all[ldmVaeName(n, 4)] = m;
    const auto text = loadAllTensors(dir + "/text_encoder/model.safetensors");
    for (const auto& [n, m] : text) all["cond_stage_model.transformer." + n] = m;
    const std::string file = ckdir + "/tiny.safetensors";
    saveAll(file, all);
    CHECK(isSingleFileCheckpoint(file) && !isSingleFileCheckpoint(dir + "/unet/diffusion_pytorch_model.safetensors"));
    const CheckpointInfo info = inspectCheckpoint(file);
    CHECK(info.family == "sd1" && info.inChannels == 4 && info.contextDim == 6 && info.hasVae && info.hasTextEncoder);
    // Every renamed tensor is the diffusers one, and the derived configs match what the tiny model needs.
    {
        const SafeTensors unet(file + "#unet");
        CHECK(unet.names().size() == loadAllTensors(dir + "/unet/diffusion_pytorch_model.safetensors").size());
        CHECK(unet.load("up_blocks.0.upsamplers.0.conv.weight").values() ==
              SafeTensors(dir + "/unet/diffusion_pytorch_model.safetensors").load("up_blocks.0.upsamplers.0.conv.weight").values());
        const ModelComponent c = modelComponent(file, "unet", "test");
        CHECK(c.config.at("layers_per_block").number == 1 && c.config.at("block_out_channels").array.size() == 2);
        CHECK(c.config.at("down_block_types").array[0].string == "CrossAttnDownBlock2D" && c.config.at("up_block_types").array[0].string == "UpBlock2D");
        CHECK(SafeTensors(file + "#vae").names().size() == loadAllTensors(dir + "/vae/diffusion_pytorch_model.safetensors").size());
        CHECK(modelComponent(file, "vae", "test").config.at("block_out_channels").array.size() == 4);
    }
    // The tiny model's unusual sizes (2 heads, 4 groups, GELU) and its scheduler and padding go in the sidecar.
    std::ofstream(ckdir + "/tiny.json") << R"({"unet": {"attention_head_dim": [2, 2], "norm_num_groups": 4},
        "vae": {"norm_num_groups": 4}, "text_encoder": {"num_attention_heads": 2, "hidden_act": "gelu"},
        "scheduler": {"beta_schedule": "scaled_linear", "timestep_spacing": "trailing"}, "tokenizer": {"pad_token": "!"}})";
    CHECK(tokenizerDirectory(file) == ckdir + "/tokenizer" && padsWithExclamation(file) && !schedulerConfig(file).empty());
    TextToImageOptions o;
    o.width = 64;
    o.height = 64;
    o.steps = 2;
    o.seed = 5;
    const Image fromFolder = textToImage(dir, "hello he", o);
    CHECK(textToImage(file, "hello he", o).pixels == fromFolder.pixels);
    // SD 2.x layout: OpenCLIP names with fused q/k/v and one layer more (SD 2 uses the penultimate one).
    {
        std::map<std::string, Matrix> two;
        for (const auto& [n, m] : all)
            if (n.rfind("cond_stage_model.", 0) != 0) two[n] = m;
        const std::string P = "cond_stage_model.model.", L = "text_model.encoder.layers.0.";
        two[P + "token_embedding.weight"] = text.at("text_model.embeddings.token_embedding.weight");
        two[P + "positional_embedding"] = text.at("text_model.embeddings.position_embedding.weight");
        two[P + "ln_final.weight"] = text.at("text_model.final_layer_norm.weight");
        two[P + "ln_final.bias"] = text.at("text_model.final_layer_norm.bias");
        for (int layer = 0; layer < 2; ++layer) {
            const std::string R = P + "transformer.resblocks." + std::to_string(layer) + ".";
            auto src = [&](const std::string& n) { return layer == 0 ? text.at(L + n) : Matrix::uniform(1, text.at(L + n).size(), -0.3f, 0.3f); };
            Matrix w(1, 0), b(1, 0);
            std::vector<float> wv, bv;
            for (const char* k : {"q_proj", "k_proj", "v_proj"}) {
                const Matrix a = src(std::string("self_attn.") + k + ".weight"), c = src(std::string("self_attn.") + k + ".bias");
                wv.insert(wv.end(), a.values().begin(), a.values().end());
                bv.insert(bv.end(), c.values().begin(), c.values().end());
            }
            two[R + "attn.in_proj_weight"] = Matrix(1, wv.size(), std::move(wv));
            two[R + "attn.in_proj_bias"] = Matrix(1, bv.size(), std::move(bv));
            const std::vector<std::pair<std::string, std::string>> map = {{"ln_1.", "layer_norm1."}, {"ln_2.", "layer_norm2."},
                {"attn.out_proj.", "self_attn.out_proj."}, {"mlp.c_fc.", "mlp.fc1."}, {"mlp.c_proj.", "mlp.fc2."}};
            for (const auto& [to, from] : map)
                for (const char* wb : {"weight", "bias"}) two[R + to + wb] = src(from + wb);
        }
        const std::string file2 = ckdir + "/tiny2.safetensors";
        saveAll(file2, two);
        std::filesystem::copy_file(ckdir + "/tiny.json", ckdir + "/tiny2.json", std::filesystem::copy_options::overwrite_existing);
        CHECK(inspectCheckpoint(file2).family == "sd2");
        CHECK(modelComponent(file2, "text_encoder", "test").config.at("num_hidden_layers").number == 1);
        CHECK(textToImage(file2, "hello he", o).pixels == fromFolder.pixels);
    }
    // Another VAE: a diffusers VAE folder, or a VAE file of its own in the original layout (no prefix).
    {
        std::map<std::string, Matrix> vae;
        for (const auto& [n, m] : all)
            if (n.rfind("first_stage_model.", 0) == 0) vae[n.substr(18)] = m;
        saveAll(ckdir + "/myvae.safetensors", vae);
        std::ofstream(ckdir + "/myvae.json") << R"({"vae": {"norm_num_groups": 4}})";
        TextToImageOptions v = o;
        v.vae = ckdir + "/myvae.safetensors";
        CHECK(textToImage(file, "hello he", v).pixels == fromFolder.pixels);
        v.vae = dir + "/vae";
        CHECK(textToImage(file, "hello he", v).pixels == fromFolder.pixels);
        const auto models = findModels(ckdir);
        std::map<std::string, std::string> kinds;
        for (const ModelInfo& m : models) kinds[std::filesystem::path(m.path).filename().string()] = m.kind;
        CHECK(kinds["tiny.safetensors"] == "sd" && kinds["tiny2.safetensors"] == "sd" && kinds["myvae.safetensors"] == "vae");
    }
    // Not a checkpoint, or one without a VAE, says so.
    CHECK_THROWS(inspectCheckpoint(dir + "/vae/diffusion_pytorch_model.safetensors"), std::runtime_error);
    {
        std::map<std::string, Matrix> noVae;
        for (const auto& [n, m] : all)
            if (n.rfind("first_stage_model.", 0) != 0) noVae[n] = m;
        saveAll(ckdir + "/novae.safetensors", noVae);
        CHECK_THROWS(modelComponent(ckdir + "/novae.safetensors", "vae", "test"), std::runtime_error);
    }
    std::filesystem::remove_all(ckdir);
    std::filesystem::remove_all(dir);
}

TEST(text_to_image_pipeline_runs_end_to_end) {
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_tiny", tok);
    TextToImageOptions o;
    o.width = 64;
    o.height = 128;
    o.steps = 2;
    o.seed = 3;
    std::vector<std::string> stages;
    o.progress = [&](const std::string& s, double) { stages.push_back(s); };
    const Image a = textToImage(dir, "hello he", o);
    CHECK(a.width == 64 && a.height == 128 && a.channels == 3 && a.pixels.size() == 64 * 128 * 3);
    CHECK(stages == std::vector<std::string>({"text encoder", "load unet", "step 1/2", "step 2/2", "vae decode"}));
    o.progress = nullptr;
    CHECK(textToImage(dir, "hello he", o).pixels == a.pixels);   // same seed, same image
    o.seed = 4;
    CHECK(textToImage(dir, "hello he", o).pixels != a.pixels);   // another seed, another image
    o.guidance = 3.0f;                                           // classifier-free guidance runs both prompts
    o.negativePrompt = "he";
    CHECK(textToImage(dir, "hello he", o).pixels.size() == a.pixels.size());
    // The tiny autoencoder decodes the same latent instead of the full VAE.
    o.guidance = 0.0f;
    o.seed = 3;
    o.tinyVae = dir + "/taesd";
    writeTinyTaesd(o.tinyVae, false);
    stages.clear();
    o.progress = [&](const std::string& s, double) { stages.push_back(s); };
    const Image tiny = textToImage(dir, "hello he", o);
    CHECK(tiny.width == 64 && tiny.height == 128 && stages.back() == "taesd decode");
    // Image to image: strength picks how many of the steps run; 0 only encodes and decodes.
    Image input;
    input.width = 100;
    input.height = 150;
    input.channels = 3;
    for (std::size_t i = 0; i < 100 * 150 * 3; ++i) input.pixels.push_back(static_cast<std::uint8_t>(i * 13 % 256));
    o.steps = 4;
    for (float strength : {0.0f, 0.5f, 1.0f}) {
        o.strength = strength;
        stages.clear();
        const Image out = imageToImage(dir, input, "hello he", o);
        CHECK(out.width == 64 && out.height == 128);
        const std::size_t steps = static_cast<std::size_t>(std::count_if(stages.begin(), stages.end(), [](const std::string& s) {
            return s.rfind("step ", 0) == 0;
        }));
        CHECK(steps == static_cast<std::size_t>(4 * strength));
        CHECK(stages.front() == "taesd encode" && stages.back() == "taesd decode");
    }
    o.strength = 1.5f;
    CHECK_THROWS(imageToImage(dir, input, "hello", o), std::invalid_argument);
    o.progress = nullptr;
    o.width = 60;
    CHECK_THROWS(textToImage(dir, "hello", o), std::invalid_argument);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(tok);
}

TEST(stable_diffusion_session_and_animations) {
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_anim", tok);
    TextToImageOptions o;
    o.width = 64;
    o.height = 64;
    o.tinyVae = dir + "/taesd";
    writeTinyTaesd(o.tinyVae, false);
    std::vector<std::string> stages;
    o.progress = [&](const std::string& s, double) { stages.push_back(s); };
    StableDiffusion sd(dir, o);
    CHECK(sd.width() == 64 && sd.height() == 64);
    CHECK(stages == std::vector<std::string>({"load text encoder", "load unet", "load autoencoder"}));
    // The session gives the same pictures as the one-shot functions.
    o.progress = nullptr;
    o.steps = 2;
    o.seed = 5;
    CHECK(sd.generate("hello he", 5, 2).pixels == textToImage(dir, "hello he", o).pixels);
    Image input;
    input.width = 80;
    input.height = 70;
    for (std::size_t i = 0; i < 80 * 70 * 3; ++i) input.pixels.push_back(static_cast<std::uint8_t>(i * 7 % 256));
    o.steps = 4;
    o.strength = 0.5f;
    CHECK(sd.transform(input, "hello", 5, 4, 0.5f).pixels == imageToImage(dir, input, "hello", o).pixels);
    CHECK_THROWS(sd.transform(input, "hello", 5, 4, 2.0f), std::invalid_argument);
    const Matrix z = sd.encodeImage(input);
    CHECK(z.rows() == 1 && z.cols() == 4 * 8 * 8);
    CHECK(sd.noise(1).values() == sd.noise(1).values() && sd.noise(1).values() != sd.noise(2).values());
    StableDiffusion decoderOnly(dir, o, false);
    CHECK_THROWS(decoderOnly.encodeImage(input), std::logic_error);

    // Morph: keyframes are hit exactly, frames per segment and looping add up.
    MorphOptions mo;
    mo.framesPerSegment = 3;
    stages.clear();
    const auto loop = morphAnimation(sd, {"hello", "he"}, {1, 2}, mo);
    CHECK(loop.size() == 6);
    CHECK(loop[0].pixels == sd.generate("hello", 1).pixels);
    CHECK(loop[3].pixels == sd.generate("he", 2).pixels);
    CHECK(loop[1].pixels != loop[0].pixels && loop[1].pixels != loop[3].pixels);
    mo.loop = false;
    const auto open = morphAnimation(sd, {"hello", "he", "hello he"}, {7}, mo);
    CHECK(open.size() == 7);
    CHECK(open.back().pixels == sd.generate("hello he", 7).pixels);
    CHECK_THROWS(morphAnimation(sd, {"a", "b"}, {1, 2, 3}, mo), std::invalid_argument);
    CHECK_THROWS(morphAnimation(sd, {}, {1}, mo), std::invalid_argument);

    // Zoom: a start image is kept as frame 0, every frame has the session's size,
    // and the run is deterministic.
    ZoomOptions zo;
    zo.frames = 3;
    const auto zoom = zoomAnimation(sd, {"hello", "he"}, zo, &input);
    CHECK(zoom.size() == 3);
    CHECK(zoom[0].pixels == fitImage(input, 64, 64).pixels);
    for (const Image& f : zoom) CHECK(f.width == 64 && f.height == 64 && f.channels == 3);
    CHECK(zoomAnimation(sd, {"hello", "he"}, zo, &input)[2].pixels == zoom[2].pixels);
    zo.frames = 2;
    CHECK(zoomAnimation(sd, {"hello"}, zo).size() == 2);  // generated first frame
    zo.frames = 0;
    CHECK_THROWS(zoomAnimation(sd, {"hello"}, zo), std::invalid_argument);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(tok);
}

// An AnimateDiff motion adapter for the tiny UNet (layers_per_block L).
std::map<std::string, Matrix> tinyMotionWeights(const std::vector<std::size_t>& ch, std::size_t L) {
    std::map<std::string, Matrix> w;
    auto module = [&](const std::string& p, std::size_t C) {
        addTensor(w, p + ".norm.weight", C, 0.8f, 1.2f);
        addTensor(w, p + ".norm.bias", C, -0.1f, 0.1f);
        for (const char* n : {".proj_in", ".proj_out"}) {
            addTensor(w, p + n + ".weight", C * C, -0.3f, 0.3f);
            addTensor(w, p + n + ".bias", C, -0.1f, 0.1f);
        }
        const std::string b = p + ".transformer_blocks.0";
        for (const char* n : {".norm1", ".norm2", ".norm3"}) {
            addTensor(w, b + n + ".weight", C, 0.8f, 1.2f);
            addTensor(w, b + n + ".bias", C, -0.1f, 0.1f);
        }
        for (const char* a : {".attn1", ".attn2"}) {
            for (const char* n : {".to_q", ".to_k", ".to_v"}) addTensor(w, b + a + n + ".weight", C * C, -0.5f, 0.5f);
            addTensor(w, b + a + ".to_out.0.weight", C * C, -0.3f, 0.3f);
            addTensor(w, b + a + ".to_out.0.bias", C, -0.1f, 0.1f);
        }
        addTensor(w, b + ".ff.net.0.proj.weight", 8 * C * C, -0.3f, 0.3f);
        addTensor(w, b + ".ff.net.0.proj.bias", 8 * C, -0.1f, 0.1f);
        addTensor(w, b + ".ff.net.2.weight", 4 * C * C, -0.3f, 0.3f);
        addTensor(w, b + ".ff.net.2.bias", C, -0.1f, 0.1f);
    };
    for (std::size_t i = 0; i < ch.size(); ++i)
        for (std::size_t j = 0; j < L; ++j) module("down_blocks." + std::to_string(i) + ".motion_modules." + std::to_string(j), ch[i]);
    module("mid_block.motion_modules.0", ch.back());
    for (std::size_t i = 0; i < ch.size(); ++i)
        for (std::size_t j = 0; j <= L; ++j)
            module("up_blocks." + std::to_string(i) + ".motion_modules." + std::to_string(j), ch[ch.size() - 1 - i]);
    return w;
}

TEST(gpu_motion_modules_match_cpu) {
    // CUDA runs a clip with motion modules as one device batch; it must agree with the CPU.
    if (!gpuAvailable() || gpuBackendName() != "CUDA") return;
    setSeed(83);
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("gpu_motion", tok);
    saveAll(dir + "/motion.safetensors", tinyMotionWeights({8, 16}, 1));
    UNet2DCondition unet = loadUNet(dir, 8, 8);
    loadMotionModules(unet, dir + "/motion.safetensors");
    const std::size_t n = 4 * 8 * 8, F = 5;
    const Matrix latents = Matrix::normal(F, n, 0.0f, 1.0f), a = Matrix::uniform(8, 6, -1, 1), b = Matrix::uniform(8, 6, -1, 1);
    const std::vector<Matrix> contexts{a, b, a, b, a};  // prompt travel: one condition per frame
    setDevice(Device::CPU);
    const Matrix cpu = unet.forwardFrames(latents, 500.0f, contexts);
    setDevice(Device::GPU);
    Matrix gpu;
    {
        GpuWeightScope scope;
        gpu = unet.forwardFrames(latents, 500.0f, contexts);
    }
    setDevice(Device::CPU);
    double scale = 1e-3, worst = 0;
    for (std::size_t i = 0; i < cpu.size(); ++i) scale = std::max(scale, static_cast<double>(std::fabs(cpu.data()[i])));
    for (std::size_t i = 0; i < cpu.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(cpu.data()[i] - gpu.data()[i])));
    if (worst > 2e-3 * scale) std::cout << "         motion frames differ by " << worst << " (scale " << scale << ")\n";
    CHECK(gpu.sameShape(cpu) && worst <= 2e-3 * scale);
}

TEST(animatediff_motion_modules_and_text_to_video) {
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_video", tok);
    std::filesystem::create_directories(dir + "/motion");
    const std::string motion = dir + "/motion/tiny.safetensors";
    saveAll(motion, tinyMotionWeights({8, 16}, 1));
    UNet2DCondition plain = loadUNet(dir, 8, 8);
    UNet2DCondition unet = loadUNet(dir, 8, 8);
    CHECK(!unet.hasMotionModules());
    loadMotionModules(unet, motion);
    CHECK(unet.hasMotionModules() && unet.parameterCount() > plain.parameterCount());
    const std::size_t n = 4 * 8 * 8, F = 4;
    const Matrix latents = Matrix::normal(F, n, 0.0f, 1.0f), ctx = Matrix::uniform(8, 6, -1.0f, 1.0f);
    // Without motion modules, frames are independent forward() calls.
    const Matrix alone = plain.forwardFrames(latents, 500.0f, ctx);
    Matrix first(1, n);
    std::copy(latents.data() + n, latents.data() + 2 * n, first.data());
    const Matrix one = plain.forward(first, 500.0f, ctx);
    for (std::size_t k = 0; k < n; ++k) CHECK_NEAR(alone(1, k), one(0, k), 1e-5);
    // Motion modules whose output projection is zero add nothing, so the batched spatial path
    // (frames stacked through resnets, transformers and convolutions) must equal per-frame forward().
    {
        auto silent = tinyMotionWeights({8, 16}, 1);
        for (auto& [name, m] : silent)
            if (name.find("proj_out") != std::string::npos) m = Matrix(m.rows(), m.cols());
        const std::string quiet = dir + "/motion_quiet.safetensors";
        saveAll(quiet, silent);
        UNet2DCondition still = loadUNet(dir, 8, 8);
        loadMotionModules(still, quiet);
        const Matrix batched = still.forwardFrames(latents, 500.0f, ctx);
        float worst = 0;
        for (std::size_t f = 0; f < F; ++f) {
            Matrix lat(1, n);
            std::copy(latents.data() + f * n, latents.data() + (f + 1) * n, lat.data());
            const Matrix single = plain.forward(lat, 500.0f, ctx);
            for (std::size_t k = 0; k < n; ++k) worst = std::max(worst, std::fabs(batched(f, k) - single(0, k)));
        }
        CHECK(worst < 1e-4f);
        std::filesystem::remove(quiet);
    }
    // With them, every frame sees the others: changing frame 0 changes frame 3.
    const Matrix mixed = unet.forwardFrames(latents, 500.0f, ctx);
    CHECK(mixed.rows() == F && mixed.cols() == n);
    Matrix changed = latents;
    for (std::size_t k = 0; k < n; ++k) changed(0, k) += 0.5f;
    const Matrix after = unet.forwardFrames(changed, 500.0f, ctx);
    double diff = 0, plainDiff = 0;
    const Matrix plainAfter = plain.forwardFrames(changed, 500.0f, ctx);
    for (std::size_t k = 0; k < n; ++k) {
        diff += std::abs(after(3, k) - mixed(3, k));
        plainDiff += std::abs(plainAfter(3, k) - alone(3, k));
    }
    CHECK(diff > 1e-3 && plainDiff < 1e-6);
    CHECK(unet.forwardFrames(latents, 500.0f, ctx).values() == mixed.values());  // deterministic
    CHECK_THROWS(unet.forwardFrames(Matrix(33, n), 500.0f, ctx), std::invalid_argument);
    // Per-frame text conditions: identical ones equal the shared one; another one changes its frame.
    CHECK(unet.forwardFrames(latents, 500.0f, std::vector<Matrix>(F, ctx)).values() == mixed.values());
    std::vector<Matrix> travel(F, ctx);
    travel[3] = Matrix::uniform(8, 6, -1.0f, 1.0f);
    const Matrix travelled = unet.forwardFrames(latents, 500.0f, travel);
    double changed3 = 0;
    for (std::size_t k = 0; k < n; ++k) changed3 += std::abs(travelled(3, k) - mixed(3, k));
    CHECK(changed3 > 1e-3);
    CHECK_THROWS(unet.forwardFrames(latents, 500.0f, std::vector<Matrix>(2, ctx)), std::invalid_argument);
    // An adapter without the mid block module (AnimateDiff v1) is refused.
    auto v1 = tinyMotionWeights({8, 16}, 1);
    for (auto it = v1.begin(); it != v1.end();) it = it->first.rfind("mid_block", 0) == 0 ? v1.erase(it) : std::next(it);
    saveAll(dir + "/v1.safetensors", v1);
    UNet2DCondition other = loadUNet(dir, 8, 8);
    CHECK_THROWS(loadMotionModules(other, dir + "/v1.safetensors"), std::runtime_error);

    // The pipeline: frames of the requested size, reproducible by seed.
    TextToVideoOptions o;
    o.width = o.height = 64;
    o.frames = 3;
    o.steps = 2;
    std::vector<std::string> stages;
    o.progress = [&](const std::string& s, double) { stages.push_back(s); };
    const auto video = textToVideo(dir, "hello he", o);
    CHECK(video.size() == 3);
    for (const Image& f : video) CHECK(f.width == 64 && f.height == 64 && f.channels == 3);
    CHECK(stages == std::vector<std::string>({"text encoder", "load unet + motion", "step 1/2", "step 2/2", "vae decode"}));
    o.progress = nullptr;
    CHECK(textToVideo(dir, "hello he", o)[2].pixels == video[2].pixels);
    // Prompt travel: frame 0 keeps the main prompt, later frames move towards the keyframe.
    o.promptAt = {{2, "he"}};
    const auto travelVideo = textToVideo(dir, "hello he", o);
    CHECK(travelVideo.size() == 3 && travelVideo[2].pixels != video[2].pixels);
    o.promptAt = {{3, "he"}};
    CHECK_THROWS(textToVideo(dir, "hello he", o), std::invalid_argument);
    o.promptAt.clear();
    o.seed = 9;
    CHECK(textToVideo(dir, "hello he", o)[0].pixels != video[0].pixels);
    // Longer than the context: overlapping windows, same frame count and size.
    o.frames = 5;
    o.context = 3;
    o.overlap = 1;
    const auto windows = textToVideo(dir, "hello he", o);
    CHECK(windows.size() == 5 && windows[4].width == 64);
    o.context = 3;
    o.overlap = 3;
    CHECK_THROWS(textToVideo(dir, "hello", o), std::invalid_argument);
    o.context = 16;
    o.overlap = 4;
    // Image to video: one picture for all frames; strength 0.5 runs 1 of 2 steps.
    Image still;
    still.width = 70;
    still.height = 60;
    for (std::size_t i = 0; i < 70 * 60 * 3; ++i) still.pixels.push_back(static_cast<std::uint8_t>(i * 11 % 256));
    o.frames = 3;
    o.init = {still};
    o.strength = 0.5f;
    stages.clear();
    o.progress = [&](const std::string& s, double) { stages.push_back(s); };
    CHECK(textToVideo(dir, "hello", o).size() == 3);
    CHECK(stages == std::vector<std::string>({"text encoder", "load unet + motion", "vae encode", "step 2/2", "vae decode"}));
    o.progress = nullptr;
    o.init = {still, still, still};  // video to video
    CHECK(textToVideo(dir, "hello", o).size() == 3);
    o.init = {still, still};
    CHECK_THROWS(textToVideo(dir, "hello", o), std::invalid_argument);
    o.init.clear();
    o.frames = 2;
    o.motion = dir + "/missing.safetensors";
    CHECK_THROWS(textToVideo(dir, "hello", o), std::exception);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(tok);
}

namespace {
// A tiny CLIP text encoder (diffusers names); with projection > 0, a CLIPTextModelWithProjection.
std::map<std::string, Matrix> tinyClipWeights(std::size_t D, std::size_t layers, std::size_t projection) {
    std::map<std::string, Matrix> w;
    const std::string p = "text_model.";
    addTensor(w, p + "embeddings.token_embedding.weight", 518 * D, -1, 1);
    addTensor(w, p + "embeddings.position_embedding.weight", 8 * D, -1, 1);
    for (std::size_t i = 0; i < layers; ++i) {
        const std::string l = p + "encoder.layers." + std::to_string(i) + ".";
        for (const char* n : {"layer_norm1", "layer_norm2"}) {
            addTensor(w, l + n + ".weight", D, 0.8f, 1.2f);
            addTensor(w, l + n + ".bias", D, -0.1f, 0.1f);
        }
        for (const char* n : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            addTensor(w, l + "self_attn." + n + ".weight", D * D, -0.4f, 0.4f);
            addTensor(w, l + "self_attn." + n + ".bias", D, -0.1f, 0.1f);
        }
        addTensor(w, l + "mlp.fc1.weight", 2 * D * D, -0.4f, 0.4f);
        addTensor(w, l + "mlp.fc1.bias", 2 * D, -0.1f, 0.1f);
        addTensor(w, l + "mlp.fc2.weight", 2 * D * D, -0.4f, 0.4f);
        addTensor(w, l + "mlp.fc2.bias", D, -0.1f, 0.1f);
    }
    addTensor(w, p + "final_layer_norm.weight", D, 0.8f, 1.2f);
    addTensor(w, p + "final_layer_norm.bias", D, -0.1f, 0.1f);
    if (projection) w["text_projection.weight"] = Matrix::uniform(projection, D, -0.5f, 0.5f);
    return w;
}

// A tiny SDXL folder: two text encoders (6 + 8 = 14 context channels, the second pooled to 5), a UNet with
// two transformer blocks in its attention level and the "text_time" added conditions.
std::string writeTinySdxl(const std::string& name) {
    std::string tok;
    const std::string dir = writeTinyStableDiffusion(name, tok);  // tokenizer, vae, scheduler
    std::filesystem::remove_all(dir + "/unet");
    std::filesystem::create_directories(dir + "/tokenizer_2");
    for (const char* f : {"vocab.json", "merges.txt", "tokenizer_config.json"})
        std::filesystem::copy_file(dir + "/tokenizer/" + f, dir + "/tokenizer_2/" + f, std::filesystem::copy_options::overwrite_existing);
    std::ofstream(dir + "/tokenizer/tokenizer_config.json") << R"({"pad_token":"<|endoftext|>"})";
    auto clip = [&](const std::string& sub, std::size_t D, std::size_t P) {
        std::filesystem::create_directories(dir + "/" + sub);
        std::ofstream(dir + "/" + sub + "/config.json") << R"({"hidden_act":"gelu","vocab_size":518,"hidden_size":)" << D
            << R"(,"num_attention_heads":2,"intermediate_size":)" << 2 * D << R"(,"num_hidden_layers":2,"max_position_embeddings":8)"
            << (P ? R"(,"projection_dim":)" + std::to_string(P) : std::string()) << "}";
        saveAll(dir + "/" + sub + "/model.safetensors", tinyClipWeights(D, 2, P));
    };
    clip("text_encoder", 6, 0);
    clip("text_encoder_2", 8, 5);
    std::filesystem::create_directories(dir + "/unet");
    std::ofstream(dir + "/unet/config.json")
        << R"({"_class_name":"UNet2DConditionModel","act_fn":"silu","attention_head_dim":[2,2],"block_out_channels":[8,16],)"
        << R"("cross_attention_dim":14,"down_block_types":["CrossAttnDownBlock2D","DownBlock2D"],"flip_sin_to_cos":true,)"
        << R"("freq_shift":0,"in_channels":4,"layers_per_block":1,"mid_block_type":"UNetMidBlock2DCrossAttn","norm_eps":1e-5,)"
        << R"("norm_num_groups":4,"out_channels":4,"up_block_types":["UpBlock2D","CrossAttnUpBlock2D"],"use_linear_projection":true,)"
        << R"("transformer_layers_per_block":[2,1],"addition_embed_type":"text_time","addition_time_embed_dim":4,)"
        << R"("projection_class_embeddings_input_dim":29})";
    std::map<std::string, Matrix> w = tinyUNetWeights({8, 16}, 1, 14);
    std::vector<std::pair<std::string, Matrix>> extra;
    for (const auto& [n, m] : w) {
        const std::size_t at = n.find(".transformer_blocks.0.");
        if (at != std::string::npos && n.rfind("mid_block", 0) != 0)  // a second block in the level-0 attentions
            extra.emplace_back(n.substr(0, at) + ".transformer_blocks.1." + n.substr(at + 22), Matrix::uniform(1, m.size(), -0.3f, 0.3f));
    }
    for (auto& [n, m] : extra) w[n] = std::move(m);
    addTensor(w, "add_embedding.linear_1.weight", 32 * 29, -0.3f, 0.3f);
    addTensor(w, "add_embedding.linear_1.bias", 32, -0.1f, 0.1f);
    addTensor(w, "add_embedding.linear_2.weight", 32 * 32, -0.3f, 0.3f);
    addTensor(w, "add_embedding.linear_2.bias", 32, -0.1f, 0.1f);
    saveAll(dir + "/unet/diffusion_pytorch_model.safetensors", w);
    std::filesystem::remove_all(tok);
    return dir;
}
} // namespace

TEST(sdxl_pipeline_with_two_text_encoders_and_added_conditions) {
    setSeed(31);
    const std::string dir = writeTinySdxl("sdxl_tiny");
    CHECK(isSdxlModel(dir));
    // The UNet needs the added conditions, and they change its prediction.
    {
        UNet2DCondition unet = loadUNet(dir, 4, 4);
        CHECK(unet.needsAddedConditions());
        const Matrix z = Matrix::uniform(1, 4 * 16, -1, 1), ctx = Matrix::uniform(8, 14, -1, 1);
        CHECK_THROWS(unet.forward(z, 500.0f, ctx), std::logic_error);
        unet.setAddedConditions(Matrix::uniform(1, 5, -1, 1), {64, 64, 0, 0, 64, 64});
        const Matrix a = unet.forward(z, 500.0f, ctx);
        unet.setAddedConditions(Matrix::uniform(1, 5, -1, 1), {64, 64, 0, 0, 64, 64});
        CHECK(!unet.forward(z, 500.0f, ctx).approxEquals(a, 1e-5f));
        CHECK_THROWS(unet.setAddedConditions(Matrix(1, 4), {1, 2, 3, 4, 5, 6}), std::invalid_argument);
    }
    // The second encoder pools the end-of-text state through its projection.
    {
        ClipTextEncoder e2 = loadClipTextEncoder(dir, false, {}, "text_encoder_2");
        CHECK(e2.projection.rows() == 8 && e2.projection.cols() == 5);
        const ClipTextEncoder::Penultimate p = e2.encodePenultimate({5, 9, 7, 0, 0, 0, 0, 0});
        CHECK(p.hidden.rows() == 8 && p.hidden.cols() == 8 && p.pooled.cols() == 5);
        CHECK(!p.hidden.approxEquals(e2.encode({5, 9, 7, 0, 0, 0, 0, 0}), 1e-4f));  // not the last, normed states
    }
    TextToImageOptions o;
    o.width = o.height = 64;
    o.steps = 2;
    o.seed = 4;
    o.guidance = 2.0f;  // the empty negative prompt is zeros
    const Image a = textToImage(dir, "hello he", o);
    CHECK(a.width == 64 && textToImage(dir, "hello he", o).pixels == a.pixels);
    CHECK(textToImage(dir, "hello", o).pixels != a.pixels);
    o.width = 128;  // the size numbers follow the picture
    CHECK(textToImage(dir, "hello he", o).width == 128);
    o.width = 64;
    CHECK(imageToImage(dir, a, "hello", o).pixels.size() == a.pixels.size());

    // The same model as one SDXL file: conditioner.embedders.0 (CLIP-L names) and .1 (OpenCLIP, x @ P projection), label_emb.
    std::map<std::string, Matrix> all;
    for (auto& [n, m] : loadAllTensors(dir + "/unet/diffusion_pytorch_model.safetensors")) {
        if (n.rfind("add_embedding.linear_1.", 0) == 0) all["model.diffusion_model.label_emb.0.0." + n.substr(23)] = m;
        else if (n.rfind("add_embedding.linear_2.", 0) == 0) all["model.diffusion_model.label_emb.0.2." + n.substr(23)] = m;
        else all[ldmUnetName(n, 1, {false, true})] = m;
    }
    for (auto& [n, m] : loadAllTensors(dir + "/vae/diffusion_pytorch_model.safetensors")) all[ldmVaeName(n, 4)] = m;
    for (auto& [n, m] : loadAllTensors(dir + "/text_encoder/model.safetensors")) all["conditioner.embedders.0.transformer." + n] = m;
    {
        const auto t = loadAllTensors(dir + "/text_encoder_2/model.safetensors");
        const std::string P = "conditioner.embedders.1.model.", T = "text_model.";
        all[P + "token_embedding.weight"] = t.at(T + "embeddings.token_embedding.weight");
        all[P + "positional_embedding"] = t.at(T + "embeddings.position_embedding.weight");
        all[P + "ln_final.weight"] = t.at(T + "final_layer_norm.weight");
        all[P + "ln_final.bias"] = t.at(T + "final_layer_norm.bias");
        { const Matrix& tp = t.at("text_projection.weight"); all[P + "text_projection"] = Matrix(5, 8, std::vector<float>(tp.data(), tp.data() + tp.size())).transpose(); }
        for (int layer = 0; layer < 2; ++layer) {
            const std::string R = P + "transformer.resblocks." + std::to_string(layer) + ".", L = T + "encoder.layers." + std::to_string(layer) + ".";
            std::vector<float> wv, bv;
            for (const char* k : {"q_proj", "k_proj", "v_proj"}) {
                const Matrix& a1 = t.at(L + "self_attn." + k + ".weight");
                const Matrix& b1 = t.at(L + "self_attn." + k + ".bias");
                wv.insert(wv.end(), a1.values().begin(), a1.values().end());
                bv.insert(bv.end(), b1.values().begin(), b1.values().end());
            }
            all[R + "attn.in_proj_weight"] = Matrix(1, wv.size(), std::move(wv));
            all[R + "attn.in_proj_bias"] = Matrix(1, bv.size(), std::move(bv));
            const std::vector<std::pair<std::string, std::string>> map = {{"ln_1.", "layer_norm1."}, {"ln_2.", "layer_norm2."},
                {"attn.out_proj.", "self_attn.out_proj."}, {"mlp.c_fc.", "mlp.fc1."}, {"mlp.c_proj.", "mlp.fc2."}};
            for (const auto& [to, from] : map)
                for (const char* wb : {"weight", "bias"}) all[R + to + wb] = t.at(L + from + wb);
        }
    }
    const std::string ckdir = tempPath("sdxl_single");
    std::filesystem::create_directories(ckdir + "/tokenizer");
    for (const char* f : {"vocab.json", "merges.txt"})
        std::filesystem::copy_file(dir + "/tokenizer/" + f, ckdir + "/tokenizer/" + f, std::filesystem::copy_options::overwrite_existing);
    const std::string file = ckdir + "/xl.safetensors";
    saveAll(file, all);
    std::ofstream(ckdir + "/xl.json") << R"({"unet": {"attention_head_dim": [2, 2], "norm_num_groups": 4, "addition_time_embed_dim": 4},
        "vae": {"norm_num_groups": 4, "scaling_factor": 0.18215}, "text_encoder": {"num_attention_heads": 2, "hidden_act": "gelu"},
        "text_encoder_2": {"num_attention_heads": 2}, "scheduler": {"beta_schedule": "scaled_linear", "timestep_spacing": "trailing"}})";
    const CheckpointInfo info = inspectCheckpoint(file);
    CHECK(info.family == "sdxl" && info.contextDim == 14 && isSdxlModel(file));
    const ModelComponent u = modelComponent(file, "unet", "test");
    CHECK(u.config.at("transformer_layers_per_block").array[0].number == 2 && u.config.at("addition_embed_type").string == "text_time");
    CHECK(u.config.at("projection_class_embeddings_input_dim").number == 29);
    CHECK(textToImage(file, "hello he", o).pixels == a.pixels);
    bool listed = false;
    for (const ModelInfo& m : findModels(ckdir))
        if (m.path == file) listed = m.kind == "sdxl";
    CHECK(listed);
    std::filesystem::remove_all(ckdir);
    std::filesystem::remove_all(dir);
}

// Every case in tests/onnx/cases (written by tests/onnx/make_cases.py with onnxruntime's outputs).
TEST(onnx_operators_match_onnxruntime) {
    namespace fs = std::filesystem;
    const fs::path root = fs::path(NEURACRAFT_SOURCE_DIR) / "tests" / "onnx" / "cases";
    std::size_t cases = 0;
    std::vector<std::string> failed;
    std::vector<fs::path> dirs;
    for (const auto& e : fs::directory_iterator(root)) dirs.push_back(e.path());
    std::sort(dirs.begin(), dirs.end());
    for (const fs::path& dir : dirs) {
        ++cases;
        try {
            const onnx::Model model = onnx::Model::load((dir / "model.onnx").string());
            std::map<std::string, onnx::Tensor> feeds;
            for (int k = 0; fs::exists(dir / ("input_" + std::to_string(k) + ".pb")); ++k) {
                std::string name;
                onnx::Tensor t = onnx::loadTensorProto((dir / ("input_" + std::to_string(k) + ".pb")).string(), &name);
                feeds[name] = std::move(t);
            }
            std::vector<std::pair<std::string, onnx::Tensor>> expected;
            std::vector<std::string> names;
            for (int k = 0; fs::exists(dir / ("output_" + std::to_string(k) + ".pb")); ++k) {
                std::string name;
                onnx::Tensor t = onnx::loadTensorProto((dir / ("output_" + std::to_string(k) + ".pb")).string(), &name);
                names.push_back(name);
                expected.emplace_back(name, std::move(t));
            }
            const auto got = model.run(feeds, names);
            for (const auto& [name, want] : expected) {
                const onnx::Tensor& g = got.at(name);
                if (g.shape != want.shape) throw std::runtime_error(name + ": shape differs");
                if (want.kind == onnx::Tensor::Kind::Int) {
                    if (g.i != want.i) throw std::runtime_error(name + ": integer values differ");
                } else {
                    for (std::size_t k = 0; k < want.f.size(); ++k)
                        if (!(std::fabs(g.f[k] - want.f[k]) <= 1e-4f + 1e-4f * std::fabs(want.f[k])))
                            throw std::runtime_error(name + "[" + std::to_string(k) + "] = " + std::to_string(g.f[k]) + ", onnxruntime " + std::to_string(want.f[k]));
                }
            }
        } catch (const std::exception& e) {
            failed.push_back(dir.filename().string() + ": " + e.what());
        }
    }
    for (const std::string& f : failed) std::cerr << "  onnx case " << f << "\n";
    CHECK(cases >= 100);
    CHECK(failed.empty());
}

TEST(onnx_picture_pipeline_and_workflow_step) {
    namespace fs = std::filesystem;
    const fs::path src = fs::path(NEURACRAFT_SOURCE_DIR) / "tests" / "onnx" / "pipeline";
    const std::string dir = tempPath("onnx_pipeline");
    fs::create_directories(dir + "/models");
    for (const char* m : {"up2_fixed8.onnx", "mean_mask.onnx", "identity.onnx"}) fs::copy_file(src / m, dir + "/models/" + m, fs::copy_options::overwrite_existing);
    Image pic;
    pic.width = 20;
    pic.height = 12;
    pic.pixels.resize(20 * 12 * 3);
    for (std::size_t k = 0; k < pic.pixels.size(); ++k) pic.pixels[k] = static_cast<std::uint8_t>((k * 37 + k / 7) % 256);
    // A fixed 8 x 8 x2 model goes over 20 x 12 in overlapping tiles: exactly the nearest-neighbour x2 picture.
    {
        const onnx::Model up = onnx::Model::load(dir + "/models/up2_fixed8.onnx");
        const onnx::OnnxPipeline p = onnx::OnnxPipeline::fromModel(up, onnx::onnxManifest(dir + "/models/up2_fixed8.onnx"));
        CHECK(p.width == 8 && p.height == 8 && p.fixedByModel);
        const Image big = p.run(up, pic);
        CHECK(big.width == 40 && big.height == 24 && big.channels == 3);
        bool same = true;
        for (std::size_t y = 0; y < 24; ++y)
            for (std::size_t x = 0; x < 40; ++x)
                for (std::size_t c = 0; c < 3; ++c) same &= big.pixels[(y * 40 + x) * 3 + c] == pic.pixels[((y / 2) * 20 + x / 2) * 3 + c];
        CHECK(same);
        onnx::OnnxPipeline whole = p;
        whole.tileOverlap = 0;  // no tiles: the picture is squeezed to 8 x 8, then comes back at 2x its size
        CHECK(whole.run(up, pic).width == 40);
    }
    // Identity with BGR input only: red and blue swap; in and out BGR: unchanged. Fixed range [0, 1].
    {
        const onnx::Model id = onnx::Model::load(dir + "/models/identity.onnx");
        std::ofstream(dir + "/models/identity.json") << R"({"input": {"channels": "BGR", "range": [-1, 1]}, "output": {"range": [-1, 1]}})";
        onnx::OnnxPipeline p = onnx::OnnxPipeline::fromModel(id, onnx::onnxManifest(dir + "/models/identity.onnx"));
        const Image swapped = p.run(id, pic);
        CHECK(swapped.width == 20 && swapped.pixels[0] == pic.pixels[2] && swapped.pixels[2] == pic.pixels[0] && swapped.pixels[1] == pic.pixels[1]);
        p.bgrOut = true;
        CHECK(p.run(id, pic).pixels == pic.pixels);
    }
    // A mask of any size, stretched min..max by default; "apply" cuts the picture out (RGBA).
    {
        const onnx::Model mm = onnx::Model::load(dir + "/models/mean_mask.onnx");
        onnx::OnnxPipeline p = onnx::OnnxPipeline::fromModel(mm, json::makeObject({}));
        CHECK(p.width == 0);
        const Image m = p.run(mm, pic);
        CHECK(m.width == 20 && m.height == 12 && m.channels == 3);
        const auto [lo, hi] = std::minmax_element(m.pixels.begin(), m.pixels.end());
        CHECK(*lo == 0 && *hi == 255);
        p.applyMask = true;
        const Image cut = p.run(mm, pic);
        CHECK(cut.channels == 4 && cut.pixels[0] == pic.pixels[0] && cut.pixels[3] == m.pixels[0]);
    }
    // The workflow step and the model listing.
    {
        savePng(dir + "/in.png", pic);
        const auto clips = runWorkflow(R"({"steps": [
            {"op": "load", "path": "in.png", "out": "pic"},
            {"op": "onnx", "model": "models/up2_fixed8.onnx", "in": "pic", "out": "big"},
            {"op": "onnx", "model": "models/mean_mask.onnx", "in": "pic", "apply": true, "out": "cut"},
            {"op": "onnx", "model": "models/mean_mask.onnx", "in": "pic", "invert": true, "out": "inv"}]})", dir);
        CHECK(clips.at("big").front().width == 40 && clips.at("cut").front().channels == 4);
        CHECK(clips.at("inv").front().pixels[0] == 255 - onnx::OnnxPipeline::fromModel(onnx::Model::load(dir + "/models/mean_mask.onnx"), json::makeObject({}))
                                                           .run(onnx::Model::load(dir + "/models/mean_mask.onnx"), pic).pixels[0]);
        CHECK(checkWorkflow(R"({"steps": [{"op": "onnx", "model": "m.onnx", "in": "x", "kind": "picture"}]})").size() >= 1);
        std::size_t listed = 0;
        for (const ModelInfo& m : findModels(dir))
            if (m.kind == "onnx") ++listed;
        CHECK(listed == 3);
    }
    fs::remove_all(dir);
}

// Running other people's workflows: files only inside the allowed folders, bounded sizes and time,
// files renamed for a service, and the readers' guards against hostile files.
TEST(model_cache_keeps_recent_models_when_asked) {
    ModelCacheScope scope;
    auto load = [](int v) { return [v] { return std::make_shared<int>(v); }; };
    cachedModel<int>("a", "A", load(1));
    cachedModel<int>("b", "B", load(2));
    cachedModel<int>("c", "C", load(3));
    cachedModel<int>("a", "A", load(9));  // a used again: the most recent
    setModelCacheKeep(1);
    releaseModelsExcept({"C"});
    CHECK(modelCacheEntries() == 2);  // C (kept) and A (most recent of the rest)
    CHECK(*cachedModel<int>("a", "A", load(7)) == 1);
    setModelCacheKeep(0);
    releaseModelsExcept({});
    CHECK(modelCacheEntries() == 0);
}

TEST(workflow_sandbox_and_limits) {
    namespace fs = std::filesystem;
    const std::string dir = tempPath("sandbox");
    fs::remove_all(dir);
    fs::create_directories(dir + "/in");
    fs::create_directories(dir + "/out");
    fs::create_directories(dir + "/secret");
    Image pic;
    pic.width = 8;
    pic.height = 6;
    pic.channels = 3;
    pic.pixels.assign(8 * 6 * 3, 100);
    savePng(dir + "/in/a.png", pic);
    savePng(dir + "/secret/b.png", pic);
    fs::create_symlink(dir + "/secret/b.png", dir + "/in/link.png");
    WorkflowLimits lim;
    lim.readRoots = {dir + "/in"};
    lim.writeRoot = dir + "/out";
    const WorkflowEvents ev;
    // allowed: read in/, write out/
    const auto ok = runWorkflow(R"({"steps": [{"op": "load", "path": "in/a.png", "out": "p"},
        {"op": "save", "in": "p", "path": "out/a.png"}]})", dir, ev, nullptr, lim);
    CHECK(ok.at("p").front().width == 8 && fs::exists(dir + "/out/a.png"));
    // refused: other folders, "..", absolute paths, symlinks out, saving outside out/
    for (const char* bad : {R"({"steps": [{"op": "load", "path": "secret/b.png"}]})", R"({"steps": [{"op": "load", "path": "in/../secret/b.png"}]})",
                            R"({"steps": [{"op": "load", "path": "in/link.png"}]})", R"({"steps": [{"op": "load", "dir": "secret"}]})"})
        CHECK_THROWS(runWorkflow(bad, dir, ev, nullptr, lim), WorkflowLimitExceeded);
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "load", "path": ")" + dir + R"(/secret/b.png"}]})", dir, ev, nullptr, lim), WorkflowLimitExceeded);
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "load", "path": "in/a.png", "out": "p"}, {"op": "save", "in": "p", "path": "in/x.png"}]})", dir, ev, nullptr, lim),
                 WorkflowLimitExceeded);
    CHECK(!fs::exists(dir + "/in/x.png"));
    // model fields are checked before anything loads (the folder need not exist)
    CHECK(!checkWorkflowLimits(R"({"steps": [{"op": "upscale", "model": "/etc/esrgan", "in": "p"}]})", dir, lim).empty());
    CHECK(!checkWorkflowLimits(R"({"steps": [{"op": "txt2img", "model": "in/sd", "prompt": "x", "lora": ["secret/l.safetensors:0.5"]}]})", dir, lim).empty());
    CHECK(checkWorkflowLimits(R"({"steps": [{"op": "txt2img", "model": "in/sd", "prompt": "x", "lora": ["in/l.safetensors:0.5"]}]})", dir, lim).empty());
    // sizes, counts, steps
    WorkflowLimits small = lim;
    small.maxPixels = 40;
    small.maxSteps = 2;
    small.maxDenoiseSteps = 8;
    small.maxFrames = 4;
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "load", "path": "in/a.png"}]})", dir, ev, nullptr, small), std::runtime_error);  // 48 pixels to decode
    CHECK(!checkWorkflowLimits(R"({"steps": [{"op": "txt2img", "model": "in/sd", "prompt": "x", "steps": 30}]})", dir, small).empty());
    CHECK(!checkWorkflowLimits(R"({"steps": [{"op": "t2v", "model": "in/sd", "prompt": "x", "frames": 16, "width": 4, "height": 4}]})", dir, small).empty());
    CHECK(!checkWorkflowLimits(R"({"steps": [{"op": "load", "path": "in/a.png"}, {"op": "load", "path": "in/a.png"}, {"op": "load", "path": "in/a.png"}]})", dir, small).empty());
    WorkflowLimits frames = lim;
    frames.maxFrames = 2;
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "load", "path": "in/a.png", "out": "p"}, {"op": "concat", "in": ["p", "p", "p"], "out": "c"}]})", dir, ev,
                             nullptr, frames),
                 WorkflowLimitExceeded);
    WorkflowLimits quick = lim;
    quick.maxSeconds = 1e-9;
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "load", "path": "in/a.png", "out": "p"}, {"op": "resize", "in": "p", "width": 4, "height": 4}]})", dir, ev,
                             nullptr, quick),
                 WorkflowLimitExceeded);
    // outside a sandbox nothing changes, and the decode bound is restored after a run
    CHECK(runWorkflow(R"({"steps": [{"op": "load", "path": "secret/b.png", "out": "p"}]})", dir).at("p").size() == 1);
    CHECK(maxDecodePixels() == (std::size_t(1) << 28));
    // file names mapped for a service (scales kept), every file seen
    std::vector<std::string> seen;
    const std::string mapped = mapWorkflowFiles(R"({"steps": [{"op": "txt2img", "model": "asset:sd", "prompt": "asset:not-a-file", "lora": ["asset:l1:0.7", "asset:l2"], "out": "x"},
        {"op": "load", "path": "asset:pic"}]})",
                                                [&](const WorkflowField& f, const std::string& v) {
                                                    seen.push_back(f.name + "=" + v);
                                                    return "/cache/" + v.substr(6);
                                                });
    CHECK(seen.size() == 4);
    const json::Value m = json::parse(mapped);
    CHECK(m.at("steps").array[0].at("model").string == "/cache/sd");
    CHECK(m.at("steps").array[0].at("prompt").string == "asset:not-a-file");
    CHECK(m.at("steps").array[0].at("lora").array[0].string == "/cache/l1:0.7");
    CHECK(m.at("steps").array[0].at("lora").array[1].string == "/cache/l2");
    CHECK(m.at("steps").array[1].at("path").string == "/cache/pic");
    // hostile files: a PNG claiming to be huge is refused before allocating, an ONNX model cannot
    // point its external data outside its folder
    {
        std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n', 0, 0, 0, 13, 'I', 'H', 'D', 'R', 0x7F, 0xFF, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF, 0xFF, 8, 2, 0, 0, 0, 0, 0, 0, 0};
        std::ofstream(dir + "/huge.png", std::ios::binary).write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
        CHECK_THROWS(loadPng(dir + "/huge.png"), std::runtime_error);
    }
    {
        const std::string escape = (fs::path(NEURACRAFT_SOURCE_DIR) / "tests" / "onnx" / "pipeline" / "external_escape.onnx").string();
        std::ofstream(fs::path(escape).parent_path().parent_path() / "escape.bin", std::ios::binary).write("0123456789abcdef", 16);
        bool refused = false;
        try {
            onnx::Model::load(escape);
        } catch (const std::exception& e) {
            refused = std::string(e.what()).find("outside the model's folder") != std::string::npos;
        }
        fs::remove(fs::path(escape).parent_path().parent_path() / "escape.bin");
        CHECK(refused);
    }
    fs::remove_all(dir);
}

TEST(lora_merging_and_lcm_sampler) {
    // LoraMerger: kohya and PEFT names, alpha / rank scaling, user scale.
    const std::string file = tempPath("lora_k.safetensors"), peft = tempPath("lora_p.safetensors");
    {
        std::map<std::string, Matrix> w;
        w["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = Matrix::uniform(2, 8, -1, 1);
        w["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = Matrix::uniform(8, 2, -1, 1);
        w["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha"] = Matrix(1, 1, std::vector<float>{1.0f});
        w["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"] = Matrix::uniform(2, 6, -1, 1);
        w["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight"] = Matrix::uniform(12, 2, -1, 1);
        saveAll(file, w);
        std::map<std::string, Matrix> p;
        p["unet.mid_block.attentions.0.proj_in.lora_A.weight"] = Matrix::uniform(3, 16, -1, 1);
        p["unet.mid_block.attentions.0.proj_in.lora_B.weight"] = Matrix::uniform(16, 3, -1, 1);
        saveAll(peft, p);
    }
    LoraMerger m({{file, 0.5f}}, "unet");
    CHECK(m.layers() == 1 && m.merged() == 0);
    Matrix weight = Matrix::uniform(8, 8, -1, 1), expected = weight;
    {
        const SafeTensors st(file);
        const Matrix down = st.load("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight");
        const Matrix up = st.load("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight");
        for (std::size_t o = 0; o < 8; ++o)
            for (std::size_t i = 0; i < 8; ++i)
                for (std::size_t r = 0; r < 2; ++r) expected(o, i) += 0.5f * (1.0f / 2.0f) * up(o, r) * down(r, i);
    }
    m.apply("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight", weight);
    for (std::size_t k = 0; k < 64; ++k) CHECK_NEAR(weight.data()[k], expected.data()[k], 1e-5);
    CHECK(m.merged() == 1 && m.unmerged().empty());
    Matrix untouched = Matrix::uniform(8, 8, -1, 1), copy = untouched;
    m.apply("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight", untouched);
    CHECK(untouched.values() == copy.values());
    Matrix wrong(4, 4);
    CHECK_THROWS(m.apply("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight", wrong), std::runtime_error);
    CHECK(LoraMerger({{file, 1.0f}}, "text_encoder").layers() == 1);

    // Loaders: a LoRA changes the UNet's output, scale 0 does not; one for no layer is refused.
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_lora", tok);
    const Matrix latent = Matrix::normal(1, 4 * 8 * 8, 0.0f, 1.0f), ctx = Matrix::uniform(8, 6, -1.0f, 1.0f);
    const Matrix base = loadUNet(dir, 8, 8).forward(latent, 300.0f, ctx);
    CHECK(loadUNet(dir, 8, 8, false, {{file, 0.0f}}).forward(latent, 300.0f, ctx).values() == base.values());
    const Matrix adapted = loadUNet(dir, 8, 8, false, {{file, 1.0f}, {peft, 1.0f}}).forward(latent, 300.0f, ctx);
    double d = 0;
    for (std::size_t k = 0; k < base.size(); ++k) d += std::abs(adapted.data()[k] - base.data()[k]);
    CHECK(d > 1e-3);
    {
        std::map<std::string, Matrix> w;
        w["lora_unet_no_such_layer.lora_down.weight"] = Matrix::uniform(2, 8, -1, 1);
        w["lora_unet_no_such_layer.lora_up.weight"] = Matrix::uniform(8, 2, -1, 1);
        saveAll(tempPath("lora_bad.safetensors"), w);
    }
    CHECK_THROWS(loadUNet(dir, 8, 8, false, {{tempPath("lora_bad.safetensors"), 1.0f}}), std::runtime_error);
    // The text encoder half (lora_te_*) changes the text hidden states.
    const std::vector<int> ids{1, 2, 3, 4, 5, 0, 0, 0};
    CHECK(loadClipTextEncoder(dir, false, {{file, 1.0f}}).encode(ids).values() != loadClipTextEncoder(dir).encode(ids).values());

    // LCM scheduler: diffusers' timesteps, unit noise in VP space, the last step returns the denoised latent.
    const EulerScheduler s = EulerScheduler::lcm(4);
    CHECK(s.consistency && s.timesteps == std::vector<double>({999, 759, 519, 279}) && s.sigmas.back() == 0.0);
    CHECK(std::abs(s.initNoiseSigma - std::sqrt(s.sigmas[0] * s.sigmas[0] + 1.0)) < 1e-9);
    Matrix x(1, 3, std::vector<float>{1.0f, -2.0f, 0.5f});
    const Matrix eps(1, 3, std::vector<float>{0.1f, 0.2f, -0.3f});
    const Matrix x0 = x;
    s.step(3, eps, x);
    const double ts = 2790.0, cSkip = 0.25 / (ts * ts + 0.25), cOut = ts / std::sqrt(ts * ts + 0.25);
    const double vp = 1.0 / std::sqrt(s.sigmas[3] * s.sigmas[3] + 1.0);
    for (std::size_t k = 0; k < 3; ++k)
        CHECK_NEAR(x.data()[k], cOut * (x0.data()[k] - s.sigmas[3] * eps.data()[k]) + cSkip * x0.data()[k] * vp, 1e-5);
    // The pipeline with the LCM sampler and a LoRA.
    TextToImageOptions o;
    o.width = o.height = 64;
    o.steps = 2;
    o.sampler = "lcm";
    o.loras = {{file, 1.0f}};
    const Image a = textToImage(dir, "hello", o);
    CHECK(a.width == 64 && textToImage(dir, "hello", o).pixels == a.pixels);
    o.sampler = "ddim";
    CHECK_THROWS(textToImage(dir, "hello", o), std::invalid_argument);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(tok);
    for (const std::string& f : {file, peft, tempPath("lora_bad.safetensors")}) std::filesystem::remove(f);
}

TEST(inpainting_and_outpaint_zoom) {
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_inpaint", tok);
    writeTinyUNet(dir, 6, 9);  // an inpainting UNet: latent + mask + masked-image latent
    TextToImageOptions o;
    o.width = o.height = 64;
    StableDiffusion sd(dir, o);
    CHECK(sd.isInpainting());
    Image input, mask;
    input.width = mask.width = 64;
    input.height = mask.height = 64;
    for (std::size_t i = 0; i < 64 * 64 * 3; ++i) input.pixels.push_back(static_cast<std::uint8_t>(i * 29 % 256));
    mask.pixels.assign(64 * 64 * 3, 0);
    for (std::size_t y = 16; y < 40; ++y)
        for (std::size_t x = 20; x < 44; ++x) std::fill_n(mask.pixels.data() + (y * 64 + x) * 3, 3, 255);
    const Image out = sd.inpaint(input, mask, "hello", 1, 2, 1.0f, 2.0f);
    CHECK(out.width == 64 && out.height == 64);
    // Far from the mask the input is kept exactly; inside it is redrawn.
    bool keptOutside = true, changedInside = false;
    for (std::size_t y = 0; y < 64; ++y)
        for (std::size_t x = 0; x < 64; ++x) {
            const bool far = y + 4 < 16 || y >= 44 || x + 4 < 20 || x >= 48, inside = y >= 18 && y < 38 && x >= 22 && x < 42;
            for (std::size_t c = 0; c < 3; ++c) {
                const std::size_t k = (y * 64 + x) * 3 + c;
                if (far && out.pixels[k] != input.pixels[k]) keptOutside = false;
                if (inside && out.pixels[k] != input.pixels[k]) changedInside = true;
            }
        }
    CHECK(keptOutside && changedInside);
    CHECK(sd.inpaint(input, mask, "hello", 1, 2, 1.0f, 2.0f).pixels == out.pixels);  // deterministic
    CHECK(sd.inpaint(input, mask, "hello", 1, 4, 0.5f, 2.0f).width == 64);            // from the noised input
    CHECK(sd.generate("hello", 3, 2).width == 64);                                     // whole picture via an all-white mask
    CHECK_THROWS(sd.inpaint(input, mask, "hello", 1, 2, 0.0f), std::invalid_argument);
    o.steps = 2;
    CHECK(inpaint(dir, input, mask, "hello", o).width == 64);

    // Outpaint zoom: nested keyframes, framesPerKey rendered frames per pair, the start picture at the end it belongs.
    OutpaintZoomOptions z;
    z.keyframes = 3;
    z.framesPerKey = 2;
    z.steps = 2;
    const auto outward = outpaintZoom(sd, {"hello", "he"}, z, &input);
    CHECK(outward.size() == 5 && outward.front().pixels == input.pixels);
    for (const Image& f : outward) CHECK(f.width == 64 && f.height == 64);
    z.zoomIn = true;
    const auto inward = outpaintZoom(sd, {"hello", "he"}, z, &input);
    CHECK(inward.size() == 5 && inward.back().pixels == input.pixels && inward.front().pixels == outward.back().pixels);
    z.shrink = 0.2f;
    CHECK_THROWS(outpaintZoom(sd, {"hello"}, z), std::invalid_argument);

    // A text-to-image UNet cannot inpaint.
    std::string tok2;
    const std::string plainDir = writeTinyStableDiffusion("sd_plain", tok2);
    StableDiffusion plain(plainDir, o);
    CHECK(!plain.isInpainting());
    CHECK_THROWS(plain.inpaint(input, mask, "hello", 1, 2), std::logic_error);
    for (const std::string& d : {dir, tok, plainDir, tok2}) std::filesystem::remove_all(d);
}

// A tiny ControlNetModel for the tiny UNet (blocks 8, 16; layers_per_block 1).
void writeTinyControlNet(const std::string& dir, bool zeroOutputs) {
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/config.json")
        << R"({"_class_name":"ControlNetModel","act_fn":"silu","attention_head_dim":[2,2],"block_out_channels":[8,16],)"
        << R"("conditioning_embedding_out_channels":[4,4,4,4],"cross_attention_dim":6,)"
        << R"("down_block_types":["CrossAttnDownBlock2D","DownBlock2D"],"flip_sin_to_cos":true,"freq_shift":0,"in_channels":4,)"
        << R"("layers_per_block":1,"norm_eps":1e-5,"norm_num_groups":4,"use_linear_projection":true})";
    std::map<std::string, Matrix> w;
    for (auto& [name, m] : tinyUNetWeights({8, 16}, 1, 6))
        if (name.rfind("up_blocks", 0) != 0 && name.rfind("conv_norm_out", 0) != 0 && name.rfind("conv_out", 0) != 0) w[name] = m;
    auto conv = [&](const std::string& n, std::size_t o, std::size_t i, std::size_t k, bool zero) {
        w[n + ".weight"] = zero ? Matrix(1, o * i * k * k) : Matrix::uniform(1, o * i * k * k, -0.3f, 0.3f);
        w[n + ".bias"] = zero ? Matrix(1, o) : Matrix::uniform(1, o, -0.1f, 0.1f);
    };
    conv("controlnet_cond_embedding.conv_in", 4, 3, 3, false);
    for (int b = 0; b < 6; ++b) conv("controlnet_cond_embedding.blocks." + std::to_string(b), 4, 4, 3, false);
    conv("controlnet_cond_embedding.conv_out", 8, 4, 3, zeroOutputs);
    const std::size_t skips[4] = {8, 8, 8, 16};
    for (int k = 0; k < 4; ++k) conv("controlnet_down_blocks." + std::to_string(k), skips[k], skips[k], 1, zeroOutputs);
    conv("controlnet_mid_block", 16, 16, 1, zeroOutputs);
    saveAll(dir + "/diffusion_pytorch_model.safetensors", w);
}

TEST(scribble_drawings_for_the_scribble_controlnet) {
    // A black line on white paper, 1 pixel wide -> a white line on black, 2 * thickness + 1 wide.
    Image d;
    d.width = 20;
    d.height = 10;
    d.pixels.assign(20 * 10 * 3, 250);
    for (std::size_t y = 0; y < 10; ++y)
        for (int c = 0; c < 3; ++c) d.pixels[(y * 20 + 10) * 3 + c] = 20;
    const Image s = scribbleImage(d);
    CHECK(s.width == 20 && s.height == 10 && s.channels == 3);
    for (std::size_t x = 0; x < 20; ++x) CHECK(s.pixels[(5 * 20 + x) * 3] == (x >= 8 && x <= 12 ? 255 : 0));
    const Image thin = scribbleImage(d, true, 128, 0);
    CHECK(thin.pixels[(5 * 20 + 10) * 3] == 255 && thin.pixels[(5 * 20 + 9) * 3] == 0);
    const Image light = scribbleImage(d, false, 128, 0);  // light strokes: here the paper is the stroke
    CHECK(light.pixels[(5 * 20 + 2) * 3] == 255 && light.pixels[(5 * 20 + 10) * 3] == 0);
    CHECK_THROWS(scribbleImage(d, true, 128, 99), std::invalid_argument);
    const std::string dir = tempPath("scribble");
    std::filesystem::create_directories(dir);
    savePng(dir + "/d.png", d);
    const auto clips = runWorkflow(R"({"steps":[{"op":"load","path":"d.png","out":"d"},{"op":"scribble","in":"d","thickness":1,"out":"s"}]})", dir);
    CHECK(clips.at("s")[0].pixels[(5 * 20 + 11) * 3] == 255 && clips.at("s")[0].pixels[(5 * 20 + 12) * 3] == 0);
    std::filesystem::remove_all(dir);
}

// With NEURACRAFT_CONTROLNET_DIR set to a ControlNet 1.1 folder (canny, scribble, ...) and python3 with
// NumPy, the residuals match scripts/controlnet_reference.py (diffusers' ControlNetModel in NumPy).
TEST(controlnet_matches_the_reference_on_real_weights) {
    const char* dirEnv = std::getenv("NEURACRAFT_CONTROLNET_DIR");
    if (!dirEnv || !*dirEnv) return;  // skipped without real weights
    const std::size_t H = 16, W = 16;
    UNet2DCondition cn = loadUNet(dirEnv, W, H);
    std::mt19937 gen(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix lat(1, 4 * H * W), ctx(77, cn.contextDim()), cond(1, 3 * 128 * 128, 0.0f);
    for (std::size_t i = 0; i < lat.cols(); ++i) lat.data()[i] = normal(gen);
    for (std::size_t i = 0; i < ctx.rows() * ctx.cols(); ++i) ctx.data()[i] = 0.5f * normal(gen);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) {  // scribble-like strokes: a wavy line, a bar and a ring
            const int r2 = (x - 30) * (x - 30) + (y - 30) * (y - 30);
            const bool on = std::abs(x - 64 - static_cast<int>(20 * std::sin(y / 12.0))) < 2 || std::abs(y - 90) < 2 || (r2 < 120 && r2 > 70);
            for (int c = 0; c < 3; ++c) cond.data()[c * 128 * 128 + y * 128 + x] = on ? 1.0f : 0.0f;
        }
    const std::string dir = tempPath("controlnet_ref");
    std::filesystem::create_directories(dir);
    auto dump = [&](const std::string& name, const float* p, std::size_t n) {
        std::ofstream(dir + "/" + name, std::ios::binary).write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n * 4));
    };
    dump("latent.f32", lat.data(), lat.cols());
    dump("context.f32", ctx.data(), ctx.rows() * ctx.cols());
    dump("cond.f32", cond.data(), cond.cols());
    const ControlResiduals r = cn.control(lat, 500.0f, ctx, cn.controlEmbedding(cond));
    std::vector<float> mine;
    for (const Matrix& m : r.down) mine.insert(mine.end(), m.data(), m.data() + m.rows() * m.cols());
    mine.insert(mine.end(), r.mid.data(), r.mid.data() + r.mid.rows() * r.mid.cols());
    const std::string script = std::string(NEURACRAFT_SOURCE_DIR) + "/scripts/controlnet_reference.py";
    const std::string cmd = "python3 '" + script + "' '" + dirEnv + "' '" + dir + "/latent.f32' '" + dir + "/context.f32' '" + dir + "/cond.f32' 500 16 16 '" +
                            dir + "/ref.f32'";
    CHECK(std::system(cmd.c_str()) == 0);
    std::ifstream f(dir + "/ref.f32", std::ios::binary);
    std::vector<float> ref(mine.size());
    f.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(ref.size() * 4));
    CHECK(static_cast<std::size_t>(f.gcount()) == ref.size() * 4);
    float diff = 0, big = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        diff = std::max(diff, std::fabs(mine[i] - ref[i]));
        big = std::max(big, std::fabs(ref[i]));
    }
    std::cerr << "    controlnet " << dirEnv << ": max difference " << diff << " (largest value " << big << ")\n";
    CHECK(diff <= 1e-5f * std::max(1.0f, big));  // the canny bar: a few 1e-6 relative
    std::filesystem::remove_all(dir);
}

TEST(controlnet_and_canny_edges) {
    // Canny: a bright square on black has edges on its border only; a flat picture has none.
    Image sq;
    sq.width = sq.height = 32;
    sq.pixels.assign(32 * 32 * 3, 0);
    for (std::size_t y = 8; y < 24; ++y)
        for (std::size_t x = 8; x < 24; ++x) std::fill_n(sq.pixels.data() + (y * 32 + x) * 3, 3, 200);
    const Image e = cannyEdges(sq);
    CHECK(e.channels == 3 && e.width == 32);
    std::size_t edges = 0, inside = 0, far = 0;
    for (std::size_t y = 0; y < 32; ++y)
        for (std::size_t x = 0; x < 32; ++x) {
            if (e.pixels[(y * 32 + x) * 3] == 0) continue;
            ++edges;
            if (y >= 11 && y < 21 && x >= 11 && x < 21) ++inside;
            if (y < 5 || y >= 27 || x < 5 || x >= 27) ++far;
        }
    CHECK(edges >= 40 && inside == 0 && far == 0);
    Image flat = sq;
    std::fill(flat.pixels.begin(), flat.pixels.end(), static_cast<std::uint8_t>(90));
    for (std::uint8_t v : cannyEdges(flat).pixels) CHECK(v == 0);

    // ControlNet: loads as a UNet without up blocks; zero convolutions give zero residuals.
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_control", tok);
    writeTinyControlNet(dir + "/cn_zero", true);
    writeTinyControlNet(dir + "/cn", false);
    UNet2DCondition unet = loadUNet(dir, 8, 8), zero = loadUNet(dir + "/cn_zero", 8, 8), cn = loadUNet(dir + "/cn", 8, 8);
    CHECK(cn.isControlNet() && !unet.isControlNet());
    const Matrix latent = Matrix::normal(1, 4 * 8 * 8, 0.0f, 1.0f), ctx = Matrix::uniform(8, 6, -1.0f, 1.0f);
    const Matrix image = Matrix::uniform(1, 3 * 64 * 64, 0.0f, 1.0f);
    CHECK_THROWS(cn.forward(latent, 10.0f, ctx), std::logic_error);
    CHECK_THROWS(unet.controlEmbedding(image), std::logic_error);
    const ControlResiduals rz = zero.control(latent, 10.0f, ctx, zero.controlEmbedding(image));
    CHECK(rz.down.size() == 4);
    const Matrix plain = unet.forward(latent, 10.0f, ctx);
    CHECK(unet.forward(latent, 10.0f, ctx, &rz).values() == plain.values());
    const ControlResiduals r = cn.control(latent, 10.0f, ctx, cn.controlEmbedding(image), 0.7f);
    const ControlResiduals r1 = cn.control(latent, 10.0f, ctx, cn.controlEmbedding(image), 1.0f);
    for (std::size_t k = 0; k < r.mid.size(); ++k) CHECK_NEAR(r.mid.data()[k], 0.7f * r1.mid.data()[k], 1e-5);
    CHECK(unet.forward(latent, 10.0f, ctx, &r).values() != plain.values());

    // The same ControlNet as one file in the original layout (control_model.*, input_hint_block, zero_convs).
    {
        std::map<std::string, Matrix> ldm;
        for (const auto& [n, m] : loadAllTensors(dir + "/cn/diffusion_pytorch_model.safetensors")) {
            std::string to;
            std::size_t k;
            std::string rest;
            if (n.rfind("controlnet_cond_embedding.conv_in.", 0) == 0) to = "input_hint_block.0." + n.substr(34);
            else if (n.rfind("controlnet_cond_embedding.conv_out.", 0) == 0) to = "input_hint_block.14." + n.substr(35);
            else if (splitIndex(n, "controlnet_cond_embedding.blocks.", k, rest)) to = "input_hint_block." + std::to_string(2 * k + 2) + "." + rest;
            else if (splitIndex(n, "controlnet_down_blocks.", k, rest)) to = "zero_convs." + std::to_string(k) + ".0." + rest;
            else if (n.rfind("controlnet_mid_block.", 0) == 0) to = "middle_block_out.0." + n.substr(21);
            else to = ldmUnetName(n, 1, {false, true}).substr(22);  // drop "model.diffusion_model."
            ldm["control_model." + to] = m;
        }
        std::filesystem::create_directories(dir + "/controlnets");
        const std::string file = dir + "/controlnets/cn_single.safetensors";
        saveAll(file, ldm);
        std::ofstream(dir + "/controlnets/cn_single.json") << R"({"family": "sd1", "unet": {"attention_head_dim": [2, 2], "norm_num_groups": 4}})";
        CHECK(inspectCheckpoint(file).controlNet && !isSingleFileCheckpoint(file));
        UNet2DCondition one = loadUNet(file, 8, 8);
        CHECK(one.isControlNet());
        const ControlResiduals a = cn.control(latent, 10.0f, ctx, cn.controlEmbedding(image)), b = one.control(latent, 10.0f, ctx, one.controlEmbedding(image));
        CHECK(a.mid.values() == b.mid.values() && a.down.size() == b.down.size() && a.down.back().values() == b.down.back().values());
        bool listed = false;
        for (const ModelInfo& m : findModels(dir))
            if (m.path == file) listed = m.kind == "controlnet";
        CHECK(listed);
    }

    // Pipelines: scale 0 equals no ControlNet; a control image steers the picture.
    TextToImageOptions o;
    o.width = o.height = 64;
    o.steps = 2;
    const Image base = textToImage(dir, "hello", o);
    o.controlNet = dir + "/cn";
    o.controlImage = cannyEdges(sq);
    o.controlScale = 0.0f;
    CHECK(textToImage(dir, "hello", o).pixels == base.pixels);
    o.controlScale = 1.0f;
    CHECK(textToImage(dir, "hello", o).pixels != base.pixels);
    o.controlImage = Image{};
    CHECK_THROWS(textToImage(dir, "hello", o), std::invalid_argument);
    // Text to video with one control image per frame.
    std::filesystem::create_directories(dir + "/motion");
    saveAll(dir + "/motion/tiny.safetensors", tinyMotionWeights({8, 16}, 1));
    TextToVideoOptions v;
    v.width = v.height = 64;
    v.frames = 3;
    v.steps = 1;
    v.controlNet = dir + "/cn";
    v.controlImages = {sq, flat, sq};
    CHECK(textToVideo(dir, "hello", v).size() == 3);
    v.controlImages = {sq, flat};
    CHECK_THROWS(textToVideo(dir, "hello", v), std::invalid_argument);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(tok);
}

TEST(upscaler_compact_esrgan) {
    // A tiny SRVGGNetCompact (x2): conv 3->4, PReLU, conv 4->4, PReLU, conv 4->12.
    const std::string file = tempPath("esr.safetensors");
    auto write = [&](bool zeroLast) {
        std::map<std::string, Matrix> w;
        w["body.0.weight"] = Matrix::uniform(1, 4 * 3 * 9, -0.3f, 0.3f);
        w["body.0.bias"] = Matrix::uniform(1, 4, -0.1f, 0.1f);
        w["body.1.weight"] = Matrix::uniform(1, 4, 0.1f, 0.3f);
        w["body.2.weight"] = Matrix::uniform(1, 4 * 4 * 9, -0.3f, 0.3f);
        w["body.2.bias"] = Matrix::uniform(1, 4, -0.1f, 0.1f);
        w["body.3.weight"] = Matrix::uniform(1, 4, 0.1f, 0.3f);
        w["body.4.weight"] = zeroLast ? Matrix(1, 12 * 4 * 9) : Matrix::uniform(1, 12 * 4 * 9, -0.3f, 0.3f);
        w["body.4.bias"] = zeroLast ? Matrix(1, 12) : Matrix::uniform(1, 12, -0.05f, 0.05f);
        saveAll(file, w);
    };
    Image img;
    img.width = 21;
    img.height = 13;
    for (std::size_t i = 0; i < 21 * 13 * 3; ++i) img.pixels.push_back(static_cast<std::uint8_t>(i * 37 % 256));
    // With a zero last layer the output is the nearest-neighbour upscale of the input.
    write(true);
    const Upscaler zero(file);
    CHECK(zero.scale() == 2);
    const Image near = zero.upscale(img);
    CHECK(near.width == 42 && near.height == 26);
    bool same = true;
    for (std::size_t y = 0; y < 26; ++y)
        for (std::size_t x = 0; x < 42; ++x)
            for (std::size_t c = 0; c < 3; ++c)
                if (near.pixels[(y * 42 + x) * 3 + c] != img.pixels[((y / 2) * 21 + x / 2) * 3 + c]) same = false;
    CHECK(same);
    // Tiles overlap by the receptive field, so tiled and whole outputs are identical.
    write(false);
    const Upscaler up(file);
    const Image whole = up.upscale(img, 0);
    CHECK(whole.pixels != near.pixels);
    CHECK(up.upscale(img, 5).pixels == whole.pixels);
    std::map<std::string, Matrix> bad;
    bad["body.0.weight"] = Matrix::uniform(1, 4 * 3, -1, 1);
    saveAll(file, bad);
    CHECK_THROWS(Upscaler(file), std::exception);
    std::filesystem::remove(file);
}

// safetensors with explicit shapes (float32), for models whose loaders check 4-D shapes.
void saveShaped(const std::string& file, const std::vector<std::pair<std::string, std::vector<std::size_t>>>& tensors,
                const std::function<float(const std::string&, std::size_t)>& value) {
    std::string header = "{";
    std::vector<float> data;
    for (const auto& [name, shape] : tensors) {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        if (header.size() > 1) header += ",";
        header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) header += (i ? "," : "") + std::to_string(shape[i]);
        header += "],\"data_offsets\":[" + std::to_string(data.size() * 4) + "," + std::to_string((data.size() + n) * 4) + "]}";
        for (std::size_t i = 0; i < n; ++i) data.push_back(value(name, i));
    }
    header += "}";
    while (header.size() % 8) header += ' ';
    std::ofstream f(file, std::ios::binary);
    const std::uint64_t len = header.size();
    f.write(reinterpret_cast<const char*>(&len), 8);
    f << header;
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 4));
}

TEST(rife_frame_interpolation) {
    const std::string file = tempPath("rife.safetensors");
    auto write = [&](bool zeroHeads) {
        std::vector<std::pair<std::string, std::vector<std::size_t>>> t;
        const std::size_t c = 8;
        for (const char* b : {"block0", "block1", "block2"}) {
            const std::string p = b;
            auto conv = [&](const std::string& n, std::size_t o, std::size_t i) {
                t.push_back({n + ".0.weight", {o, i, 3, 3}});
                t.push_back({n + ".0.bias", {o}});
                t.push_back({n + ".1.weight", {o}});
            };
            conv(p + ".conv0.0", c / 2, 11);
            conv(p + ".conv0.1", c, c / 2);
            for (int k = 0; k < 4; ++k)
                for (int j = 0; j < 2; ++j) conv(p + ".convblock" + std::to_string(k) + "." + std::to_string(j), c, c);
            for (const char* head : {".conv1", ".conv2"}) {
                t.push_back({p + head + ".0.weight", {c, c / 2, 4, 4}});
                t.push_back({p + head + ".0.bias", {c / 2}});
                t.push_back({p + head + ".1.weight", {c / 2}});
                const std::size_t out = std::string(head) == ".conv1" ? 4 : 1;
                t.push_back({p + head + ".2.weight", {c / 2, out, 4, 4}});
                t.push_back({p + head + ".2.bias", {out}});
            }
        }
        std::mt19937 rng(3);
        saveShaped(file, t, [&](const std::string& name, std::size_t) {
            const bool head = name.find(".2.") != std::string::npos && (name.find("conv1") != std::string::npos || name.find("conv2") != std::string::npos);
            if (zeroHeads && head) return 0.0f;
            if (name.find(".1.weight") != std::string::npos) return 0.25f;  // PReLU slopes
            return (static_cast<float>(rng() % 2001) / 1000.0f - 1.0f) * 0.2f;
        });
    };
    Image a, b;
    a.width = b.width = 40;  // not a multiple of 32
    a.height = b.height = 24;
    for (std::size_t i = 0; i < 40 * 24 * 3; ++i) {
        a.pixels.push_back(static_cast<std::uint8_t>(i * 7 % 256));
        b.pixels.push_back(static_cast<std::uint8_t>(i * 13 % 256));
    }
    // Zero flow and mask heads: no motion, an even blend of the two frames.
    write(true);
    const FrameInterpolator blend(file);
    const Image m = blend.middle(a, b);
    CHECK(m.width == 40 && m.height == 24);
    bool even = true;
    for (std::size_t i = 0; i < m.pixels.size(); ++i)
        if (std::abs(m.pixels[i] - (a.pixels[i] + b.pixels[i]) / 2.0) > 1.0) even = false;
    CHECK(even);
    // With flow: frames in between, originals kept, deterministic.
    write(false);
    const FrameInterpolator fi(file);
    const std::vector<Image> frames{a, b, a};
    const auto doubled = fi.interpolate(frames, 1);
    CHECK(doubled.size() == 5 && doubled[0].pixels == a.pixels && doubled[2].pixels == b.pixels && doubled[4].pixels == a.pixels);
    CHECK(doubled[1].pixels == fi.middle(a, b).pixels && doubled[1].pixels != m.pixels);
    CHECK(fi.interpolate(frames, 2).size() == 9);
    CHECK(fi.interpolate(frames, 1, true).size() == 6);
    Image small = a;
    small.width = 20;
    small.pixels.resize(20 * 24 * 3);
    CHECK_THROWS(fi.middle(a, small), std::invalid_argument);
    std::filesystem::remove(file);
}

TEST(depth_anything_and_parallax) {
    // A tiny Depth Anything: hidden 8, 2 heads, 4 layers, 4 x 4 position grid.
    const std::string dir = tempPath("depth_tiny");
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/config.json")
        << R"({"backbone_config":{"hidden_size":8,"num_attention_heads":2,"num_hidden_layers":4,"patch_size":14,)"
        << R"("out_indices":[1,2,3,4]},"fusion_hidden_size":8,"head_hidden_size":4,"neck_hidden_sizes":[4,4,4,4],)"
        << R"("patch_size":14,"reassemble_factors":[4,2,1,0.5],"reassemble_hidden_size":8})";
    std::vector<std::pair<std::string, std::vector<std::size_t>>> t;
    const std::string b = "backbone.";
    t.push_back({b + "embeddings.cls_token", {1, 1, 8}});
    t.push_back({b + "embeddings.patch_embeddings.projection.weight", {8, 3, 14, 14}});
    t.push_back({b + "embeddings.patch_embeddings.projection.bias", {8}});
    t.push_back({b + "embeddings.position_embeddings", {1, 17, 8}});
    for (int i = 0; i < 4; ++i) {
        const std::string p = b + "encoder.layer." + std::to_string(i) + ".";
        for (const char* n : {"norm1", "norm2"}) {
            t.push_back({p + n + ".weight", {8}});
            t.push_back({p + n + ".bias", {8}});
        }
        t.push_back({p + "layer_scale1.lambda1", {8}});
        t.push_back({p + "layer_scale2.lambda1", {8}});
        for (const char* n : {"attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense"}) {
            t.push_back({p + n + ".weight", {8, 8}});
            t.push_back({p + n + ".bias", {8}});
        }
        t.push_back({p + "mlp.fc1.weight", {32, 8}});
        t.push_back({p + "mlp.fc1.bias", {32}});
        t.push_back({p + "mlp.fc2.weight", {8, 32}});
        t.push_back({p + "mlp.fc2.bias", {8}});
    }
    t.push_back({b + "layernorm.weight", {8}});
    t.push_back({b + "layernorm.bias", {8}});
    const std::size_t resize[4][3] = {{4, 4, 1}, {4, 4, 1}, {0, 0, 0}, {4, 4, 1}};
    for (int i = 0; i < 4; ++i) {
        const std::string p = "neck.reassemble_stage.layers." + std::to_string(i) + ".";
        t.push_back({p + "projection.weight", {4, 8, 1, 1}});
        t.push_back({p + "projection.bias", {4}});
        if (i == 0) t.push_back({p + "resize.weight", {4, 4, 4, 4}});
        if (i == 1) t.push_back({p + "resize.weight", {4, 4, 2, 2}});
        if (i == 3) t.push_back({p + "resize.weight", {4, 4, 3, 3}});
        if (i != 2) t.push_back({p + "resize.bias", {4}});
        t.push_back({"neck.convs." + std::to_string(i) + ".weight", {8, 4, 3, 3}});
        const std::string q = "neck.fusion_stage.layers." + std::to_string(i) + ".";
        t.push_back({q + "projection.weight", {8, 8, 1, 1}});
        t.push_back({q + "projection.bias", {8}});
        for (const char* r : {"residual_layer1.", "residual_layer2."})
            for (const char* c : {"convolution1", "convolution2"}) {
                t.push_back({q + r + c + ".weight", {8, 8, 3, 3}});
                t.push_back({q + r + c + ".bias", {8}});
            }
    }
    (void)resize;
    t.push_back({"head.conv1.weight", {4, 8, 3, 3}});
    t.push_back({"head.conv1.bias", {4}});
    t.push_back({"head.conv2.weight", {4, 4, 3, 3}});
    t.push_back({"head.conv2.bias", {4}});
    t.push_back({"head.conv3.weight", {1, 4, 1, 1}});
    t.push_back({"head.conv3.bias", {1}});
    std::mt19937 rng(9);
    saveShaped(dir + "/model.safetensors", t, [&](const std::string& name, std::size_t) {
        if (name.find("norm") != std::string::npos && name.find(".weight") != std::string::npos) return 1.0f;
        if (name == "head.conv3.bias") return 0.5f;  // keep the ReLU output above zero somewhere
        return (static_cast<float>(rng() % 2001) / 1000.0f - 1.0f) * 0.3f;
    });
    DepthEstimator de(dir);
    // Native grid (4 x 4 patches) and an interpolated one (3 x 5).
    std::vector<float> x(3 * 56 * 56);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.01f * static_cast<float>(i));
    const auto d = de.forward(x, 56, 56);
    CHECK(d.size() == 56 * 56);
    float mx = 0;
    for (float v : d) {
        CHECK(v >= 0.0f);
        mx = std::max(mx, v);
    }
    CHECK(mx > 0.0f);
    std::vector<float> y(3 * 42 * 70, 0.1f);
    CHECK(de.forward(y, 42, 70).size() == 42 * 70);
    CHECK_THROWS(de.forward(y, 40, 70), std::invalid_argument);
    Image img;
    img.width = 50;
    img.height = 30;
    for (std::size_t i = 0; i < 50 * 30 * 3; ++i) img.pixels.push_back(static_cast<std::uint8_t>(i * 17 % 256));
    const auto e = de.estimate(img, 28);
    CHECK(e.size() == 50 * 30);
    const Image di = de.depthImage(img, 28);
    CHECK(di.width == 50 && di.height == 30 && di.channels == 3);

    // Parallax: the first frame of a loop is the picture itself; later frames move near pixels.
    std::vector<float> depth(50 * 30);
    for (std::size_t yy = 0; yy < 30; ++yy)
        for (std::size_t xx = 0; xx < 50; ++xx) depth[yy * 50 + xx] = xx > 20 && xx < 30 ? 1.0f : 0.0f;  // a near stripe
    ParallaxOptions po;
    po.frames = 8;
    po.amplitude = 0.1f;
    const auto frames = parallaxAnimation(img, depth, po);
    CHECK(frames.size() == 8 && frames[0].pixels == img.pixels && frames[2].pixels != img.pixels);
    CHECK_THROWS(parallaxAnimation(img, std::vector<float>(10), po), std::invalid_argument);

    // depth_mesh and render in a workflow: picture -> 3D relief -> .glb / .ply and a turntable clip.
    const std::string out = tempPath("depth_mesh_out");
    std::filesystem::create_directories(out);
    savePng(out + "/in.png", img);
    std::vector<std::string> seen;
    WorkflowEvents ev;
    ev.output = [&](std::size_t, const std::string& op, const std::string& name, const Clip& frames, double) {
        if (op == "depth_mesh") CHECK(name == "relief" && frames.size() == 1 && frames[0].width == 256);  // a preview view
        seen.push_back(op);
    };
    const std::string wf = R"({"steps":[{"op":"load","path":"in.png","out":"pic"},)"
                           R"({"op":"depth_mesh","model":")" + dir + R"(","in":"pic","grid":24,"size":28,"edges":0,"out":"relief"},)"
                           R"({"op":"render","in":"relief","width":64,"height":48,"frames":3,"out":"turn"},)"
                           R"({"op":"save","in":"relief","path":"relief.glb"},{"op":"save","in":"relief","path":"relief.ply"},)"
                           R"({"op":"save","in":"turn","path":"turn.gif"}]})";
    WorkflowResultCache results;
    const auto clips = runWorkflow(wf, out, ev, &results);
    CHECK(clips.count("turn") && clips.at("turn").size() == 3 && clips.at("turn")[0].width == 64 && !clips.count("relief"));
    const Mesh glb = loadMesh(out + "/relief.glb"), ply = loadMesh(out + "/relief.ply");
    CHECK(glb.vertexCount() > 100 && glb.triangleCount() == ply.triangleCount() && glb.texture.width == 50 && !ply.colors.empty());
    CHECK(seen.size() == 6);
    // Run again: the mesh comes from the result cache.
    const std::size_t hits = results.hits();
    runWorkflow(wf, out, ev, &results);
    CHECK(results.hits() > hits);
    // A clip is not a mesh, and a mesh is not saved as a picture.
    CHECK_THROWS(runWorkflow(R"({"steps":[{"op":"load","path":"in.png","out":"pic"},{"op":"render","in":"pic"}]})", out), std::invalid_argument);
    CHECK_THROWS(runWorkflow(R"({"steps":[{"op":"load","path":"relief.glb","out":"m"},{"op":"save","in":"m","path":"m.png"}]})", out), std::invalid_argument);
    CHECK(checkWorkflow(R"({"steps":[{"op":"render","in":"x","background":"pink"}]})").size() == 1);
    // Stereo 3D: a picture through its depth, and a mesh rendered for both eyes.
    const auto s3d = runWorkflow(R"({"steps":[{"op":"load","path":"in.png","out":"pic"},)"
                                 R"({"op":"stereo","model":")" + dir + R"(","in":"pic","layout":"sbs","size":28,"out":"sbs"},)"
                                 R"({"op":"stereo","model":")" + dir + R"(","in":"pic","layout":"anaglyph","size":28,"out":"red"},)"
                                 R"({"op":"load","path":"relief.glb","out":"relief"},)"
                                 R"({"op":"render","in":"relief","width":32,"height":24,"stereo":"tb","out":"pair"}]})",
                                 out, {}, &results);
    CHECK(s3d.at("sbs")[0].width == 100 && s3d.at("sbs")[0].height == 30 && s3d.at("red")[0].width == 50);
    CHECK(s3d.at("pair")[0].width == 32 && s3d.at("pair")[0].height == 48);
    CHECK(checkWorkflow(R"({"steps":[{"op":"stereo","model":"m","in":"x","layout":"3d"}]})").size() == 1);
    std::filesystem::remove_all(out);
    std::filesystem::remove_all(dir);
}

TEST(stereo_views_and_layouts) {
    // 40 x 10, every column its own colour; columns 15..19 are near, the rest far.
    Image img;
    img.width = 40;
    img.height = 10;
    std::vector<float> near(400, 0.0f);
    for (std::size_t y = 0; y < 10; ++y)
        for (std::size_t x = 0; x < 40; ++x) {
            for (int c = 0; c < 3; ++c) img.pixels.push_back(static_cast<std::uint8_t>(x * 6 + c));
            if (x >= 15 && x <= 19) near[y * 40 + x] = 1.0f;
        }
    const auto px = [](const Image& im, std::size_t x, std::size_t y) { return im.pixels[(y * im.width + x) * 3]; };
    StereoOptions so;
    so.strength = 0.2f;  // 8 pixels between near and far; each eye moves near pixels by 4
    so.focus = 0.0f;     // far stays on the screen plane
    auto [left, right] = stereoViews(img, near, so);
    CHECK(left.width == 40 && right.height == 10);
    CHECK(px(left, 19, 3) == px(img, 15, 3) && px(right, 11, 3) == px(img, 15, 3));  // near: crossed, in front of the screen
    CHECK(px(left, 5, 3) == px(img, 5, 3) && px(right, 30, 3) == px(img, 30, 3));    // far: in place
    CHECK(px(left, 15, 3) == px(img, 14, 3));  // uncovered: filled from the far side
    so.focus = 1.0f;  // near on the screen plane: it stays, far moves the other way
    auto [l2, r2] = stereoViews(img, near, so);
    CHECK(px(l2, 17, 3) == px(img, 17, 3) && px(l2, 1, 3) == px(img, 5, 3));
    CHECK_THROWS(stereoViews(img, std::vector<float>(3), so), std::invalid_argument);

    CHECK(stereoFrame(left, right, StereoLayout::SideBySide).width == 80);
    const Image half = stereoFrame(left, right, StereoLayout::HalfSideBySide);
    CHECK(half.width == 40 && half.height == 10 && px(half, 20, 0) == (px(right, 0, 0) + px(right, 1, 0) + 1) / 2);
    const Image tb = stereoFrame(left, right, StereoLayout::TopBottom);
    CHECK(tb.width == 40 && tb.height == 20 && px(tb, 11, 13) == px(right, 11, 3));
    Image white = img, black = img;
    std::fill(white.pixels.begin(), white.pixels.end(), 255);
    std::fill(black.pixels.begin(), black.pixels.end(), 0);
    const Image red = stereoFrame(white, black, StereoLayout::Anaglyph), cyan = stereoFrame(black, white, StereoLayout::Anaglyph);
    CHECK(red.pixels[0] > 150 && red.pixels[1] < 20 && red.pixels[2] < 20);
    CHECK(cyan.pixels[0] < 20 && cyan.pixels[1] > 150 && cyan.pixels[2] > 150);
    CHECK_THROWS(stereoFrame(left, tb, StereoLayout::SideBySide), std::invalid_argument);
    CHECK(stereoLayout("tb") == StereoLayout::TopBottom && stereoLayoutName(StereoLayout::HalfSideBySide) == "half_sbs");
    CHECK_THROWS(stereoLayout("3d"), std::invalid_argument);

    // Depth range: robust to outliers, smoothed from frame to frame.
    std::vector<float> depth(1000);
    for (std::size_t i = 0; i < depth.size(); ++i) depth[i] = static_cast<float>(i % 100);
    depth[7] = 1e9f;  // one wild value does not flatten the rest
    StereoDepthRange range(0.5f);
    const auto n1 = range.nearness(depth);
    CHECK(n1[0] == 0.0f && n1[98] == 1.0f && n1[50] > 0.45f && n1[50] < 0.55f && n1[7] == 1.0f);
    for (float& d : depth) d *= 2;  // the next frame's range jumps; the range moves halfway
    const auto n2 = range.nearness(depth);
    CHECK(n2[50] > 0.6f && n2[50] < 0.75f);
}

namespace {
// A stand-in 3D service on a local port: answers /health, and /v1/image-to-3d with `glb` (or a 500
// with a message when the seed is 13), recording what it was asked.
struct FakeService {
    int fd = -1;
    int port = 0;
    std::thread thread;
    std::vector<std::uint8_t> glb;
    std::vector<std::string> paths;
    std::vector<std::size_t> bodies;
    std::atomic<bool> stop{false};
    explicit FakeService(std::vector<std::uint8_t> answer) : glb(std::move(answer)) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        ::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a));
        socklen_t len = sizeof(a);
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
        port = ntohs(a.sin_port);
        ::listen(fd, 8);
        thread = std::thread([this] {
            while (!stop) {
                const int c = ::accept(fd, nullptr, nullptr);
                if (c < 0) break;
                std::string req;
                char buf[65536];
                std::size_t need = std::string::npos;
                for (;;) {
                    const ssize_t k = ::recv(c, buf, sizeof(buf), 0);
                    if (k <= 0) break;
                    req.append(buf, static_cast<std::size_t>(k));
                    const std::size_t end = req.find("\r\n\r\n");
                    if (end != std::string::npos && need == std::string::npos) {
                        const std::size_t cl = req.find("Content-Length: ");
                        need = end + 4 + (cl != std::string::npos && cl < end ? std::stoul(req.substr(cl + 16)) : 0);
                    }
                    if (need != std::string::npos && req.size() >= need) break;
                }
                const std::string path = req.substr(req.find(' ') + 1, req.find(" HTTP/") - req.find(' ') - 1);
                paths.push_back(path);
                bodies.push_back(need == std::string::npos ? 0 : req.size() - req.find("\r\n\r\n") - 4);
                std::string head;
                std::string body;
                if (path == "/health") {
                    head = "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n";
                    body = "{\"ok\":true}";
                } else if (path.find("seed=13") != std::string::npos) {
                    head = "HTTP/1.0 500 Internal Server Error\r\nContent-Type: application/json\r\n";
                    body = "{\"error\":\"out of memory\"}";
                } else {
                    head = "HTTP/1.1 200 OK\r\nContent-Type: model/gltf-binary\r\nTransfer-Encoding: chunked\r\n";
                    // chunked, in two pieces, as servers stream large answers
                    const std::string g(glb.begin(), glb.end());
                    const std::size_t half = g.size() / 2;
                    auto hex = [](std::size_t n) { std::ostringstream o; o << std::hex << n; return o.str(); };
                    body = hex(half) + "\r\n" + g.substr(0, half) + "\r\n" + hex(g.size() - half) + "\r\n" + g.substr(half) + "\r\n0\r\n\r\n";
                    head += "\r\n";
                    const std::string all = head + body;
                    ::send(c, all.data(), all.size(), MSG_NOSIGNAL);
                    ::close(c);
                    continue;
                }
                const std::string all = head + "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
                ::send(c, all.data(), all.size(), MSG_NOSIGNAL);
                ::close(c);
            }
        });
    }
    ~FakeService() {
        stop = true;
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        thread.join();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port); }
};
} // namespace

TEST(image_to_3d_goes_through_the_3d_service) {
    Mesh q;
    q.positions = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    q.uvs = {0, 1, 1, 1, 1, 0, 0, 0};
    q.indices = {0, 1, 2, 0, 2, 3};
    q.texture.width = q.texture.height = 2;
    q.texture.pixels = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0};
    const std::vector<std::uint8_t> answer = encodeGlb(q);
    FakeService service(answer);
    setImageTo3DService(service.url());
    CHECK(imageTo3DAvailable());

    const std::string dir = tempPath("image_to_3d");
    std::filesystem::create_directories(dir);
    Image pic;
    pic.width = pic.height = 8;
    pic.channels = 4;
    pic.pixels.assign(8 * 8 * 4, 200);
    savePng(dir + "/in.png", pic);
    const std::string wf = R"({"steps":[{"op":"load","path":"in.png","out":"pic"},)"
                           R"({"op":"image_to_3d","in":"pic","quality":"fast","seed":7,"faces":20000,"out":"model"},)"
                           R"({"op":"render","in":"model","width":32,"height":32,"out":"view"},)"
                           R"({"op":"save","in":"model","path":"model.glb"},{"op":"save","in":"model","path":"model.ply"}]})";
    const auto clips = runWorkflow(wf, dir);
    CHECK(clips.count("view") && clips.at("view")[0].width == 32);
    // The service's glTF is kept byte for byte; other formats are written from the mesh.
    std::ifstream f(dir + "/model.glb", std::ios::binary);
    const std::vector<std::uint8_t> saved((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(saved == answer);
    CHECK(loadMesh(dir + "/model.ply").triangleCount() == 2);
    const std::string asked = service.paths.size() >= 2 ? service.paths[1] : "";
    CHECK(asked.find("/v1/image-to-3d?quality=fast&seed=7&texture=2048&faces=20000") == 0);
    CHECK(service.bodies.size() >= 2 && service.bodies[1] > 50);  // the PNG went with it
    // The service's own message comes back; without a service the step says what is missing.
    bool told = false;
    try {
        runWorkflow(R"({"steps":[{"op":"load","path":"in.png","out":"pic"},{"op":"image_to_3d","in":"pic","seed":13}]})", dir);
    } catch (const std::runtime_error& e) {
        told = std::string(e.what()).find("out of memory") != std::string::npos;
    }
    CHECK(told);
    // Rigging: a model read from a glTF goes to the rigging service as it was read; one from another
    // format goes as a glTF the engine writes. The answer is kept as it came.
    CHECK_THROWS(runWorkflow(R"({"steps":[{"op":"load","path":"model.glb","out":"m"},{"op":"rig","in":"m"}]})", dir), std::runtime_error);
    setRigService(service.url());
    CHECK(rigAvailable());
    const std::size_t before = service.paths.size();
    runWorkflow(R"({"steps":[{"op":"load","path":"model.glb","out":"m"},{"op":"rig","in":"m","seed":5,"postprocess":false,"out":"r"},)"
                R"({"op":"save","in":"r","path":"rigged.glb"},{"op":"load","path":"model.ply","out":"p"},{"op":"rig","in":"p"}]})",
                dir);
    CHECK(service.paths.size() == before + 2);
    CHECK(service.paths[before] == "/v1/rig?seed=5&postprocess=0" && service.bodies[before] == answer.size());
    CHECK(service.paths[before + 1] == "/v1/rig?seed=42&postprocess=1" && service.bodies[before + 1] > 100);
    std::ifstream rf(dir + "/rigged.glb", std::ios::binary);
    CHECK(std::vector<std::uint8_t>((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>()) == answer);
    setRigService("");
    CHECK(!rigAvailable());
    setImageTo3DService("");
    CHECK(!imageTo3DAvailable());
    CHECK_THROWS(runWorkflow(R"({"steps":[{"op":"load","path":"in.png","out":"pic"},{"op":"image_to_3d","in":"pic"}]})", dir), std::runtime_error);
    setImageTo3DService("http://127.0.0.1:1");  // nothing listens there
    CHECK_THROWS(imageTo3D(pic), std::runtime_error);
    std::filesystem::remove_all(dir);
}

namespace {
// A rigged stick figure as a glTF binary: 13 joints (hips, chest, head; knees and feet; shoulders,
// elbows and hands), a small box on each joint, every box bound to its joint.
std::vector<std::uint8_t> stickFigureGlb(bool waving = false) {
    const char* names[] = {"hips", "chest", "head", "knee_l", "foot_l", "knee_r", "foot_r", "shoulder_l", "elbow_l", "hand_l", "shoulder_r", "elbow_r", "hand_r"};
    const int parents[] = {-1, 0, 1, 0, 3, 0, 5, 1, 7, 8, 1, 10, 11};
    const float worldA[][3] = {{0, 1, 0},          {0, 1.5f, 0},       {0, 2, 0},          {0.2f, 0.5f, 0},     {0.2f, 0, 0},
                              {-0.2f, 0.5f, 0},   {-0.2f, 0, 0},      {0.15f, 1.5f, 0},   {0.45f, 1.2f, 0},    {0.7f, 0.9f, 0},
                              {-0.15f, 1.5f, 0},  {-0.45f, 1.2f, 0},  {-0.7f, 0.9f, 0}};  // arms hang down (A-pose)
    float world[13][3];
    std::memcpy(world, worldA, sizeof(world));
    if (waving) {  // the right hand raised above the head
        const float elbow[3] = {-0.35f, 1.9f, 0}, hand[3] = {-0.4f, 2.3f, 0};
        std::memcpy(world[11], elbow, sizeof(elbow));
        std::memcpy(world[12], hand, sizeof(hand));
    }
    const int J = 13;
    std::vector<float> pos, w;
    std::vector<std::uint16_t> jn;
    std::vector<std::uint32_t> idx;
    for (int j = 0; j < J; ++j) {
        const std::uint32_t base = static_cast<std::uint32_t>(pos.size() / 3);
        for (int c = 0; c < 8; ++c) {
            pos.push_back(world[j][0] + (c & 1 ? 0.03f : -0.03f));
            pos.push_back(world[j][1] + (c & 2 ? 0.03f : -0.03f));
            pos.push_back(world[j][2] + (c & 4 ? 0.03f : -0.03f));
            jn.insert(jn.end(), {static_cast<std::uint16_t>(j), 0, 0, 0});
            w.insert(w.end(), {1, 0, 0, 0});
        }
        const std::uint32_t faces[] = {0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3};
        for (std::uint32_t f : faces) idx.push_back(base + f);
    }
    std::vector<float> ibm;
    for (int j = 0; j < J; ++j) ibm.insert(ibm.end(), {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -world[j][0], -world[j][1], -world[j][2], 1});
    std::vector<std::uint8_t> bin;
    auto put = [&](const void* p, std::size_t n) {
        const std::size_t at = bin.size();
        bin.insert(bin.end(), static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + n);
        while (bin.size() % 4) bin.push_back(0);
        return at;
    };
    const std::size_t oPos = put(pos.data(), pos.size() * 4), oJ = put(jn.data(), jn.size() * 2), oW = put(w.data(), w.size() * 4),
                      oI = put(idx.data(), idx.size() * 4), oB = put(ibm.data(), ibm.size() * 4);
    const std::size_t V = pos.size() / 3;
    std::ostringstream js;
    js.precision(9);
    js << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],"nodes":[{"mesh":0,"skin":0})";
    for (int j = 0; j < J; ++j) {
        const int p = parents[j];
        const float* pw = p < 0 ? nullptr : world[p];
        js << R"(,{"name":")" << names[j] << R"(","translation":[)" << world[j][0] - (pw ? pw[0] : 0) << "," << world[j][1] - (pw ? pw[1] : 0) << ","
           << world[j][2] - (pw ? pw[2] : 0) << "]";
        std::string kids;
        for (int k = 0; k < J; ++k)
            if (parents[k] == j) kids += (kids.empty() ? "" : ",") + std::to_string(k + 1);
        if (!kids.empty()) js << R"(,"children":[)" << kids << "]";
        js << "}";
    }
    js << R"(],"skins":[{"inverseBindMatrices":4,"joints":[)";
    for (int j = 0; j < J; ++j) js << (j ? "," : "") << j + 1;
    js << R"(]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"JOINTS_0":1,"WEIGHTS_0":2},"indices":3}]}],)";
    js << R"("buffers":[{"byteLength":)" << bin.size() << "}],";
    js << R"("bufferViews":[{"buffer":0,"byteOffset":)" << oPos << R"(,"byteLength":)" << V * 12 << "},"
       << R"({"buffer":0,"byteOffset":)" << oJ << R"(,"byteLength":)" << V * 8 << "},"
       << R"({"buffer":0,"byteOffset":)" << oW << R"(,"byteLength":)" << V * 16 << "},"
       << R"({"buffer":0,"byteOffset":)" << oI << R"(,"byteLength":)" << idx.size() * 4 << "},"
       << R"({"buffer":0,"byteOffset":)" << oB << R"(,"byteLength":)" << J * 64 << "}],";
    float lo[3] = {1e9f, 1e9f, 1e9f}, hi[3] = {-1e9f, -1e9f, -1e9f};
    for (std::size_t i = 0; i < pos.size(); ++i) {
        lo[i % 3] = std::min(lo[i % 3], pos[i]);
        hi[i % 3] = std::max(hi[i % 3], pos[i]);
    }
    js << R"("accessors":[{"bufferView":0,"componentType":5126,"count":)" << V << R"(,"type":"VEC3","min":[)" << lo[0] << "," << lo[1] << "," << lo[2]
       << R"(],"max":[)" << hi[0] << "," << hi[1] << "," << hi[2] << "]},"
       << R"({"bufferView":1,"componentType":5123,"count":)" << V << R"(,"type":"VEC4"},)"
       << R"({"bufferView":2,"componentType":5126,"count":)" << V << R"(,"type":"VEC4"},)"
       << R"({"bufferView":3,"componentType":5125,"count":)" << idx.size() << R"(,"type":"SCALAR"},)"
       << R"({"bufferView":4,"componentType":5126,"count":)" << J << R"(,"type":"MAT4"}]})";
    std::string json = js.str();
    while (json.size() % 4) json.push_back(' ');
    std::vector<std::uint8_t> out;
    auto u32 = [&](std::uint32_t v) {
        for (int k = 0; k < 4; ++k) out.push_back(static_cast<std::uint8_t>(v >> (8 * k)));
    };
    u32(0x46546C67);
    u32(2);
    u32(static_cast<std::uint32_t>(28 + json.size() + bin.size()));
    u32(static_cast<std::uint32_t>(json.size()));
    u32(0x4E4F534A);
    out.insert(out.end(), json.begin(), json.end());
    u32(static_cast<std::uint32_t>(bin.size()));
    u32(0x004E4942);
    out.insert(out.end(), bin.begin(), bin.end());
    return out;
}

// A BVH stick figure in T-pose (arms out along x, in other units) that steps forward and lifts its
// left arm over the frames; the first frame is CMU's T-pose, which the motion jumps away from.
std::string stickFigureBvh() {
    std::ostringstream b;
    b << "HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
      << " JOINT LeftUpLeg\n {\n  OFFSET 2 -1 0\n  CHANNELS 3 Zrotation Yrotation Xrotation\n  JOINT LeftLeg\n  {\n   OFFSET 0 -9 0\n"
      << "   CHANNELS 3 Zrotation Yrotation Xrotation\n   End Site\n   {\n    OFFSET 0 -10 0\n   }\n  }\n }\n"
      << " JOINT RightUpLeg\n {\n  OFFSET -2 -1 0\n  CHANNELS 3 Zrotation Yrotation Xrotation\n  JOINT RightLeg\n  {\n   OFFSET 0 -9 0\n"
      << "   CHANNELS 3 Zrotation Yrotation Xrotation\n   End Site\n   {\n    OFFSET 0 -10 0\n   }\n  }\n }\n"
      << " JOINT Spine\n {\n  OFFSET 0 10 0\n  CHANNELS 3 Zrotation Yrotation Xrotation\n"
      << "  JOINT Head\n  {\n   OFFSET 0 10 0\n   CHANNELS 3 Zrotation Yrotation Xrotation\n   End Site\n   {\n    OFFSET 0 5 0\n   }\n  }\n"
      << "  JOINT LeftArm\n  {\n   OFFSET 3 8 0\n   CHANNELS 3 Zrotation Yrotation Xrotation\n   JOINT LeftForeArm\n   {\n    OFFSET 8 0 0\n"
      << "    CHANNELS 3 Zrotation Yrotation Xrotation\n    End Site\n    {\n     OFFSET 8 0 0\n    }\n   }\n  }\n"
      << "  JOINT RightArm\n  {\n   OFFSET -3 8 0\n   CHANNELS 3 Zrotation Yrotation Xrotation\n   JOINT RightForeArm\n   {\n    OFFSET -8 0 0\n"
      << "    CHANNELS 3 Zrotation Yrotation Xrotation\n    End Site\n    {\n     OFFSET -8 0 0\n    }\n   }\n  }\n }\n}\n";
    // 10 channels groups: hips 6, then 3 for each of the 8 joints: LeftUpLeg LeftLeg RightUpLeg RightLeg Spine Head LeftArm LeftForeArm RightArm RightForeArm
    b << "MOTION\nFrames: 31\nFrame Time: 0.0333333\n";
    b << "0 20 0 0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0\n";  // T-pose
    for (int f = 1; f <= 30; ++f) {
        const double t = (f - 1) / 29.0;
        // the left arm swings from hanging down (-80 degrees around z) to straight up (+80); hips walk 12 units along +z
        b << "0 20 " << 12 * t << " 0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  0 0 0  " << (-80 + 160 * t) << " 0 0  0 0 0  -80 0 0  0 0 0\n";
    }
    return b.str();
}
} // namespace

TEST(motion_capture_animates_a_rigged_figure) {
    const std::vector<std::uint8_t> glb = stickFigureGlb();
    Mesh fig = decodeGlb(glb, "stick figure");
    fig.glb = glb;  // as loadMesh keeps it
    CHECK(fig.rig.jointCount() == 13 && fig.rig.frameCount() == 0 && fig.rig.names[0] == "hips");
    CHECK(fig.vertexCount() == 104 && std::fabs(fig.positions[1] - 0.97f) < 1e-4f);  // the rest pose: boxes on their joints

    const MotionCapture mo = parseBvh(stickFigureBvh());
    CHECK(mo.jointCount() == 16 && mo.frameCount() == 31 && std::fabs(mo.fps - 30) < 0.01);
    CHECK_THROWS(parseBvh("HIERARCHY\nROOT a\n{\n OFFSET 0 0\n"), std::runtime_error);
    CHECK_THROWS(parseBvh("MOTION"), std::runtime_error);

    std::vector<std::string> report;
    AnimateOptions ao;
    ao.fps = 10;
    const Mesh moving = animateMesh(fig, mo, ao, &report);
    CHECK(report.size() == 6 && report[0] == "spine: hips > chest > head" && report[1] == "left leg: hips > knee_l > foot_l");
    CHECK(report[2] == "left arm: chest > shoulder_l > elbow_l > hand_l");
    CHECK(moving.rig.frameCount() == 10 && !moving.glb.empty());  // 29 frames at 30 fps (the T-pose skipped) -> 0.93 s at 10 fps
    // The saved glTF has the animation; reading it back gives the same frames.
    const Mesh again = decodeGlb(moving.glb, "animated");
    CHECK(again.rig.frameCount() == 10);
    // Start: the left arm hangs down and to the side (as the motion's, -80 degrees); end: it points up.
    auto hand = [](const Mesh& m) {  // centre of the left hand's box (vertices 72..79)
        double c[3] = {0, 0, 0};
        for (int v = 72; v < 80; ++v)
            for (int k = 0; k < 3; ++k) c[k] += m.positions[3 * v + k] / 8.0;
        return std::array<double, 3>{c[0], c[1], c[2]};
    };
    const auto h0 = hand(poseMesh(moving, 0)), h1 = hand(poseMesh(moving, moving.rig.duration()));
    CHECK(h0[1] < 1.0 && h0[0] > 0.15 && h1[1] > 2.0 && h1[0] > 0.15 && h1[0] < 0.5);  // up above the shoulder, on its side (+x)
    // The hips walk forward (+z), scaled from the motion's 12 units by the hip heights (1 / 20).
    const auto hipsZ = [](const Mesh& m) { return (m.positions[2] + m.positions[3 * 7 + 2]) / 2; };
    CHECK(std::fabs(hipsZ(poseMesh(moving, moving.rig.duration())) - hipsZ(poseMesh(moving, 0)) - 0.6 * 28.0 / 29.0 * 0.93 / 0.9667) < 0.05);
    ao.inPlace = true;
    const Mesh still = animateMesh(fig, mo, ao);
    CHECK(std::fabs(hipsZ(poseMesh(still, still.rig.duration())) - hipsZ(poseMesh(still, 0))) < 1e-3);
    // Without a rig, or with no legs to find, it says so.
    Mesh plain = fig;
    plain.rig = MeshRig{};
    CHECK_THROWS(animateMesh(plain, mo), std::invalid_argument);
    // A raised hand higher than the head is still an arm (the head is the highest end over the hips).
    const std::vector<std::uint8_t> wavingGlb = stickFigureGlb(true);
    Mesh waving = decodeGlb(wavingGlb);
    waving.glb = wavingGlb;
    report.clear();
    animateMesh(waving, mo, ao, &report);
    CHECK(report[0] == "spine: hips > chest > head" && report[4] == "right arm: chest > shoulder_r > elbow_r > hand_r");
    // In a workflow: animate, render the animation (camera still), save the animated model.
    const std::string dir = tempPath("animate");
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/fig.glb", std::ios::binary).write(reinterpret_cast<const char*>(glb.data()), static_cast<std::streamsize>(glb.size()));
    std::ofstream(dir + "/move.bvh") << stickFigureBvh();
    const auto clips = runWorkflow(R"({"steps":[{"op":"load","path":"fig.glb","out":"f"},{"op":"animate","in":"f","capture":"move.bvh","fps":10,"out":"a"},)"
                                   R"({"op":"render","in":"a","width":48,"height":48,"out":"v"},{"op":"save","in":"a","path":"a.glb"}]})",
                                   dir);
    CHECK(clips.at("v").size() == 10 && clips.at("v")[0].pixels != clips.at("v")[9].pixels);
    CHECK(loadMesh(dir + "/a.glb").rig.frameCount() == 10);
    CHECK(checkWorkflow(R"({"steps":[{"op":"animate","in":"m"}]})").size() == 1);  // the capture is required
    const auto found = findModels(dir);  // motion captures are listed with the models (kind "mocap")
    CHECK(std::any_of(found.begin(), found.end(), [](const ModelInfo& m) { return m.kind == "mocap" && m.name == "move.bvh"; }));
    std::filesystem::remove_all(dir);
}

TEST(meshes_round_trip_render_and_come_from_depth) {
    // A textured quad leaning back, and a coloured triangle.
    Mesh q;
    q.positions = {-1, -1, 0, 1, -1, 0, 1, 1, -0.5f, -1, 1, -0.5f};
    q.uvs = {0, 1, 1, 1, 1, 0, 0, 0};
    q.indices = {0, 1, 2, 0, 2, 3};
    q.texture.width = 4;
    q.texture.height = 4;
    for (std::size_t i = 0; i < 16; ++i) q.texture.pixels.insert(q.texture.pixels.end(), {static_cast<std::uint8_t>(i * 16), 40, 200});
    computeNormals(q);
    const std::string dir = tempPath("meshes");
    std::filesystem::create_directories(dir);
    for (const char* ext : {".glb", ".obj", ".ply"}) {
        saveMesh(dir + "/q" + ext, q);
        const Mesh r = loadMesh(dir + "/q" + ext);
        CHECK(r.triangleCount() == 2);
        // Positions survive (OBJ / PLY may re-index; compare as sets of corners).
        std::vector<float> a = q.positions, b = r.positions;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        bool same = a.size() == b.size();
        for (std::size_t i = 0; same && i < a.size(); ++i) same = std::fabs(a[i] - b[i]) < 1e-5f;
        if (!same) std::cout << "         " << ext << " positions differ\n";
        CHECK(same);
        CHECK(r.normals.size() == r.positions.size());
    }
    const Mesh g = loadMesh(dir + "/q.glb");
    CHECK(g.texture.width == 4 && g.texture.pixels == q.texture.pixels && g.uvs == q.uvs);
    CHECK(!loadMesh(dir + "/q.ply").colors.empty() && std::filesystem::exists(dir + "/q_albedo.png") && std::filesystem::exists(dir + "/q.mtl"));
    // Broken files are refused, not trusted.
    std::vector<std::uint8_t> bad = encodeGlb(q);
    bad.resize(bad.size() / 2);
    CHECK_THROWS(decodeGlb(bad), std::runtime_error);
    CHECK_THROWS(parseObj("v 0 0 0\nf 1 2 3\n"), std::runtime_error);
    CHECK_THROWS(decodePly(std::vector<std::uint8_t>{'p', 'l', 'y', '\n'}), std::runtime_error);
    Mesh wrong = q;
    wrong.indices.push_back(9);
    CHECK_THROWS(checkMesh(wrong), std::invalid_argument);

    // Rendering: the quad covers the middle, the corners keep the background; a turn changes the view.
    MeshRenderOptions ro;
    ro.width = ro.height = 48;
    ro.pitch = 0;
    ro.background[0] = ro.background[1] = ro.background[2] = 0;
    const Image front = renderMesh(q, ro);
    auto at = [](const Image& im, std::size_t x, std::size_t y, int c) { return im.pixels[3 * (y * im.width + x) + c]; };
    CHECK(at(front, 24, 24, 2) > 60 && at(front, 0, 0, 0) == 0 && at(front, 0, 0, 2) == 0);
    const auto turn = renderTurntable(q, 4, 360, ro);
    CHECK(turn.size() == 4 && turn[0].pixels == front.pixels && turn[1].pixels != front.pixels);

    // From depth: a near square on a far background; edges split them apart unless edges = 0.
    Image pic;
    pic.width = 40;
    pic.height = 30;
    pic.pixels.assign(40 * 30 * 3, 128);
    std::vector<float> depth(40 * 30, 0.0f);
    for (std::size_t y = 10; y < 20; ++y)
        for (std::size_t x = 15; x < 25; ++x) depth[y * 40 + x] = 1.0f;
    DepthMeshOptions mo;
    mo.grid = 40;
    const Mesh cut = meshFromDepth(pic, depth, mo);
    mo.edges = 0;
    const Mesh whole = meshFromDepth(pic, depth, mo);
    CHECK(whole.triangleCount() == 2 * 39 * 29 && cut.triangleCount() < whole.triangleCount() && cut.texture.width == 40);
    float zmax = 0;
    for (std::size_t i = 0; i < whole.vertexCount(); ++i) zmax = std::max(zmax, whole.positions[3 * i + 2]);
    CHECK(std::fabs(zmax - mo.depth) < 1e-6f);
    CHECK_THROWS(meshFromDepth(pic, std::vector<float>(5), mo), std::invalid_argument);
    std::filesystem::remove_all(dir);
}

TEST(ip_adapter_image_prompts) {
    std::mt19937 rng(21);
    auto rnd = [&](const std::string& name, std::size_t) {
        if (name.find("norm") != std::string::npos && name.find(".weight") != std::string::npos) return 1.0f;
        if (name.find("layer_norm") != std::string::npos && name.find(".weight") != std::string::npos) return 1.0f;
        if (name.find("layrnorm") != std::string::npos && name.find(".weight") != std::string::npos) return 1.0f;
        return (static_cast<float>(rng() % 2001) / 1000.0f - 1.0f) * 0.4f;
    };
    // A tiny CLIP vision encoder: 28 x 28 input, 14-pixel patches, hidden 8, 2 layers, projection 5.
    const std::string enc = tempPath("clip_vision");
    std::filesystem::create_directories(enc);
    std::ofstream(enc + "/config.json") << R"({"hidden_size":8,"num_attention_heads":2,"num_hidden_layers":2,"patch_size":14,)"
                                       << R"("image_size":28,"projection_dim":5,"hidden_act":"gelu","layer_norm_eps":1e-5})";
    std::vector<std::pair<std::string, std::vector<std::size_t>>> v;
    const std::string p = "vision_model.";
    v.push_back({p + "embeddings.class_embedding", {8}});
    v.push_back({p + "embeddings.patch_embedding.weight", {8, 3, 14, 14}});
    v.push_back({p + "embeddings.position_embedding.weight", {5, 8}});
    for (const char* n : {"pre_layrnorm", "post_layernorm"}) {
        v.push_back({p + n + ".weight", {8}});
        v.push_back({p + n + ".bias", {8}});
    }
    for (int i = 0; i < 2; ++i) {
        const std::string l = p + "encoder.layers." + std::to_string(i) + ".";
        for (const char* n : {"layer_norm1", "layer_norm2"}) {
            v.push_back({l + n + ".weight", {8}});
            v.push_back({l + n + ".bias", {8}});
        }
        for (const char* n : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"}) {
            v.push_back({l + n + ".weight", {8, 8}});
            v.push_back({l + n + ".bias", {8}});
        }
        v.push_back({l + "mlp.fc1.weight", {32, 8}});
        v.push_back({l + "mlp.fc1.bias", {32}});
        v.push_back({l + "mlp.fc2.weight", {8, 32}});
        v.push_back({l + "mlp.fc2.bias", {8}});
    }
    v.push_back({"visual_projection.weight", {5, 8}});
    saveShaped(enc + "/model.safetensors", v, rnd);
    ClipImageEncoder clip(enc);
    Image pic;
    pic.width = 40;
    pic.height = 30;
    for (std::size_t i = 0; i < 40 * 30 * 3; ++i) pic.pixels.push_back(static_cast<std::uint8_t>(i * 23 % 256));
    const Matrix e = clip.embed(pic);
    CHECK(e.rows() == 1 && e.cols() == 5 && clip.imageSize() == 28);
    CHECK_THROWS(clip.forward(std::vector<float>(10)), std::invalid_argument);

    // The adapter for the tiny UNet (cross-attention channels 8, 8, 8, 16; context 6): 2 tokens.
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_ip", tok);
    const std::string adapter = dir + "/ip.safetensors";
    std::vector<std::pair<std::string, std::vector<std::size_t>>> a{{"image_proj.proj.weight", {12, 5}},
                                                                   {"image_proj.proj.bias", {12}},
                                                                   {"image_proj.norm.weight", {6}},
                                                                   {"image_proj.norm.bias", {6}}};
    const std::size_t ch[4] = {8, 8, 8, 16};
    for (int i = 0; i < 4; ++i)
        for (const char* n : {".to_k_ip.weight", ".to_v_ip.weight"}) a.push_back({"ip_adapter." + std::to_string(2 * i + 1) + n, {ch[i], 6}});
    saveShaped(adapter, a, rnd);
    const IpAdapter ip(adapter);
    const Matrix tokens = ip.tokens(e);
    CHECK(tokens.rows() == 2 && tokens.cols() == 6);
    double mean = 0;
    for (std::size_t c = 0; c < 6; ++c) mean += tokens(0, c);
    CHECK(std::abs(mean / 6) < 0.5);  // layer-normed (weight 1, small bias)

    UNet2DCondition unet = loadUNet(dir, 8, 8);
    const Matrix latent = Matrix::normal(1, 4 * 8 * 8, 0.0f, 1.0f), ctx = Matrix::uniform(8, 6, -1.0f, 1.0f);
    const Matrix plain = unet.forward(latent, 100.0f, ctx);
    CHECK_THROWS(unet.setImagePrompt(tokens), std::logic_error);
    loadIpAdapter(unet, adapter);
    CHECK(unet.hasImagePrompt());
    CHECK(unet.forward(latent, 100.0f, ctx).values() == plain.values());  // no tokens set yet
    unet.setImagePrompt(tokens, 0.0f);
    CHECK(unet.forward(latent, 100.0f, ctx).values() == plain.values());
    unet.setImagePrompt(tokens, 1.0f);
    CHECK(unet.forward(latent, 100.0f, ctx).values() != plain.values());
    unet.setImagePrompt(Matrix(), 1.0f);
    CHECK(unet.forward(latent, 100.0f, ctx).values() == plain.values());
    CHECK_THROWS(unet.setImagePrompt(Matrix(2, 5)), std::invalid_argument);

    // Pipeline: scale 0 equals no image prompt.
    TextToImageOptions o;
    o.width = o.height = 64;
    o.steps = 2;
    const Image base = textToImage(dir, "hello", o);
    o.ipAdapter = adapter;
    o.imageEncoder = enc;
    o.imagePrompt = pic;
    o.ipScale = 0.0f;
    CHECK(textToImage(dir, "hello", o).pixels == base.pixels);
    o.ipScale = 1.0f;
    CHECK(textToImage(dir, "hello", o).pixels != base.pixels);
    // The session uses the image prompt for every generation.
    StableDiffusion withIp(dir, o, false);
    o.ipScale = 0.0f;
    StableDiffusion zeroIp(dir, o, false);
    TextToImageOptions plainOptions;
    plainOptions.width = plainOptions.height = 64;
    StableDiffusion plainSd(dir, plainOptions, false);
    const Image p0 = plainSd.generate("hello", 3, 2);
    CHECK(zeroIp.generate("hello", 3, 2).pixels == p0.pixels);
    CHECK(withIp.generate("hello", 3, 2).pixels != p0.pixels);
    for (const std::string& d : {dir, tok, enc}) std::filesystem::remove_all(d);
}

TEST(vector_exp_erf_and_softmax_match_std) {
    std::vector<float> x;
    for (double v = -30; v <= 30; v += 0.00731) x.push_back(static_cast<float>(v));
    for (float v : {-1000.0f, -105.0f, -87.0f, 0.0f, 88.0f, 89.5f, -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity()})
        x.push_back(v);
    std::vector<float> e(x.size()), r(x.size());
    expArray(x.data(), e.data(), x.size());
    erfArray(x.data(), r.data(), x.size());
    double expWorst = 0, erfWorst = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double ref = std::exp(static_cast<double>(x[i]));
        if (ref > 1e-37 && ref < 1e38) expWorst = std::max(expWorst, std::fabs(e[i] - ref) / ref);
        if (std::isfinite(x[i])) erfWorst = std::max(erfWorst, std::fabs(r[i] - std::erf(static_cast<double>(x[i]))));
    }
    CHECK(expWorst < 2.5e-7 && erfWorst < 4e-7);
    const std::size_t n = x.size();
    CHECK(e[n - 8] == 0.0f && e[n - 7] == 0.0f && e[n - 2] == 0.0f);  // exp(-1000), exp(-105), exp(-inf)
    CHECK(std::isinf(e[n - 3]) && std::isinf(e[n - 1]));                  // exp(89.5), exp(inf)
    CHECK(r[n - 2] == -1.0f && r[n - 1] == 1.0f && r[n - 5] == 0.0f);
    // softmaxRow: sums to one, matches the direct formula, -inf entries get exactly zero.
    std::vector<float> row{1.0f, 2.0f, -std::numeric_limits<float>::infinity(), 0.5f, 3.0f};
    softmaxRow(row.data(), row.size(), 0.5f);
    double z = 0;
    for (float v : {1.0f, 2.0f, 0.5f, 3.0f}) z += std::exp(0.5 * v);
    CHECK_NEAR(row[1], std::exp(1.0) / z, 1e-6);
    CHECK(row[2] == 0.0f);
    CHECK_NEAR(row[0] + row[1] + row[3] + row[4], 1.0, 1e-6);
}

TEST(model_cache_loads_once_and_releases) {
    CHECK(!modelCaching());
    int loads = 0;
    auto load = [&] {
        ++loads;
        return std::make_shared<int>(7);
    };
    // Off: every call loads.
    cachedModel<int>("k", "t", load);
    cachedModel<int>("k", "t", load);
    CHECK(loads == 2 && modelCacheEntries() == 0);
    {
        ModelCacheScope scope;
        const auto a = cachedModel<int>("k", "data/x@64x64", load);
        const auto b = cachedModel<int>("k", "data/x@64x64", load);
        cachedModel<int>("other", "data/y", load);
        CHECK(loads == 4 && a == b && modelCacheEntries() == 2);
        releaseModelsExcept({"data/y"});
        CHECK(modelCacheEntries() == 1 && *a == 7);  // a caller's copy outlives the entry
        cachedModel<int>("k", "data/x@64x64", load);
        CHECK(loads == 5);
    }
    CHECK(!modelCaching() && modelCacheEntries() == 0);
    CHECK(modelTag("data/sd/../sd/") == "data/sd" && modelTag("data/sd", 512, 768) == "data/sd@512x768");
}

TEST(workflow_schema_checks_and_lists_models) {
    // The schema is valid JSON with every op, and each op's field names are unique.
    const json::Value schema = json::parse(workflowSchemaJson());
    CHECK(schema.at("ops").array.size() == workflowOps().size());
    for (const WorkflowOp& op : workflowOps()) {
        std::set<std::string> names;
        for (const WorkflowField& f : op.fields) CHECK(names.insert(f.name).second);
        // Every op is one runWorkflow knows: only its missing fields are reported.
        for (const std::string& p : checkWorkflow("{\"steps\": [{\"op\": \"" + op.name + "\"}]}"))
            CHECK(p.find("unknown op") == std::string::npos && p.find("needs") != std::string::npos);
    }
    CHECK(checkWorkflow(R"({"steps": [{"op": "upscale", "model": "m", "in": "a", "ui": {"x": 10}, "label": "x", "_note": 1}]})").empty());
    const auto bad = checkWorkflow(R"({"steps": [{"op": "txt2imgg"}, {"op": "txt2img", "model": "m", "promt": "a", "steps": 0, "sampler": "ddim", "int8": 1}]})");
    CHECK(bad.size() == 6);
    CHECK(bad[0].find("did you mean \"txt2img\"") != std::string::npos);
    CHECK(bad[1].find("did you mean \"prompt\"") != std::string::npos);
    CHECK(checkWorkflow("{nope").size() == 1 && checkWorkflow("{}").size() == 1);
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "upscale", "modle": "m", "in": "a"}]})"), std::invalid_argument);

    // JSON writer round trip.
    const json::Value v = json::parse(R"({"a": [1, 2.5, "x\"y\n", true, null], "b": {"c": -3}})");
    CHECK(json::dump(v) == R"({"a":[1,2.5,"x\"y\n",true,null],"b":{"c":-3}})");
    CHECK(json::dump(json::parse(json::dump(v))) == json::dump(v));

    // Model discovery.
    const std::string dir = (std::filesystem::temp_directory_path() / "nc_models_test").string();
    std::filesystem::remove_all(dir);
    std::string tok;
    std::filesystem::create_directories(dir);
    std::filesystem::rename(writeTinyStableDiffusion("sd_for_models", tok), dir + "/sd");
    std::filesystem::create_directories(dir + "/loras");
    std::map<std::string, Matrix> lora{{"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", Matrix(2, 4)}};
    saveAll(dir + "/loras/style.safetensors", lora);
    std::map<std::string, Matrix> junk{{"something.weight", Matrix(2, 2)}};
    saveAll(dir + "/loras/other.safetensors", junk);
    const auto models = findModels(dir);
    std::map<std::string, std::string> kinds;
    for (const ModelInfo& m : models) kinds[std::filesystem::path(m.path).filename().string()] = m.kind;
    CHECK(kinds["sd"] == "sd" && kinds["style.safetensors"] == "lora" && kinds.count("other.safetensors") == 0);
    CHECK(json::parse(modelsJson(models)).array.size() == models.size());
    std::filesystem::remove_all(dir);
}

#ifndef _WIN32
namespace {
// One HTTP/1.1 request to 127.0.0.1:port; returns the status line's code and the body (and the raw head).
struct HttpReply {
    int status = 0;
    std::string head, body;
};
HttpReply http(int port, const std::string& method, const std::string& path, const std::string& body = "",
               const std::string& extraHeaders = "Content-Type: application/json\r\n") {
    HttpReply r;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<std::uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        ::close(fd);
        return r;
    }
    const std::string req = method + " " + path + " HTTP/1.1\r\nHost: localhost\r\n" + extraHeaders +
                            "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    ::send(fd, req.data(), req.size(), 0);
    std::string data;
    char buf[4096];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof buf, 0)) > 0) data.append(buf, static_cast<std::size_t>(n));
    ::close(fd);
    const std::size_t end = data.find("\r\n\r\n");
    if (end == std::string::npos) return r;
    r.head = data.substr(0, end);
    r.body = data.substr(end + 4);
    r.status = std::atoi(r.head.c_str() + 9);
    return r;
}
} // namespace

TEST(studio_server_api) {
    const std::string work = tempPath("studio_server");
    std::filesystem::create_directories(work);
    Image a;
    a.width = 40;
    a.height = 30;
    for (std::size_t i = 0; i < 40 * 30 * 3; ++i) a.pixels.push_back(static_cast<std::uint8_t>(i * 7 % 256));
    savePng(work + "/a.png", a);
    ServerOptions o;
    o.port = 0;
    o.baseDir = work;
    o.dataDir = work;
    o.keepJobs = 2;
    StudioServer server(o);
    server.start();
    const int port = server.port();
    CHECK(port > 0);

    // Read-only endpoints.
    HttpReply info = http(port, "GET", "/api/info");
    CHECK(info.status == 200 && json::parse(info.body).at("api").number == 1);
    CHECK(json::parse(http(port, "GET", "/api/ops").body).at("ops").array.size() == workflowOps().size());
    CHECK(json::parse(http(port, "GET", "/api/models").body).isArray());
    CHECK(http(port, "GET", "/api/nothing").status == 404);
    {
        const json::Value cache = json::parse(http(port, "GET", "/api/cache").body);
        CHECK(cache.at("enabled").boolean && cache.at("entries").number == 0);
        CHECK(http(port, "POST", "/api/cache/clear", "{}").status == 200 && http(port, "POST", "/api/cache/nope", "{}").status == 404);
    }
    const json::Value check = json::parse(http(port, "POST", "/api/check", R"({"steps": [{"op": "fly"}]})").body);
    CHECK(check.at("problems").array.size() == 1);

    // Browser rules: JSON bodies only, local origins only; preflight answers local origins.
    CHECK(http(port, "POST", "/api/run", "{}", "Content-Type: text/plain\r\n").status == 415);
    CHECK(http(port, "GET", "/api/info", "", "Origin: http://example.com\r\n").status == 403);
    const HttpReply pre = http(port, "OPTIONS", "/api/run", "", "Origin: http://localhost:5173\r\n");
    CHECK(pre.status == 204 && pre.head.find("Access-Control-Allow-Origin: http://localhost:5173") != std::string::npos);

    // A job, followed on the event stream.
    std::string events;
    std::atomic<bool> listening{false};
    std::thread listener([&] {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<std::uint16_t>(port));
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) return;
        const std::string req = "GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n";
        ::send(fd, req.data(), req.size(), 0);
        timeval tv{5, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        char buf[4096];
        ssize_t n;
        while ((n = ::recv(fd, buf, sizeof buf, 0)) > 0) {
            events.append(buf, static_cast<std::size_t>(n));
            if (events.find(": connected") != std::string::npos) listening = true;
            if (events.find("\"state\":\"done\"") != std::string::npos) break;
        }
        ::close(fd);
    });
    for (int i = 0; i < 500 && !listening; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(listening.load());
    const std::string wf = R"({"steps": [{"op": "load", "path": "a.png", "out": "a"},
        {"op": "resize", "in": "a", "width": 20, "height": 16, "out": "small"}]})";
    const HttpReply run = http(port, "POST", "/api/run", wf);
    CHECK(run.status == 200);
    const int id = static_cast<int>(json::parse(run.body).at("job").number);
    json::Value job;
    for (int i = 0; i < 500; ++i) {
        job = json::parse(http(port, "GET", "/api/jobs/" + std::to_string(id)).body);
        if (job.at("state").string == "done" || job.at("state").string == "failed") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(job.at("state").string == "done" && job.at("outputs").array.size() == 2);
    listener.join();
    for (const char* e : {"event: job", "event: progress", "event: output", "event: log"}) CHECK(events.find(e) != std::string::npos);
    const HttpReply frame = http(port, "GET", "/api/jobs/" + std::to_string(id) + "/clips/small/0.jpg");
    CHECK(frame.status == 200 && frame.head.find("image/jpeg") != std::string::npos && frame.body.size() > 100);
    CHECK(static_cast<unsigned char>(frame.body[0]) == 0xFF && static_cast<unsigned char>(frame.body[1]) == 0xD8);
    CHECK(http(port, "GET", "/api/jobs/" + std::to_string(id) + "/clips/small/5.jpg").status == 404);
    CHECK(http(port, "GET", "/api/jobs/999").status == 404);
    CHECK(json::parse(http(port, "GET", "/api/jobs").body).array.size() == 1);
    // A workflow with problems is refused before it is queued.
    CHECK(http(port, "POST", "/api/run", R"({"steps": [{"op": "resize"}]})").status == 400);
    // A failing job reports its error.
    const int bad = static_cast<int>(json::parse(http(port, "POST", "/api/run", R"({"steps": [{"op": "load", "path": "missing.png"}]})").body)
                                         .at("job")
                                         .number);
    for (int i = 0; i < 500; ++i) {
        job = json::parse(http(port, "GET", "/api/jobs/" + std::to_string(bad)).body);
        if (job.at("state").string != "queued" && job.at("state").string != "running") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(job.at("state").string == "failed" && !job.at("error").string.empty());
    // Uploads: base64 PNG saved under <base>/uploads, usable by a workflow; anything else is refused.
    {
        std::ifstream f(work + "/a.png", std::ios::binary);
        const std::string png((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string b64;
        for (std::size_t i = 0; i < png.size(); i += 3) {
            const unsigned n = (static_cast<unsigned char>(png[i]) << 16) | (i + 1 < png.size() ? static_cast<unsigned char>(png[i + 1]) << 8 : 0) |
                               (i + 2 < png.size() ? static_cast<unsigned char>(png[i + 2]) : 0);
            b64 += abc[(n >> 18) & 63];
            b64 += abc[(n >> 12) & 63];
            b64 += i + 1 < png.size() ? abc[(n >> 6) & 63] : '=';
            b64 += i + 2 < png.size() ? abc[n & 63] : '=';
        }
        const HttpReply up = http(port, "POST", "/api/upload", R"({"name": "my photo.png", "data": ")" + b64 + R"("})");
        CHECK(up.status == 200);
        const std::string path = json::parse(up.body).at("path").string;
        CHECK(path.rfind("uploads/", 0) == 0 && path.find(' ') == std::string::npos);
        CHECK(loadImage(work + "/" + path).pixels == a.pixels);
        CHECK(http(port, "POST", "/api/upload", R"({"data": "aGVsbG8="})").status == 400);  // "hello", not a PNG
    }
    // The same workflow again comes from the result cache; with keepJobs = 2 the first job gives its frames back.
    {
        const int again = static_cast<int>(json::parse(http(port, "POST", "/api/run", wf).body).at("job").number);
        for (int i = 0; i < 500; ++i) {
            job = json::parse(http(port, "GET", "/api/jobs/" + std::to_string(again)).body);
            if (job.at("state").string == "done" || job.at("state").string == "failed") break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(job.at("state").string == "done" && job.at("outputs").array.size() == 2);
        for (const json::Value& out : job.at("outputs").array) CHECK(out.at("seconds").number == 0);
        CHECK(json::parse(http(port, "GET", "/api/cache").body).at("hits").number == 2);
        for (int i = 0; i < 100 && json::parse(http(port, "GET", "/api/jobs/" + std::to_string(id)).body).at("clipsKept").boolean; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(!json::parse(http(port, "GET", "/api/jobs/" + std::to_string(id)).body).at("clipsKept").boolean);
        CHECK(http(port, "GET", "/api/jobs/" + std::to_string(id) + "/clips/small/0.jpg").status == 404);
        CHECK(http(port, "GET", "/api/jobs/" + std::to_string(again) + "/clips/small/0.jpg").status == 200);
    }
    server.stop();
    CHECK(http(port, "GET", "/api/info").status == 0);  // closed
}
#endif

TEST(json_workflows) {
    std::string tok;
    const std::string dir = writeTinyStableDiffusion("sd_workflow", tok);
    const std::string work = tempPath("workflow");
    std::filesystem::create_directories(work + "/in");
    Image a;
    a.width = 40;
    a.height = 30;
    for (std::size_t i = 0; i < 40 * 30 * 3; ++i) a.pixels.push_back(static_cast<std::uint8_t>(i * 19 % 256));
    savePng(work + "/in/a.png", a);
    savePng(work + "/in/b.png", resizeImage(a, 40, 30));
    const std::string json = R"({"steps": [
        {"op": "load", "path": "in/a.png", "out": "pic"},
        {"op": "load", "dir": "in", "out": "frames"},
        {"op": "crop", "in": "pic", "width": 64, "height": 64, "out": "square"},
        {"op": "canny", "in": "square", "out": "edges"},
        {"op": "txt2img", "model": ")" + dir + R"(", "prompt": "hello", "width": 64, "height": 64, "steps": 2, "out": "gen"},
        {"op": "img2img", "model": ")" + dir + R"(", "prompt": "he", "in": "square", "width": 64, "height": 64, "steps": 2, "out": "gen2"},
        {"op": "concat", "in": ["gen", "gen2", "square"], "out": "clip"},
        {"op": "save", "in": "clip", "path": "clip.gif", "fps": 4},
        {"op": "save", "in": "gen", "path": "gen.jpg"},
        {"op": "save", "in": "frames", "path": "frames/"}
    ]})";
    std::vector<std::string> log;
    std::vector<std::size_t> cached;  // models held after each step
    const auto clips = runWorkflow(json, work, [&](const std::string& l) {
        log.push_back(l);
        cached.push_back(modelCacheEntries());
    });
    CHECK(log.size() == 10 && log[4].rfind("step 5 txt2img", 0) == 0);
    // The text encoder, UNet and VAE stay loaded from txt2img to img2img (which adds the VAE encoder), then go.
    CHECK(cached[3] == 0 && cached[4] >= 3 && cached[5] == 0 && cached[9] == 0);
    CHECK(!modelCaching() && modelCacheEntries() == 0);
    // Same pictures without the cache.
    std::string uncached = json;
    uncached.replace(uncached.find("{\"steps\""), 1, "{\"cache\": false, ");
    const auto again = runWorkflow(uncached, work);
    CHECK(again.at("gen").front().pixels == clips.at("gen").front().pixels);
    CHECK(again.at("gen2").front().pixels == clips.at("gen2").front().pixels);
    CHECK(clips.at("frames").size() == 2 && clips.at("square").front().width == 64 && clips.at("edges").front().height == 64);
    CHECK(clips.at("clip").size() == 3 && clips.at("gen").front().width == 64);
    CHECK(std::filesystem::exists(work + "/clip.gif") && std::filesystem::exists(work + "/gen.jpg"));
    CHECK(std::filesystem::exists(work + "/frames/frame_0001.png"));
    // Events: progress with fractions, live previews (with a TAESD), outputs, cancellation.
    {
        writeTinyTaesd(dir + "/taesd", false);
        const std::string two = R"({"steps": [
            {"op": "load", "path": "in/a.png", "out": "pic"},
            {"op": "txt2img", "model": ")" + dir + R"(", "tiny_vae": ")" + dir + R"(/taesd", "prompt": "hello", "width": 64,
             "height": 64, "steps": 3, "out": "gen"}]})";
        WorkflowEvents ev;
        std::vector<std::string> stages;
        std::vector<double> fractions;
        std::size_t previews = 0;
        std::vector<std::string> outputs;
        ev.progress = [&](std::size_t step, std::size_t steps, const std::string& op, const std::string& stage, double f) {
            CHECK(steps == 2 && step >= 1 && step <= 2 && (op == "load" || op == "txt2img"));
            stages.push_back(op + ":" + stage);
            fractions.push_back(f);
        };
        ev.preview = [&](std::size_t step, const Image& im) {
            CHECK(step == 2 && im.width == 32 && im.height == 32);
            ++previews;
        };
        ev.output = [&](std::size_t, const std::string& op, const std::string& clip, const Clip& frames, double seconds) {
            CHECK(frames.size() == 1 && seconds >= 0);
            outputs.push_back(op + "->" + clip);
        };
        const auto got = runWorkflow(two, work, ev);
        CHECK(got.at("gen").front().width == 64);
        CHECK(previews == 3);
        CHECK((outputs == std::vector<std::string>{"load->pic", "txt2img->gen"}));
        CHECK(stages.front() == "load:start" && fractions.front() < 0);
        const auto at = std::find(stages.begin(), stages.end(), "txt2img:step 2/3");
        CHECK(at != stages.end() && std::fabs(fractions[static_cast<std::size_t>(at - stages.begin())] - 2.0 / 3.0) < 1e-9);

        // Cancelled after the first denoising step: WorkflowCancelled, and the model cache is let go.
        std::size_t seen = 0;
        WorkflowEvents stop;
        stop.progress = [&](std::size_t, std::size_t, const std::string&, const std::string& stage, double) {
            if (stage.rfind("step ", 0) == 0) ++seen;
        };
        stop.cancelled = [&] { return seen >= 1; };
        CHECK_THROWS(runWorkflow(two, work, stop), WorkflowCancelled);
        CHECK(seen == 1 && modelCacheEntries() == 0 && !modelCaching());
    }
    // Result cache: an unchanged step (and what it reads) is reused; a changed field, a changed input
    // file or a changed upstream step re-runs it and what follows.
    {
        WorkflowResultCache results;
        const std::string wf = R"({"steps": [
            {"op": "load", "path": "in/a.png", "out": "pic"},
            {"op": "crop", "in": "pic", "width": 64, "height": 64, "out": "square", "ui": {"x": 1}},
            {"op": "txt2img", "model": ")" + dir + R"(", "prompt": "hello", "width": 64, "height": 64, "steps": 2, "out": "gen"},
            {"op": "canny", "in": "square", "low": LOW, "out": "edges"},
            {"op": "save", "in": "edges", "path": "edges.png"}]})";
        auto withLow = [&](const std::string& low) {
            std::string t = wf;
            t.replace(t.find("LOW"), 3, low);
            return t;
        };
        auto run = [&](const std::string& t, std::vector<std::string>& stages) {
            WorkflowEvents ev;
            ev.progress = [&](std::size_t, std::size_t, const std::string& op, const std::string& stage, double) {
                if (stage == "start" || stage == "cached") stages.push_back(op + ":" + stage);
            };
            return runWorkflow(t, work, ev, &results);
        };
        std::vector<std::string> first, second, third, fourth;
        const auto c1 = run(withLow("100"), first);
        CHECK((first == std::vector<std::string>{"load:start", "crop:start", "txt2img:start", "canny:start", "save:start"}));
        CHECK(results.size() == 4 && results.bytes() > 0);
        // Same workflow with only editor data and a clip name changed: everything but save is reused.
        std::string renamed = withLow("100");
        renamed.replace(renamed.find("{\"x\": 1}"), 8, "{\"x\": 2}");
        for (std::size_t at; (at = renamed.find("\"square\"")) != std::string::npos;) renamed.replace(at, 8, "\"sq\"");
        const auto c2 = run(renamed, second);
        CHECK((second == std::vector<std::string>{"load:cached", "crop:cached", "txt2img:cached", "canny:cached", "save:start"}));
        CHECK(c2.at("gen").front().pixels == c1.at("gen").front().pixels && c2.at("edges").front().pixels == c1.at("edges").front().pixels);
        CHECK(c2.count("sq") && !c2.count("square"));
        // A changed threshold re-runs canny only.
        run(withLow("60"), third);
        CHECK((third == std::vector<std::string>{"load:cached", "crop:cached", "txt2img:cached", "canny:start", "save:start"}));
        // A changed input file re-runs load and the steps that read it, not txt2img.
        std::filesystem::last_write_time(work + "/in/a.png", std::filesystem::last_write_time(work + "/in/a.png") + std::chrono::seconds(5));
        run(withLow("60"), fourth);
        CHECK((fourth == std::vector<std::string>{"load:start", "crop:start", "txt2img:cached", "canny:start", "save:start"}));
        CHECK(results.hits() == 4 + 3 + 1 && modelCacheEntries() == 0);
        // A tiny limit keeps nothing too big; clear empties it.
        WorkflowResultCache tiny(16);
        tiny.put("k", c1.at("gen"));
        CHECK(tiny.size() == 0);
        results.clear();
        CHECK(results.size() == 0 && results.bytes() == 0);
    }
    // blend: a skip connection from an earlier step's clip, with a mask and a broadcast single frame.
    const auto mixed = runWorkflow(R"({"steps": [
        {"op": "load", "path": "in/a.png", "out": "a"},
        {"op": "crop", "in": "a", "width": 20, "height": 30, "out": "small"},
        {"op": "load", "dir": "in", "out": "two"},
        {"op": "blend", "in": ["two", "small"], "weight": 1.0, "out": "all_b"},
        {"op": "blend", "in": ["a", "a"], "mode": "difference", "weight": 1.0, "out": "zero"},
        {"op": "blend", "in": ["a", "zero"], "mask": "zero", "out": "same"}]})", work);
    CHECK(mixed.at("all_b").size() == 2 && mixed.at("all_b")[1].width == 40);
    CHECK(mixed.at("all_b")[0].pixels == resizeImage(mixed.at("small")[0], 40, 30).pixels);
    for (std::uint8_t p : mixed.at("zero")[0].pixels) CHECK(p == 0);
    CHECK(mixed.at("same")[0].pixels == a.pixels);
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "load", "path": "in/a.png", "out": "a"},
        {"op": "blend", "in": ["a", "a"], "mode": "burn"}]})", work), std::invalid_argument);
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "fly"}]})", work), std::invalid_argument);
    CHECK_THROWS(runWorkflow(R"({"steps": [{"op": "save", "in": "missing", "path": "x.png"}]})", work), std::invalid_argument);
    CHECK_THROWS(runWorkflow(R"({"nothing": 1})", work), std::invalid_argument);
    for (const std::string& d : {dir, tok, work}) std::filesystem::remove_all(d);
}

// ---- Training robustness ------------------------------------------------------------------

TEST(cosine_schedule_and_gradient_clipping) {
    CosineSchedule s;
    s.base = 1.0f;
    s.warmupSteps = 10;
    s.totalSteps = 110;
    s.minRatio = 0.1f;
    CHECK_NEAR(s(0), 0.1, 1e-6);   // warmup starts small
    CHECK_NEAR(s(9), 1.0, 1e-6);   // reaches the base rate
    CHECK_NEAR(s(60), 0.55, 1e-4); // halfway through the cosine
    CHECK_NEAR(s(110), 0.1, 1e-6);
    CHECK_NEAR(s(500), 0.1, 1e-6);

    Matrix w(1, 2), g{{3, 4}};
    std::vector<Param> params{{&w, &g}};
    CHECK_NEAR(clipGradientNorm(params, 1.0f), 5.0, 1e-5);
    CHECK_NEAR(gradientNorm(params), 1.0, 1e-4);
    CHECK_NEAR(clipGradientNorm(params, 10.0f), 1.0, 1e-4);  // below the limit: unchanged
    CHECK_NEAR(g(0, 0), 0.6, 1e-4);
}

namespace {
Network resumeNet() {
    Network net;
    net.add<Dense>(2, 16, Init::He);
    net.add<ReLU>();
    net.add<Dropout>(0.1f);
    net.add<Dense>(16, 3);
    return net;
}
} // namespace

TEST(checkpoint_resume_is_exact) {
    const Dataset d = [] { setSeed(50); return makeSpiral(30, 3, 0.1f); }();
    CosineSchedule schedule;
    schedule.base = 0.02f;
    schedule.warmupSteps = 3;
    schedule.totalSteps = 24;
    auto options = [&](std::size_t epochs, std::size_t startStep) {
        FitOptions o;
        o.epochs = epochs;
        o.batchSize = 8;
        o.verbose = false;
        o.learningRate = schedule;
        o.clipGradNorm = 0.5f;
        o.startStep = startStep;
        return o;
    };
    SoftmaxCrossEntropy loss;

    setSeed(51);
    Network straight = resumeNet();
    Adam optA(0.02f);
    straight.fit(d.x, d.y, loss, optA, options(2, 0));

    setSeed(51);
    Network first = resumeNet();
    Adam optB(0.02f);
    const History h = first.fit(d.x, d.y, loss, optB, options(1, 0));
    CHECK(h.back().step == 12);
    const std::string path = tempPath("resume.ckpt");
    saveCheckpoint(path, first, optB, h.back().step);

    setSeed(999);  // different init and random state: everything must come from the checkpoint
    Network resumed = resumeNet();
    Adam optC(0.02f);
    const std::size_t step = loadCheckpoint(path, resumed, optC);
    CHECK(step == 12);
    resumed.fit(d.x, d.y, loss, optC, options(1, step));

    const auto a = straight.parameters(), b = resumed.parameters();
    for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].value->approxEquals(*b[i].value, 0.0f));
    CHECK(optA.stepCount() == optC.stepCount());

    Network other;
    other.add<Dense>(2, 3);
    Adam optD;
    CHECK_THROWS(loadCheckpoint(path, other, optD), std::runtime_error);
    SGD sgd(0.1f, 0.9f);
    CHECK_THROWS(loadCheckpoint(path, resumed, sgd), std::runtime_error);  // Adam state into SGD
    std::remove(path.c_str());
}

TEST(sgd_state_roundtrip_and_max_steps) {
    setSeed(52);
    const Dataset d = makeSpiral(20, 3, 0.1f);  // 60 rows: 6 steps per epoch
    Network net = resumeNet();
    SGD opt(0.05f, 0.9f);
    FitOptions o;
    o.epochs = 100;
    o.batchSize = 10;
    o.verbose = false;
    o.maxSteps = 7;
    std::size_t calls = 0;
    o.onStep = [&](std::size_t) { ++calls; };
    SoftmaxCrossEntropy loss;
    const History h = net.fit(d.x, d.y, loss, opt, o);
    CHECK(calls == 7 && h.back().step == 7 && h.size() == 2);

    std::stringstream state;
    opt.saveState(state, net.parameters());
    SGD copy(0.05f, 0.9f);
    copy.loadState(state, net.parameters());
    std::stringstream again;
    copy.saveState(again, net.parameters());
    std::stringstream first;
    opt.saveState(first, net.parameters());
    CHECK(first.str() == again.str());
}

TEST(gpt_config_roundtrip) {
    GPTConfig c;
    c.vocabSize = 77;
    c.contextLength = 32;
    c.dim = 48;
    c.heads = 6;
    c.layers = 3;
    c.mlpRatio = 2;
    c.dropout = 0.25f;
    const std::string path = tempPath("gpt.config");
    saveGPTConfig(path, c);
    const GPTConfig r = loadGPTConfig(path);
    CHECK(r.vocabSize == 77 && r.contextLength == 32 && r.dim == 48 && r.heads == 6 && r.layers == 3 &&
          r.mlpRatio == 2);
    CHECK_NEAR(r.dropout, 0.25, 1e-6);
    std::remove(path.c_str());
}

// ---- Backends & threading -----------------------------------------------------------

namespace {
// Compares an implementation against the double-precision reference for all
// transpose combinations, ragged sizes and accumulate mode.
template <typename Fn>
void checkGemmImplementation(Fn fn, const char* name) {
    setSeed(11);
    const std::size_t shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {17, 33, 9}, {64, 64, 64}, {70, 129, 45}, {5, 300, 260},
                                     {1, 777, 300}, {4, 129, 513}};  // last two: few-row (GEMV) path
    for (const auto& s : shapes) {
        const std::size_t M = s[0], N = s[1], K = s[2];
        for (int ta = 0; ta < 2; ++ta)
            for (int tb = 0; tb < 2; ++tb)
                for (int acc = 0; acc < 2; ++acc) {
                    const Matrix a = ta ? Matrix::uniform(K, M, -1, 1) : Matrix::uniform(M, K, -1, 1);
                    const Matrix b = tb ? Matrix::uniform(N, K, -1, 1) : Matrix::uniform(K, N, -1, 1);
                    const Matrix c0 = Matrix::uniform(M, N, -1, 1);
                    Matrix expected = c0, got = c0;
                    gemmReference(ta, tb, M, N, K, a.data(), a.cols(), b.data(), b.cols(), expected.data(), N, acc);
                    fn(ta, tb, M, N, K, a.data(), a.cols(), b.data(), b.cols(), got.data(), N, acc);
                    if (!got.approxEquals(expected, 1e-4f * static_cast<float>(K + 1))) {
                        std::ostringstream os;
                        os << name << " gemm mismatch for M=" << M << " N=" << N << " K=" << K << " transA=" << ta
                           << " transB=" << tb << " accumulate=" << acc;
                        throw Failure(os.str());
                    }
                }
    }
}
} // namespace

TEST(gemm_blocked_matches_reference) {
    checkGemmImplementation(gemmBlocked, "blocked");
}

TEST(gemm_cpu_backend_matches_reference) {
    checkGemmImplementation(gemmCpu, cpuBackendName().c_str());
}

TEST(gemm_gpu_matches_reference_when_available) {
    if (!gpuAvailable()) {
        std::cout << "         (no GPU in this build/machine: skipped)\n";
        CHECK(!setDevice(Device::GPU));
        CHECK(currentDevice() == Device::CPU);
        CHECK_THROWS(gemmGpu(false, false, 1, 1, 1, nullptr, 1, nullptr, 1, nullptr, 1), std::runtime_error);
        return;
    }
    std::cout << "         (GPU: " << gpuName() << ")\n";
    checkGemmImplementation(gemmGpu, gpuBackendName().c_str());
}

TEST(gpu_device_trains_same_as_cpu) {
    auto train = [](Device device) {
        setDevice(device);
        const std::size_t saved = gpuMinWork();
        setGpuMinWork(1);  // force every product onto the selected device
        setSeed(12);
        const Dataset d = makeSpiral(40, 3, 0.1f);
        Network net;
        net.add<Dense>(2, 32, Init::He);
        net.add<ReLU>();
        net.add<Dense>(32, 3);
        SoftmaxCrossEntropy loss;
        Adam opt(0.01f);
        FitOptions opts;
        opts.epochs = 30;
        opts.batchSize = 40;
        opts.verbose = false;
        const History h = net.fit(d.x, d.y, loss, opt, opts);
        setGpuMinWork(saved);
        setDevice(Device::CPU);
        return h.back().loss;
    };
    const float cpu = train(Device::CPU);
    if (!gpuAvailable()) return;
    const float gpu = train(Device::GPU);
    CHECK_NEAR(cpu, gpu, 1e-3);
}

TEST(matrix_storage_is_page_backed_when_large) {
    const Matrix small(4, 4);
    CHECK(!small.pageBacked());
    const Matrix large(128, 256);  // 128 KB
    CHECK(large.pageBacked());
    CHECK(reinterpret_cast<std::uintptr_t>(large.data()) % kStoragePage == 0);
    CHECK(large.storageBytes() % kStoragePage == 0 && large.storageBytes() >= large.size() * sizeof(float));
    const Matrix copy = large;
    CHECK(reinterpret_cast<std::uintptr_t>(copy.data()) % kStoragePage == 0);
    const Matrix fromVector(2, 2, std::vector<float>{1, 2, 3, 4});
    CHECK(fromVector.approxEquals(Matrix{{1, 2}, {3, 4}}));
}

TEST(gpu_zero_copy_matmul_matches_cpu) {
    if (!gpuAvailable()) return;
    setSeed(42);
    const std::size_t saved = gpuMinWork();
    setGpuMinWork(1);
    for (int ta = 0; ta < 2; ++ta)
        for (int tb = 0; tb < 2; ++tb) {
            // Large enough to be page-backed; ragged sizes exercise tile edges.
            const std::size_t M = 150, N = 133, K = 270;
            const Matrix a = ta ? Matrix::uniform(K, M, -1, 1) : Matrix::uniform(M, K, -1, 1);
            const Matrix b = tb ? Matrix::uniform(N, K, -1, 1) : Matrix::uniform(K, N, -1, 1);
            CHECK(a.pageBacked() && b.pageBacked());
            Matrix cpu(M, N), gpu(M, N);
            setDevice(Device::CPU);
            matmul(a, ta, b, tb, cpu);
            setDevice(Device::GPU);
            matmul(a, ta, b, tb, gpu);
            setDevice(Device::CPU);
            CHECK(gpu.approxEquals(cpu, 1e-3f));
        }
    setGpuMinWork(saved);
}

TEST(gpu_int8_products_match_cpu) {
    if (!gpuAvailable()) return;
    std::cout << "         (int8 GPU kernel: " << (gpuInt8Available() ? "yes" : "no") << ")\n";
    CHECK(gpuInt8Available());
    setSeed(77);
    const std::size_t saved = gpuMinWork();
    setGpuMinWork(1);
    // Dense layout (x * W^T) and the convolution one (W * b); 200 inputs leave a partial last group,
    // ragged sizes exercise the tile edges, and the large case is page-backed (mapped in place).
    for (std::size_t in : {200, 576})
        for (std::size_t out : {70, 130}) {
            const QuantizedWeights w = QuantizedWeights::fromOutIn(Matrix::uniform(out, in, -1, 1));
            const Matrix x = Matrix::uniform(QuantizedWeights::kGemmRows + 37, in, -1, 1);
            setDevice(Device::CPU);
            const Matrix cpu = w.multiplyFloat(x);
            CHECK(!w.gpuProducts(x.rows()));
            setDevice(Device::GPU);
            CHECK(w.gpuProducts(x.rows()));
            const Matrix gpu = w.multiply(x);
            CHECK(gpu.approxEquals(cpu, 1e-3f));

            const std::size_t n = x.rows();
            const Matrix b = Matrix::uniform(in, n, -1, 1);
            Matrix cg(out, n), cc(out, n);
            w.multiplyLeft(n, b.data(), n, cg.data(), n, b.pageBacked() ? b.storageBytes() : 0,
                           cg.pageBacked() ? cg.storageBytes() : 0);
            setDevice(Device::CPU);
            w.multiplyLeft(n, b.data(), n, cc.data(), n);
            CHECK(cg.approxEquals(cc, 1e-3f));
            CHECK(cc.approxEquals(w.multiplyFloat(b.transpose()).transpose(), 1e-3f));
        }
    // Int8 convolutions (3x3 strips and 1x1 on the image itself) give the same maps on both devices.
    struct Case { ImageShape in; std::size_t out, k, stride, pad; };
    for (const Case& c : {Case{{16, 20, 20}, 8, 3, 1, 1}, Case{{8, 24, 24}, 6, 1, 1, 0}, Case{{4, 34, 34}, 5, 3, 2, 1}}) {
        Conv2D conv(c.in, c.out, c.k, c.stride, c.pad);
        conv.bias() = Matrix::uniform(1, c.out, -1, 1);
        conv.quantize();
        const Matrix x = Matrix::uniform(2, c.in.size(), -1, 1);
        setDevice(Device::CPU);
        const Matrix cpu = conv.forward(x, false);
        setDevice(Device::GPU);
        const Matrix gpu = conv.forward(x, false);
        setDevice(Device::CPU);
        CHECK(gpu.approxEquals(cpu, 1e-3f));
    }
    setGpuMinWork(saved);
}

TEST(gpu_graphs_match_layer_by_layer_passes) {
    // CUDA keeps whole VAE and UNet passes on the device; they must agree with the CPU.
    if (!gpuAvailable() || gpuBackendName() != "CUDA") return;
    setSeed(61);
    const std::string dir = tempPath("gpu_graph");
    std::filesystem::create_directories(dir + "/vae");
    std::ofstream(dir + "/vae/config.json") << R"({"_class_name":"AutoencoderKL","act_fn":"silu","block_out_channels":[8,16],)"
                                            << R"("in_channels":3,"out_channels":3,"latent_channels":2,"layers_per_block":1,)"
                                            << R"("norm_num_groups":4,"scaling_factor":0.5})";
    std::map<std::string, Matrix> owned;
    for (const auto& [name, shape] : vaeTensorShapes({8, 16}, 1, 2)) {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        const bool scale = name.find("norm") != std::string::npos && name.size() > 7 && name.substr(name.size() - 7) == ".weight";
        owned[name] = scale ? Matrix::uniform(1, n, 0.8f, 1.2f) : Matrix::uniform(1, n, -0.2f, 0.2f);
    }
    std::vector<std::pair<std::string, const Matrix*>> refs;
    for (const auto& [n, m] : owned) refs.emplace_back(n, &m);
    saveSafeTensors(dir + "/vae/diffusion_pytorch_model.safetensors", refs);
    writeTinyUNet(dir, 6);

    auto close = [](const Matrix& a, const Matrix& b) {
        double scale = 1e-3, worst = 0;
        for (std::size_t i = 0; i < a.size(); ++i) scale = std::max(scale, static_cast<double>(std::fabs(b.data()[i])));
        for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(a.data()[i] - b.data()[i])));
        if (worst > 2e-3 * scale) std::cout << "         gpu graph differs by " << worst << " (scale " << scale << ")\n";
        return a.sameShape(b) && worst <= 2e-3 * scale;
    };
    Vae vae = loadVae(dir, 16, 8);
    UNet2DCondition unet = loadUNet(dir, 4, 6);
    const Matrix img = Matrix::uniform(2, 3 * 16 * 8, -1, 1), lat = Matrix::uniform(2, 2 * 4 * 8, -2, 2);
    const Matrix z = Matrix::uniform(1, 4 * 6 * 4, -1, 1), ctx = Matrix::uniform(5, 6, -1, 1);
    setDevice(Device::CPU);
    const Matrix encCpu = vae.encode(img), decCpu = vae.decode(lat), epsCpu = unet.forward(z, 500.0f, ctx);
    setDevice(Device::GPU);
    Matrix encGpu, decGpu, epsGpu, epsAgain;
    {
        GpuWeightScope scope;
        encGpu = vae.encode(img);
        decGpu = vae.decode(lat);
        epsGpu = unet.forward(z, 500.0f, ctx);
        epsAgain = unet.forward(z, 500.0f, ctx);  // weights from the resident copies
    }
    setDevice(Device::CPU);
    CHECK(close(encGpu, encCpu));
    CHECK(close(decGpu, decCpu));
    CHECK(close(epsGpu, epsCpu));
    CHECK(close(epsAgain, epsCpu));

    // Batches: two conditions (classifier-free guidance) and frames, against the CPU.
    const Matrix neg = Matrix::uniform(5, 6, -1, 1), frames = Matrix::uniform(3, 4 * 6 * 4, -1, 1);
    std::vector<UNetCondition> conds(2);
    conds[0].context = &ctx;
    conds[1].context = &neg;
    const Matrix negCpu = unet.forward(z, 500.0f, neg);
    const Matrix framesCpu = unet.forwardFrames(frames, 500.0f, std::vector<Matrix>{ctx, neg, ctx});
    setDevice(Device::GPU);
    std::vector<Matrix> both;
    Matrix framesGpu;
    {
        GpuWeightScope scope;
        both = unet.forwardBatch(z, 500.0f, conds);
        framesGpu = unet.forwardFrames(frames, 500.0f, std::vector<Matrix>{ctx, neg, ctx});
    }
    setDevice(Device::CPU);
    CHECK(both.size() == 2 && close(both[0], epsCpu) && close(both[1], negCpu));
    CHECK(close(framesGpu, framesCpu));
}

TEST(gpu_resident_weights_follow_their_memory) {
    // Inside a weight scope the GPU keeps constant operands; a freed and reused buffer (a model
    // swapped for another of the same shape) must not be served from the old copy.
    if (!gpuAvailable()) return;
    setSeed(5);
    const std::size_t saved = gpuMinWork();
    setGpuMinWork(1);
    setDevice(Device::GPU);
    {
        GpuWeightScope scope;
        for (const std::size_t n : {std::size_t(80), std::size_t(256)}) {  // 30 KB (not kept) and 96 KB (kept) weights
            const Matrix x = Matrix::uniform(64, 96, -1, 1);
            for (int round = 0; round < 3; ++round) {
                const Matrix w = Matrix::uniform(96, n, -1, 1);  // likely the same address each round
                Matrix gpu(64, n), cpu(64, n);
                gemmConst(false, false, false, 64, n, 96, x.data(), 96, w.data(), n, gpu.data(), n);
                gemmConst(false, false, false, 64, n, 96, x.data(), 96, w.data(), n, gpu.data(), n);  // from the resident copy
                gemmReference(false, false, 64, n, 96, x.data(), 96, w.data(), n, cpu.data(), n);
                if (!gpu.approxEquals(cpu, 1e-3f)) std::cout << "         resident gemm " << n << " round " << round << " differs\n";
                CHECK(gpu.approxEquals(cpu, 1e-3f));
            }
        }
        // A convolution against the CPU one, weights resident (whole convolutions: CUDA only).
        if (gpuBackendName() != "CUDA") {
            setDevice(Device::CPU);
            setGpuMinWork(saved);
            return;
        }
        const std::size_t C = 24, H = 13, W = 11, O = 16, k = 3;
        const Matrix img = Matrix::uniform(2, C * H * W, -1, 1), wt = Matrix::uniform(C * k * k, O, -1, 1), bias = Matrix::uniform(1, O, -1, 1);
        Matrix out(2, O * H * W);
        CHECK(gpuConvolution(img.data(), 2, C, H, W, wt.data(), false, bias.data(), O, k, 1, 1, out.data()));
        CHECK(gpuConvolution(img.data(), 2, C, H, W, wt.data(), false, bias.data(), O, k, 1, 1, out.data()));
        float worst = 0;
        for (std::size_t b = 0; b < 2; ++b)
            for (std::size_t o = 0; o < O; ++o)
                for (std::size_t y = 0; y < H; ++y)
                    for (std::size_t xx = 0; xx < W; ++xx) {
                        double acc = bias(0, o);
                        for (std::size_t c = 0; c < C; ++c)
                            for (std::size_t ky = 0; ky < k; ++ky)
                                for (std::size_t kx = 0; kx < k; ++kx) {
                                    const long iy = static_cast<long>(y + ky) - 1, ix = static_cast<long>(xx + kx) - 1;
                                    if (iy < 0 || ix < 0 || iy >= static_cast<long>(H) || ix >= static_cast<long>(W)) continue;
                                    acc += img(b, (c * H + iy) * W + ix) * wt((c * k + ky) * k + kx, o);
                                }
                        worst = std::max(worst, std::fabs(static_cast<float>(acc) - out(b, (o * H + y) * W + xx)));
                    }
        if (worst >= 1e-3f) std::cout << "         gpu convolution max diff " << worst << "\n";
        CHECK(worst < 1e-3f);
    }
    setDevice(Device::CPU);
    setGpuMinWork(saved);
}

TEST(gpu_fused_attention_matches_cpu) {
    if (!gpuAvailable()) return;
    std::cout << "         (attention kernels: head 40 " << gpuAttentionKernel(40) << ", head 64 " << gpuAttentionKernel(64)
              << ", head 36 " << gpuAttentionKernel(36) << ")\n";
    if (gpuBackendName() == "Metal") CHECK(gpuAttentionKernel(36) == "scalar" && gpuAttentionKernel(512).empty());
    else CHECK(!gpuAttentionKernel(36).empty() && gpuAttentionKernel(512).empty());
    setSeed(91);
    const std::size_t saved = gpuMinWork();
    setGpuMinWork(1);
    struct Case { std::size_t N, M, C, heads; bool gpu; };
    // Self- and cross-attention shapes of the UNet (head size 40 / 80 / 160), a ragged key count,
    // a 64-wide head, and a 512-wide one (VAE) that stays on the CPU.
    for (const Case& c : {Case{300, 300, 320, 8, true}, Case{257, 77, 640, 8, true}, Case{100, 33, 320, 2, true},
                          Case{70, 129, 320, 5, true}, Case{13, 5, 288, 8, true}, Case{1, 1, 80, 2, true},
                          Case{64, 64, 512, 1, false}}) {
        const Matrix q = Matrix::uniform(c.N, c.C, -2, 2), k = Matrix::uniform(c.M, c.C, -2, 2), v = Matrix::uniform(c.M, c.C, -1, 1);
        setDevice(Device::CPU);
        const Matrix cpu = multiHeadAttention(q, k, v, c.heads);
        setDevice(Device::GPU);
        Matrix direct(c.N, c.C);
        CHECK(gpuAttention(q, k, v, c.heads, direct) == c.gpu);
        const Matrix gpu = multiHeadAttention(q, k, v, c.heads);
        setDevice(Device::CPU);
        // Without the fused kernel the GPU path is GEMMs on the GPU + CPU softmax: a looser bound
        // (512-wide dot products accumulate in a different order).
        const float tol = c.gpu ? 1e-4f : 1e-3f;
        float worst = 0;
        for (std::size_t i = 0; i < cpu.size(); ++i) worst = std::max(worst, std::fabs(gpu.data()[i] - cpu.data()[i]));
        if (worst > tol) std::cout << "         attention " << c.N << "x" << c.M << "x" << c.C << "/" << c.heads << ": max diff " << worst << "\n";
        CHECK(worst <= tol);
        if (c.gpu) CHECK(direct.approxEquals(cpu, 1e-4f));
        // Hybrid: CPU device, attention alone on the GPU.
        setGpuAttention(true);
        Matrix hybrid(c.N, c.C);
        CHECK(gpuAttention(q, k, v, c.heads, hybrid) == c.gpu);
        CHECK(multiHeadAttention(q, k, v, c.heads).approxEquals(cpu, 1e-4f));
        setGpuAttention(false);
    }
    setGpuMinWork(saved);
}

TEST(gpu_attention_kernels_agree_and_repeat) {
    if (!gpuAvailable()) return;
    setSeed(91);
    const std::size_t saved = gpuMinWork();
    setGpuMinWork(1);
    setDevice(Device::GPU);
    // Both kernels (simdgroup only where the simdgroup GEMM runs), several threadgroups per head, ragged
    // sizes, repeated: a race between lanes shows up as rows that change from run to run.
    for (const char* kernel : {"scalar", "simdgroup"}) {
        setGpuAttentionKernel(kernel);
        if (gpuAttentionKernel(40) != kernel) continue;
        for (std::size_t N : {32, 300})
            for (std::size_t M : {16, 77, 300}) {
                const std::size_t C = 320, heads = 8;
                const Matrix q = Matrix::uniform(N, C, -2, 2), k = Matrix::uniform(M, C, -2, 2), v = Matrix::uniform(M, C, -1, 1);
                setDevice(Device::CPU);
                const Matrix cpu = multiHeadAttention(q, k, v, heads);
                setDevice(Device::GPU);
                for (int run = 0; run < 3; ++run) {
                    Matrix out(N, C);
                    CHECK(gpuAttention(q, k, v, heads, out));
                    std::size_t bad = 0;
                    for (std::size_t i = 0; i < out.size(); ++i) bad += std::fabs(out.data()[i] - cpu.data()[i]) > 1e-4f;
                    if (bad) std::cout << "         " << kernel << " N=" << N << " M=" << M << " run " << run << ": " << bad << " wrong\n";
                    CHECK(bad == 0);
                }
            }
    }
    setGpuAttentionKernel("");
    setDevice(Device::CPU);
    setGpuMinWork(saved);
}

TEST(matrix_products_independent_of_thread_count) {
    setSeed(13);
    const Matrix a = Matrix::uniform(97, 150, -1, 1), b = Matrix::uniform(150, 61, -1, 1);
    const unsigned saved = threadCount();
    setThreadCount(1);
    const Matrix single = a * b;
    setThreadCount(4);
    const Matrix multi = a * b;
    setThreadCount(saved);
    CHECK(single.approxEquals(multi, 1e-5f));
}

TEST(parallel_for_covers_range_and_propagates_errors) {
    const unsigned saved = threadCount();
    setThreadCount(4);
    std::vector<int> hits(10007, 0);
    parallelFor(0, hits.size(), 1, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) ++hits[i];
    });
    for (int h : hits) CHECK(h == 1);

    std::atomic<int> nestedCalls{0};
    parallelFor(0, 8, 1, [&](std::size_t, std::size_t) {
        parallelFor(0, 4, 1, [&](std::size_t b, std::size_t e) { nestedCalls += static_cast<int>(e - b); });
    });
    CHECK(nestedCalls.load() == 32);

    CHECK_THROWS(parallelFor(0, 100, 1,
                             [](std::size_t b, std::size_t) {
                                 if (b >= 50) throw std::runtime_error("boom");
                             }),
                 std::runtime_error);
    // The pool must still work after an exception.
    std::atomic<std::size_t> total{0};
    parallelFor(0, 1000, 10, [&](std::size_t b, std::size_t e) { total += e - b; });
    CHECK(total.load() == 1000);
    setThreadCount(saved);
}

// ---- Runner ------------------------------------------------------------------------------

int main() {
    int failed = 0;
    for (const auto& t : registry()) {
        try {
            t.fn();
            std::cout << "[ PASS ] " << t.name << '\n';
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[ FAIL ] " << t.name << "\n         " << e.what() << '\n';
        }
    }
    std::cout << '\n' << registry().size() - static_cast<std::size_t>(failed) << "/" << registry().size()
              << " tests passed\n";
    return failed == 0 ? 0 : 1;
}
