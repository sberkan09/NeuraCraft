// NeuraCraft unit tests. Dependency-free: a tiny TEST/CHECK harness below.
#include "NeuraCraft.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    checkGemmImplementation(gemmGpu, "metal");
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
