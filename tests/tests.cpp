// NeuraCraft unit tests. Dependency-free: a tiny TEST/CHECK harness below.
#include "NeuraCraft.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
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
    const Matrix out = net.forward(x, false);
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
