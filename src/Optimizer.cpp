#include "Optimizer.h"

#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace nc {

SGD::SGD(float learningRate, float momentum, float weightDecay)
    : Optimizer(learningRate), momentum_(momentum), weightDecay_(weightDecay) {
    if (learningRate <= 0.0f) throw std::invalid_argument("SGD: learning rate must be > 0");
    if (momentum < 0.0f || momentum >= 1.0f) throw std::invalid_argument("SGD: momentum must be in [0, 1)");
}

void SGD::step(const std::vector<Param>& params) {
    for (const auto& p : params) {
        Matrix& w = *p.value;
        const Matrix& g = *p.grad;
        if (!w.sameShape(g)) throw std::logic_error("SGD: parameter/gradient shape mismatch");

        if (momentum_ == 0.0f) {
            for (std::size_t i = 0; i < w.size(); ++i) {
                const float grad = g.data()[i] + weightDecay_ * w.data()[i];
                w.data()[i] -= learningRate_ * grad;
            }
            continue;
        }

        auto it = velocity_.find(p.value);
        if (it == velocity_.end()) it = velocity_.emplace(p.value, Matrix(w.rows(), w.cols())).first;
        Matrix& v = it->second;
        for (std::size_t i = 0; i < w.size(); ++i) {
            const float grad = g.data()[i] + weightDecay_ * w.data()[i];
            v.data()[i] = momentum_ * v.data()[i] + grad;
            w.data()[i] -= learningRate_ * v.data()[i];
        }
    }
}

Adam::Adam(float learningRate, float beta1, float beta2, float epsilon, float weightDecay)
    : Optimizer(learningRate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weightDecay_(weightDecay) {
    if (learningRate <= 0.0f) throw std::invalid_argument("Adam: learning rate must be > 0");
    if (beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f)
        throw std::invalid_argument("Adam: betas must be in [0, 1)");
}

void Adam::step(const std::vector<Param>& params) {
    ++t_;
    const float correction1 = 1.0f - static_cast<float>(std::pow(beta1_, static_cast<double>(t_)));
    const float correction2 = 1.0f - static_cast<float>(std::pow(beta2_, static_cast<double>(t_)));

    for (const auto& p : params) {
        Matrix& w = *p.value;
        const Matrix& g = *p.grad;
        if (!w.sameShape(g)) throw std::logic_error("Adam: parameter/gradient shape mismatch");

        auto mi = m_.find(p.value);
        if (mi == m_.end()) mi = m_.emplace(p.value, Matrix(w.rows(), w.cols())).first;
        auto vi = v_.find(p.value);
        if (vi == v_.end()) vi = v_.emplace(p.value, Matrix(w.rows(), w.cols())).first;
        float* m = mi->second.data();
        float* v = vi->second.data();

        for (std::size_t i = 0; i < w.size(); ++i) {
            const float grad = g.data()[i];
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad;
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * grad * grad;
            const float mHat = m[i] / correction1;
            const float vHat = v[i] / correction2;
            w.data()[i] -= learningRate_ * (mHat / (std::sqrt(vHat) + epsilon_) + weightDecay_ * w.data()[i]);
        }
    }
}

// ---- State persistence ---------------------------------------------------------

namespace {
void writeTensor(std::ostream& out, const Matrix* m, std::size_t size) {
    out << size << '\n';
    for (std::size_t i = 0; i < size; ++i) out << (m ? m->data()[i] : 0.0f) << (i + 1 < size ? ' ' : '\n');
}

Matrix readTensor(std::istream& in, const Matrix& shape, const char* who) {
    std::size_t size = 0;
    if (!(in >> size) || size != shape.size())
        throw std::runtime_error(std::string(who) + ": optimizer state does not match the parameters");
    Matrix m(shape.rows(), shape.cols());
    for (auto& v : m.values())
        if (!(in >> v)) throw std::runtime_error(std::string(who) + ": truncated optimizer state");
    return m;
}

void expectHeader(std::istream& in, const std::string& name, std::size_t count) {
    std::string got;
    std::size_t n = 0;
    if (!(in >> got >> n) || got != name || n != count)
        throw std::runtime_error("optimizer state is for " + got + " with " + std::to_string(n) +
                                 " tensors, expected " + name + " with " + std::to_string(count));
}
} // namespace

void SGD::saveState(std::ostream& out, const std::vector<Param>& params) const {
    out << name() << ' ' << params.size() << '\n' << std::setprecision(9);
    for (const auto& p : params) {
        auto it = velocity_.find(p.value);
        writeTensor(out, it == velocity_.end() ? nullptr : &it->second, p.value->size());
    }
}

void SGD::loadState(std::istream& in, const std::vector<Param>& params) {
    expectHeader(in, name(), params.size());
    std::unordered_map<const Matrix*, Matrix> loaded;
    for (const auto& p : params) loaded.emplace(p.value, readTensor(in, *p.value, "SGD"));
    velocity_ = std::move(loaded);
}

void Adam::saveState(std::ostream& out, const std::vector<Param>& params) const {
    out << name() << ' ' << params.size() << '\n' << t_ << '\n' << std::setprecision(9);
    for (const auto& p : params) {
        auto mi = m_.find(p.value);
        auto vi = v_.find(p.value);
        writeTensor(out, mi == m_.end() ? nullptr : &mi->second, p.value->size());
        writeTensor(out, vi == v_.end() ? nullptr : &vi->second, p.value->size());
    }
}

void Adam::loadState(std::istream& in, const std::vector<Param>& params) {
    expectHeader(in, name(), params.size());
    long long t = 0;
    if (!(in >> t) || t < 0) throw std::runtime_error("Adam: invalid step count in optimizer state");
    std::unordered_map<const Matrix*, Matrix> m, v;
    for (const auto& p : params) {
        m.emplace(p.value, readTensor(in, *p.value, "Adam"));
        v.emplace(p.value, readTensor(in, *p.value, "Adam"));
    }
    t_ = t;
    m_ = std::move(m);
    v_ = std::move(v);
}

// ---- Schedules and clipping -------------------------------------------------------

float CosineSchedule::operator()(std::size_t step) const {
    if (warmupSteps > 0 && step < warmupSteps)
        return base * static_cast<float>(step + 1) / static_cast<float>(warmupSteps);
    if (totalSteps <= warmupSteps || step >= totalSteps) return base * minRatio;
    const double progress = static_cast<double>(step - warmupSteps) / static_cast<double>(totalSteps - warmupSteps);
    const double cosine = 0.5 * (1.0 + std::cos(3.14159265358979 * progress));
    return base * static_cast<float>(minRatio + (1.0 - minRatio) * cosine);
}

float gradientNorm(const std::vector<Param>& params) {
    double sq = 0.0;
    for (const auto& p : params)
        for (float g : p.grad->values()) sq += static_cast<double>(g) * g;
    return static_cast<float>(std::sqrt(sq));
}

float clipGradientNorm(const std::vector<Param>& params, float maxNorm) {
    const float norm = gradientNorm(params);
    if (maxNorm > 0.0f && norm > maxNorm) {
        const float scale = maxNorm / (norm + 1e-6f);
        for (const auto& p : params) *p.grad *= scale;
    }
    return norm;
}

} // namespace nc
