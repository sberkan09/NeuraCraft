#include "Optimizer.h"

#include <cmath>
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

} // namespace nc
