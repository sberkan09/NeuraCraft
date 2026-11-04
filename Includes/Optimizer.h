#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace nc {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    // Updates every parameter in place using its accumulated gradient.
    virtual void step(const std::vector<Param>& params) = 0;
    virtual std::string name() const = 0;

    float learningRate() const { return learningRate_; }
    void setLearningRate(float lr) { learningRate_ = lr; }

protected:
    explicit Optimizer(float lr) : learningRate_(lr) {}
    float learningRate_;
};

// Stochastic gradient descent with optional momentum and L2 weight decay.
class SGD : public Optimizer {
public:
    explicit SGD(float learningRate = 0.01f, float momentum = 0.0f, float weightDecay = 0.0f);
    void step(const std::vector<Param>& params) override;
    std::string name() const override { return "SGD"; }

private:
    float momentum_;
    float weightDecay_;
    std::unordered_map<const Matrix*, Matrix> velocity_;
};

// Adam (Kingma & Ba, 2015). weightDecay > 0 gives decoupled decay (AdamW).
class Adam : public Optimizer {
public:
    explicit Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
                  float epsilon = 1e-8f, float weightDecay = 0.0f);
    void step(const std::vector<Param>& params) override;
    std::string name() const override { return "Adam"; }

private:
    float beta1_;
    float beta2_;
    float epsilon_;
    float weightDecay_;
    long long t_ = 0;
    std::unordered_map<const Matrix*, Matrix> m_;
    std::unordered_map<const Matrix*, Matrix> v_;
};

} // namespace nc

#endif // OPTIMIZER_H
