#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"

#include <iosfwd>
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

    // Optimizer state (momentum buffers, Adam moments and step count), in the
    // order of `params`, so training can resume exactly where it stopped.
    virtual void saveState(std::ostream& out, const std::vector<Param>& params) const = 0;
    virtual void loadState(std::istream& in, const std::vector<Param>& params) = 0;

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
    void saveState(std::ostream& out, const std::vector<Param>& params) const override;
    void loadState(std::istream& in, const std::vector<Param>& params) override;

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
    void saveState(std::ostream& out, const std::vector<Param>& params) const override;
    void loadState(std::istream& in, const std::vector<Param>& params) override;
    long long stepCount() const { return t_; }

private:
    float beta1_;
    float beta2_;
    float epsilon_;
    float weightDecay_;
    long long t_ = 0;
    std::unordered_map<const Matrix*, Matrix> m_;
    std::unordered_map<const Matrix*, Matrix> v_;
};

// Learning rate per optimizer step: linear warmup from 0 to `base` over
// `warmupSteps`, then cosine decay to `minRatio * base` at `totalSteps`.
struct CosineSchedule {
    float base = 1e-3f;
    std::size_t warmupSteps = 0;
    std::size_t totalSteps = 1;
    float minRatio = 0.1f;
    float operator()(std::size_t step) const;
};

// Scales every gradient so the global L2 norm is at most maxNorm; returns the
// norm before clipping.
float clipGradientNorm(const std::vector<Param>& params, float maxNorm);
float gradientNorm(const std::vector<Param>& params);

} // namespace nc

#endif // OPTIMIZER_H
