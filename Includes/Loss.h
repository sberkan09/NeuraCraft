#ifndef LOSS_H
#define LOSS_H

#include "Matrix.h"

#include <string>
#include <vector>

namespace nc {

// A loss compares predictions with targets (both batch x outputs) and returns
// a scalar averaged over the batch. gradient() returns dL/d(predictions) for
// the most recent call to compute().
class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(const Matrix& predictions, const Matrix& targets) = 0;
    virtual Matrix gradient() const = 0;
    virtual std::string name() const = 0;

protected:
    static void checkShapes(const Matrix& predictions, const Matrix& targets, const char* who);
};

// Mean squared error, averaged over every element. For regression.
class MSE : public Loss {
public:
    float compute(const Matrix& predictions, const Matrix& targets) override;
    Matrix gradient() const override;
    std::string name() const override { return "MSE"; }

private:
    Matrix diff_;
};

// Binary cross-entropy on probabilities (put a Sigmoid layer before it).
class BinaryCrossEntropy : public Loss {
public:
    float compute(const Matrix& predictions, const Matrix& targets) override;
    Matrix gradient() const override;
    std::string name() const override { return "BinaryCrossEntropy"; }

private:
    Matrix predictions_;
    Matrix targets_;
};

// Categorical cross-entropy on probabilities (put a Softmax layer before it).
class CategoricalCrossEntropy : public Loss {
public:
    float compute(const Matrix& predictions, const Matrix& targets) override;
    Matrix gradient() const override;
    std::string name() const override { return "CategoricalCrossEntropy"; }

private:
    Matrix predictions_;
    Matrix targets_;
};

// Softmax + categorical cross-entropy fused, applied to raw logits.
// Preferred for classification: gradient is simply (softmax - target) / N.
class SoftmaxCrossEntropy : public Loss {
public:
    float compute(const Matrix& logits, const Matrix& targets) override;
    Matrix gradient() const override;
    std::string name() const override { return "SoftmaxCrossEntropy"; }

private:
    Matrix probabilities_;
    Matrix targets_;
};

// Softmax cross-entropy against integer class ids instead of one-hot rows.
// `targets` may have any shape as long as it holds one id per logits row
// (e.g. batch x seqLen next-token ids for a language model).
class SparseSoftmaxCrossEntropy : public Loss {
public:
    float compute(const Matrix& logits, const Matrix& targets) override;
    Matrix gradient() const override;
    std::string name() const override { return "SparseSoftmaxCrossEntropy"; }

private:
    Matrix probabilities_;
    std::vector<std::size_t> targets_;
};

} // namespace nc

#endif // LOSS_H
