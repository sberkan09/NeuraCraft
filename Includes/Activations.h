#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "Layer.h"

namespace nc {

class ReLU : public Layer {
public:
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "ReLU"; }

private:
    Matrix input_;
};

class LeakyReLU : public Layer {
public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha_(alpha) {}
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "LeakyReLU"; }

private:
    float alpha_;
    Matrix input_;
};

class Sigmoid : public Layer {
public:
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "Sigmoid"; }

private:
    Matrix output_;
};

class Tanh : public Layer {
public:
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "Tanh"; }

private:
    Matrix output_;
};

// Row-wise softmax. If you train with SoftmaxCrossEntropy, leave this layer
// out: that loss applies softmax to the logits itself (numerically safer).
class Softmax : public Layer {
public:
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "Softmax"; }

private:
    Matrix output_;
};

// Inverted dropout: active only when training == true.
class Dropout : public Layer {
public:
    explicit Dropout(float rate);
    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& gradOutput) override;
    std::string name() const override { return "Dropout"; }
    std::string describe() const override;

private:
    float rate_;
    Matrix mask_;
};

// Numerically stable row-wise softmax.
Matrix softmax(const Matrix& logits);
float sigmoid(float x);

} // namespace nc

#endif // ACTIVATIONS_H
