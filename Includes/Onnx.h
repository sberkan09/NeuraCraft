#ifndef NEURACRAFT_ONNX_H
#define NEURACRAFT_ONNX_H

// Running ONNX models: a model exported from PyTorch, TensorFlow, ... as a .onnx file is read
// (a small protobuf reader, no dependency; weights inline or in external data files) and run by a
// generic graph executor over n-dimensional tensors. So a network whose architecture NeuraCraft has
// no code for - a background remover, a depth estimator, an upscaler, a face restorer - can be
// used without changing the library. `NeuraCraft onnx model.onnx` shows what a model holds and runs
// it on TensorProto inputs; NEURACRAFT_ONNX_PROFILE=1 prints the time per operator type.
//
// Supported: opsets 7-21 of the default domain for about 110 operators (elementwise maths and
// activations, MatMul / Gemm / Conv / ConvTranspose on the library's GEMM, pooling, Resize, the
// normalisations, Softmax, shape operators, reductions, comparisons, Where, Einsum's common forms),
// float / float16 / bfloat16 / double values (computed in float) and integer / bool tensors. Not
// supported: control flow (If / Loop / Scan), quantised operators, sequences and strings - a model
// using one fails to load with the operator and node named. tests/onnx/make_cases.py writes the
// operator cases the tests compare with onnxruntime; mobilenet v2, u2netp, Depth Anything v2 small
// and Real-ESRGAN x4plus match it to about 1e-6.

#include "Image.h"
#include "Json.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nc::onnx {

// An n-dimensional tensor: float values (any floating ONNX type) or integers (int*, uint*, bool).
struct Tensor {
    enum class Kind { Float, Int };
    Kind kind = Kind::Float;
    int elemType = 1;                // the ONNX type it stands for (1 float, 7 int64, 9 bool, ...)
    std::vector<std::int64_t> shape;
    std::vector<float> f;            // Kind::Float
    std::vector<std::int64_t> i;     // Kind::Int

    std::size_t size() const;        // number of elements (1 for a scalar)
    static Tensor floats(std::vector<std::int64_t> shape, std::vector<float> values);
    static Tensor ints(std::vector<std::int64_t> shape, std::vector<std::int64_t> values, int elemType = 7);
};

struct ValueInfo {
    std::string name;
    int elemType = 0;
    std::vector<std::int64_t> shape;  // -1 for a dimension without a fixed size
    std::vector<std::string> dimNames; // the symbolic names of those dimensions ("" if fixed)
};

class Model {
public:
    // Reads a .onnx file (external data is looked up next to it). Throws, naming the operator and
    // node, if the graph uses something this executor does not run.
    static Model load(const std::string& path);
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    ~Model();

    const std::vector<ValueInfo>& inputs() const;   // graph inputs that are not initializers
    const std::vector<ValueInfo>& outputs() const;
    int opset() const;
    std::size_t nodeCount() const;
    std::map<std::string, std::size_t> operatorCounts() const;
    std::size_t parameterCount() const;              // elements in the initializers

    // Runs the graph. feeds: one tensor per input (by name); returns the requested outputs (all
    // graph outputs when `wanted` is empty).
    std::map<std::string, Tensor> run(const std::map<std::string, Tensor>& feeds, const std::vector<std::string>& wanted = {}) const;

    struct Impl;

private:
    Model();
    std::unique_ptr<Impl> impl_;
};

// A TensorProto file (the ONNX test-data format: test_data_set_0/input_0.pb); `name` gets its name.
Tensor loadTensorProto(const std::string& path, std::string* name = nullptr);
void saveTensorProto(const std::string& path, const Tensor& tensor, const std::string& name = "");

// The operators the executor runs (sorted), for listings and error messages.
std::vector<std::string> supportedOperators();

} // namespace nc::onnx

#endif // NEURACRAFT_ONNX_H
