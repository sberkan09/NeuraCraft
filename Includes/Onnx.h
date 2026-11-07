#ifndef NEURACRAFT_ONNX_H
#define NEURACRAFT_ONNX_H

// Running ONNX models: a model exported from PyTorch, TensorFlow, ... as a .onnx file is read
// (a small protobuf reader, no dependency; weights inline or in external data files) and run by a
// generic graph executor over n-dimensional tensors. So a network whose architecture NeuraCraft has
// no code for - a background remover, a depth estimator, an upscaler, a face restorer - can be
// used without changing the library. `NeuraCraft onnx model.onnx` shows what a model holds and runs
// it on TensorProto inputs; NEURACRAFT_ONNX_PROFILE=1 prints the time per operator type. Workflows
// run image models with the "onnx" step, whose pre- and post-processing come from a manifest next
// to the model (OnnxPipeline below).
//
// Supported: opsets 7-21 of the default domain for about 115 operators (elementwise maths and
// activations, MatMul / Gemm / Conv / ConvTranspose / LSTM / GRU / RNN on the library's GEMM, pooling, Resize, the
// normalisations, Softmax, shape operators, reductions, comparisons, Where, Einsum's common forms),
// float / float16 / bfloat16 / double values (computed in float) and integer / bool tensors, and the
// quantised operators (QuantizeLinear / DequantizeLinear, DynamicQuantizeLinear, MatMulInteger,
// ConvInteger, QLinearMatMul, QLinearConv; com.microsoft MatMulNBits and MatMulBnb4 for 4-bit
// weights), and control flow: If, Loop and Scan, whose subgraphs may read the values of the graphs
// around them. Not supported: sequences and strings - a model using one fails to load with the
// operator and node named. tests/onnx/make_cases.py writes the operator
// cases the tests compare with onnxruntime; mobilenet v2, u2netp, Depth Anything v2 small (float,
// q4, bnb4) and Real-ESRGAN x4plus match it to about 1e-6.

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

// ---- Image models: what a workflow's "onnx" step does around Model::run ----
//
// The manifest is <model>.json next to <model>.onnx (every key optional):
// {
//   "input":  {"name": "input", "layout": "NCHW" | "NHWC", "channels": "RGB" | "BGR" | "GRAY",
//              "size": [H, W] (resize to) | "multiple_of": 8 (pad to) | "fixed": true (as the model says),
//              "range": [0, 1] (pixel 0..255 maps to this), "mean": [..], "std": [..],
//              "tile": 0 (run a fixed-size model over the picture in tiles with this much overlap)},
//   "output": {"name": "...", "kind": "image" | "mask" | "depth",
//              "range": [0, 1] (values that map to 0..255; "auto" stretches min..max),
//              "layout": "NCHW" | "NHWC", "channels": "RGB" | "BGR", "invert": false,
//              "apply": true (kind mask: cut the input out with it - RGBA - instead of returning it),
//              "labels": ["normal", "nsfw"], "activation": "softmax" | "sigmoid" | "none" (classifiers)}
// }
// Without a manifest: one NCHW RGB input in 0..1, the model's fixed size if it has one, the output
// an image (3 channels) or a mask / depth (1 channel) stretched to 0..255. An image output comes back
// at the input size times the model's own scale (an x4 upscaler gives 4x the input, a fixed-size
// style model gives the input size); a fixed-size model whose output is bigger than its input (a
// super-resolution model) runs over a larger picture in overlapping tiles.
struct OnnxPipeline {
    std::string inputName, outputName;
    bool nhwcIn = false, nhwcOut = false, bgrIn = false, bgrOut = false, grayIn = false;
    int width = 0, height = 0;        // resize the input to this (0: keep)
    int multipleOf = 1;               // pad the input up to a multiple of this
    float inLow = 0.0f, inHigh = 1.0f;
    std::vector<float> mean, stdev;
    std::string kind = "auto";        // image, mask, depth or auto (by channel count)
    bool autoRange = true;
    float outLow = 0.0f, outHigh = 1.0f;
    bool invert = false;
    bool applyMask = false;           // kind "mask": cut the input out with it (alpha) instead of returning the mask
    int tileOverlap = -1;             // tiles for fixed-size models: overlap in input pixels (-1: automatic, 0: never)
    bool rangeSet = false;            // the manifest gave output.range
    bool fixedByModel = false;        // width / height are the model's own (not the manifest's "size")
    std::vector<std::string> labels;  // classifiers: a name per output value ("0", "1", ... when absent)
    std::string activation = "softmax";  // classifiers: softmax, sigmoid or none (the raw outputs)

    static OnnxPipeline fromModel(const Model& model, const json::Value& manifest);
    Image run(const Model& model, const Image& in) const;
    // An image classifier: the picture squeezed to the model's size (it needs one), the output values
    // through the activation, one per label. Throws when the output does not match the labels.
    std::vector<float> classify(const Model& model, const Image& in) const;
};

// The manifest of a model (an empty object when there is none).
json::Value onnxManifest(const std::string& modelPath);

} // namespace nc::onnx

#endif // NEURACRAFT_ONNX_H
