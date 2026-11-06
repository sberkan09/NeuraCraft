"""Writes the ONNX operator test cases in tests/onnx/cases: for each, model.onnx plus inputs and the
outputs onnxruntime computes (TensorProto .pb files, the ONNX test-data format). The C++ test
(onnx_operators_match_onnxruntime) runs every case and compares. Regenerate with:
    pip install onnx onnxruntime numpy && python3 tests/onnx/make_cases.py
"""
import os, shutil
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
rng = np.random.default_rng(7)
cases = []

def f32(*shape, lo=-1.0, hi=1.0):
    return rng.uniform(lo, hi, shape).astype(np.float32)

def case(name, nodes, inputs, outputs=None, inits=(), opset=17, out_types=None):
    """inputs: dict name -> array; outputs: list of names (default: last node's outputs)."""
    outputs = outputs or list(nodes[-1].output)
    vin = [h.make_tensor_value_info(k, h.np_dtype_to_tensor_dtype(v.dtype), v.shape) for k, v in inputs.items()]
    vout = [h.make_tensor_value_info(o, (out_types or {}).get(o, T.FLOAT), None) for o in outputs]
    g = h.make_graph(nodes, name, vin, vout, initializer=[nh.from_array(v, k) for k, v in inits])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", opset)])
    m.ir_version = 8
    cases.append((name, m, inputs, outputs))

def n(op, ins, outs, **attrs):
    return h.make_node(op, ins, outs, **attrs)

x34 = f32(3, 4); y34 = f32(3, 4); y4 = f32(4)
img = f32(1, 3, 8, 8)
# elementwise and broadcasting
for op in ["Add", "Sub", "Mul", "Div", "Max", "Min", "Pow"]:
    a = f32(2, 3, 4, lo=0.5, hi=2.0) if op == "Pow" else f32(2, 3, 4)
    case(op.lower() + "_broadcast", [n(op, ["a", "b"], ["y"])], {"a": a, "b": f32(3, 1, lo=0.5, hi=2.0)})
for op in ["Abs", "Neg", "Exp", "Tanh", "Sigmoid", "Relu", "Erf", "Floor", "Ceil", "Sin", "Cos", "Softplus", "Softsign", "Sign", "HardSwish", "Round"]:
    case(op.lower(), [n(op, ["x"], ["y"])], {"x": f32(2, 5) * 3})
case("sqrt_log_reciprocal", [n("Sqrt", ["x"], ["s"]), n("Log", ["s"], ["l"]), n("Reciprocal", ["l"], ["y"])], {"x": f32(2, 5, lo=2, hi=5)})
case("leakyrelu", [n("LeakyRelu", ["x"], ["y"], alpha=0.2)], {"x": f32(2, 5)})
case("elu_selu_celu", [n("Elu", ["x"], ["a"], alpha=0.7), n("Selu", ["a"], ["b"]), n("Celu", ["b"], ["y"], alpha=1.3)], {"x": f32(2, 5) * 2})
case("hardsigmoid", [n("HardSigmoid", ["x"], ["y"], alpha=0.3, beta=0.4)], {"x": f32(2, 5) * 3})
case("prelu", [n("PRelu", ["x", "s"], ["y"])], {"x": f32(1, 3, 2, 2)}, inits=[("s", f32(3, 1, 1))])
case("gelu_exact", [n("Gelu", ["x"], ["y"])], {"x": f32(3, 4) * 3}, opset=20)
case("gelu_tanh", [n("Gelu", ["x"], ["y"], approximate="tanh")], {"x": f32(3, 4) * 3}, opset=20)
case("clip11", [n("Clip", ["x", "lo", "hi"], ["y"])], {"x": f32(3, 4) * 2}, inits=[("lo", np.array(-0.5, np.float32)), ("hi", np.array(0.7, np.float32))])
case("clip6", [n("Clip", ["x"], ["y"], min=-0.3, max=0.4)], {"x": f32(3, 4)}, opset=6)
case("sum_mean_variadic", [n("Sum", ["a", "b", "c"], ["s"]), n("Mean", ["s", "a"], ["y"])], {"a": f32(2, 3), "b": f32(3), "c": f32(2, 1)})
case("compare_where", [n("Greater", ["a", "b"], ["g"]), n("Where", ["g", "a", "b"], ["y"])], {"a": f32(2, 3, 4), "b": f32(4)})
case("equal_not_and", [n("Equal", ["a", "b"], ["e"]), n("Not", ["e"], ["ne"]), n("Less", ["a", "b"], ["l"]), n("And", ["ne", "l"], ["y"])],
     {"a": rng.integers(0, 3, (3, 4)).astype(np.int64), "b": rng.integers(0, 3, (4,)).astype(np.int64)}, out_types={"y": T.BOOL})
case("mod", [n("Mod", ["a", "b"], ["y"], fmod=1)], {"a": f32(2, 3) * 5, "b": f32(2, 3, lo=1, hi=2)})
case("int_arith", [n("Add", ["a", "b"], ["s"]), n("Mul", ["s", "b"], ["m"]), n("Div", ["m", "c"], ["y"])],
     {"a": np.array([3, -7, 9], np.int64), "b": np.array([2, 5, -4], np.int64), "c": np.array([2, 3, 5], np.int64)}, out_types={"y": T.INT64})
# products
case("matmul_2d", [n("MatMul", ["a", "b"], ["y"])], {"a": f32(5, 7)}, inits=[("b", f32(7, 3))])
case("matmul_batched_broadcast", [n("MatMul", ["a", "b"], ["y"])], {"a": f32(2, 1, 4, 5), "b": f32(3, 5, 6)})
case("matmul_vec", [n("MatMul", ["a", "b"], ["y"])], {"a": f32(5), "b": f32(2, 5, 3)})
case("gemm_trans", [n("Gemm", ["a", "b", "c"], ["y"], alpha=0.5, beta=2.0, transA=1, transB=1)], {"a": f32(4, 3)}, inits=[("b", f32(5, 4)), ("c", f32(5))])
case("gemm_plain", [n("Gemm", ["a", "b"], ["y"])], {"a": f32(3, 4)}, inits=[("b", f32(4, 2))])
case("einsum_attention", [n("Einsum", ["q", "k"], ["y"], equation="bhqd,bhkd->bhqk")], {"q": f32(1, 2, 3, 4), "k": f32(1, 2, 5, 4)})
case("einsum_ellipsis", [n("Einsum", ["a", "b"], ["y"], equation="...ij,...jk->...ik")], {"a": f32(2, 3, 4), "b": f32(2, 4, 5)})
case("einsum_transpose_sum", [n("Einsum", ["a"], ["y"], equation="ijk->ki")], {"a": f32(2, 3, 4)})
case("einsum_outer", [n("Einsum", ["a", "b"], ["y"], equation="i,j->ij")], {"a": f32(3), "b": f32(4)})
# convolution and pooling
case("conv_pad_stride", [n("Conv", ["x", "w", "b"], ["y"], pads=[1, 1, 1, 1], strides=[2, 2])], {"x": img}, inits=[("w", f32(4, 3, 3, 3)), ("b", f32(4))])
case("conv_group_dilation", [n("Conv", ["x", "w"], ["y"], group=3, dilations=[2, 2], pads=[2, 2, 2, 2])], {"x": f32(1, 6, 7, 9)}, inits=[("w", f32(6, 2, 3, 3))])
case("conv_same_upper", [n("Conv", ["x", "w"], ["y"], auto_pad="SAME_UPPER", strides=[2, 2])], {"x": f32(1, 3, 7, 6)}, inits=[("w", f32(2, 3, 4, 4))])
case("conv_pointwise", [n("Conv", ["x", "w", "b"], ["y"])], {"x": f32(2, 5, 4, 3)}, inits=[("w", f32(7, 5, 1, 1)), ("b", f32(7))])
case("conv_1d", [n("Conv", ["x", "w"], ["y"], pads=[2, 1], strides=[2])], {"x": f32(1, 3, 11)}, inits=[("w", f32(4, 3, 5))])
case("convtranspose", [n("ConvTranspose", ["x", "w", "b"], ["y"], strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1])], {"x": f32(1, 4, 5, 6)},
     inits=[("w", f32(4, 3, 4, 4)), ("b", f32(3))])
case("convtranspose_group", [n("ConvTranspose", ["x", "w"], ["y"], strides=[2, 2], group=2)], {"x": f32(1, 4, 3, 3)}, inits=[("w", f32(4, 2, 3, 3))])
case("convtranspose_output_shape", [n("ConvTranspose", ["x", "w"], ["y"], strides=[2, 2], output_shape=[10, 8], auto_pad="SAME_UPPER")], {"x": f32(1, 2, 5, 4)},
     inits=[("w", f32(2, 3, 3, 3))])
case("maxpool", [n("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])], {"x": img})
case("maxpool_ceil", [n("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)], {"x": f32(1, 2, 7, 5)})
case("avgpool_pad", [n("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], count_include_pad=0)], {"x": img})
case("avgpool_pad_count", [n("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], count_include_pad=1)], {"x": img})
case("global_pools", [n("GlobalAveragePool", ["x"], ["a"]), n("GlobalMaxPool", ["x"], ["m"]), n("Add", ["a", "m"], ["y"])], {"x": img})
# normalisation
case("batchnorm", [n("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], epsilon=1e-3)], {"x": img},
     inits=[("s", f32(3)), ("b", f32(3)), ("m", f32(3)), ("v", f32(3, lo=0.5, hi=2))])
case("instancenorm", [n("InstanceNormalization", ["x", "s", "b"], ["y"])], {"x": img}, inits=[("s", f32(3)), ("b", f32(3))])
case("layernorm", [n("LayerNormalization", ["x", "s", "b"], ["y"], axis=-1, epsilon=1e-5)], {"x": f32(2, 3, 8)}, inits=[("s", f32(8)), ("b", f32(8))])
case("layernorm_axis1", [n("LayerNormalization", ["x", "s"], ["y"], axis=1)], {"x": f32(2, 3, 4)}, inits=[("s", f32(3, 4))])
case("groupnorm", [n("GroupNormalization", ["x", "s", "b"], ["y"], num_groups=2, epsilon=1e-5)], {"x": f32(2, 4, 3, 3)}, inits=[("s", f32(4)), ("b", f32(4))], opset=21)
case("lpnorm", [n("LpNormalization", ["x"], ["y"], axis=1, p=2)], {"x": f32(3, 5)})
case("softmax13", [n("Softmax", ["x"], ["y"], axis=1)], {"x": f32(2, 5, 3) * 3})
case("softmax11", [n("Softmax", ["x"], ["y"], axis=1)], {"x": f32(2, 5, 3) * 3}, opset=11)
case("logsoftmax", [n("LogSoftmax", ["x"], ["y"])], {"x": f32(4, 6)})
# shape operators
case("reshape_flatten", [n("Reshape", ["x", "s"], ["r"]), n("Flatten", ["r"], ["y"], axis=2)], {"x": f32(2, 3, 4)}, inits=[("s", np.array([0, -1, 2], np.int64))])
case("squeeze_unsqueeze13", [n("Unsqueeze", ["x", "a"], ["u"]), n("Squeeze", ["u", "b"], ["y"])], {"x": f32(3, 1, 4)},
     inits=[("a", np.array([0, 3], np.int64)), ("b", np.array([2], np.int64))])
case("squeeze_unsqueeze11", [n("Unsqueeze", ["x"], ["u"], axes=[-1]), n("Squeeze", ["u"], ["y"], axes=[1])], {"x": f32(3, 1, 4)}, opset=11)
case("transpose", [n("Transpose", ["x"], ["y"], perm=[2, 0, 3, 1])], {"x": f32(2, 3, 4, 5)})
case("concat_split", [n("Concat", ["a", "b"], ["c"], axis=1), n("Split", ["c", "s"], ["y0", "y1", "y2"], axis=1)], {"a": f32(2, 3, 2), "b": f32(2, 4, 2)},
     inits=[("s", np.array([1, 4, 2], np.int64))], outputs=["y0", "y1", "y2"])
case("split_equal18", [n("Split", ["x"], ["y0", "y1", "y2"], axis=-1, num_outputs=3)], {"x": f32(2, 7)}, outputs=["y0", "y1", "y2"], opset=18)
case("slice_steps", [n("Slice", ["x", "st", "en", "ax", "sp"], ["y"])], {"x": f32(5, 6, 7)},
     inits=[("st", np.array([1, -1], np.int64)), ("en", np.array([100, 0], np.int64)), ("ax", np.array([0, 2], np.int64)), ("sp", np.array([2, -2], np.int64))])
case("gather_axis1", [n("Gather", ["x", "i"], ["y"], axis=1)], {"x": f32(3, 5, 2)}, inits=[("i", np.array([[0, 4], [-1, 2]], np.int64))])
case("gather_elements", [n("GatherElements", ["x", "i"], ["y"], axis=1)], {"x": f32(3, 4)}, inits=[("i", np.array([[0, 3], [1, 1], [2, 0]], np.int64))])
case("scatter_nd", [n("ScatterND", ["x", "i", "u"], ["y"])], {"x": f32(4, 3)}, inits=[("i", np.array([[1], [3]], np.int64)), ("u", f32(2, 3))])
case("expand_tile", [n("Expand", ["x", "s"], ["e"]), n("Tile", ["e", "r"], ["y"])], {"x": f32(3, 1)},
     inits=[("s", np.array([2, 3, 4], np.int64)), ("r", np.array([1, 2, 1], np.int64))])
for mode in ["constant", "reflect", "edge"]:
    case("pad_" + mode, [n("Pad", ["x", "p", "v"] if mode == "constant" else ["x", "p"], ["y"], mode=mode)], {"x": f32(1, 2, 4, 5)},
         inits=[("p", np.array([0, 0, 1, 2, 0, 0, 2, 1], np.int64))] + ([("v", np.array(0.5, np.float32))] if mode == "constant" else []))
case("depth_to_space", [n("DepthToSpace", ["x"], ["d"], blocksize=2), n("DepthToSpace", ["x"], ["c"], blocksize=2, mode="CRD"), n("Add", ["d", "c"], ["y"])],
     {"x": f32(1, 8, 3, 2)})
case("space_to_depth", [n("SpaceToDepth", ["x"], ["y"], blocksize=2)], {"x": f32(1, 2, 4, 6)})
case("shape_gather_concat", [n("Shape", ["x"], ["s"]), n("Gather", ["s", "i"], ["g"], axis=0), n("Concat", ["g", "k"], ["c"], axis=0), n("Reshape", ["x", "c"], ["y"])],
     {"x": f32(2, 3, 4)}, inits=[("i", np.array([0], np.int64)), ("k", np.array([-1], np.int64))])
case("constant_of_shape_range", [n("ConstantOfShape", ["s"], ["c"], value=nh.from_array(np.array([2.5], np.float32))),
                                  n("Range", ["a", "b", "d"], ["r"]), n("Add", ["c", "r"], ["y"])],
     {"s": np.array([3, 4], np.int64)}, inits=[("a", np.array(1.0, np.float32)), ("b", np.array(3.0, np.float32)), ("d", np.array(0.5, np.float32))])
case("cast_chain", [n("Cast", ["x"], ["i"], to=T.INT64), n("Cast", ["i"], ["y"], to=T.FLOAT)], {"x": f32(3, 4) * 5})
case("constant_node", [n("Constant", [], ["c"], value=nh.from_array(f32(4), "c")), n("Add", ["x", "c"], ["y"])], {"x": f32(3, 4)})
case("trilu", [n("Trilu", ["x", "k"], ["y"], upper=0)], {"x": f32(2, 4, 5)}, inits=[("k", np.array(1, np.int64))])
case("cumsum", [n("CumSum", ["x", "a"], ["y"], exclusive=1, reverse=1)], {"x": f32(3, 4)}, inits=[("a", np.array(1, np.int64))])
case("topk", [n("TopK", ["x", "k"], ["v", "i"], axis=1)], {"x": f32(3, 6)}, inits=[("k", np.array([2], np.int64))], outputs=["v", "i"], out_types={"i": T.INT64})
case("argmax_argmin", [n("ArgMax", ["x"], ["a"], axis=1, keepdims=0), n("ArgMin", ["x"], ["b"], axis=-1), n("Cast", ["a"], ["y"], to=T.FLOAT)], {"x": f32(3, 5, 4)},
     outputs=["y", "b"], out_types={"b": T.INT64})
# reductions
for op in ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL2", "ReduceSumSquare", "ReduceLogSumExp", "ReduceL1"]:
    case(op.lower() + "18", [n(op, ["x", "a"], ["y"], keepdims=1)], {"x": f32(2, 3, 4, lo=0.5, hi=1.5)}, inits=[("a", np.array([1, -1], np.int64))], opset=18)
case("reducemean11", [n("ReduceMean", ["x"], ["y"], axes=[1], keepdims=0)], {"x": f32(2, 3, 4)}, opset=11)
case("reducesum_all", [n("ReduceSum", ["x"], ["y"], keepdims=0)], {"x": f32(2, 3, 4)}, opset=13)
# resize
for mode, coord in [("nearest", "asymmetric"), ("nearest", "half_pixel"), ("linear", "half_pixel"), ("linear", "align_corners"),
                    ("linear", "pytorch_half_pixel"), ("cubic", "half_pixel"), ("cubic", "align_corners")]:
    case(f"resize_{mode}_{coord}_scales", [n("Resize", ["x", "", "s"], ["y"], mode=mode, coordinate_transformation_mode=coord)], {"x": f32(1, 2, 5, 4)},
         inits=[("s", np.array([1, 1, 2, 1.5], np.float32))], opset=13)
case("resize_linear_sizes_down", [n("Resize", ["x", "", "", "z"], ["y"], mode="linear")], {"x": f32(1, 2, 9, 7)}, inits=[("z", np.array([1, 2, 4, 3], np.int64))], opset=13)
case("resize_nearest_floor", [n("Resize", ["x", "", "s"], ["y"], mode="nearest", nearest_mode="floor")], {"x": f32(1, 1, 3, 3)},
     inits=[("s", np.array([1, 1, 2.5, 2], np.float32))], opset=13)
case("upsample9", [n("Upsample", ["x", "s"], ["y"], mode="nearest")], {"x": f32(1, 2, 3, 3)}, inits=[("s", np.array([1, 1, 2, 2], np.float32))], opset=9)
# a small network end to end: conv - norm - act - attention-ish matmul - softmax
case("tiny_net", [
    n("Conv", ["x", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1]), n("InstanceNormalization", ["c1", "s1", "o1"], ["n1"]), n("Relu", ["n1"], ["r1"]),
    n("Reshape", ["r1", "shp"], ["t"]), n("Transpose", ["t"], ["tt"], perm=[0, 2, 1]), n("MatMul", ["tt", "wq"], ["q"]),
    n("MatMul", ["q", "t"], ["att"]), n("Softmax", ["att"], ["p"], axis=-1), n("MatMul", ["p", "tt"], ["o"]), n("ReduceMean", ["o", "ax"], ["y"], keepdims=0)],
    {"x": img}, inits=[("w1", f32(4, 3, 3, 3)), ("b1", f32(4)), ("s1", f32(4)), ("o1", f32(4)), ("shp", np.array([1, 4, 64], np.int64)),
                       ("wq", f32(4, 4)), ("ax", np.array([1], np.int64))], opset=18)

# convolution + activation (fused by the executor)
case("conv_leakyrelu_fused", [n("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]), n("LeakyRelu", ["c"], ["y"], alpha=0.2)], {"x": f32(1, 16, 9, 7)},
     inits=[("w", f32(8, 16, 3, 3)), ("b", f32(8))])
case("conv_clip_relu_fused", [n("Conv", ["x", "w"], ["c"], pads=[2, 2, 2, 2], dilations=[2, 2]), n("Clip", ["c", "lo", "hi"], ["d"]),
                              n("Conv", ["d", "w2", "b2"], ["e"], pads=[1, 1, 1, 1]), n("Relu", ["e"], ["y"])], {"x": f32(1, 20, 6, 8)},
     inits=[("w", f32(18, 20, 3, 3)), ("lo", np.array(-0.2, np.float32)), ("hi", np.array(0.3, np.float32)), ("w2", f32(5, 18, 3, 3)), ("b2", f32(5))])
case("conv_shared_output_not_fused", [n("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), n("Relu", ["c"], ["r"]), n("Add", ["c", "r"], ["y"])],
     {"x": f32(1, 16, 5, 5)}, inits=[("w", f32(4, 16, 3, 3))])

# quantised operators
u8 = lambda *s: rng.integers(0, 256, s).astype(np.uint8)
i8 = lambda *s: rng.integers(-128, 128, s).astype(np.int8)
case("dynamic_quantize", [n("DynamicQuantizeLinear", ["x"], ["y", "s", "z"])], {"x": f32(3, 7) * 3}, outputs=["y", "s", "z"],
     out_types={"y": T.UINT8, "z": T.UINT8}, opset=13)
case("matmul_integer", [n("MatMulInteger", ["a", "b", "az", "bz"], ["y"])], {"a": u8(3, 8)},
     inits=[("b", i8(8, 5)), ("az", np.array(7, np.uint8)), ("bz", i8(5))], out_types={"y": T.INT32}, opset=13)
case("conv_integer", [n("ConvInteger", ["x", "w", "xz", "wz"], ["y"], pads=[1, 1, 1, 1])], {"x": u8(1, 2, 5, 5)},
     inits=[("w", u8(3, 2, 3, 3)), ("xz", np.array(100, np.uint8)), ("wz", np.array(128, np.uint8))], out_types={"y": T.INT32}, opset=13)
case("quantize_dequantize_axis", [n("QuantizeLinear", ["x", "s", "z"], ["q"], axis=1), n("DequantizeLinear", ["q", "s", "z"], ["y"], axis=1)],
     {"x": f32(2, 3, 4) * 2}, inits=[("s", np.array([0.01, 0.02, 0.05], np.float32)), ("z", np.array([0, 10, -5], np.int8))], outputs=["q", "y"],
     out_types={"q": T.INT8}, opset=13)
case("dequantize_scalar_u8", [n("DequantizeLinear", ["x", "s", "z"], ["y"])], {"x": u8(4, 4)},
     inits=[("s", np.array(0.1, np.float32)), ("z", np.array(128, np.uint8))], opset=13)
case("qlinear_matmul", [n("QLinearMatMul", ["a", "as", "az", "b", "bs", "bz", "ys", "yz"], ["y"])], {"a": u8(3, 6)},
     inits=[("as", np.array(0.02, np.float32)), ("az", np.array(120, np.uint8)), ("b", u8(6, 4)), ("bs", np.array(0.03, np.float32)),
            ("bz", np.array(130, np.uint8)), ("ys", np.array(0.05, np.float32)), ("yz", np.array(128, np.uint8))], out_types={"y": T.UINT8}, opset=13)
case("qlinear_conv", [n("QLinearConv", ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"], ["y"], pads=[1, 1, 1, 1])], {"x": u8(1, 2, 5, 5)},
     inits=[("xs", np.array(0.02, np.float32)), ("xz", np.array(110, np.uint8)), ("w", i8(3, 2, 3, 3)), ("ws", np.array([0.01, 0.02, 0.015], np.float32)),
            ("wz", np.zeros(3, np.int8)), ("ys", np.array(0.1, np.float32)), ("yz", np.array(128, np.uint8)), ("b", rng.integers(-500, 500, 3).astype(np.int32))],
     out_types={"y": T.UINT8}, opset=13)
def nbits_case(name, K, N, bs, with_zp, bias):
    W = f32(K, N)
    blocks = (K + bs - 1) // bs
    Wp = np.zeros((blocks * bs, N), np.float32); Wp[:K] = W
    q = np.zeros((N, blocks, bs), np.uint8); scales = np.zeros((N, blocks), np.float32); zps = np.zeros((N, blocks), np.uint8)
    for c in range(N):
        for b in range(blocks):
            v = Wp[b * bs:(b + 1) * bs, c]
            lo, hi = min(v.min(), 0), max(v.max(), 0)
            s = (hi - lo) / 15 if hi > lo else 1.0
            z = int(np.clip(np.round(-lo / s), 0, 15)) if with_zp else 8
            if not with_zp: s = max(abs(lo), abs(hi)) / 7 if hi > lo else 1.0
            scales[c, b] = s; zps[c, b] = z
            q[c, b] = np.clip(np.round(v / s) + z, 0, 15)
    packed = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).astype(np.uint8)
    ins = ["a", "b", "s"] + (["z"] if with_zp else ([""] if bias else [])) + (["", "bias"] if bias else [])
    inits = [("b", packed), ("s", scales.reshape(-1))]
    if with_zp:
        zb = (blocks + 1) // 2
        zpk = np.zeros((N, zb), np.uint8)
        for b in range(blocks): zpk[:, b // 2] |= (zps[:, b] << ((b % 2) * 4)).astype(np.uint8)
        inits.append(("z", zpk.reshape(-1)))
    if bias: inits.append(("bias", f32(N)))
    node = h.make_node("MatMulNBits", ins, ["y"], domain="com.microsoft", K=K, N=N, bits=4, block_size=bs)
    vin = [h.make_tensor_value_info("a", T.FLOAT, [2, 3, K])]
    g = h.make_graph([node], name, vin, [h.make_tensor_value_info("y", T.FLOAT, None)], initializer=[nh.from_array(v, k) for k, v in inits])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17), h.make_opsetid("com.microsoft", 1)]); m.ir_version = 8
    cases.append((name, m, {"a": f32(2, 3, K)}, ["y"]))
nbits_case("matmul_nbits_zp", 40, 6, 16, True, False)
nbits_case("matmul_nbits_symmetric_bias", 64, 5, 32, False, True)
def bnb4_case(name, K, N, bs, quant_type):
    absmax_n = (N * K + bs - 1) // bs
    codes = rng.integers(0, 16, N * K + (N * K) % 2).astype(np.uint8)
    packed = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
    node = h.make_node("MatMulBnb4", ["a", "b", "m"], ["y"], domain="com.microsoft", K=K, N=N, block_size=bs, quant_type=quant_type)
    g = h.make_graph([node], name, [h.make_tensor_value_info("a", T.FLOAT, [3, K])], [h.make_tensor_value_info("y", T.FLOAT, None)],
                     initializer=[nh.from_array(packed, "b"), nh.from_array(f32(absmax_n, lo=0.1, hi=1.0), "m")])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17), h.make_opsetid("com.microsoft", 1)]); m.ir_version = 8
    cases.append((name, m, {"a": f32(3, K)}, ["y"]))
bnb4_case("matmul_bnb4_nf4", 32, 6, 64, 1)
bnb4_case("matmul_bnb4_fp4", 16, 4, 16, 0)

# control flow: If branches, Loop bodies and Scan bodies are subgraphs that may read outer values
def vi(name, t=T.FLOAT, shape=None): return h.make_tensor_value_info(name, t, shape)
def graph(name, nodes, ins, outs): return h.make_graph(nodes, name, ins, outs)
then_g = graph("then", [n("Mul", ["x", "w"], ["t_out"])], [], [vi("t_out")])          # reads outer x and w
else_g = graph("else", [n("Sub", ["x", "w"], ["e1"]), n("Abs", ["e1"], ["e_out"])], [], [vi("e_out")])
for tag, thr in [("then", -10.0), ("else", 10.0)]:
    case("if_" + tag, [n("ReduceSum", ["x"], ["s"], keepdims=0), n("Greater", ["s", "thr"], ["c"]),
                       n("If", ["c"], ["y"], then_branch=then_g, else_branch=else_g), n("Add", ["y", "x"], ["z"])],
         {"x": f32(2, 3)}, outputs=["z"], inits=[("w", f32(2, 3)), ("thr", np.array(thr, np.float32))])
# Loop: 4 trips, a carried accumulator, a scan output, and the body reading an outer value
loop_body = graph("body", [n("Add", ["acc_in", "x"], ["acc_out"]), n("Cast", ["iter"], ["itf"], to=T.FLOAT), n("Mul", ["acc_out", "itf"], ["scan"]),
                           n("Identity", ["cond_in"], ["cond_out"])],
                  [vi("iter", T.INT64, []), vi("cond_in", T.BOOL, []), vi("acc_in")], [vi("cond_out", T.BOOL), vi("acc_out"), vi("scan")])
case("loop_carried_scan", [n("Loop", ["trips", "", "a0"], ["acc", "scans"], body=loop_body)], {"x": f32(2, 3), "a0": f32(2, 3)},
     outputs=["acc", "scans"], inits=[("trips", np.array(4, np.int64))])
# Loop stopped by its condition (no trip count): keep going while the sum stays below a limit
while_body = graph("while", [n("Mul", ["v_in", "two"], ["v_out"]), n("ReduceSum", ["v_out"], ["s"], keepdims=0), n("Less", ["s", "lim"], ["cond_out"])],
                   [vi("iter", T.INT64, []), vi("cond_in", T.BOOL, []), vi("v_in")], [vi("cond_out", T.BOOL), vi("v_out")])
case("loop_while", [n("Loop", ["", "c0", "v"], ["vf"], body=while_body)], {"v": f32(3, lo=0.5, hi=1.0)}, outputs=["vf"],
     inits=[("c0", np.array(True)), ("two", np.array(2.0, np.float32)), ("lim", np.array(40.0, np.float32))])
# nested: a Loop whose body holds an If on the iteration number
odd = graph("odd", [n("Neg", ["acc_in"], ["o_out"])], [], [vi("o_out")])
even = graph("even", [n("Add", ["acc_in", "x"], ["ev_out"])], [], [vi("ev_out")])
nest_body = graph("nest", [n("Mod", ["iter", "two_i"], ["r"]), n("Equal", ["r", "one_i"], ["isodd"]), n("If", ["isodd"], ["acc_out"], then_branch=odd, else_branch=even),
                           n("Identity", ["cond_in"], ["cond_out"])],
                  [vi("iter", T.INT64, []), vi("cond_in", T.BOOL, []), vi("acc_in")], [vi("cond_out", T.BOOL), vi("acc_out")])
case("loop_nested_if", [n("Loop", ["trips", "", "a0"], ["acc"], body=nest_body)], {"x": f32(4), "a0": f32(4)}, outputs=["acc"],
     inits=[("trips", np.array(5, np.int64)), ("two_i", np.array(2, np.int64)), ("one_i", np.array(1, np.int64))])
# Scan: a running sum over the rows (one reversed), with a scan output per step
scan_body = graph("scan", [n("Add", ["st_in", "row"], ["st_out"]), n("Mul", ["st_out", "rrow"], ["out"])],
                  [vi("st_in", T.FLOAT, [3]), vi("row", T.FLOAT, [3]), vi("rrow", T.FLOAT, [3])], [vi("st_out", T.FLOAT, [3]), vi("out", T.FLOAT, [3])])
case("scan_running_sum", [n("Scan", ["s0", "xs", "xs"], ["sf", "outs"], body=scan_body, num_scan_inputs=2, scan_input_directions=[0, 1])],
     {"s0": f32(3), "xs": f32(5, 3)}, outputs=["sf", "outs"], opset=11)

# recurrent layers
T_, N_, I_, H_ = 5, 3, 4, 6
xs_ = f32(T_, N_, I_)
case("lstm_peephole", [n("LSTM", ["x", "w", "r", "b", "", "h0", "c0", "p"], ["y", "yh", "yc"], hidden_size=H_)], {"x": xs_}, outputs=["y", "yh", "yc"],
     inits=[("w", f32(1, 4 * H_, I_)), ("r", f32(1, 4 * H_, H_)), ("b", f32(1, 8 * H_)), ("h0", f32(1, N_, H_)), ("c0", f32(1, N_, H_)), ("p", f32(1, 3 * H_))])
case("lstm_bidirectional_lens", [n("LSTM", ["x", "w", "r", "b", "lens"], ["y", "yh", "yc"], hidden_size=H_, direction="bidirectional")], {"x": xs_},
     outputs=["y", "yh", "yc"], inits=[("w", f32(2, 4 * H_, I_)), ("r", f32(2, 4 * H_, H_)), ("b", f32(2, 8 * H_)), ("lens", np.array([5, 2, 4], np.int32))])
case("lstm_reverse_clip", [n("LSTM", ["x", "w", "r"], ["y", "yh"], hidden_size=H_, clip=0.5, direction="reverse")],
     {"x": f32(T_, N_, I_) * 3}, outputs=["y", "yh"], inits=[("w", f32(1, 4 * H_, I_)), ("r", f32(1, 4 * H_, H_))])
for lbr in [0, 1]:
    case(f"gru_lbr{lbr}", [n("GRU", ["x", "w", "r", "b", "", "h0"], ["y", "yh"], hidden_size=H_, linear_before_reset=lbr, direction="bidirectional")],
         {"x": xs_}, outputs=["y", "yh"], inits=[("w", f32(2, 3 * H_, I_)), ("r", f32(2, 3 * H_, H_)), ("b", f32(2, 6 * H_)), ("h0", f32(2, N_, H_))])
case("rnn_relu", [n("RNN", ["x", "w", "r", "b"], ["y", "yh"], hidden_size=H_, activations=["Relu"])], {"x": xs_}, outputs=["y", "yh"],
     inits=[("w", f32(1, H_, I_)), ("r", f32(1, H_, H_) * 0.5), ("b", f32(1, 2 * H_))])

# a channel Concat read only by a Conv is folded into it (direct stride-1 path, a strided path, and a
# Concat read twice, which stays)
case("concat_into_conv", [n("Conv", ["x", "w0"], ["a"], pads=[1, 1, 1, 1]), n("LeakyRelu", ["a"], ["la"], alpha=0.2), n("Concat", ["x", "la"], ["c1"], axis=1),
                          n("Conv", ["c1", "w1", "b1"], ["y1"], pads=[1, 1, 1, 1]), n("Concat", ["x", "la", "y1"], ["c2"], axis=1),
                          n("Conv", ["c2", "w2"], ["y"], pads=[1, 1, 1, 1], strides=[2, 2])],
     {"x": f32(1, 16, 9, 11)}, inits=[("w0", f32(8, 16, 3, 3) * 0.3), ("w1", f32(12, 24, 3, 3) * 0.3), ("b1", f32(12)), ("w2", f32(5, 36, 3, 3) * 0.3)])
case("concat_read_twice", [n("Concat", ["a", "b"], ["c"], axis=1), n("Conv", ["c", "w"], ["y1"], pads=[1, 1, 1, 1]), n("Add", ["y1", "y1"], ["s"]),
                           n("ReduceSum", ["c"], ["r"], axes=[1], keepdims=1), n("Add", ["s", "r"], ["y"])],
     {"a": f32(2, 10, 6, 7), "b": f32(2, 14, 6, 7)}, inits=[("w", f32(3, 24, 3, 3))], opset=11)

# depthwise convolutions (one input channel per group)
case("conv_depthwise_stride", [n("Conv", ["x", "w", "b"], ["y"], group=6, pads=[1, 1, 1, 1], strides=[2, 2])], {"x": f32(2, 6, 9, 8)},
     inits=[("w", f32(6, 1, 3, 3)), ("b", f32(6))])
case("conv_depthwise_multiplier_dilation", [n("Conv", ["x", "w"], ["y"], group=4, dilations=[2, 2], pads=[2, 1, 2, 3])], {"x": f32(1, 4, 7, 10)},
     inits=[("w", f32(8, 1, 3, 3))])
case("conv_depthwise_1d_clip", [n("Conv", ["x", "w"], ["c"], group=5, pads=[2, 2]), n("Clip", ["c", "lo", "hi"], ["y"])], {"x": f32(1, 5, 13)},
     inits=[("w", f32(5, 1, 5)), ("lo", np.array(0.0, np.float32)), ("hi", np.array(0.6, np.float32))])

shutil.rmtree(OUT, ignore_errors=True)
for name, model, inputs, outputs in cases:
    # (outputs are declared without shapes, which onnx.checker refuses; onnxruntime accepts them)
    d = os.path.join(OUT, name)
    os.makedirs(d)
    onnx.save(model, os.path.join(d, "model.onnx"))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    res = sess.run(outputs, inputs)
    for k, (key, v) in enumerate(inputs.items()):
        with open(os.path.join(d, f"input_{k}.pb"), "wb") as f: f.write(nh.from_array(v, key).SerializeToString())
    for k, (key, v) in enumerate(zip(outputs, res)):
        with open(os.path.join(d, f"output_{k}.pb"), "wb") as f: f.write(nh.from_array(np.asarray(v), key).SerializeToString())
print(len(cases), "cases in", OUT)

# Picture models for the "onnx" workflow step (no reference outputs: the C++ test knows what they do).
PIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline")
shutil.rmtree(PIPE, ignore_errors=True)
os.makedirs(PIPE)
def save_model(name, nodes, inp, out, inits=(), opset=17):
    g = h.make_graph(nodes, name, [inp], [out], initializer=[nh.from_array(v, k) for k, v in inits])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", opset)])
    m.ir_version = 8
    onnx.save(m, os.path.join(PIPE, name + ".onnx"))
# a fixed-size x2 "upscaler" (nearest): run over bigger pictures in tiles
save_model("up2_fixed8", [n("Resize", ["x", "", "s"], ["y"], mode="nearest")], h.make_tensor_value_info("x", T.FLOAT, [1, 3, 8, 8]),
           h.make_tensor_value_info("y", T.FLOAT, [1, 3, 16, 16]), inits=[("s", np.array([1, 1, 2, 2], np.float32))])
# a "mask" of any size: the mean of the channels
save_model("mean_mask", [n("ReduceMean", ["x"], ["y"], axes=[1], keepdims=1)], h.make_tensor_value_info("x", T.FLOAT, [1, 3, "h", "w"]),
           h.make_tensor_value_info("y", T.FLOAT, [1, 1, "h", "w"]), opset=13)
# identity, for channel order and value-range checks
save_model("identity", [n("Identity", ["x"], ["y"])], h.make_tensor_value_info("x", T.FLOAT, [1, 3, "h", "w"]),
           h.make_tensor_value_info("y", T.FLOAT, [1, 3, "h", "w"]))
print("pipeline models in", PIPE)
