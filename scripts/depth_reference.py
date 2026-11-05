#!/usr/bin/env python3
"""Independent NumPy implementation of Depth Anything V2 (transformers
DepthAnythingForDepthEstimation with a DINOv2 backbone), used to check
NeuraCraft's DepthEstimator::forward.

    pip install numpy
    scripts/depth_reference.py MODEL_DIR input.f32 H W out.f32

input.f32: 3 x H x W normalized pixels (ImageNet mean / std), H and W
multiples of 14; writes the depth map (H x W).
"""
import json
import math
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()
    h.pop("__metadata__", None)
    return {k: np.frombuffer(data[v["data_offsets"][0]:v["data_offsets"][1]], np.float32).reshape(v["shape"]) for k, v in h.items()}


def conv(x, w, b, stride=1, pad=0):
    c, H, W = x.shape
    o, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    OH, OW = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    out = np.zeros((o, OH * OW), np.float32)
    for ky in range(k):
        for kx in range(k):
            out += w[:, :, ky, kx] @ xp[:, ky:ky + stride * OH:stride, kx:kx + stride * OW:stride].reshape(c, -1)
    out = out.reshape(o, OH, OW)
    return out + b[:, None, None] if b is not None else out


def conv_t_blocks(x, w, b):  # kernel = stride = s, padding 0
    cin, H, W = x.shape
    s = w.shape[2]
    out = np.einsum("iokl,ihw->ohkwl", w, x).reshape(w.shape[1], H * s, W * s)
    return out + b[:, None, None]


def bilinear(x, H, W, align):
    c, h, w = x.shape
    if align:
        ys = np.arange(H) * ((h - 1) / (H - 1) if H > 1 else 0)
        xs = np.arange(W) * ((w - 1) / (W - 1) if W > 1 else 0)
    else:
        ys = np.maximum((np.arange(H) + 0.5) * h / H - 0.5, 0)
        xs = np.maximum((np.arange(W) + 0.5) * w / W - 0.5, 0)
    y0 = np.minimum(np.floor(ys).astype(int), h - 1); y1 = np.minimum(y0 + 1, h - 1); ay = (ys - y0)[None, :, None]
    x0 = np.minimum(np.floor(xs).astype(int), w - 1); x1 = np.minimum(x0 + 1, w - 1); ax = (xs - x0)[None, None, :]
    top = x[:, y0][:, :, x0] * (1 - ax) + x[:, y0][:, :, x1] * ax
    bot = x[:, y1][:, :, x0] * (1 - ax) + x[:, y1][:, :, x1] * ax
    return (top * (1 - ay) + bot * ay).astype(np.float32)


def bicubic(x, H, W, A=-0.75):
    def cub(t):
        t = np.abs(t)
        return np.where(t <= 1, ((A + 2) * t - (A + 3)) * t * t + 1, np.where(t < 2, ((A * t - 5 * A) * t + 8 * A) * t - 4 * A, 0))
    c, h, w = x.shape

    def weights(n_out, n_in):
        src = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        i0 = np.floor(src).astype(int)
        t = src - i0
        M = np.zeros((n_out, n_in))
        for m in range(-1, 3):
            idx = np.clip(i0 + m, 0, n_in - 1)
            np.add.at(M, (np.arange(n_out), idx), cub(m - t))
        return M
    My, Mx = weights(H, h), weights(W, w)
    return np.einsum("yh,chw,xw->cyx", My, x, Mx).astype(np.float32)


def layer_norm(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def main():
    d, inp, H, W, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    cfg = json.load(open(d + "/config.json"))
    bc = cfg["backbone_config"]
    w = load(d + "/model.safetensors")
    D, heads, P = bc.get("hidden_size", 384), bc.get("num_attention_heads", 6), bc.get("patch_size", 14)
    eps = bc.get("layer_norm_eps", 1e-6)
    x = np.fromfile(inp, np.float32).reshape(3, H, W)
    ph, pw = H // P, W // P
    b = "backbone."
    pe = conv(x, w[b + "embeddings.patch_embeddings.projection.weight"], w[b + "embeddings.patch_embeddings.projection.bias"], P)
    tokens = pe.reshape(D, -1).T
    pos = w[b + "embeddings.position_embeddings"][0]
    side = int(round(math.sqrt(pos.shape[0] - 1)))
    grid = pos[1:].T.reshape(D, side, side)
    if (ph, pw) != (side, side):
        grid = bicubic(grid, ph, pw)
    t = np.concatenate([w[b + "embeddings.cls_token"][0] + pos[:1], tokens + grid.reshape(D, -1).T])
    lin = lambda v, n: v @ w[n + ".weight"].T + w[n + ".bias"]
    feats = []
    for i in range(bc.get("num_hidden_layers", 12)):
        p = b + f"encoder.layer.{i}."
        a = layer_norm(t, w[p + "norm1.weight"], w[p + "norm1.bias"], eps)
        q, k, v = (lin(a, p + "attention.attention." + n).reshape(-1, heads, D // heads) for n in ("query", "key", "value"))
        s = np.einsum("nhd,mhd->hnm", q, k) / math.sqrt(D // heads)
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        att = lin(np.einsum("hnm,mhd->nhd", s, v).reshape(-1, D), p + "attention.output.dense")
        t = t + w[p + "layer_scale1.lambda1"] * att
        a = layer_norm(t, w[p + "norm2.weight"], w[p + "norm2.bias"], eps)
        hdn = lin(a, p + "mlp.fc1")
        hdn = 0.5 * hdn * (1 + np.vectorize(math.erf)(hdn / math.sqrt(2)))
        t = t + w[p + "layer_scale2.lambda1"] * lin(hdn, p + "mlp.fc2")
        if i + 1 in bc["out_indices"]:
            n = layer_norm(t, w[b + "layernorm.weight"], w[b + "layernorm.bias"], eps)
            feats.append(n[1:].T.reshape(D, ph, pw).astype(np.float32))
    fs = []
    for i, (f, factor) in enumerate(zip(feats, cfg["reassemble_factors"])):
        p = f"neck.reassemble_stage.layers.{i}."
        f = conv(f, w[p + "projection.weight"], w[p + "projection.bias"])
        if factor > 1:
            f = conv_t_blocks(f, w[p + "resize.weight"], w[p + "resize.bias"])
        elif factor < 1:
            f = conv(f, w[p + "resize.weight"], w[p + "resize.bias"], int(round(1 / factor)), 1)
        fs.append(conv(f, w[f"neck.convs.{i}.weight"], None, 1, 1))

    def preact(h, p):
        r = conv(np.maximum(h, 0), w[p + "convolution1.weight"], w[p + "convolution1.bias"], 1, 1)
        r = conv(np.maximum(r, 0), w[p + "convolution2.weight"], w[p + "convolution2.bias"], 1, 1)
        return r + h
    fs = fs[::-1]
    fused = None
    for s, f in enumerate(fs):
        p = f"neck.fusion_stage.layers.{s}."
        if fused is None:
            h = f
        else:
            res = f if f.shape == fused.shape else bilinear(f, fused.shape[1], fused.shape[2], False)
            h = fused + preact(res, p + "residual_layer1.")
        h = preact(h, p + "residual_layer2.")
        size = fs[s + 1].shape[1:] if s + 1 < len(fs) else (h.shape[1] * 2, h.shape[2] * 2)
        h = bilinear(h, size[0], size[1], True)
        fused = conv(h, w[p + "projection.weight"], w[p + "projection.bias"])
    dmap = conv(fused, w["head.conv1.weight"], w["head.conv1.bias"], 1, 1)
    dmap = bilinear(dmap, ph * P, pw * P, True)
    dmap = np.maximum(conv(dmap, w["head.conv2.weight"], w["head.conv2.bias"], 1, 1), 0)
    dmap = np.maximum(conv(dmap, w["head.conv3.weight"], w["head.conv3.bias"]), 0) * cfg.get("max_depth", 1)
    dmap[0].astype(np.float32).tofile(out)


if __name__ == "__main__":
    main()
