#!/usr/bin/env python3
"""Independent NumPy TAESD (diffusers AutoencoderTiny), used to check NeuraCraft's.

    pip install numpy
    scripts/taesd_reference.py data/taesd decode latent.f32 H W out.f32   # latent: 4 x H/8 x W/8
    scripts/taesd_reference.py data/taesd encode image.f32 H W out.f32    # image: 3 x H x W in [-1, 1]
"""
import json
import os
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    d = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    kinds = {"F16": np.float16, "F32": np.float32}
    return {k: d[t["data_offsets"][0]:t["data_offsets"][1]].view(kinds[t["dtype"]]).astype(np.float32).reshape(t["shape"])
            for k, t in h.items()}


def conv(x, w, b=None, stride=1):
    x = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    o, c, k, _ = w.shape
    h = (x.shape[1] - k) // stride + 1
    wd = (x.shape[2] - k) // stride + 1
    out = np.zeros((o, h * wd), dtype=np.float32)
    for ky in range(k):
        for kx in range(k):
            out += w[:, :, ky, kx] @ x[:, ky:ky + stride * h:stride, kx:kx + stride * wd:stride].reshape(c, -1)
    out = out.reshape(o, h, wd)
    return out + b[:, None, None] if b is not None else out


relu = lambda x: np.maximum(x, 0.0)


def block(w, x, p):
    h = relu(conv(x, w[p + ".conv.0.weight"], w[p + ".conv.0.bias"]))
    h = relu(conv(h, w[p + ".conv.2.weight"], w[p + ".conv.2.bias"]))
    h = conv(h, w[p + ".conv.4.weight"], w[p + ".conv.4.bias"])
    return relu(h + x)


def decode(w, cfg, z):
    x = np.tanh(z / 3) * 3
    x = relu(conv(x, w["decoder.layers.0.weight"], w["decoder.layers.0.bias"]))
    i = 2
    ch = cfg["decoder_block_out_channels"]
    for level, n in enumerate(cfg["num_decoder_blocks"]):
        for _ in range(n):
            x = block(w, x, f"decoder.layers.{i}")
            i += 1
        last = level + 1 == len(ch)
        if not last:
            x = x.repeat(2, axis=1).repeat(2, axis=2)
            i += 1
        x = conv(x, w[f"decoder.layers.{i}.weight"], w.get(f"decoder.layers.{i}.bias"))
        i += 1
    return x * 2 - 1


def encode(w, cfg, img):
    x = (img + 1) / 2
    i = 0
    for level, n in enumerate(cfg["num_encoder_blocks"]):
        x = conv(x, w[f"encoder.layers.{i}.weight"], w.get(f"encoder.layers.{i}.bias"), 1 if level == 0 else 2)
        i += 1
        for _ in range(n):
            x = block(w, x, f"encoder.layers.{i}")
            i += 1
    return conv(x, w[f"encoder.layers.{i}.weight"], w[f"encoder.layers.{i}.bias"])


def main():
    d, mode, src, H, W, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    cfg = json.load(open(os.path.join(d, "config.json")))
    w = load(os.path.join(d, "diffusion_pytorch_model.safetensors"))
    a = np.fromfile(src, dtype=np.float32)
    if mode == "decode":
        y = decode(w, cfg, a.reshape(4, H // 8, W // 8))
    else:
        y = encode(w, cfg, a.reshape(3, H, W))
    y.astype(np.float32).tofile(out)


if __name__ == "__main__":
    main()
