#!/usr/bin/env python3
"""Independent NumPy implementation of Stable Diffusion's VAE (diffusers
AutoencoderKL), used to check NeuraCraft's.

    pip install numpy
    scripts/vae_reference.py data/sd-turbo image.ppm out_prefix

Writes out_prefix.latent.f32 (scaled latent mean, 4 x H/8 x W/8) and
out_prefix.recon.f32 (decoded image, 3 x H x W in [-1, 1]); with --decode
latent.f32 it decodes the given latent instead of encoding the image.
Arrays are raw little-endian float32 in (C, H, W) order.
"""
import json
import os
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, t in header.items():
        a, b = t["data_offsets"]
        raw = data[a:b]
        kind = {"F16": np.float16, "F32": np.float32}[t["dtype"]] if t["dtype"] != "BF16" else None
        arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32) if kind is None else raw.view(kind)
        out[name] = arr.astype(np.float32).reshape(t["shape"])
    return out


def conv2d(x, w, b, stride=1, pad=(1, 1, 1, 1)):
    """x: (C, H, W); w: (O, C, k, k); pad: (top, bottom, left, right)."""
    x = np.pad(x, ((0, 0), (pad[0], pad[1]), (pad[2], pad[3])))
    o, c, k, _ = w.shape
    h = (x.shape[1] - k) // stride + 1
    wd = (x.shape[2] - k) // stride + 1
    out = np.zeros((o, h * wd), dtype=np.float32)
    wm = w.reshape(o, c, k * k)
    for ky in range(k):
        for kx in range(k):
            patch = x[:, ky:ky + stride * h:stride, kx:kx + stride * wd:stride].reshape(c, -1)
            out += wm[:, :, ky * k + kx] @ patch
    return out.reshape(o, h, wd) + b[:, None, None]


def group_norm(x, gamma, beta, groups=32, eps=1e-6):
    c, h, w = x.shape
    g = x.reshape(groups, -1).astype(np.float64)
    g = (g - g.mean(1, keepdims=True)) / np.sqrt(g.var(1, keepdims=True) + eps)
    return (g.reshape(c, h, w) * gamma[:, None, None] + beta[:, None, None]).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


class VAE:
    def __init__(self, model_dir):
        d = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else os.path.join(model_dir, "vae")
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        f = os.path.join(d, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(f):
            f = os.path.join(d, "diffusion_pytorch_model.fp16.safetensors")
        self.w = load_safetensors(f)

    def conv(self, x, name, stride=1, pad=None):
        w = self.w[name + ".weight"]
        k = w.shape[-1]
        if pad is None:
            pad = (k // 2,) * 4
        return conv2d(x, w, self.w[name + ".bias"], stride, pad)

    def norm(self, x, name):
        return group_norm(x, self.w[name + ".weight"], self.w[name + ".bias"], self.cfg.get("norm_num_groups", 32))

    def resnet(self, x, p):
        h = self.conv(silu(self.norm(x, p + ".norm1")), p + ".conv1")
        h = self.conv(silu(self.norm(h, p + ".norm2")), p + ".conv2")
        if p + ".conv_shortcut.weight" in self.w:
            x = self.conv(x, p + ".conv_shortcut")
        return x + h

    def attention(self, x, p):
        c, h, w = x.shape
        t = self.norm(x, p + ".group_norm").reshape(c, -1).T  # (HW, C)
        lin = lambda v, n: v @ self.w[n + ".weight"].reshape(c, c).T + self.w[n + ".bias"]
        q, k, v = lin(t, p + ".to_q"), lin(t, p + ".to_k"), lin(t, p + ".to_v")
        s = q @ k.T / np.sqrt(c)
        s = np.exp(s - s.max(1, keepdims=True))
        s /= s.sum(1, keepdims=True)
        o = lin(s @ v, p + ".to_out.0")
        return x + o.T.reshape(c, h, w)

    def encode(self, img):
        ch = self.cfg["block_out_channels"]
        x = self.conv(img, "encoder.conv_in")
        for i in range(len(ch)):
            for j in range(self.cfg.get("layers_per_block", 2)):
                x = self.resnet(x, f"encoder.down_blocks.{i}.resnets.{j}")
            if i + 1 < len(ch):
                x = self.conv(x, f"encoder.down_blocks.{i}.downsamplers.0.conv", stride=2, pad=(0, 1, 0, 1))
        x = self.resnet(x, "encoder.mid_block.resnets.0")
        x = self.attention(x, "encoder.mid_block.attentions.0")
        x = self.resnet(x, "encoder.mid_block.resnets.1")
        x = self.conv(silu(self.norm(x, "encoder.conv_norm_out")), "encoder.conv_out")
        moments = self.conv(x, "quant_conv")
        return moments[: moments.shape[0] // 2] * self.cfg.get("scaling_factor", 0.18215)

    def decode(self, z):
        ch = self.cfg["block_out_channels"]
        x = self.conv(z / self.cfg.get("scaling_factor", 0.18215), "post_quant_conv")
        x = self.conv(x, "decoder.conv_in")
        x = self.resnet(x, "decoder.mid_block.resnets.0")
        x = self.attention(x, "decoder.mid_block.attentions.0")
        x = self.resnet(x, "decoder.mid_block.resnets.1")
        for i in range(len(ch)):
            for j in range(self.cfg.get("layers_per_block", 2) + 1):
                x = self.resnet(x, f"decoder.up_blocks.{i}.resnets.{j}")
            if i + 1 < len(ch):
                x = x.repeat(2, axis=1).repeat(2, axis=2)
                x = self.conv(x, f"decoder.up_blocks.{i}.upsamplers.0.conv")
        return self.conv(silu(self.norm(x, "decoder.conv_norm_out")), "decoder.conv_out")


def load_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(maxsplit=4)
    assert parts[0] == b"P6" and int(parts[3]) == 255, "binary 8-bit PPM expected"
    w, h = int(parts[1]), int(parts[2])
    pix = np.frombuffer(parts[4][: w * h * 3], dtype=np.uint8).reshape(h, w, 3)
    return pix.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0


def main():
    model_dir, image, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    vae = VAE(model_dir)
    if "--decode" in sys.argv:
        z = np.fromfile(sys.argv[sys.argv.index("--decode") + 1], dtype=np.float32)
        img = load_ppm(image)
        z = z.reshape(4, img.shape[1] // 8, img.shape[2] // 8)
    else:
        z = vae.encode(load_ppm(image)).astype(np.float32)
        z.tofile(prefix + ".latent.f32")
    vae.decode(z).astype(np.float32).tofile(prefix + ".recon.f32")
    print("latent", z.shape, "mean", float(z.mean()), "std", float(z.std()))


if __name__ == "__main__":
    main()
