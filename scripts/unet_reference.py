#!/usr/bin/env python3
"""Independent NumPy implementation of Stable Diffusion's UNet (diffusers
UNet2DConditionModel), used to check NeuraCraft's.

    pip install numpy
    scripts/unet_reference.py data/sd-turbo latent.f32 context.f32 999 H W out.f32

latent.f32: 4 x H x W float32 (H, W = latent size); context.f32: 77 x 1024
float32 text hidden states; writes the predicted noise (4 x H x W) to out.f32.
Weights are converted to float32 as they are used, to keep memory low.
"""
import json
import math
import os
import struct
import sys

import numpy as np


class Lazy:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.h.pop("__metadata__", None)
        self.d = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)

    def __contains__(self, k):
        return k in self.h

    def __getitem__(self, k):
        t = self.h[k]
        a, b = t["data_offsets"]
        kind = {"F16": np.float16, "F32": np.float32}[t["dtype"]]
        return self.d[a:b].view(kind).astype(np.float32).reshape(t["shape"])


def conv2d(x, w, b, stride=1, pad=1):
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    o, c, k, _ = w.shape
    h = (x.shape[1] - k) // stride + 1
    wd = (x.shape[2] - k) // stride + 1
    out = np.zeros((o, h * wd), dtype=np.float32)
    for ky in range(k):
        for kx in range(k):
            out += w[:, :, ky, kx] @ x[:, ky:ky + stride * h:stride, kx:kx + stride * wd:stride].reshape(c, -1)
    return out.reshape(o, h, wd) + b[:, None, None]


def group_norm(x, g, b, groups, eps):
    c, h, w = x.shape
    y = x.reshape(groups, -1).astype(np.float64)
    y = (y - y.mean(1, keepdims=True)) / np.sqrt(y.var(1, keepdims=True) + eps)
    return (y.reshape(c, h, w) * g[:, None, None] + b[:, None, None]).astype(np.float32)


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


silu = lambda x: x / (1.0 + np.exp(-x))
erf = np.vectorize(math.erf, otypes=[np.float32])


class UNet:
    def __init__(self, model_dir):
        d = os.path.join(model_dir, "unet")
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        f = os.path.join(d, "diffusion_pytorch_model.safetensors")
        self.w = Lazy(f if os.path.exists(f) else os.path.join(d, "diffusion_pytorch_model.fp16.safetensors"))
        self.groups = self.cfg.get("norm_num_groups", 32)
        self.eps = self.cfg.get("norm_eps", 1e-5)
        heads = self.cfg.get("num_attention_heads") or self.cfg["attention_head_dim"]
        n = len(self.cfg["block_out_channels"])
        self.heads = heads if isinstance(heads, list) else [heads] * n

    def lin(self, x, name):
        w = self.w[name + ".weight"]
        y = x @ w.reshape(w.shape[0], -1).T
        return y + self.w[name + ".bias"] if name + ".bias" in self.w else y

    def conv(self, x, name, stride=1):
        w = self.w[name + ".weight"]
        return conv2d(x, w, self.w[name + ".bias"], stride, w.shape[-1] // 2)

    def norm(self, x, name, eps=None):
        return group_norm(x, self.w[name + ".weight"], self.w[name + ".bias"], self.groups, self.eps if eps is None else eps)

    def resnet(self, x, temb, p):
        h = self.conv(silu(self.norm(x, p + ".norm1")), p + ".conv1")
        h = h + self.lin(silu(temb), p + ".time_emb_proj")[:, None, None]
        h = self.conv(silu(self.norm(h, p + ".norm2")), p + ".conv2")
        if p + ".conv_shortcut.weight" in self.w:
            x = self.conv(x, p + ".conv_shortcut")
        return x + h

    def attention(self, x, ctx, heads, p):
        q = self.lin(x, p + ".to_q")
        k = self.lin(ctx, p + ".to_k")
        v = self.lin(ctx, p + ".to_v")
        n, c = q.shape
        dh = c // heads
        out = np.zeros_like(q)
        for h in range(heads):
            sl = slice(h * dh, (h + 1) * dh)
            s = q[:, sl] @ k[:, sl].T / math.sqrt(dh)
            s = np.exp(s - s.max(1, keepdims=True))
            out[:, sl] = (s / s.sum(1, keepdims=True)) @ v[:, sl]
        return self.lin(out, p + ".to_out.0")

    def transformer(self, x, ctx, heads, p):
        c, hh, ww = x.shape
        t = self.norm(x, p + ".norm", 1e-6).reshape(c, -1).T
        t = self.lin(t, p + ".proj_in")
        b = p + ".transformer_blocks.0"
        ln = lambda v, n: layer_norm(v, self.w[n + ".weight"], self.w[n + ".bias"])
        a = ln(t, b + ".norm1")
        t = t + self.attention(a, a, heads, b + ".attn1")
        t = t + self.attention(ln(t, b + ".norm2"), ctx, heads, b + ".attn2")
        g = self.lin(ln(t, b + ".norm3"), b + ".ff.net.0.proj")
        hid, gate = np.split(g, 2, axis=1)
        t = t + self.lin(hid * 0.5 * gate * (1.0 + erf(gate / math.sqrt(2.0))), b + ".ff.net.2")
        return x + self.lin(t, p + ".proj_out").T.reshape(c, hh, ww)

    def __call__(self, x, t, ctx):
        cfg = self.cfg
        ch = cfg["block_out_channels"]
        half = ch[0] // 2
        freqs = np.exp(-math.log(10000.0) * np.arange(half) / (half - cfg.get("freq_shift", 0)))
        e = t * freqs
        emb = np.concatenate([np.cos(e), np.sin(e)] if cfg.get("flip_sin_to_cos", True) else [np.sin(e), np.cos(e)]).astype(np.float32)
        temb = self.lin(silu(self.lin(emb, "time_embedding.linear_1")), "time_embedding.linear_2")
        h = self.conv(x, "conv_in")
        skips = [h]
        L = cfg.get("layers_per_block", 2)
        for i, kind in enumerate(cfg["down_block_types"]):
            for j in range(L):
                h = self.resnet(h, temb, f"down_blocks.{i}.resnets.{j}")
                if kind.startswith("CrossAttn"):
                    h = self.transformer(h, ctx, self.heads[i], f"down_blocks.{i}.attentions.{j}")
                skips.append(h)
            if i + 1 < len(ch):
                h = self.conv(h, f"down_blocks.{i}.downsamplers.0.conv", stride=2)
                skips.append(h)
        h = self.resnet(h, temb, "mid_block.resnets.0")
        h = self.transformer(h, ctx, self.heads[-1], "mid_block.attentions.0")
        h = self.resnet(h, temb, "mid_block.resnets.1")
        rheads = list(reversed(self.heads))
        for i, kind in enumerate(cfg["up_block_types"]):
            for j in range(L + 1):
                h = np.concatenate([h, skips.pop()], axis=0)
                h = self.resnet(h, temb, f"up_blocks.{i}.resnets.{j}")
                if kind.startswith("CrossAttn"):
                    h = self.transformer(h, ctx, rheads[i], f"up_blocks.{i}.attentions.{j}")
            if i + 1 < len(ch):
                h = h.repeat(2, axis=1).repeat(2, axis=2)
                h = self.conv(h, f"up_blocks.{i}.upsamplers.0.conv")
        return self.conv(silu(self.norm(h, "conv_norm_out")), "conv_out")


def main():
    model_dir, latent, context, t, H, W, out = sys.argv[1:8]
    H, W = int(H), int(W)
    x = np.fromfile(latent, dtype=np.float32).reshape(4, H, W)
    ctx = np.fromfile(context, dtype=np.float32).reshape(77, -1)
    UNet(model_dir)(x, float(t), ctx).astype(np.float32).tofile(out)


if __name__ == "__main__":
    main()
