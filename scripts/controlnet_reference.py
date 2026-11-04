#!/usr/bin/env python3
"""Independent NumPy implementation of diffusers' ControlNetModel, used to
check NeuraCraft's UNet2DCondition::controlEmbedding / control.

    pip install numpy
    scripts/controlnet_reference.py CONTROLNET_DIR latent.f32 context.f32 cond.f32 T H W out.f32

latent.f32: 4 x H x W (latent size); context.f32: 77 x D; cond.f32: 3 x 8H x 8W
conditioning image in [0, 1]; writes every down residual and then the mid
residual (conditioning scale 1), concatenated, float32. The down / mid blocks
come from unet_reference.py.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_reference import UNet, Lazy, conv2d, silu  # noqa: E402


class ControlNet(UNet):
    def __init__(self, d):
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        f = os.path.join(d, "diffusion_pytorch_model.safetensors")
        self.w = Lazy(f if os.path.exists(f) else os.path.join(d, "diffusion_pytorch_model.fp16.safetensors"))
        self.groups = self.cfg.get("norm_num_groups", 32)
        self.eps = self.cfg.get("norm_eps", 1e-5)
        heads = self.cfg.get("num_attention_heads") or self.cfg["attention_head_dim"]
        n = len(self.cfg["block_out_channels"])
        self.heads = heads if isinstance(heads, list) else [heads] * n

    def residuals(self, x, t, ctx, cond):
        cfg = self.cfg
        ch = cfg["block_out_channels"]
        half = ch[0] // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(half) / (half - cfg.get("freq_shift", 0)))
        e = t * freqs
        emb = np.concatenate([np.cos(e), np.sin(e)]).astype(np.float32)
        temb = self.lin(silu(self.lin(emb, "time_embedding.linear_1")), "time_embedding.linear_2")
        # Conditioning embedding: conv_in, then pairs of (3x3, 3x3 stride 2), SiLU after each, conv_out.
        c = silu(self.conv(cond, "controlnet_cond_embedding.conv_in"))
        k = 0
        while "controlnet_cond_embedding.blocks.%d.weight" % k in self.w:
            c = silu(self.conv(c, "controlnet_cond_embedding.blocks.%d" % k, stride=1 + (k % 2)))
            k += 1
        c = self.conv(c, "controlnet_cond_embedding.conv_out")
        h = self.conv(x, "conv_in") + c
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
        out = [self.conv(s, "controlnet_down_blocks.%d" % k) for k, s in enumerate(skips)]
        out.append(self.conv(h, "controlnet_mid_block"))
        return np.concatenate([o.ravel() for o in out])


def main():
    d, latent, context, cond, t, H, W, out = sys.argv[1:9]
    H, W = int(H), int(W)
    x = np.fromfile(latent, dtype=np.float32).reshape(4, H, W)
    ctx = np.fromfile(context, dtype=np.float32).reshape(77, -1)
    c = np.fromfile(cond, dtype=np.float32).reshape(3, 8 * H, 8 * W)
    ControlNet(d).residuals(x, float(t), ctx, c).astype(np.float32).tofile(out)


if __name__ == "__main__":
    main()
