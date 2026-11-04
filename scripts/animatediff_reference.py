#!/usr/bin/env python3
"""Independent NumPy implementation of AnimateDiff's UNet (diffusers
UNetMotionModel = UNet2DConditionModel + MotionAdapter), used to check
NeuraCraft's UNet2DCondition::forwardFrames.

    pip install numpy
    scripts/animatediff_reference.py MODEL_DIR motion.safetensors latents.f32 context.f32 T F H W out.f32

latents.f32: F x 4 x H x W float32 (H, W = latent size); context.f32: 77 x D
text hidden states (the same for every frame); writes the predicted noise
(F x 4 x H x W) to out.f32. The spatial layers come from unet_reference.py.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_reference import UNet, Lazy, layer_norm, silu, erf  # noqa: E402


class MotionUNet(UNet):
    def __init__(self, model_dir, motion_file):
        super().__init__(model_dir)
        self.m = Lazy(motion_file)

    def mlin(self, x, name):
        w = self.m[name + ".weight"]
        y = x @ w.T
        return y + self.m[name + ".bias"] if name + ".bias" in self.m else y

    def frame_attention(self, x, p, heads=8):
        """x: (pixels, frames, C); self-attention across frames for each pixel."""
        q, k, v = (self.mlin(x, p + n) for n in (".to_q", ".to_k", ".to_v"))
        P, F, C = q.shape
        dh = C // heads
        q, k, v = (a.reshape(P, F, heads, dh) for a in (q, k, v))
        s = np.einsum("pfhd,pghd->phfg", q, k) / math.sqrt(dh)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("phfg,pghd->pfhd", s, v).reshape(P, F, C)
        return self.mlin(o, p + ".to_out.0")

    def motion(self, frames, p):
        x = np.stack(frames)  # F, C, H, W
        F, C, H, W = x.shape
        # GroupNorm over (channels of the group, all frames, all pixels), eps 1e-6.
        y = x.transpose(1, 0, 2, 3).reshape(self.groups, -1).astype(np.float64)
        y = (y - y.mean(1, keepdims=True)) / np.sqrt(y.var(1, keepdims=True) + 1e-6)
        y = y.reshape(C, F, H, W).astype(np.float32)
        y = y * self.m[p + ".norm.weight"][:, None, None, None] + self.m[p + ".norm.bias"][:, None, None, None]
        t = self.mlin(y.transpose(2, 3, 1, 0).reshape(H * W, F, C), p + ".proj_in")
        b = p + ".transformer_blocks.0"
        # Sinusoidal frame positions (computed here, not read from the file).
        pos = np.arange(F)[:, None]
        div = np.exp(np.arange(0, C, 2) * (-math.log(10000.0) / C))
        pe = np.zeros((F, C), dtype=np.float32)
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        ln = lambda v, n: layer_norm(v, self.m[n + ".weight"], self.m[n + ".bias"])
        t = t + self.frame_attention(ln(t, b + ".norm1") + pe, b + ".attn1")
        t = t + self.frame_attention(ln(t, b + ".norm2") + pe, b + ".attn2")
        g = self.mlin(ln(t, b + ".norm3"), b + ".ff.net.0.proj")
        hid, gate = np.split(g, 2, axis=-1)
        t = t + self.mlin(hid * 0.5 * gate * (1.0 + erf(gate / math.sqrt(2.0))), b + ".ff.net.2")
        out = self.mlin(t, p + ".proj_out").reshape(H, W, F, C).transpose(2, 3, 0, 1)
        return list(out + x)

    def frames(self, xs, t, ctx):
        cfg = self.cfg
        ch = cfg["block_out_channels"]
        half = ch[0] // 2
        freqs = np.exp(-math.log(10000.0) * np.arange(half) / (half - cfg.get("freq_shift", 0)))
        e = t * freqs
        emb = np.concatenate([np.cos(e), np.sin(e)]).astype(np.float32)
        temb = self.lin(silu(self.lin(emb, "time_embedding.linear_1")), "time_embedding.linear_2")
        each = lambda hs, fn: [fn(h) for h in hs]
        h = each(xs, lambda x: self.conv(x, "conv_in"))
        skips = [h]
        L = cfg.get("layers_per_block", 2)
        for i, kind in enumerate(cfg["down_block_types"]):
            for j in range(L):
                h = each(h, lambda x: self.resnet(x, temb, f"down_blocks.{i}.resnets.{j}"))
                if kind.startswith("CrossAttn"):
                    h = each(h, lambda x: self.transformer(x, ctx, self.heads[i], f"down_blocks.{i}.attentions.{j}"))
                h = self.motion(h, f"down_blocks.{i}.motion_modules.{j}")
                skips.append(h)
            if i + 1 < len(ch):
                h = each(h, lambda x: self.conv(x, f"down_blocks.{i}.downsamplers.0.conv", stride=2))
                skips.append(h)
        h = each(h, lambda x: self.resnet(x, temb, "mid_block.resnets.0"))
        h = each(h, lambda x: self.transformer(x, ctx, self.heads[-1], "mid_block.attentions.0"))
        h = self.motion(h, "mid_block.motion_modules.0")
        h = each(h, lambda x: self.resnet(x, temb, "mid_block.resnets.1"))
        rheads = list(reversed(self.heads))
        for i, kind in enumerate(cfg["up_block_types"]):
            for j in range(L + 1):
                s = skips.pop()
                h = [self.resnet(np.concatenate([a, b], axis=0), temb, f"up_blocks.{i}.resnets.{j}") for a, b in zip(h, s)]
                if kind.startswith("CrossAttn"):
                    h = each(h, lambda x: self.transformer(x, ctx, rheads[i], f"up_blocks.{i}.attentions.{j}"))
                h = self.motion(h, f"up_blocks.{i}.motion_modules.{j}")
            if i + 1 < len(ch):
                h = each(h, lambda x: self.conv(x.repeat(2, axis=1).repeat(2, axis=2), f"up_blocks.{i}.upsamplers.0.conv"))
        return np.stack(each(h, lambda x: self.conv(silu(self.norm(x, "conv_norm_out")), "conv_out")))


def main():
    model_dir, motion, latents, context, t, F, H, W, out = sys.argv[1:10]
    F, H, W = int(F), int(H), int(W)
    xs = np.fromfile(latents, dtype=np.float32).reshape(F, 4, H, W)
    ctx = np.fromfile(context, dtype=np.float32).reshape(77, -1)
    MotionUNet(model_dir, motion).frames(list(xs), float(t), ctx).astype(np.float32).tofile(out)


if __name__ == "__main__":
    main()
