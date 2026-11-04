#!/usr/bin/env python3
"""Independent NumPy CLIP text encoder (Hugging Face CLIPTextModel), used to
check NeuraCraft's.

    pip install numpy transformers
    scripts/clip_reference.py data/sd-turbo "a photo of a cat" out.f32

Tokenizes with transformers' CLIPTokenizer (padding to 77) and writes the
final hidden states (77 x hidden, float32, row-major) to out.f32.
"""
import json
import math
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
        kind = {"F16": np.float16, "F32": np.float32}[t["dtype"]]
        out[name] = data[a:b].view(kind).astype(np.float32).reshape(t["shape"])
    return out


def layer_norm(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


erf = np.vectorize(math.erf, otypes=[np.float64])


def act(x, kind):
    if kind == "gelu":
        return (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(np.float32)
    return x / (1.0 + np.exp(-1.702 * x))  # quick_gelu


def encode(w, cfg, ids):
    D, H = cfg["hidden_size"], cfg["num_attention_heads"]
    hd, eps, T = D // H, cfg.get("layer_norm_eps", 1e-5), len(ids)
    p = "text_model."
    x = w[p + "embeddings.token_embedding.weight"][ids] + w[p + "embeddings.position_embedding.weight"][:T]
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    lin = lambda v, n: v @ w[n + ".weight"].T + w[n + ".bias"]
    for i in range(cfg["num_hidden_layers"]):
        l = f"{p}encoder.layers.{i}."
        h = layer_norm(x, w[l + "layer_norm1.weight"], w[l + "layer_norm1.bias"], eps)
        q = lin(h, l + "self_attn.q_proj").reshape(T, H, hd)
        k = lin(h, l + "self_attn.k_proj").reshape(T, H, hd)
        v = lin(h, l + "self_attn.v_proj").reshape(T, H, hd)
        heads = []
        for j in range(H):
            s = q[:, j] @ k[:, j].T / math.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            heads.append((s / s.sum(-1, keepdims=True)) @ v[:, j])
        x = x + lin(np.concatenate(heads, -1), l + "self_attn.out_proj")
        h = layer_norm(x, w[l + "layer_norm2.weight"], w[l + "layer_norm2.bias"], eps)
        x = x + lin(act(lin(h, l + "mlp.fc1"), cfg.get("hidden_act", "quick_gelu")), l + "mlp.fc2")
    return layer_norm(x, w[p + "final_layer_norm.weight"], w[p + "final_layer_norm.bias"], eps)


def main():
    from transformers import CLIPTokenizer

    model_dir, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    d = os.path.join(model_dir, "text_encoder")
    cfg = json.load(open(os.path.join(d, "config.json")))
    f = os.path.join(d, "model.safetensors")
    w = load_safetensors(f if os.path.exists(f) else os.path.join(d, "model.fp16.safetensors"))
    tok = CLIPTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    ids = tok(prompt, padding="max_length", max_length=cfg.get("max_position_embeddings", 77), truncation=True).input_ids
    print("ids:", " ".join(map(str, ids[: ids.index(tok.eos_token_id) + 1])))
    encode(w, cfg, ids).astype(np.float32).tofile(out)


if __name__ == "__main__":
    main()
