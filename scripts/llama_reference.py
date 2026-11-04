#!/usr/bin/env python3
"""Independent NumPy Llama forward pass (float32), used to check NeuraCraft.

    pip install numpy tokenizers
    scripts/llama_reference.py data/tinyllama "What is the capital of Turkey?" 20 --chat

Prints the prompt's token ids, the top-5 next-token logits and a greedy
continuation, in the same format as `NeuraCraft llama ... --check --fp32`.
Handles TinyLlama / Llama 2 and Llama 3 (llama3 RoPE scaling, tied
embeddings, the Llama 3 chat template). Weights stay memory-mapped and are
converted to float32 when used, so memory stays near one layer's worth plus
the embedding table.
"""
import json
import math
import struct
import sys
import time

import numpy as np
from tokenizers import Tokenizer


class LazyTensors:
    """F32 / F16 / BF16 safetensors, converted to float32 on each access."""

    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)

    def __contains__(self, name):
        return name in self.header

    def __getitem__(self, name):
        t = self.header[name]
        a, b = t["data_offsets"]
        raw = self.data[a:b]
        if t["dtype"] == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif t["dtype"] == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        else:
            arr = np.array(raw.view(np.float32))
        return arr.reshape(t["shape"])


def inv_frequencies(cfg, hd):
    """RoPE frequencies, with Hugging Face's "llama3" rope_scaling if configured."""
    theta = cfg.get("rope_theta", 10000.0)
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    rs = cfg.get("rope_scaling")
    if not rs:
        return inv
    if rs.get("rope_type", rs.get("type")) != "llama3":
        raise SystemExit("unsupported rope_scaling")
    factor, low, high = rs["factor"], rs.get("low_freq_factor", 1.0), rs.get("high_freq_factor", 4.0)
    old = rs.get("original_max_position_embeddings", 8192)
    wavelen = 2 * math.pi / inv
    scaled = np.where(wavelen > old / low, inv / factor, inv)
    smooth = (old / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < old / high) & ~(wavelen > old / low)
    return np.where(medium, smoothed, scaled)


def rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, positions, inv):
    # x: (T, heads, hd); "rotate half" layout as in Hugging Face Llama.
    hd = x.shape[-1]
    ang = positions[:, None].astype(np.float64) * inv[None, :]
    cos = np.cos(ang).astype(np.float32)[:, None, :]
    sin = np.sin(ang).astype(np.float32)[:, None, :]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def forward(w, cfg, ids):
    D, H = cfg["hidden_size"], cfg["num_attention_heads"]
    KV = cfg.get("num_key_value_heads", H)
    hd, eps = cfg.get("head_dim", D // H), cfg["rms_norm_eps"]
    inv = inv_frequencies(cfg, hd)
    T = len(ids)
    pos = np.arange(T)
    x = w["model.embed_tokens.weight"][ids].astype(np.float32)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms_norm(x, w[p + "input_layernorm.weight"], eps)
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, H, hd)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, KV, hd)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, KV, hd)
        q, k = rope(q, pos, inv), rope(k, pos, inv)
        heads = []
        for hh in range(H):
            g = hh // (H // KV)
            s = q[:, hh] @ k[:, g].T / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            heads.append((s / s.sum(-1, keepdims=True)) @ v[:, g])
        x = x + np.concatenate(heads, axis=-1) @ w[p + "self_attn.o_proj.weight"].T
        h = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
        gate = h @ w[p + "mlp.gate_proj.weight"].T
        up = h @ w[p + "mlp.up_proj.weight"].T
        x = x + (gate / (1.0 + np.exp(-gate)) * up) @ w[p + "mlp.down_proj.weight"].T
    x = rms_norm(x, w["model.norm.weight"], eps)
    head = w["model.embed_tokens.weight"] if cfg.get("tie_word_embeddings") else w["lm_head.weight"]
    return x @ head.T


def main():
    model_dir = sys.argv[1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    chat = "--chat" in sys.argv
    prompt = args[0] if args else "What is the capital of Turkey?"
    steps = int(args[1]) if len(args) > 1 else 20
    cfg = json.load(open(f"{model_dir}/config.json"))
    w = LazyTensors(f"{model_dir}/model.safetensors")
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    text = prompt
    if chat and tok.token_to_id("<|start_header_id|>") is not None:  # Llama 3
        today = time.strftime("%d %b %Y")
        text = ("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
                f"Today Date: {today}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    elif chat:
        text = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    ids = tok.encode(text).ids  # adds <s> / <|begin_of_text|>
    stops = cfg.get("eos_token_id", 2)
    try:
        stops = json.load(open(f"{model_dir}/generation_config.json")).get("eos_token_id", stops)
    except OSError:
        pass
    stops = set(stops if isinstance(stops, list) else [stops])
    print("prompt ids:", " ".join(map(str, ids)))
    logits = forward(w, cfg, ids)[-1]
    top = np.argsort(-logits)[:5]
    print("top-5 next:", " ".join(f"{t}:{logits[t]:.4f}" for t in top))
    out = []
    for _ in range(steps):
        nxt = int(np.argmax(forward(w, cfg, ids + out)[-1]))
        out.append(nxt)
        if nxt in stops:
            break
    print("greedy:", repr(tok.decode(out, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
