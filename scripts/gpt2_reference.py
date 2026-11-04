#!/usr/bin/env python3
"""Independent NumPy GPT-2 forward pass, used to check NeuraCraft's GPT-2 output.

    pip install numpy safetensors tokenizers
    scripts/gpt2_reference.py data/gpt2 "The meaning of life is" 20

Prints the prompt's token ids, the top-5 next-token logits and a greedy
continuation, in the same format as `NeuraCraft gpt2 ... --check`.
"""
import json
import sys

import numpy as np
from safetensors.numpy import load_file
from tokenizers import ByteLevelBPETokenizer


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def forward(w, cfg, ids):
    n_head, eps = cfg["n_head"], cfg["layer_norm_epsilon"]
    T = len(ids)
    x = w["wte.weight"][ids] + w["wpe.weight"][:T]
    mask = np.triu(np.full((T, T), -1e10, dtype=np.float32), 1)
    for i in range(cfg["n_layer"]):
        p = f"h.{i}."
        h = layer_norm(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"], eps)
        qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        heads = []
        for hq, hk, hv in zip(np.split(q, n_head, -1), np.split(k, n_head, -1), np.split(v, n_head, -1)):
            att = softmax(hq @ hk.T / np.sqrt(hq.shape[-1]) + mask)
            heads.append(att @ hv)
        x = x + np.hstack(heads) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        h = layer_norm(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"], eps)
        h = gelu(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"])
        x = x + h @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
    x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"], eps)
    return x @ w["wte.weight"].T  # tied output


def main():
    model_dir = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The meaning of life is"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    cfg = json.load(open(f"{model_dir}/config.json"))
    w = load_file(f"{model_dir}/model.safetensors")
    tok = ByteLevelBPETokenizer(f"{model_dir}/vocab.json", f"{model_dir}/merges.txt")

    ids = tok.encode(prompt).ids
    print("prompt ids:", " ".join(map(str, ids)))
    logits = forward(w, cfg, ids)[-1]
    top = np.argsort(-logits)[:5]
    print("top-5 next:", " ".join(f"{t}:{logits[t]:.4f}" for t in top))
    for _ in range(steps):
        ids.append(int(np.argmax(forward(w, cfg, ids)[-1])))
    print("greedy:", repr(tok.decode(ids)))


if __name__ == "__main__":
    main()
