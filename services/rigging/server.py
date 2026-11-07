#!/usr/bin/env python3
"""The rigging service: gives a 3D model a skeleton and skin weights with SkinTokens / TokenRig
(VAST-AI, MIT), for the engine's rig step (NEURACRAFT_RIG_URL).

  GET  /health                         -> 200 {"ok": true, ...} once the model is loaded (500 with the error if it failed)
  POST /v1/rig?seed=N&postprocess=0|1  body: a glTF binary (.glb)
                                       -> 200 model/gltf-binary (the model, its materials kept, with an armature and
                                          skin weights), or 4xx/5xx {"error": "..."}

One job at a time on the GPU. Requests from other machines need the TOKEN (X-Token header); the engine
connects locally. Environment: SKINTOKENS_DIR (the SkinTokens checkout, with its models downloaded),
HOST (127.0.0.1), PORT (8189), TOKEN, FILES_DIR (files served at /files/<name>, for checks).
"""

import hmac
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = os.path.abspath(os.environ.get("SKINTOKENS_DIR", os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)  # SkinTokens finds its checkpoints and configs relative to its folder
sys.path.insert(0, ROOT)

TOKEN = os.environ.get("TOKEN", "")
FILES_DIR = os.environ.get("FILES_DIR", "")
MAX_BYTES = 256 << 20

gpu = threading.Lock()
demo = None  # SkinTokens' demo module, once loaded
ckpt = None
loaded_at = None
load_error = None


def load():
    global demo, ckpt, loaded_at, load_error
    try:
        start = time.time()
        import demo as d  # SkinTokens' own entry point: its model loading and rigging

        d.start_bpy_server()  # Blender (bpy) reads and writes the models, in a helper process
        d.wait_for_bpy_server(timeout=120)
        ckpt = d.MODEL_CKPTS[0]
        d.load_model(ckpt, None)
        demo = d
        loaded_at = time.time()
        print(f"[rig] TokenRig loaded in {loaded_at - start:.1f} s", flush=True)
    except BaseException as e:
        traceback.print_exc()
        load_error = f"{type(e).__name__}: {e}"
        print(f"[rig] could not load TokenRig: {load_error}", flush=True)


ATTEMPTS = 3  # the rig is sampled; now and then a sample ends without skin weights, and the next seed works


def rig(glb: bytes, seed: int, postprocess: bool) -> bytes:
    import numpy as np
    import torch

    start = time.time()
    with gpu, tempfile.TemporaryDirectory(prefix="rig_") as tmp:
        src = Path(tmp) / "in.glb"
        src.write_bytes(glb)
        torch.cuda.reset_peak_memory_stats()
        for attempt in range(ATTEMPTS):
            s = (seed + attempt * 1_000_003) % 2**31
            dst = Path(tmp) / f"out{attempt}.glb"
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            try:
                demo.run_rig([src], top_k=5, top_p=0.95, temperature=1.0, repetition_penalty=2.0, num_beams=10,
                             use_skeleton=False, use_transfer=True, use_postprocess=postprocess, output_paths=[dst],
                             model_ckpt=ckpt, hf_path=None)
            except AssertionError:  # TokenRig's check that the sample has skin weights
                print(f"[rig] seed {s}: no complete rig, trying another seed", flush=True)
                continue
            if dst.is_file():
                break
        else:
            raise RuntimeError(f"no complete rig in {ATTEMPTS} samples (try another seed)")
        data = dst.read_bytes()
    print(f"[rig] seed {seed} (sample {attempt + 1}): {len(glb) / 1e6:.1f} MB -> {len(data) / 1e6:.1f} MB in "
          f"{time.time() - start:.1f} s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        print("[rig] " + (fmt % args), flush=True)

    def reply(self, status, body: bytes, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message):
        self.reply(status, json.dumps({"error": message}).encode())

    def allowed(self):
        if self.client_address[0] in ("127.0.0.1", "::1"):
            return True
        return bool(TOKEN) and hmac.compare_digest(self.headers.get("X-Token", ""), TOKEN)

    def do_GET(self):
        if not self.allowed():
            return self.error(403, "forbidden")
        url = urlparse(self.path)
        if url.path == "/health":
            state = {"ok": demo is not None, "model": "VAST-AI/SkinTokens", "busy": gpu.locked()}
            if load_error:
                state["error"] = load_error
            return self.reply(200 if load_error is None else 500, json.dumps(state).encode())
        m = re.fullmatch(r"/files/([A-Za-z0-9._-]{1,128})", url.path)
        if m and FILES_DIR:
            path = os.path.join(FILES_DIR, m.group(1))
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return self.reply(200, f.read(), "application/octet-stream")
        self.error(404, "not found")

    def do_POST(self):
        if not self.allowed():
            return self.error(403, "forbidden")
        url = urlparse(self.path)
        if url.path != "/v1/rig":
            return self.error(404, "not found")
        q = parse_qs(url.query)
        try:
            seed = int(q.get("seed", ["42"])[0])
            postprocess = q.get("postprocess", ["1"])[0] not in ("0", "false")
            if not 0 <= seed < 2**31:
                raise ValueError("seed is 0 .. 2^31-1")
            length = int(self.headers.get("Content-Length", "0"))
            if not 12 < length <= MAX_BYTES:
                raise ValueError("send a glTF binary of at most 256 MB")
            glb = self.rfile.read(length)
            if glb[:4] != b"glTF":
                raise ValueError("the body is not a glTF binary (.glb)")
        except ValueError as e:
            return self.error(400, str(e))
        if demo is None:
            return self.error(503, f"the rigging model could not be loaded ({load_error})" if load_error else "the model is still loading")
        try:
            data = rig(glb, seed, postprocess)
        except Exception as e:  # the engine shows the message to the user
            traceback.print_exc()
            try:
                import torch

                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    return self.error(507, "out of GPU memory (try a model with fewer triangles)")
            except ImportError:
                pass
            return self.error(500, f"{type(e).__name__}: {e}")
        self.reply(200, data, "model/gltf-binary")


def main():
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8189"))
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=load, daemon=True).start()  # /health answers (ok: false) while loading
    print(f"[rig] listening on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
