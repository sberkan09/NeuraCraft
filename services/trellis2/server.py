"""NeuraCraft's 3D service: one picture -> a textured glTF model, with TRELLIS.2 (Microsoft, MIT).

The engine's image_to_3d step (Includes/ImageTo3D.h) talks to it over plain HTTP:
  GET  /health                      -> 200 {"ok": true, ...}
  POST /v1/image-to-3d?quality=fast|standard|high&seed=N&texture=N&faces=N   (body: PNG)
                                    -> 200 model/gltf-binary, or 4xx/5xx {"error": "..."}
A PNG with transparency is used as it is (its alpha is the object); an opaque one goes through the
pipeline's background removal (BiRefNet, MIT).

Environment: TRELLIS_DIR (the TRELLIS.2 checkout), MODEL (default microsoft/TRELLIS.2-4B; needs
access to the gated facebook/dinov3-vitl16-pretrain-lvd1689m), HOST (127.0.0.1), PORT (8188).
Requests from other machines need TOKEN in an X-Token header; FILES_DIR serves its files at
/files/<name> to token holders (for checks on a rented GPU). One request runs at a time.
"""

import hmac
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.environ.get("TRELLIS_DIR", os.path.dirname(os.path.abspath(__file__))))

try:  # TRELLIS.2's sparse attention needs FlashAttention or xFormers (install.sh installs one)
    import flash_attn  # noqa: F401
except ImportError:
    os.environ.setdefault("ATTN_BACKEND", "xformers")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

PIPELINE_TYPES = {"fast": "512", "standard": "1024_cascade", "high": "1536_cascade"}
MAX_BODY = 64 << 20
TOKEN = os.environ.get("TOKEN", "")
FILES_DIR = os.environ.get("FILES_DIR", "")
MODEL = os.environ.get("MODEL", "microsoft/TRELLIS.2-4B")

gpu = threading.Lock()
pipeline = None
loaded_at = None


def load():
    global pipeline, loaded_at
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    start = time.time()
    p = Trellis2ImageTo3DPipeline.from_pretrained(MODEL)
    p.cuda()
    pipeline = p
    loaded_at = time.time()
    print(f"[3d] {MODEL} loaded in {loaded_at - start:.1f} s", flush=True)


def generate(png: bytes, quality: str, seed: int, texture: int, faces: int) -> bytes:
    import o_voxel

    image = Image.open(io.BytesIO(png))
    image.load()
    image = image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    with gpu:
        mesh = pipeline.run(image, seed=seed, pipeline_type=PIPELINE_TYPES[quality])[0]
        mesh.simplify(16777216)  # nvdiffrast's limit
        model = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords, attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=faces,
            texture_size=texture, remesh=True, remesh_band=1, remesh_project=0, verbose=False,
        )
        del mesh
        torch.cuda.empty_cache()
    data = model.export(file_type="glb")  # PNG textures (the engine reads PNG, not WebP)
    print(f"[3d] {quality} seed {seed}: {len(model.faces)} faces, {len(data) / 1e6:.1f} MB in {time.time() - start:.1f} s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # one line per request on stdout
        print("[3d] " + (fmt % args), flush=True)

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
            return self.reply(200, json.dumps({"ok": pipeline is not None, "model": MODEL, "busy": gpu.locked()}).encode())
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
        if url.path != "/v1/image-to-3d":
            return self.error(404, "not found")
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            quality = q.get("quality", "standard")
            if quality not in PIPELINE_TYPES:
                raise ValueError("quality is fast, standard or high")
            seed = int(q.get("seed", 42))
            texture = int(q.get("texture", 2048))
            faces = int(q.get("faces", 500000))
            if not (0 <= seed < 2**31 and 512 <= texture <= 4096 and 10000 <= faces <= 2000000):
                raise ValueError("seed, texture (512..4096) or faces (10000..2000000) out of range")
            length = int(self.headers.get("Content-Length", "0"))
            if not (0 < length <= MAX_BODY):
                raise ValueError("send a PNG of at most 64 MB")
            png = self.rfile.read(length)
        except ValueError as e:
            return self.error(400, str(e))
        if pipeline is None:
            return self.error(503, "the model is still loading")
        try:
            data = generate(png, quality, seed, texture, faces)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self.error(507, "out of GPU memory (try quality fast or fewer faces)")
        except Exception as e:  # the engine shows the message to the user
            traceback.print_exc()
            return self.error(500, f"{type(e).__name__}: {e}")
        self.reply(200, data, "model/gltf-binary")


def main():
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8188"))
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=load, daemon=True).start()  # /health answers (ok: false) while loading
    print(f"[3d] listening on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
