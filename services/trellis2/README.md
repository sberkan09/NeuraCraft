# 3D service: TRELLIS.2

The engine's `image_to_3d` workflow step makes a whole textured 3D model from one picture by asking
this service (`Includes/ImageTo3D.h`). It runs [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
(Microsoft, MIT; 4B parameters) on an NVIDIA GPU with 24 GB or more, and answers with a glTF binary
that keeps TRELLIS.2's PBR materials (base colour, metallic / roughness). The engine saves that file
byte for byte and uses its base colour for previews and `render`.

```bash
services/trellis2/install.sh /opt/trellis2   # Linux, CUDA 12.4 toolkit; ~25 min (builds 5 CUDA extensions)
HF_TOKEN=... /opt/trellis2/run.sh            # listens on 127.0.0.1:8188, loads the model (~1 min)
NEURACRAFT_3D_URL=http://127.0.0.1:8188 bin/NeuraCraft workflow my3d.json
```

* **Models**: `microsoft/TRELLIS.2-4B` (MIT), `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated: the
  Hugging Face account behind `HF_TOKEN` must have been granted access; DINOv3 License) and
  `ZhengPeng7/BiRefNet` (MIT; background removal, only for pictures without transparency).
* **Quality**: `fast` = 512³ voxels, `standard` = 1024³ (cascade), `high` = 1536³ (cascade).
* **Protocol**: `GET /health`; `POST /v1/image-to-3d?quality=&seed=&texture=&faces=` with a PNG body →
  `model/gltf-binary`, or an error status with `{"error": "..."}`. Requests from other machines need
  the `TOKEN` in an `X-Token` header; the engine connects locally.
* `pod_check.sh` installs and checks everything on a rented GPU (see the comment at its top).
