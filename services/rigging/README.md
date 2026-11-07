# Rigging service (SkinTokens / TokenRig)

The engine's `rig` step gives a 3D model a skeleton and skin weights, ready to pose and animate. The
rigging itself runs here, in a separate Python process on an NVIDIA GPU: **TokenRig** with
**SkinTokens** (VAST-AI, MIT; UniRig's successor). It generates the whole rig (bone hierarchy and the
weights of each vertex) as one token sequence. The engine reaches the service only at
`NEURACRAFT_RIG_URL`; workflows cannot point it elsewhere.

```bash
services/rigging/install.sh /opt/skintokens   # Linux, Python 3.11, NVIDIA GPU with 14 GB or more
/opt/skintokens/run.sh                        # listens on 127.0.0.1:8189, loads the model
NEURACRAFT_RIG_URL=http://127.0.0.1:8189 bin/NeuraCraft workflow rig.json
```

```json
{"steps": [
  {"op": "load", "path": "witch.glb", "out": "model"},
  {"op": "rig", "in": "model", "seed": 42, "out": "rigged"},
  {"op": "save", "in": "rigged", "path": "witch_rigged.glb"}]}
```

* **Input**: any mesh the engine holds. A model read from a glTF (or made by `image_to_3d`) is sent as
  it is, so its textures and materials come back on the rigged model.
* **Output**: a glTF binary with an armature and four bone weights per vertex. It opens in Blender,
  three.js, Unity, Unreal and Godot. The engine keeps it byte for byte (`.glb`); its own renderer draws
  the rest pose.
* **Seeds**: the skeleton is sampled, so another `seed` gives another skeleton. `postprocess` (on by
  default) smooths the weights over the model's volume.
* **Protocol**: `GET /health`; `POST /v1/rig?seed=&postprocess=0|1` with a `.glb` body →
  `model/gltf-binary`, or an error status with `{"error": "..."}`. Requests from other machines need
  `TOKEN` in an `X-Token` header.
* `pod_check.sh` installs and checks everything on a rented GPU (see the comment at its top).
