#!/usr/bin/env python3
"""Converts a PyTorch .pth / .pt checkpoint (zip format, tensors only) to
.safetensors without PyTorch, so NeuraCraft can load it.

    scripts/convert_pth.py model.pth model.safetensors [--key params_ema]

Only a whitelist of classes may be unpickled (OrderedDict and PyTorch's tensor
rebuild helpers), so a malicious file cannot run code. With --key, the state
dict is taken from that entry of the checkpoint dict (e.g. Real-ESRGAN's
"params_ema" / "params"); by default the first of those found, else the dict
itself.
"""
import collections
import json
import pickle
import struct
import sys
import zipfile

import numpy as np

DTYPES = {"FloatStorage": (np.float32, "F32"), "HalfStorage": (np.float16, "F16"), "BFloat16Storage": (None, "BF16")}


class StorageRef:
    def __init__(self, kind, key):
        self.kind, self.key = kind, key


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def rebuild_tensor_v2(storage, offset, size, stride, *_):
    return Tensor(storage, offset, size, stride)


class StorageType:
    def __init__(self, name):
        self.name = name


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return rebuild_tensor_v2
        if module == "torch" and name in DTYPES:
            return StorageType(name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, kind, key = pid[0], pid[1], pid[2]
        return StorageRef(kind.name if isinstance(kind, StorageType) else str(kind), key)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    key = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else None
    z = zipfile.ZipFile(src)
    root = [n for n in z.namelist() if n.endswith("data.pkl")][0][: -len("data.pkl")]
    obj = SafeUnpickler(z.open(root + "data.pkl")).load()
    if key:
        obj = obj[key]
    elif isinstance(obj, dict):
        for k in ("params_ema", "params", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    header, blobs, offset = {}, [], 0
    for name, t in obj.items():
        if not isinstance(t, Tensor):
            continue
        np_type, tag = DTYPES[t.storage.kind]
        raw = z.read(root + "data/" + t.storage.key)
        if np_type is None:  # bf16: keep the raw 16-bit values
            arr = np.frombuffer(raw, dtype=np.uint16)
        else:
            arr = np.frombuffer(raw, dtype=np_type)
        itemsize = arr.itemsize
        view = np.lib.stride_tricks.as_strided(arr[t.offset:], shape=t.size, strides=[s * itemsize for s in t.stride])
        data = np.ascontiguousarray(view).tobytes()
        header[name] = {"dtype": tag, "shape": list(t.size), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    print(f"{len(header)} tensors -> {dst}")


if __name__ == "__main__":
    main()
