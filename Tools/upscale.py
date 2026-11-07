"""4x super-resolution of textures with 4xNomos8kSC (ESRGAN, Philip Hofmann, CC-BY-4.0), loaded with spandrel and run
on Apple MPS, CUDA or the CPU. The weights download on first use into ~/.cache/hd-textures/models (HD_TEXTURES_MODELS).

Tiled axes are padded by wrapping before the model sees them and cropped after, so a texture that tiles keeps tiling;
other borders are padded by reflection. Large inputs run in overlapping tiles. Alpha is upscaled by the same model as
a grey image, which keeps cut-out edges sharp.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from functools import lru_cache

import numpy as np
import torch
from spandrel import ImageModelDescriptor, ModelLoader

MODEL_DIR = os.environ.get("HD_TEXTURES_MODELS", os.path.expanduser("~/.cache/hd-textures/models"))
MODEL_FILE = "4xNomos8kSC.pth"
MODEL_URL = "https://github.com/Phhofm/models/releases/download/4xNomos8kSC/4xNomos8kSC.pth"
MODEL_SHA256 = "d73b89267320411c7eb0ec7ab56ab17f1044f2408fed211588cee845bee48407"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _weights() -> str:
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.isfile(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Downloading {MODEL_URL}", flush=True)
        urllib.request.urlretrieve(MODEL_URL, path + ".part")
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"{path} has SHA-256 {digest}, expected {MODEL_SHA256}: delete it to download it again")
    return path


@lru_cache(maxsize=1)
def model() -> ImageModelDescriptor:
    m = ModelLoader().load_from_file(_weights())
    assert isinstance(m, ImageModelDescriptor) and m.scale == 4, MODEL_FILE
    m.to(device()).eval()
    if device().type != "cpu":
        m.model.half()  # half precision on the GPU: about a quarter faster, at most 2/255 off
    return m


def _pad(img: np.ndarray, pad: int, axes: str) -> np.ndarray:
    """Pads H and W by `pad`, wrapping along tiled axes and reflecting along the others."""
    h, w = img.shape[:2]
    py, px = min(pad, h - 1), min(pad, w - 1)
    out = np.pad(img, ((py, py), (0, 0), (0, 0)), mode="wrap" if "v" in axes else "reflect")
    return np.pad(out, ((0, 0), (px, px), (0, 0)), mode="wrap" if "u" in axes else "reflect")


def _ramp(n: int, overlap: int, start: bool, end: bool) -> np.ndarray:
    r = np.ones(n, np.float32)
    if start:
        r[:overlap] = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
    if end:
        r[n - overlap:] = np.linspace(1.0, 0.0, overlap + 2, dtype=np.float32)[1:-1]
    return r


@torch.no_grad()
def _run(rgb: np.ndarray, tile: int = 640, overlap: int = 32) -> np.ndarray:
    """Model output for an HxWx3 float image, in overlapping tiles blended with linear ramps (512 px sources with
    their padding fit one tile)."""
    desc = model()
    s = desc.scale
    h, w, _ = rgb.shape
    dtype = next(desc.model.parameters()).dtype
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(device(), dtype)
    if h <= tile and w <= tile:
        return desc.model(x).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    out = np.zeros((h * s, w * s, 3), np.float32)
    weight = np.zeros((h * s, w * s, 1), np.float32)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for x0 in xs:
            th, tw = min(tile, h - y), min(tile, w - x0)
            part = desc.model(x[:, :, y:y + th, x0:x0 + tw]).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            wgt = (_ramp(th * s, overlap * s, y > 0, y + th < h)[:, None] *
                   _ramp(tw * s, overlap * s, x0 > 0, x0 + tw < w)[None, :])[..., None]
            out[y * s:(y + th) * s, x0 * s:(x0 + tw) * s] += part * wgt
            weight[y * s:(y + th) * s, x0 * s:(x0 + tw) * s] += wgt
    return out / np.maximum(weight, 1e-6)


def upscale(img: np.ndarray, axes: str = "", pad: int = 16) -> np.ndarray:
    """`img` (HxWxC float 0..1, C = 3 or 4) upscaled 4x; `axes` ("u", "v", "uv" or "") are the tiled axes."""
    h, w, c = img.shape
    padded = _pad(img, pad, axes)
    py, px = (padded.shape[0] - h) // 2, (padded.shape[1] - w) // 2
    channels = [_run(padded[..., :3])]
    if c == 4:
        channels.append(_run(np.repeat(padded[..., 3:4], 3, axis=2)).mean(axis=2, keepdims=True))
    out = np.concatenate(channels, axis=2)
    s = out.shape[0] // padded.shape[0]
    return out[py * s:(py + h) * s, px * s:(px + w) * s]
