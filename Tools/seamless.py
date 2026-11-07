"""Makes textures tile seamlessly along U (horizontal), V (vertical) or both.

cut (default, for generated textures that do not tile at all): each tiled axis wraps by overlapping the image with
itself, the last `overlap` part of the axis laid over the first part, and the two are joined along the minimum-error
cut through the overlap (image quilting, Efros & Freeman 2001), softened by a few pixels of feathering. The cut
follows where both copies already look alike, so it keeps the detail sharp instead of crossfading it into ghosts. The
result is shorter by the overlap along each tiled axis; resize_periodic scales it back without breaking the wrap.

heal (for textures whose layout must stay): size and scale stay, a band of `overlap` width across the seam is
replaced by the stretch of the texture whose ends continue both sides of the band best, joined along minimum-error
cuts.

blend: a linear crossfade across the overlap, like most online seamless texture makers.

A periodic plus smooth decomposition (Moisan 2011) then removes what is left of colour steps at the borders, and an
optional flattening takes out large brightness gradients that would show as a grid once the texture repeats.

    uv run seamless.py in.png out.png --axes uv --overlap 0.2 --size 1024 [--flatten 1] [--preview tiled.png]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d


def load(path: str) -> np.ndarray:
    """An image file as float32 HxWxC in 0..1, C = 3 or 4 (alpha kept)."""
    im = Image.open(path)
    im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P", "PA") else "RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def save(path: str, img: np.ndarray, quality: int = 92) -> None:
    arr = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGBA" if arr.shape[2] == 4 else "RGB")
    if path.lower().endswith((".jpg", ".jpeg")):
        im.convert("RGB").save(path, quality=quality, subsampling=0, optimize=True)
    else:
        im.save(path, optimize=True)


def _cut_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel error between two overlapping strips: colour difference plus a gradient term, lightly blurred so the
    cut avoids noisy single-pixel matches."""
    rgb_a, rgb_b = a[..., :3], b[..., :3]
    diff = np.sum((rgb_a - rgb_b) ** 2, axis=2)
    gy_a, gx_a = np.gradient(rgb_a.mean(axis=2))
    gy_b, gx_b = np.gradient(rgb_b.mean(axis=2))
    diff += 0.5 * ((gx_a - gx_b) ** 2 + (gy_a - gy_b) ** 2)
    if a.shape[2] == 4:
        diff += (a[..., 3] - b[..., 3]) ** 2
    return gaussian_filter(diff, 1.0)


def _min_cut(cost: np.ndarray, margin: int, cyclic: bool) -> np.ndarray:
    """Minimum-cost top-to-bottom path through `cost` (rows x overlap columns), moving at most one column per row,
    as the column index per row. Cyclic paths end next to where they start (the rows wrap around)."""
    rows, cols = cost.shape
    lo, hi = margin, cols - margin
    if hi - lo < 3:
        lo, hi = 0, cols
    band = cost[:, lo:hi]
    width = band.shape[1]
    inf = np.float32(np.inf)

    def solve(starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # starts: K candidate first columns (or -1 for free start). Returns cost (K,) and paths (K, rows).
        k = len(starts)
        acc = np.full((k, width), inf, dtype=np.float32)
        for i, s in enumerate(starts):
            if s < 0:
                acc[i] = band[0]
            else:
                acc[i, s] = band[0, s]
        back = np.zeros((rows, k, width), dtype=np.int8)
        for y in range(1, rows):
            left = np.concatenate([np.full((k, 1), inf, np.float32), acc[:, :-1]], axis=1)
            right = np.concatenate([acc[:, 1:], np.full((k, 1), inf, np.float32)], axis=1)
            stack = np.stack([left, acc, right])
            choice = np.argmin(stack, axis=0)
            acc = np.take_along_axis(stack, choice[None], axis=0)[0] + band[y]
            back[y] = (choice - 1).astype(np.int8)
        if cyclic:
            # The last row must sit next to the first: only columns within one step of the start count.
            idx = np.arange(width)[None, :]
            ok = np.abs(idx - starts[:, None]) <= 1
            acc = np.where(ok, acc, inf)
        end = np.argmin(acc, axis=1)
        total = acc[np.arange(k), end]
        paths = np.zeros((k, rows), dtype=np.int64)
        paths[:, -1] = end
        for y in range(rows - 1, 0, -1):
            paths[:, y - 1] = paths[:, y] + back[y, np.arange(k), paths[:, y]]
        return total, paths

    if not cyclic:
        _, paths = solve(np.array([-1]))
        return paths[0] + lo
    # Cyclic: try evenly spaced starting columns, then refine around the best one.
    coarse = np.unique(np.linspace(0, width - 1, min(width, 24)).astype(np.int64))
    total, paths = solve(coarse)
    best = coarse[int(np.argmin(total))]
    step = max(1, width // 24)
    fine = np.arange(max(0, best - step), min(width, best + step + 1))
    total2, paths2 = solve(fine)
    if total2.min() <= total.min():
        return paths2[int(np.argmin(total2))] + lo
    return paths[int(np.argmin(total))] + lo


def wrap_axis(img: np.ndarray, axis: int, overlap: float, feather: float, method: str = "cut",
              cyclic: bool = False) -> np.ndarray:
    """Makes `img` wrap along `axis` (1 = U/horizontal, 0 = V/vertical). The axis gets shorter by the overlap.
    `cyclic`: the other axis already wraps, so the cut must end where it starts to keep that wrap intact."""
    if axis == 0:
        return wrap_axis(img.transpose(1, 0, 2), 1, overlap, feather, method, cyclic).transpose(1, 0, 2)
    h, w, _ = img.shape
    o = max(4, int(round(w * overlap)))
    a = img[:, w - o:]  # end of the axis, continues into the start of the tile
    b = img[:, :o]      # start of the axis, continues into the middle of the tile
    cols = np.arange(o, dtype=np.float32)[None, :]
    if method == "blend":
        # Linear crossfade across the whole overlap (what most online seamless makers do).
        mask = np.repeat(1.0 - (cols + 0.5) / o, h, axis=0)
    else:
        cut = _min_cut(_cut_cost(a, b), margin=max(2, int(feather * 2)), cyclic=cyclic)
        mask = _feathered((cols < cut[:, None]).astype(np.float32), feather, cyclic)
    mask = mask[..., None]
    seam = a * mask + b * (1.0 - mask)
    return np.concatenate([seam, img[:, o:w - o]], axis=1)


def _feathered(mask: np.ndarray, feather: float, cyclic: bool) -> np.ndarray:
    if feather <= 0:
        return mask
    mask = gaussian_filter1d(mask, feather, axis=1, mode="nearest")
    return gaussian_filter1d(mask, feather, axis=0, mode="wrap" if cyclic else "nearest")


def heal_axis(img: np.ndarray, axis: int, band: float = 0.1, feather: float | None = None,
              cyclic: bool = False) -> np.ndarray:
    """Hides the wrap seam along `axis` (1 = U, 0 = V) without changing size or scale: a band across the seam is
    replaced by the stretch of the texture whose ends best continue both sides of the band, joined along
    minimum-error cuts. `cyclic`: the other axis already wraps and keeps wrapping."""
    if axis == 0:
        return heal_axis(img.transpose(1, 0, 2), 1, band, feather, cyclic).transpose(1, 0, 2)
    h, w, _ = img.shape
    if feather is None:
        feather = max(1.0, min(h, w) / 256.0)
    half = w // 2
    rolled = np.roll(img, half, axis=1)  # the seam now lies between columns half - 1 and half
    b = max(2, int(round(w * band / 2)))  # half the replaced band
    m = max(3, int(round(w * band / 2)))  # overlap with the kept image on each side
    width = 2 * (b + m)
    if width >= w // 2:
        return img
    x0 = half - b - m
    left, right = rolled[:, x0:x0 + m], rolled[:, half + b:half + b + m]
    doubled = np.concatenate([rolled, rolled], axis=1)
    best, best_cost = None, np.inf
    for s in range(w):
        if 0 < (half - s) % w < width:  # the strip would contain the seam itself
            continue
        strip = doubled[:, s:s + width]
        cost = np.mean((strip[:, :m, :3] - left[..., :3]) ** 2) + np.mean((strip[:, -m:, :3] - right[..., :3]) ** 2)
        if cost < best_cost:
            best, best_cost = s, cost
    if best is None:
        return img
    strip = doubled[:, best:best + width]
    margin = max(1, int(feather))
    cut_l = _min_cut(_cut_cost(left, strip[:, :m]), margin=margin, cyclic=cyclic)
    cut_r = _min_cut(_cut_cost(strip[:, -m:], right), margin=margin, cyclic=cyclic)
    cols = np.arange(m)[None, :]
    mask = np.concatenate([
        (cols >= cut_l[:, None]).astype(np.float32),  # the kept image left of the cut, the strip right of it
        np.ones((h, 2 * b), np.float32),
        (cols < cut_r[:, None]).astype(np.float32),   # the strip left of the cut, the kept image right of it
    ], axis=1)
    mask = _feathered(mask, feather, cyclic)[..., None]
    out = rolled.copy()
    out[:, x0:x0 + width] = rolled[:, x0:x0 + width] * (1.0 - mask) + strip * mask
    return np.roll(out, -half, axis=1)


def periodic_smooth(img: np.ndarray, axes: str) -> np.ndarray:
    """Periodic component of Moisan's periodic plus smooth decomposition, restricted to the tiled axes: removes the
    smooth image whose Laplacian equals the border jumps, so opposite borders meet without a colour step."""
    h, w, c = img.shape
    out = img.copy()
    yy = np.cos(2.0 * np.pi * np.arange(h) / h)[:, None]
    xx = np.cos(2.0 * np.pi * np.arange(w) / w)[None, :]
    denom = 2.0 * xx + 2.0 * yy - 4.0
    denom[0, 0] = 1.0
    for ch in range(c):
        u = img[..., ch].astype(np.float64)
        v = np.zeros_like(u)
        if "u" in axes:
            jump = u[:, -1] - u[:, 0]
            v[:, 0] += jump
            v[:, -1] -= jump
        if "v" in axes:
            jump = u[-1, :] - u[0, :]
            v[0, :] += jump
            v[-1, :] -= jump
        s_hat = np.fft.fft2(v) / denom
        s_hat[0, 0] = 0.0
        out[..., ch] = (u - np.real(np.fft.ifft2(s_hat))).astype(np.float32)
    return out


def flatten(img: np.ndarray, axes: str, strength: float = 1.0, scale: float = 0.25) -> np.ndarray:
    """Takes out brightness gradients larger than `scale` of the texture (vignetting, uneven lighting) along the tiled
    axes, keeping the mean colour, so repeats do not form a visible grid."""
    h, w, _ = img.shape
    sigma = (h * scale if "v" in axes else 0.0, w * scale if "u" in axes else 0.0, 0.0)
    mode = ("wrap" if "v" in axes else "nearest", "wrap" if "u" in axes else "nearest", "nearest")
    low = gaussian_filter(img[..., :3], sigma, mode=mode)
    mean = img[..., :3].mean(axis=(0, 1), keepdims=True)
    out = img.copy()
    out[..., :3] = img[..., :3] - strength * (low - mean)
    return np.clip(out, 0.0, 1.0)


def resize_periodic(img: np.ndarray, width: int, height: int, axes: str) -> np.ndarray:
    """Resizes with Lanczos, wrapping around the tiled axes so the result still tiles."""
    ty, tx = (3 if "v" in axes else 1), (3 if "u" in axes else 1)
    big = np.tile(img, (ty, tx, 1))
    chans = []
    for ch in range(img.shape[2]):
        band = Image.fromarray(big[..., ch].astype(np.float32), "F")
        chans.append(np.asarray(band.resize((width * tx, height * ty), Image.LANCZOS), dtype=np.float32))
    out = np.stack(chans, axis=2)
    y0 = height if ty == 3 else 0
    x0 = width if tx == 3 else 0
    return np.clip(out[y0:y0 + height, x0:x0 + width], 0.0, 1.0)


def make_seamless(img: np.ndarray, axes: str = "uv", overlap: float = 0.2, feather: float | None = None,
                  method: str = "cut", flat: float = 0.0) -> np.ndarray:
    """The texture made to wrap along `axes` ("u", "v" or "uv"). "cut" and "blend" overlap the ends of each axis and
    make it shorter by `overlap`; "heal" keeps size and scale and replaces a band of `overlap` width across the seam."""
    axes = axes.lower()
    if feather is None:
        feather = max(1.0, min(img.shape[:2]) / 256.0)
    out = img
    for axis, name in ((1, "u"), (0, "v")):
        if name not in axes:
            continue
        cyclic = name == "v" and "u" in axes
        if method == "heal":
            out = heal_axis(out, axis, overlap, feather, cyclic)
        else:
            out = wrap_axis(out, axis, overlap, feather, method, cyclic)
    if axes:
        out = periodic_smooth(out, axes)
    if flat > 0 and axes:
        out = flatten(out, axes, flat)
    return np.clip(out, 0.0, 1.0)


def seam_score(img: np.ndarray, axes: str) -> float:
    """How much the wrap borders jump compared to ordinary neighbouring pixels (about 1 for a seamless texture)."""
    rgb = img[..., :3]
    scores = []
    if "u" in axes:
        inner = np.abs(np.diff(rgb, axis=1)).mean()
        scores.append(np.abs(rgb[:, 0] - rgb[:, -1]).mean() / max(inner, 1e-6))
    if "v" in axes:
        inner = np.abs(np.diff(rgb, axis=0)).mean()
        scores.append(np.abs(rgb[0] - rgb[-1]).mean() / max(inner, 1e-6))
    return float(max(scores)) if scores else 0.0


def seam_dominance(img: np.ndarray, axis: str) -> float:
    """The wrap jump along `axis` against the strongest jump between neighbouring rows (columns for "u") inside the
    texture. Above 1 the wrap is the hardest edge of the texture, a seam; a texture whose own content has edges as
    hard (windows, beams, frames) ends on a drawn edge instead."""
    rgb = img[..., :3]
    if axis == "u":
        rgb = rgb.transpose(1, 0, 2)
    inner = np.abs(np.diff(rgb, axis=0)).mean(axis=(1, 2))
    return float(np.abs(rgb[0] - rgb[-1]).mean() / max(float(inner.max()), 1e-6))


def tiled_preview(img: np.ndarray, axes: str, n: int = 3) -> np.ndarray:
    return np.tile(img, (n if "v" in axes else 1, n if "u" in axes else 1, 1))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--axes", default="uv", help="u, v or uv (default)")
    p.add_argument("--overlap", type=float, default=0.2,
                   help="share of each tiled axis that is overlapped (cut, blend) or replaced (heal), default 0.2")
    p.add_argument("--feather", type=float, default=None, help="cut softening in pixels (default size/256)")
    p.add_argument("--method", choices=["cut", "heal", "blend"], default="cut",
                   help="cut: overlap the ends along a minimum-error cut (default, for generated textures); heal: keep "
                        "size and scale, patch the seam with the best matching stretch; blend: linear crossfade")
    p.add_argument("--flatten", type=float, default=0.0, help="strength of large-gradient removal, 0..1")
    p.add_argument("--size", type=str, default=None, help="output size WxH or one number for both (default: input size)")
    p.add_argument("--preview", default=None, help="also write a 3x3 tiled preview here")
    p.add_argument("--quality", type=int, default=92, help="JPG quality")
    a = p.parse_args(argv)
    img = load(a.input)
    h, w, _ = img.shape
    out = make_seamless(img, a.axes, a.overlap, a.feather, a.method, a.flatten)
    if a.size:
        tw, th = (int(v) for v in (a.size.split("x") if "x" in a.size else (a.size, a.size)))
    else:
        tw, th = w, h
    out = resize_periodic(out, tw, th, a.axes)
    save(a.output, out, a.quality)
    print(f"{a.output}: {tw}x{th}, seam score {seam_score(out, a.axes):.2f} (input {seam_score(img, a.axes):.2f})")
    if a.preview:
        save(a.preview, tiled_preview(out, a.axes), a.quality)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
