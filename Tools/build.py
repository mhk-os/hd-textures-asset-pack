"""Builds the pack's Assets/ folder: HD versions of the track textures of the core mods (MHK 2, MHK 3, MHK Extra,
MHK Thunder, TINcan).

Sources are mod folders as a mod build, an installed mod or a mod release zip holds them: the bundles
(<mod>.mhkmod, <mod>.scenes.mhkmod) and the loose replaceable files in Assets/, so every output lands on the virtual
path the game replaces. Per texture:

1. Tiling: the axes along which the game repeats the texture, from the meshes and materials in the bundles (uvs.py);
   for a texture no mesh shows, the axes whose opposite borders already meet. Tiled axes wrap through every later
   step, so the HD texture repeats like the original.
2. Seams: where the tracks often show a tiled border and the original jumps there harder than at any edge inside it,
   the seam is healed (seamless.py, size and scale kept). Drawn edges (outlines, window frames, beams) and the
   textures in keep-seams.txt stay.
3. Remaster: a generated remaster in Tools/generated/<virtual path>.png (from an image model, any size, opaque
   textures) is colour matched to the original and made seamless along the tiled axes. Without one the texture is
   upscaled 4x by an ESRGAN model (upscale.py) and keeps the broad colours of the original, the model adds detail.
4. Size: 256 px and smaller sources become 4x, 512 px sources 2x; 1024 px and larger ones are already HD and stay out,
   as do near-uniform ones that would only grow.
5. Encoding: JPG for opaque textures, PNG where alpha is used.

    uv run build.py                      # every core mod, from the local builds in the sibling mod repositories
    uv run build.py --mod mhk2=/path/to/Mods/mhk2 --mods mhk2 --only 'mhk2/Tracks/mhk2_egypt/*'
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

import seamless

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
ASSETS = os.path.join(ROOT, "Assets")
GENERATED = os.path.join(TOOLS, "generated")
MANIFEST = os.path.join(TOOLS, "manifest.json")
KEEP_SEAMS = os.path.join(TOOLS, "keep-seams.txt")

MODS = ("mhk2", "mhk3", "mhke", "mhkt", "tincan")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# Bump when a change to the pipeline should rebuild every texture.
PIPELINE_VERSION = 1
JPG_QUALITY = 90
SEAM_SCORE = 2.0      # a border jumping more than this times the neighbouring pixels is a seam
SEAM_SCORE_MAX = 15.0 # ... up to this much: a harder jump is a drawn edge (an outline, a panel border)
SEAM_DOMINANCE = 1.2  # ... and is healed when it also jumps this much more than any edge inside the texture
HEAL_BAND = 0.12      # share of the axis a healed seam replaces
HEAL_SHARE = 0.1      # seams are healed where at least this share of a texture's triangles shows them
GENERATED_OVERLAP = 0.2
MIN_SIDE, MAX_SOURCE_SIDE = 96, 512
MIN_DETAIL = 1.5      # luminance standard deviation (0..255) below which a texture is near uniform


def default_mod_folder(mod: str) -> str:
    """The local build of the mod in the sibling repository mhk-os-<mod>-mod."""
    return os.path.join(os.path.dirname(ROOT), f"mhk-os-{mod}-mod", "Builds", "Mods", "StandaloneOSX", mod)


def is_track_texture(path: str) -> bool:
    """Textures of the tracks: everything under <mod>/Tracks/ except the menu previews, and the shared track props
    and materials in <mod>/Shared/Textures/."""
    parts = path.split("/")
    if not path.lower().endswith(IMAGE_EXTENSIONS) or len(parts) < 3:
        return False
    if parts[1] == "Tracks":
        return not os.path.splitext(parts[-1])[0].endswith("_preview")
    return parts[1] == "Shared" and parts[2] == "Textures"


def sources(mod: str, folder: str) -> list[tuple[str, str]]:
    """(virtual path, file) of every track texture in a mod folder, sorted by virtual path."""
    assets = os.path.join(folder, "Assets")
    found = []
    for dirpath, _, files in os.walk(assets):
        for name in files:
            file = os.path.join(dirpath, name)
            path = os.path.relpath(file, assets).replace(os.sep, "/")
            if path.startswith(mod + "/") and is_track_texture(path):
                found.append((path, file))
    return sorted(found)


def target_size(w: int, h: int) -> tuple[int, int] | None:
    side = max(w, h)
    if side < MIN_SIDE or side > MAX_SOURCE_SIDE:
        return None
    scale = 4 if side <= 256 else 2
    return w * scale, h * scale


def blur(img: np.ndarray, sigma: float, axes: str) -> np.ndarray:
    modes = ("wrap" if "v" in axes else "nearest", "wrap" if "u" in axes else "nearest", "nearest")
    return gaussian_filter(img, (sigma, sigma, 0), mode=modes)


def lock_colours(out: np.ndarray, original: np.ndarray, axes: str, scale: float) -> np.ndarray:
    """`out` with the broad colours (features of about 4 source pixels and larger) of `original`."""
    h, w = out.shape[:2]
    ref = seamless.resize_periodic(original[..., :3], w, h, axes)
    locked = out.copy()
    locked[..., :3] = out[..., :3] + blur(ref - out[..., :3], 4.0 * scale, axes)
    return np.clip(locked, 0.0, 1.0)


def match_stats(img: np.ndarray, original: np.ndarray) -> np.ndarray:
    """`img` with the per-channel mean and spread of `original` (a generated remaster takes the original's colours)."""
    out = img.copy()
    for c in range(3):
        src, ref = img[..., c], original[..., c]
        out[..., c] = (src - src.mean()) / max(float(src.std()), 1e-4) * ref.std() + ref.mean()
    return np.clip(out, 0.0, 1.0)


def generated_for(path: str) -> str | None:
    base = os.path.join(GENERATED, os.path.splitext(path)[0])
    for ext in IMAGE_EXTENSIONS:
        if os.path.isfile(base + ext):
            return base + ext
    return None


def remastered(original: np.ndarray, generated: str, axes: str, size: tuple[int, int]) -> np.ndarray:
    img = match_stats(seamless.load(generated)[..., :3], original)
    if axes:
        img = seamless.make_seamless(img, axes, GENERATED_OVERLAP)
    return seamless.resize_periodic(img, size[0], size[1], axes)


def upscaled(original: np.ndarray, axes: str, size: tuple[int, int]) -> np.ndarray:
    import upscale  # torch loads only when something is upscaled

    big = upscale.upscale(original, axes)
    if big.shape[1] != size[0] or big.shape[0] != size[1]:
        big = seamless.resize_periodic(big, size[0], size[1], axes)
    return lock_colours(big, original, axes, size[0] / original.shape[1])


def sha256(file: str) -> str:
    with open(file, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def output_file(path: str, alpha: bool) -> str:
    return os.path.join(ASSETS, os.path.splitext(path)[0] + (".png" if alpha else ".jpg"))


def keep_seams() -> set[str]:
    """Virtual paths whose seams stay (keep-seams.txt)."""
    if not os.path.isfile(KEEP_SEAMS):
        return set()
    with open(KEEP_SEAMS, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def heals(img: np.ndarray, axis: str, share: float) -> bool:
    """Whether the seam along `axis` is healed: the tracks show it often, it is the hardest edge of the texture and
    not so hard that it is drawn."""
    score = seamless.seam_score(img, axis)
    return share >= HEAL_SHARE and SEAM_SCORE < score <= SEAM_SCORE_MAX and \
        seamless.seam_dominance(img, axis) > SEAM_DOMINANCE


def build(path: str, file: str, tiling, keep_seam: bool, entry: dict | None, force: bool) -> dict | None:
    """Builds one texture (`tiling`: how the meshes show it, None when none does; `keep_seam`: it is never healed),
    returns its manifest entry (None: the texture stays out of the pack)."""
    img = seamless.load(file)
    h, w = img.shape[:2]
    size = target_size(w, h)
    if size is None or (img[..., :3].mean(axis=2) * 255.0).std() < MIN_DETAIL:
        return None
    alpha = img.shape[2] == 4 and float(img[..., 3].min()) < 0.99
    if not alpha:
        img = img[..., :3]
    if tiling is None:
        axes = "".join(a for a in "uv" if seamless.seam_score(img, a) <= SEAM_SCORE)
        healed = ""
    else:
        axes = tiling.axes
        healed = "" if keep_seam else "".join(a for a, share in zip("uv", tiling) if heals(img, a, share))
    generated = generated_for(path) if not alpha else None
    key = f"v{PIPELINE_VERSION}:q{JPG_QUALITY}:axes={axes}:healed={healed}:{sha256(file)}" + \
          (f":generated={sha256(generated)}" if generated else "")
    if not force and entry and entry.get("key") == key and os.path.isfile(os.path.join(ROOT, entry["file"])):
        return entry
    if healed:
        img = seamless.make_seamless(img, healed, HEAL_BAND, method="heal")
    if generated:
        out, method = remastered(img, generated, axes, size), "generated"
    else:
        out, method = upscaled(img, axes, size), "upscaled"
    target = output_file(path, alpha)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    stale = output_file(path, not alpha)
    if os.path.exists(stale):
        os.remove(stale)
    seamless.save(target, out, JPG_QUALITY)
    return {
        "file": os.path.relpath(target, ROOT).replace(os.sep, "/"),
        "key": key,
        "from": [w, h],
        "size": [size[0], size[1]],
        "axes": axes,
        "axesFrom": "border" if tiling is None else "mesh",
        "healed": healed,
        "method": method,
    }


def remove(entry: dict) -> None:
    file = os.path.join(ROOT, entry["file"])
    if os.path.exists(file):
        os.remove(file)


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST + ".part", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=1)
        f.write("\n")
    os.replace(MANIFEST + ".part", MANIFEST)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mod", action="append", default=[], metavar="ID=DIR",
                   help="a mod folder with the bundles and Assets/ (default: the local build in ../../mhk-os-<id>-mod)")
    p.add_argument("--mods", default=",".join(MODS), help="comma separated pack ids (default: every core mod)")
    p.add_argument("--only", action="append", default=[], metavar="GLOB", help="only virtual paths matching a glob")
    p.add_argument("--force", action="store_true", help="rebuild textures that are up to date")
    a = p.parse_args(argv)

    import uvs

    folders = {m: default_mod_folder(m) for m in a.mods.split(",") if m}
    for item in a.mod:
        mod, _, folder = item.partition("=")
        if mod in folders:
            folders[mod] = folder
    for mod, folder in folders.items():
        if not os.path.isdir(os.path.join(folder, "Assets")):
            print(f"{mod}: no Assets folder in {folder}", file=sys.stderr)
            return 1

    manifest = {}
    if os.path.isfile(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)
    start, built, kept = time.time(), 0, keep_seams()
    for mod, folder in folders.items():
        found = sources(mod, folder)
        tiling = uvs.tilings(folder, mod)
        print(f"{mod}: {len(found)} track textures, {len(tiling)} shown by meshes", flush=True)
        for i, (path, file) in enumerate(found):
            if a.only and not any(fnmatch.fnmatch(path, g) for g in a.only):
                continue
            old = manifest.get(path)
            entry = build(path, file, tiling.get(path.lower()), path in kept, old, a.force)
            if entry is None:
                if old:
                    remove(manifest.pop(path))
                continue
            if entry is not old:
                built += 1
                manifest[path] = entry
                save_manifest(manifest)  # an interrupted build resumes where it stopped
                print(f"[{i + 1}/{len(found)}] {path} {entry['from'][0]}x{entry['from'][1]} -> "
                      f"{entry['size'][0]}x{entry['size'][1]} {entry['method']} axes={entry['axes'] or '-'}"
                      f"{' healed=' + entry['healed'] if entry['healed'] else ''}", flush=True)
        # Textures gone from the mod leave the pack.
        present = {path for path, _ in found}
        for path in [p for p in manifest if p.split("/")[0] == mod and p not in present]:
            remove(manifest.pop(path))
    save_manifest(manifest)
    print(f"{built} built, {len(manifest)} in the pack, {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
