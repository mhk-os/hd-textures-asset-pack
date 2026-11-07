"""Which axes each texture of a mod repeats along in the game, read from the mod's bundles (<mod>.mhkmod and
<mod>.scenes.mhkmod, as in a mod build, an installed mod or a mod release zip).

Every mesh renderer pairs submeshes of its mesh (its range of the scene's combined mesh when statically batched) with
its materials; a texture slot of a material maps the submesh's UV0 through the slot's tiling and offset. A texture
repeats along U (V) where it shows its opposite borders side by side: a triangle's mapped UVs reach across a whole
number, triangles meet at a whole number from both sides, or two triangles share an edge in space whose UVs differ by
a whole number (each quad of a road mapped onto the whole texture).
"""

from __future__ import annotations

import glob
import os
from typing import NamedTuple

import numpy as np
import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

TOLERANCE = 0.02
POSITION_STEP = 1e-4
FLAT = 0.85  # cosine between the normals of two triangles that continue one surface (about 30 degrees)


class Submesh(NamedTuple):
    lo: np.ndarray     # T x 2, UV0 minimum of each triangle
    hi: np.ndarray     # T x 2, UV0 maximum of each triangle
    jumps: np.ndarray  # E x 2, UV0 difference across edges two triangles share in space but not in UV


def _virtual_paths(env, mod: str) -> dict[tuple[str, int], str]:
    """(serialized file name, path id) of each texture in the mod's content bundle -> virtual path (lower case)."""
    paths = {}
    for obj in env.objects:
        if obj.type.name != "AssetBundle":
            continue
        tree = obj.read_typetree()
        for asset_path, info in tree.get("m_Container", []):
            if not asset_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ptr = info["asset"]
            if ptr["m_FileID"] != 0:
                continue
            parts = asset_path.split("/")
            # packages/org.mhkose.mod.<mod>/... -> <mod>/...
            if len(parts) > 2 and parts[0].lower() == "packages" and parts[1].lower().endswith("." + mod):
                paths[(obj.assets_file.name, ptr["m_PathID"])] = (mod + "/" + "/".join(parts[2:])).lower()
    return paths


def _jumps(idx: np.ndarray, position_ids: np.ndarray, uv: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """UV differences across the edges that triangles `idx` share by position but not by UV, where the surface
    continues flat (a box's corners show its texture's borders side by side without a visible seam)."""
    corners = pos[idx]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    a = idx[:, [0, 1, 2]].ravel()
    b = idx[:, [1, 2, 0]].ravel()
    tri = np.repeat(np.arange(len(idx)), 3)
    pa, pb = position_ids[a], position_ids[b]
    swap = pa > pb
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    pa, pb = np.minimum(pa, pb), np.maximum(pa, pb)
    keep = pa != pb
    a, b, pa, pb, tri = a[keep], b[keep], pa[keep], pb[keep], tri[keep]
    order = np.lexsort((pb, pa))
    a, b, pa, pb, tri = a[order], b[order], pa[order], pb[order], tri[order]
    same = (pa[1:] == pa[:-1]) & (pb[1:] == pb[:-1])
    flat = np.abs(np.einsum("ij,ij->i", normals[tri[1:]], normals[tri[:-1]])) > FLAT
    d1 = uv[a[1:]] - uv[a[:-1]]
    d2 = uv[b[1:]] - uv[b[:-1]]
    rigid = same & flat & (np.abs(d1 - d2) < TOLERANCE).all(axis=1) & (np.abs(d1) > TOLERANCE).any(axis=1)
    return d1[rigid]


def _submeshes(mesh_obj) -> list[Submesh | None] | None:
    """Per submesh, its triangles' UV0 bounds and the UV0 jumps across its shared edges, None without UVs."""
    handler = MeshHandler(mesh_obj.read())
    handler.process()
    if not handler.m_UV0 or not handler.m_Vertices:
        return None
    uv = np.asarray(handler.m_UV0, dtype=np.float64).reshape(-1, 2)
    pos = np.asarray(handler.m_Vertices, dtype=np.float64).reshape(-1, 3)[:len(uv)]
    _, position_ids = np.unique(np.round(pos / POSITION_STEP).astype(np.int64), axis=0, return_inverse=True)
    position_ids = position_ids.ravel()
    out = []
    for tris in handler.get_triangles():
        idx = np.asarray(tris, dtype=np.int64).reshape(-1, 3) if tris else np.zeros((0, 3), np.int64)
        idx = idx[(idx < len(position_ids)).all(axis=1)]
        idx = idx[np.isfinite(uv[idx]).all(axis=(1, 2))]
        if len(idx) == 0:
            out.append(None)
            continue
        corners = uv[idx]  # T x 3 x 2
        out.append(Submesh(corners.min(axis=1), corners.max(axis=1), _jumps(idx, position_ids, uv, pos)))
    return out


def _repeating(lo: np.ndarray, hi: np.ndarray, jumps: np.ndarray) -> int:
    """How many triangles (with these mapped UV bounds and edge jumps along one axis) show the texture's opposite
    borders side by side: crossing a whole number, meeting other triangles at one, or across a flat edge's jump."""
    crossing = np.ceil(lo + TOLERANCE) < hi - TOLERANCE
    near_hi, near_lo = np.round(hi), np.round(lo)
    from_below = (np.abs(hi - near_hi) < TOLERANCE) & (lo < near_hi - TOLERANCE)
    from_above = (np.abs(lo - near_lo) < TOLERANCE) & (hi > near_lo + TOLERANCE)
    met = np.intersect1d(near_hi[from_below], near_lo[from_above])
    meeting = (from_below & np.isin(near_hi, met)) | (from_above & np.isin(near_lo, met))
    whole = np.round(jumps)
    jumping = (np.abs(jumps - whole) < TOLERANCE) & (np.abs(whole) >= 1)
    return int(np.count_nonzero(crossing | meeting) + 2 * np.count_nonzero(jumping))


class Tiling(NamedTuple):
    """How a texture shows on the mod's meshes: per axis the share of its triangles that show its opposite borders
    side by side (0 when it never repeats along the axis)."""
    u: float
    v: float

    @property
    def axes(self) -> str:
        return ("u" if self.u > 0 else "") + ("v" if self.v > 0 else "")


def tilings(mod_folder: str, mod: str) -> dict[str, Tiling]:
    """Virtual path (lower case) -> tiling of every texture a mesh of the mod shows."""
    bundles = sorted(glob.glob(os.path.join(mod_folder, "*.mhkmod")))
    if not bundles:
        return {}
    env = UnityPy.load(*bundles)
    by_name = {(file_name.lower(), path_id): path for (file_name, path_id), path in _virtual_paths(env, mod).items()}
    objects = {(obj.assets_file.name, obj.path_id): obj for obj in env.objects}
    files = {f.name.lower(): f for f in {obj.assets_file for obj in env.objects}}

    def deref(owner_file, ptr):
        if ptr["m_PathID"] == 0:
            return None
        if ptr["m_FileID"] == 0:
            return objects.get((owner_file.name, ptr["m_PathID"]))
        ext = os.path.basename(owner_file.externals[ptr["m_FileID"] - 1].path).lower()
        target = files.get(ext)
        return objects.get((target.name, ptr["m_PathID"])) if target else None

    submesh_cache: dict[tuple[str, int], list[Submesh | None] | None] = {}
    counts: dict[str, np.ndarray] = {}  # path -> [triangles repeating along U, along V, all triangles]
    for obj in env.objects:
        if obj.type.name not in ("MeshRenderer", "SkinnedMeshRenderer"):
            continue
        renderer = obj.read_typetree()
        if obj.type.name == "MeshRenderer":
            go = deref(obj.assets_file, renderer["m_GameObject"])
            if go is None:
                continue
            mesh_ptr = None
            for comp in go.read_typetree()["m_Component"]:
                c = deref(obj.assets_file, comp["component"])
                if c is not None and c.type.name == "MeshFilter":
                    mesh_ptr = (c.assets_file, c.read_typetree()["m_Mesh"])
                    break
            if mesh_ptr is None:
                continue
            mesh_obj = deref(*mesh_ptr)
        else:
            mesh_obj = deref(obj.assets_file, renderer["m_Mesh"])
        if mesh_obj is None:
            continue
        key = (mesh_obj.assets_file.name, mesh_obj.path_id)
        if key not in submesh_cache:
            try:
                submesh_cache[key] = _submeshes(mesh_obj)
            except Exception:  # meshes UnityPy cannot decode count as unknown
                submesh_cache[key] = None
        submeshes = submesh_cache[key]
        if not submeshes:
            continue
        materials = renderer["m_Materials"]
        # A statically batched renderer draws its range of the scene's combined mesh, material j on submesh first + j;
        # otherwise material j draws submesh j (submeshes without a material are not drawn).
        batch = renderer.get("m_StaticBatchInfo") or {}
        first, count = batch.get("firstSubMesh", 0), batch.get("subMeshCount", 0)
        if count == 0:
            first, count = 0, len(submeshes)
        for j in range(min(count, len(materials))):
            sub = submeshes[first + j] if first + j < len(submeshes) else None
            if sub is None:
                continue
            mat = deref(obj.assets_file, materials[j])
            if mat is None or mat.type.name != "Material":
                continue
            for _, slot in mat.read_typetree()["m_SavedProperties"]["m_TexEnvs"]:
                tex = deref(mat.assets_file, slot["m_Texture"])
                if tex is None:
                    continue
                path = by_name.get((tex.assets_file.name.lower(), tex.path_id))
                if path is None:
                    continue
                scale = np.array([slot["m_Scale"]["x"], slot["m_Scale"]["y"]])
                offset = np.array([slot["m_Offset"]["x"], slot["m_Offset"]["y"]])
                a, b = sub.lo * scale + offset, sub.hi * scale + offset
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                jumps = sub.jumps * scale
                total = counts.setdefault(path, np.zeros(3, np.int64))
                total += (_repeating(lo[:, 0], hi[:, 0], jumps[:, 0]), _repeating(lo[:, 1], hi[:, 1], jumps[:, 1]),
                          len(lo))
    return {path: Tiling(min(1.0, c[0] / c[2]), min(1.0, c[1] / c[2])) for path, c in counts.items() if c[2] > 0}
