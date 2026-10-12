"""tifxyz windings: load, list and save.

A tifxyz folder holds x.tif, y.tif, z.tif (float32, one value per grid node, in level-0 voxels of the volume;
-1 = no node) and meta.json. The spiral fitter writes one folder per winding, named w<NNN>_<tag>, grid rows along the
scroll axis (z) and columns around the winding (theta); the grid step is 1/scale voxels (meta.json "scale", usually
0.05 = one node every 20 voxels).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import tifffile

_WINDING_RE = re.compile(r"w(\d+)")


@dataclass
class Winding:
    name: str
    path: str
    xyz: np.ndarray          # (H, W, 3) float64, NaN where there is no node; order x, y, z
    meta: dict = field(default_factory=dict)

    @property
    def index(self) -> int | None:
        m = _WINDING_RE.search(self.name)
        return int(m.group(1)) if m else None

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.xyz).all(-1)

    @property
    def grid_step(self) -> float:
        scale = self.meta.get("scale", [0.05, 0.05])
        return 1.0 / float(scale[0])

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """Valid nodes as (N, 3) x, y, z and their (row, col) grid indices (N, 2)."""
        v = self.valid
        rc = np.stack(np.nonzero(v), -1)
        return self.xyz[v], rc


def load_winding(path: str, name: str | None = None) -> Winding:
    xs, ys, zs = (tifffile.imread(os.path.join(path, f"{c}.tif")).astype(np.float64) for c in "xyz")
    xyz = np.stack([xs, ys, zs], -1)
    bad = (xyz < -0.5).any(-1) | ~np.isfinite(xyz).all(-1)
    xyz[bad] = np.nan
    meta_path = os.path.join(path, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return Winding(name or os.path.basename(os.path.normpath(path)), path, xyz, meta)


def list_windings(folder: str, skip_substrings: tuple[str, ...] = ("spliced",), subdir: str = "") -> list[Winding]:
    """All tifxyz subfolders named w<NNN>..., sorted by winding index.

    ``subdir``: where the tifxyz sits inside each winding folder, e.g. "tifxyz/flatten.tifxyz" for lasagna flatten
    output (w070_tag/tifxyz/flatten.tifxyz/x.tif); the winding keeps the outer folder's name."""
    out = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name, subdir) if subdir else os.path.join(folder, name)
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "x.tif")):
            continue
        if any(s in name for s in skip_substrings) or _WINDING_RE.search(name) is None:
            continue
        out.append(load_winding(p, name))
    out.sort(key=lambda w: w.index)
    return out


def save_winding(w: Winding, out_dir: str) -> str:
    """Write a winding as a tifxyz folder (NaN nodes become -1), copying meta.json."""
    os.makedirs(out_dir, exist_ok=True)
    arr = np.where(np.isfinite(w.xyz), w.xyz, -1.0).astype(np.float32)
    for i, c in enumerate("xyz"):
        tifffile.imwrite(os.path.join(out_dir, f"{c}.tif"), arr[..., i])
    meta = dict(w.meta)
    v = w.valid
    if v.any():
        meta["bbox"] = [w.xyz[v].min(0).tolist(), w.xyz[v].max(0).tolist()]
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return out_dir
