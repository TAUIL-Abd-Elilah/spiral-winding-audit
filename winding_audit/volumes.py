"""Volume sources: OME-Zarr arrays, and a local chunk cache of raw .bin chunks (one file per 128^3 uint8 chunk,
named ``<z>_<y>_<x>.bin``), wrapped so that the samplers in ``sheets`` read either the same way."""
from __future__ import annotations

import os
import re

import numpy as np

from .sheets import open_surface

_BIN = re.compile(r"^\d+_\d+_\d+\.bin$")


class ChunkCacheArray:
    """Read-only, array-like view of a folder of raw chunks. Missing chunks read as 0; ``available`` tells which
    points fall in chunks that are present."""

    def __init__(self, path: str, shape, chunk: int = 128, dtype=np.uint8):
        self.path = path
        self.shape = tuple(int(s) for s in shape)
        self.chunks = (chunk, chunk, chunk)
        self.dtype = np.dtype(dtype)
        self._present = {tuple(int(v) for v in f[:-4].split("_")) for f in os.listdir(path) if _BIN.match(f)}

    def _chunk(self, key):
        if key not in self._present:
            return None
        c = self.chunks[0]
        b = np.fromfile(os.path.join(self.path, f"{key[0]}_{key[1]}_{key[2]}.bin"), self.dtype)
        return b.reshape(c, c, c) if b.size == c ** 3 else None

    def __getitem__(self, sl):
        if not isinstance(sl, tuple):
            sl = (sl,)
        sl = tuple(sl) + (slice(None),) * (3 - len(sl))
        idx = [s.indices(n) for s, n in zip(sl, self.shape)]
        if any(st != 1 for _, _, st in idx):
            raise ValueError("ChunkCacheArray supports unit-step slices only")
        lo = np.array([i[0] for i in idx])
        hi = np.array([i[1] for i in idx])
        out = np.zeros(np.maximum(hi - lo, 0), self.dtype)
        c = self.chunks[0]
        for kz in range(lo[0] // c, (hi[0] - 1) // c + 1):
            for ky in range(lo[1] // c, (hi[1] - 1) // c + 1):
                for kx in range(lo[2] // c, (hi[2] - 1) // c + 1):
                    blk = self._chunk((kz, ky, kx))
                    if blk is None:
                        continue
                    k = np.array([kz, ky, kx]) * c
                    a = np.maximum(lo, k)
                    b = np.minimum(hi, k + c)
                    out[a[0] - lo[0]:b[0] - lo[0], a[1] - lo[1]:b[1] - lo[1], a[2] - lo[2]:b[2] - lo[2]] = \
                        blk[a[0] - k[0]:b[0] - k[0], a[1] - k[1]:b[1] - k[1], a[2] - k[2]:b[2] - k[2]]
        return out

    def available(self, pts_xyz: np.ndarray) -> np.ndarray:
        """True where a point (x, y, z) lies in a chunk that is present."""
        k = np.floor(np.asarray(pts_xyz)[:, ::-1] / self.chunks[0]).astype(np.int64)
        return np.array([tuple(r) in self._present for r in k], bool)


def is_chunk_cache(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as it:
        for i, e in enumerate(it):
            if _BIN.match(e.name):
                return True
            if i > 50:
                break
    return False


def open_volume(path: str, group: str = "0", like=None):
    """An OME-Zarr array (group path or array path), or a raw chunk cache; a chunk cache takes its shape from
    ``like`` (an array on the same scan grid, e.g. the surface prediction)."""
    if is_chunk_cache(path):
        if like is None:
            raise ValueError(f"{path} is a chunk cache: its shape must come from an array on the same grid")
        return ChunkCacheArray(path, like.shape)
    return open_surface(path, group)
