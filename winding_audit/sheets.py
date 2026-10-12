"""Sheet count between adjacent windings, read from the surface prediction.

The physical definition of "winding k+1 is the next wrap after winding k": the straight segment from a node of k to
the nearest point of k+1 crosses exactly two predicted sheets, the one each winding sits on, and nothing in between.
Along that segment (extended by a small margin at both ends) the surface prediction (e.g. m7, uint8 probability) is
sampled every voxel and thresholded; the number of runs of sheet voxels is the sheet count:

    2 runs  -> adjacent wraps (ok)
    1 run   -> both windings on one sheet, or the segment never leaves one sheet (merged)
    3+ runs -> at least one wrap between them was skipped
    0 runs  -> neither end is on a predicted sheet (no surface here; unassessed)

Whether each endpoint itself lies on a sheet is reported separately (the snap may have left nodes in the gap).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zarr

from .spacing import SpacingResult, UNASSESSED
from .tifxyz import Winding


@dataclass
class SheetCountResult:
    runs: np.ndarray            # (H, W) int8 number of sheet runs along the segment, -1 where not assessed
    start_on_sheet: np.ndarray  # (H, W) bool: the node of A is on a predicted sheet
    end_on_sheet: np.ndarray    # (H, W) bool: the nearest point on B is on a predicted sheet

    def histogram(self) -> dict[str, int]:
        r = self.runs[self.runs >= 0]
        return {"n": int(r.size), "runs0": int((r == 0).sum()), "runs1": int((r == 1).sum()),
                "runs2": int((r == 2).sum()), "runs3plus": int((r >= 3).sum()),
                "start_on_sheet": int(self.start_on_sheet[self.runs >= 0].sum()),
                "end_on_sheet": int(self.end_on_sheet[self.runs >= 0].sum())}


def open_surface(path: str, group: str = "0"):
    """A zarr array of the surface prediction (uint8, z, y, x), from a group path or an array path."""
    node = zarr.open(path, mode="r")
    if hasattr(node, "shape"):
        return node
    return node[group]


def sample_volume(arr, pts_xyz: np.ndarray, batch: int = 2_000_000) -> np.ndarray:
    """Nearest-voxel samples of ``arr`` (z, y, x) at float x, y, z coordinates; 0 outside the volume."""
    Z, Y, X = arr.shape
    out = np.zeros(len(pts_xyz), np.uint8)
    zi = np.rint(pts_xyz[:, 2]).astype(np.int64)
    yi = np.rint(pts_xyz[:, 1]).astype(np.int64)
    xi = np.rint(pts_xyz[:, 0]).astype(np.int64)
    ok = (zi >= 0) & (zi < Z) & (yi >= 0) & (yi < Y) & (xi >= 0) & (xi < X)
    idx = np.nonzero(ok)[0]
    for s in range(0, len(idx), batch):
        sel = idx[s:s + batch]
        # sort by chunk-major order so the store reads each chunk once
        order = np.lexsort((xi[sel], yi[sel], zi[sel]))
        sel = sel[order]
        out[sel] = arr.get_coordinate_selection((zi[sel], yi[sel], xi[sel]))
    return out


def count_runs(mask: np.ndarray) -> np.ndarray:
    """Number of runs of True along the last axis, per row."""
    m = mask.astype(np.int8)
    starts = (m[:, 1:] == 1) & (m[:, :-1] == 0)
    return starts.sum(1) + m[:, 0]


def sheet_counts(a: Winding, spacing: SpacingResult, surface, threshold: int = 128, step: float = 1.0,
                 margin: float = 3.0, pitch_cap: float = 4.0) -> SheetCountResult:
    """Sheet runs along node(A) -> nearest(B) segments, for the nodes that spacing assessed."""
    H, W = a.xyz.shape[:2]
    runs = np.full((H, W), -1, np.int8)
    start_on = np.zeros((H, W), bool)
    end_on = np.zeros((H, W), bool)
    assess = (spacing.flag != UNASSESSED) & np.isfinite(spacing.distance)
    rc = np.stack(np.nonzero(assess), -1)
    if len(rc) == 0:
        return SheetCountResult(runs, start_on, end_on)
    p = a.xyz[rc[:, 0], rc[:, 1]]
    q = spacing.nearest[rc[:, 0], rc[:, 1]]
    d = spacing.distance[rc[:, 0], rc[:, 1]]
    # segment length is capped so a skipped-sheet segment is still sampled end to end but a stray huge distance is not
    L = np.minimum(d, pitch_cap * spacing.pitch)
    u = (q - p) / np.maximum(d, 1e-9)[:, None]
    n = int(np.ceil((L.max() + 2 * margin) / step)) + 1
    t = -margin + step * np.arange(n)                      # (n,)
    samples = p[:, None, :] + t[None, :, None] * u[:, None, :]   # (N, n, 3)
    inside = t[None, :] <= (L + margin)[:, None]
    vals = sample_volume(surface, samples.reshape(-1, 3)).reshape(len(rc), n)
    sheet = (vals >= threshold) & inside
    r = count_runs(sheet).astype(np.int8)
    runs[rc[:, 0], rc[:, 1]] = r
    k0 = int(round(margin / step))
    start_on[rc[:, 0], rc[:, 1]] = sheet[:, k0]
    kend = np.clip(np.rint((L + margin) / step).astype(int), 0, n - 1)
    end_on[rc[:, 0], rc[:, 1]] = sheet[np.arange(len(rc)), kend]
    return SheetCountResult(runs, start_on, end_on)
