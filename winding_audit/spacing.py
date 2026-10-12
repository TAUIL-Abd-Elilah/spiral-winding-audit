"""Neighbour spacing: how far every node of winding k is from winding k+1.

Adjacent windings of a scroll are one sheet pitch apart everywhere. A node much closer than the pitch to the next
winding means the two windings sit on the same sheet there (a merger); a node much farther means a wrap was skipped.
The pitch is estimated from the data itself (the median spacing of the pair), so no scroll-specific value is needed.

Distances are to the neighbour's *surface*, not its nodes: on a 20-voxel grid the nearest node of the next winding
can be 10 or more voxels to the side, so two windings that touch would read as a pitch apart. The nearest node is
found first (with an angular window, below), then the node is projected onto the neighbour's tangent plane there
(the plane through that node, normal to the neighbour's grid). The foot of that perpendicular is the nearest point;
if it slides more than a grid step from the node it was built on (a curved or folded neighbour), the node itself is
used.

A node that is far from the neighbour but whose nearest point lies on the neighbour's own boundary (its band edge, or
the edge of a hole) is not called "skipped": the true nearest sheet point may be outside the neighbour's coverage, so a
large distance says nothing there. A small or normal distance to a boundary point is still a real measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .tifxyz import Winding, grid_normals

OK, MERGED, SKIPPED, UNASSESSED = 0, 1, 2, 3
FLAG_NAMES = {OK: "ok", MERGED: "merged", SKIPPED: "skipped", UNASSESSED: "unassessed"}


@dataclass
class SpacingResult:
    distance: np.ndarray     # (H, W) distance from each node of A to winding B, NaN where A has no node
    nearest: np.ndarray      # (H, W, 3) nearest point on B
    flag: np.ndarray         # (H, W) int8: OK / MERGED / SKIPPED / UNASSESSED (UNASSESSED also where A is invalid)
    pitch: float             # estimated sheet pitch for this pair (voxels)
    on_plane: np.ndarray | None = None   # (H, W) bool: the nearest point is the foot on B's tangent plane

    def fractions(self) -> dict[str, float]:
        assessed = (self.flag == OK) | (self.flag == MERGED) | (self.flag == SKIPPED)
        n = max(int(assessed.sum()), 1)
        return {
            "n_assessed": int(assessed.sum()),
            "frac_merged": float((self.flag == MERGED).sum() / n),
            "frac_skipped": float((self.flag == SKIPPED).sum() / n),
            "frac_unassessed": float(((self.flag == UNASSESSED) & np.isfinite(self.distance)).sum()
                                     / max(int(np.isfinite(self.distance).sum()), 1)),
            "pitch": float(self.pitch),
        }


def boundary_mask(valid: np.ndarray) -> np.ndarray:
    """Valid nodes that touch an invalid node or the grid edge."""
    inner = ndimage.binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=0)
    return valid & ~inner


def neighbour_spacing(a: Winding, b: Winding, merge_frac: float = 0.3, skip_frac: float = 1.7,
                      pitch: float | None = None, angular_weight: float = 10.0,
                      to_surface: bool = True) -> SpacingResult:
    """Distance from every node of ``a`` to the surface of ``b``, with merged / skipped flags.

    Each winding is one turn of the spiral, columns ordered by angle, so the end of winding k physically meets the
    start of winding k+1 (one sheet continuing). A plain nearest-point distance reads ~0 there and would call it a
    merger. The query therefore compares a node only with neighbour nodes at a similar angular position: the column
    fraction t = col / (W - 1) joins the coordinates as a fourth axis scaled by ``angular_weight`` sheet pitches per
    turn (10 pitches per full turn by default: a node 1/20 turn away costs half a pitch, the seam costs 10).
    """
    H, W = a.xyz.shape[:2]
    pa, rca = a.points()
    pb, rcb = b.points()
    distance = np.full((H, W), np.nan)
    nearest = np.full((H, W, 3), np.nan)
    flag = np.full((H, W), UNASSESSED, np.int8)
    if len(pa) == 0 or len(pb) == 0:
        return SpacingResult(distance, nearest, flag, float("nan"), np.zeros((H, W), bool))
    d0, _ = cKDTree(pb).query(pa, workers=1)            # pass 1: plain 3D, only to scale the angular axis
    pitch0 = pitch if pitch is not None else float(np.median(d0))
    ta = rca[:, 1] / max(a.xyz.shape[1] - 1, 1)
    tb = rcb[:, 1] / max(b.xyz.shape[1] - 1, 1)
    lam = angular_weight * pitch0
    qa = np.column_stack([pa, lam * ta])
    qb = np.column_stack([pb, lam * tb])
    _, j = cKDTree(qb).query(qa, workers=1)             # pass 2: nearest neighbour node at a similar angle
    q = pb[j]
    plane = np.zeros(len(pa), bool)
    if to_surface:
        nb = grid_normals(b.xyz)[rcb[j, 0], rcb[j, 1]]
        foot = pa - ((pa - q) * nb).sum(1, keepdims=True) * nb
        plane = np.isfinite(foot).all(1) & (np.linalg.norm(foot - q, axis=1) <= b.grid_step)
        q = np.where(plane[:, None], foot, q)
    d = np.linalg.norm(q - pa, axis=1)                  # plain 3D distance to the surface point
    distance[rca[:, 0], rca[:, 1]] = d
    nearest[rca[:, 0], rca[:, 1]] = q
    on_plane = np.zeros((H, W), bool)
    on_plane[rca[:, 0], rca[:, 1]] = plane
    on_edge = boundary_mask(b.valid)[rcb[j, 0], rcb[j, 1]]
    if pitch is None:
        core = d[~on_edge]
        pitch = float(np.median(core)) if len(core) else float(np.median(d))
    f = np.where(d < merge_frac * pitch, MERGED, np.where(d > skip_frac * pitch, SKIPPED, OK)).astype(np.int8)
    # a large distance to a point on the neighbour's own border may only mean the neighbour ends there (a hole or
    # the band edge), so "skipped" cannot be read off it; a small or normal distance to a border point is still real
    f[on_edge & (f == SKIPPED)] = UNASSESSED
    flag[rca[:, 0], rca[:, 1]] = f
    return SpacingResult(distance, nearest, flag, pitch, on_plane)
