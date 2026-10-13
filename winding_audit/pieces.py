"""Single-sheet pieces: the parts of each winding that stay on one predicted sheet and that no other winding covers.

A winding from a fit that is denser than the sheets runs between two sheets, and a nearest-sheet snap puts parts of
it on each: around the turn it switches sheets, and where it shares a sheet with its neighbour the same papyrus is
read twice. A re-snap cannot fix that (there is no free sheet); a re-fit can. Until then, this module marks what can
be trusted:

1. **On a sheet.** A node within ``node_tol`` voxels of a predicted-sheet voxel.
2. **Continuous.** Two grid neighbours stay on one sheet when the step between them along the normal is under
   ``jump_frac`` pitches and the points at 1/4, 1/2 and 3/4 of the edge are on a sheet too. A winding that crosses
   the gap between two sheets breaks here.
3. **Cut where windings collapse.** A winding can only reach another sheet by crossing a gap (2 cuts it) or by
   landing on a neighbour's sheet. So the nodes within ``merge_frac`` pitches of a neighbouring winding's surface
   form their own pieces, separate from the rest of the winding.
4. **Once.** Going from the inside out, a node within ``merge_frac`` pitches of the inner neighbour's surface, where
   that neighbour's node is kept, is a duplicate and is dropped: the shared sheet is kept in the inner winding.

The connected components of the kept nodes over the continuous edges are the pieces; pieces smaller than
``min_nodes`` are dropped. The output is a mask per winding (and a piece label), so an existing render of the
winding can be read through it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .sheets import node_sheet_distance, prediction_pitch
from .spacing import neighbour_spacing
from .tifxyz import Winding, grid_normals


@dataclass
class PieceResult:
    label: np.ndarray          # (H, W) int32: piece id 1..n (largest first), 0 = not kept
    on_sheet: np.ndarray       # (H, W) bool
    duplicate: np.ndarray      # (H, W) bool: on the inner neighbour's kept surface
    shared_out: np.ndarray     # (H, W) bool: on the outer neighbour's surface (kept here, dropped there)
    pitch: float
    stats: dict = field(default_factory=dict)

    @property
    def kept(self) -> np.ndarray:
        return self.label > 0


def _near(a: Winding, b: Winding | None, pitch: float, merge_frac: float):
    """Nodes of ``a`` within merge_frac pitches of ``b``'s surface, and the index of b's node they were found on."""
    H, W = a.xyz.shape[:2]
    if b is None or not b.valid.any():
        return np.zeros((H, W), bool), None
    sp = neighbour_spacing(a, b, pitch=pitch)
    near = np.isfinite(sp.distance) & (sp.distance < merge_frac * pitch)
    pb, rcb = b.points()
    j = np.full((H, W), -1, np.int64)
    if near.any():
        _, jj = cKDTree(pb).query(sp.nearest[near], workers=1)
        j[near] = jj
    return near, (rcb, j)


def winding_pieces(a: Winding, inner: Winding | None, outer: Winding | None, surface,
                   inner_kept: np.ndarray | None = None, pitch: float | None = None, threshold: int = 128,
                   node_tol: float = 2.0, jump_frac: float = 0.35, merge_frac: float = 0.3, min_nodes: int = 50,
                   radius: float = 4.0) -> PieceResult:
    H, W = a.xyz.shape[:2]
    X = a.xyz
    v = a.valid
    if pitch is None or not np.isfinite(pitch):
        pitch = prediction_pitch(a, surface, threshold)
    pitch = float(pitch)
    n = grid_normals(X)
    nd = np.full((H, W), np.inf)
    if v.any():
        nd[v] = node_sheet_distance(surface, X[v], threshold, max(radius, node_tol))
    on = v & (nd <= node_tol) & np.isfinite(n).all(-1)

    near_in, ref_in = _near(a, inner, pitch, merge_frac)
    duplicate = np.zeros((H, W), bool)
    if ref_in is not None and near_in.any():
        rcb, j = ref_in
        kept_b = inner_kept if inner_kept is not None else inner.valid
        r, c = np.nonzero(near_in)
        duplicate[r, c] = kept_b[rcb[j[r, c], 0], rcb[j[r, c], 1]]
    shared_out, _ = _near(a, outer, pitch, merge_frac)
    collapsed = near_in | shared_out              # pieces never cross the edge of a collapsed region

    keep = on & ~duplicate
    idx = np.full((H, W), -1, np.int64)
    idx[keep] = np.arange(int(keep.sum()))
    rows, cols = [], []
    n_jump = n_gap = n_edges = 0
    for dr, dc in ((0, 1), (1, 0)):
        both = keep[:H - dr, :W - dc] & keep[dr:, dc:]
        same = collapsed[:H - dr, :W - dc] == collapsed[dr:, dc:]
        r, c = np.nonzero(both & same)
        if len(r) == 0:
            continue
        p, q = X[r, c], X[r + dr, c + dc]
        nn = n[r, c] + n[r + dr, c + dc]
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        step = np.abs(((q - p) * nn).sum(1))
        mids = np.concatenate([p + (q - p) * t for t in (0.25, 0.5, 0.75)])
        md = node_sheet_distance(surface, mids, threshold, max(radius, node_tol)).reshape(3, -1).max(0)
        ok = (step < jump_frac * pitch) & (md <= node_tol)
        n_jump += int((step >= jump_frac * pitch).sum())
        n_gap += int((md > node_tol).sum())
        n_edges += len(r)
        rows.append(idx[r[ok], c[ok]])
        cols.append(idx[r[ok] + dr, c[ok] + dc])
    N = int(keep.sum())
    label = np.zeros((H, W), np.int32)
    sizes = np.zeros(0, int)
    if N:
        rr = np.concatenate(rows) if rows else np.zeros(0, int)
        cc = np.concatenate(cols) if cols else np.zeros(0, int)
        _, comp = connected_components(coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(N, N)), directed=False)
        sizes = np.bincount(comp)
        big = np.nonzero(sizes >= min_nodes)[0]
        order = big[np.argsort(-sizes[big])]
        remap = np.zeros(len(sizes), np.int32)
        remap[order] = np.arange(1, len(order) + 1)
        label[keep] = remap[comp]
        sizes = sizes[order]
    stats = {"n": int(v.sum()), "on_sheet": int(on.sum()), "duplicate": int((duplicate & v).sum()),
             "shared_out": int((shared_out & v).sum()), "kept": int((label > 0).sum()), "pieces": int(len(sizes)),
             "largest": int(sizes[0]) if len(sizes) else 0, "edges": n_edges, "edges_jump": n_jump,
             "edges_gap": n_gap, "pitch": pitch}
    return PieceResult(label, on, duplicate, shared_out, pitch, stats)


def band_pieces(windings: list[Winding], surface, **kw) -> list[PieceResult]:
    """Pieces for every winding, from the inside out, each deduplicated against its inner neighbour's kept nodes."""
    out: list[PieceResult] = []
    for i, w in enumerate(windings):
        inner = windings[i - 1] if i > 0 else None
        outer = windings[i + 1] if i + 1 < len(windings) else None
        res = winding_pieces(w, inner, outer, surface, inner_kept=out[-1].kept if out else None, **kw)
        out.append(res)
    return out
