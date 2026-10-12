"""Sheet count between adjacent windings, read from the surface prediction.

The physical definition of "winding k+1 is the next wrap after winding k": the straight segment from a node of k to
the nearest point of k+1 crosses exactly two predicted sheets, the one each winding sits on, and nothing in between.
Along that segment the surface prediction (e.g. m7, uint8, binary at threshold 128 in the published level-0 stores)
is sampled every voxel and thresholded. First each endpoint is checked: the distance from the node to the nearest
predicted-sheet voxel (exact, within ``radius`` voxels) must be at most ``node_tol``, else the node is off any sheet
(`node_off_sheet`: the snap or the flatten resampling left it in a gap, or there is no prediction there). The
tolerance matters: a render samples a slab of tens of voxels around the node, so a node 1-2 voxels off its sheet
still renders that sheet; the distance itself is kept per node so the reader can choose. When both ends are on a
sheet, the runs of sheet voxels from the node to the neighbour decide the verdict:

    1 run, neighbour closer than half a pitch -> both windings on one sheet (merged: a fit error)
    1 run, neighbour about a pitch away       -> the prediction fills the gap between the wraps (fused: a model error)
    2 runs  -> adjacent wraps (ok)
    3+ runs -> at least one wrap between them was skipped

`runs` keeps the raw count over the whole segment (with a margin at both ends) for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zarr

zarr.config.set({"async.concurrency": 2})   # this tool shares the machine; one chunk at a time is enough

from .spacing import SpacingResult, UNASSESSED
from .tifxyz import Winding


S_ADJACENT, S_MERGED, S_SKIPPED, S_START_OFF, S_END_OFF, S_UNASSESSED, S_FUSED = 0, 1, 2, 3, 4, 5, 6
SHEET_VERDICT_NAMES = {S_ADJACENT: "adjacent", S_MERGED: "merged", S_SKIPPED: "skipped",
                       S_START_OFF: "node_off_sheet", S_END_OFF: "neighbour_off_sheet", S_UNASSESSED: "unassessed",
                       S_FUSED: "fused"}
DIST_BINS = (1.0, 2.0, 3.0)   # reported fractions of nodes within 1, 2, 3 voxels of a predicted sheet


@dataclass
class SheetCountResult:
    runs: np.ndarray            # (H, W) int8 number of sheet runs along the segment, -1 where not assessed
    start_on_sheet: np.ndarray  # (H, W) bool: the node of A is within node_tol of a predicted sheet
    end_on_sheet: np.ndarray    # (H, W) bool: the nearest point on B is within node_tol of a predicted sheet
    verdict: np.ndarray         # (H, W) int8 S_* : the sheet-based reading of the pair at this node
    node_dist: np.ndarray       # (H, W) float32 distance from the node to the nearest sheet voxel (nan: unassessed)
    neighbour_dist: np.ndarray  # (H, W) float32 the same for the nearest point on B
    node_tol: float = 2.0

    def histogram(self) -> dict[str, int]:
        v = self.verdict
        assessed = v != S_UNASSESSED
        both = assessed & (v != S_START_OFF) & (v != S_END_OFF)
        h = {"n": int(assessed.sum()), "node_off_sheet": int((v == S_START_OFF).sum()),
             "neighbour_off_sheet": int((v == S_END_OFF).sum()), "both_on_sheet": int(both.sum()),
             "adjacent": int((v == S_ADJACENT).sum()), "merged": int((v == S_MERGED).sum()),
             "fused": int((v == S_FUSED).sum()), "skipped": int((v == S_SKIPPED).sum())}
        d = self.node_dist[assessed]
        for b in DIST_BINS:
            h[f"node_within_{b:g}"] = int((d <= b).sum())
        return h


def open_surface(path: str, group: str = "0"):
    """A zarr array of the surface prediction (uint8, z, y, x), from a group path or an array path."""
    node = zarr.open(path, mode="r")
    if hasattr(node, "shape"):
        return node
    return node[group]


def _chunk_groups(keys: np.ndarray):
    """Unique chunk keys and, per key, the indices of the rows that map to it."""
    uk, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(uk) + 1))
    for k, key in enumerate(uk):
        yield key, order[bounds[k]:bounds[k + 1]]


def _read_chunk(arr, key):
    cs = np.array(arr.chunks)
    lo = key * cs
    hi = np.minimum(lo + cs, arr.shape)
    return lo, hi, np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])


def sample_volume(arr, pts_xyz: np.ndarray) -> np.ndarray:
    """Nearest-voxel samples of ``arr`` (z, y, x) at float x, y, z coordinates; 0 outside the volume.

    Reads one chunk at a time (zarr's own coordinate selection decodes every touched chunk at once, which on a
    192^3-chunk prediction means gigabytes), so memory stays at one decoded chunk plus the sample arrays.
    """
    out = np.zeros(len(pts_xyz), np.uint8)
    p = np.rint(pts_xyz[:, ::-1]).astype(np.int64)             # z, y, x
    ok = np.all((p >= 0) & (p < np.array(arr.shape)), 1)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return out
    cs = np.array(arr.chunks)
    for key, grp in _chunk_groups(p[idx] // cs):
        sel = idx[grp]
        lo, _, block = _read_chunk(arr, key)
        q = p[sel] - lo
        out[sel] = block[q[:, 0], q[:, 1], q[:, 2]]
    return out


def node_sheet_distance(arr, pts_xyz: np.ndarray, threshold: int = 128, radius: float = 4.0,
                        batch: int = 4096) -> np.ndarray:
    """Euclidean distance (voxels) from each point to the nearest voxel of ``arr`` >= threshold, exact up to
    ``radius`` and ``radius + 1`` beyond it (or outside the volume). Every chunk is decoded once: a point's
    neighbourhood is gathered from each chunk it overlaps and the minimum is kept."""
    r = int(np.ceil(radius))
    g = np.arange(-r, r + 1)
    oz, oy, ox = np.meshgrid(g, g, g, indexing="ij")
    offs = np.stack([oz.ravel(), oy.ravel(), ox.ravel()], 1)
    norms = np.sqrt((offs ** 2).sum(1))
    keep = norms <= radius
    offs, norms = offs[keep], norms[keep].astype(np.float32)
    out = np.full(len(pts_xyz), radius + 1, np.float32)
    p = np.rint(pts_xyz[:, ::-1]).astype(np.int64)             # z, y, x
    shape = np.array(arr.shape)
    ok = np.all((p >= 0) & (p < shape), 1)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return out
    cs = np.array(arr.chunks)
    # every chunk a point's cube overlaps: the keys of the cube's 8 corners (the cube is far smaller than a chunk)
    signs = np.array([[a, b, c] for a in (-r, r) for b in (-r, r) for c in (-r, r)])
    corners = np.clip(p[idx][:, None, :] + signs[None, :, :], 0, shape - 1)        # (N, 8, 3)
    keys = corners // cs
    pairs = np.unique(np.concatenate([np.repeat(idx, 8)[:, None], keys.reshape(-1, 3)], 1), axis=0)
    for key, grp in _chunk_groups(pairs[:, 1:]):
        sel = pairs[grp, 0]
        lo, hi, block = _read_chunk(arr, key)
        sheet = block >= threshold
        if not sheet.any():
            continue
        for s in range(0, len(sel), batch):
            ss = sel[s:s + batch]
            q = p[ss][:, None, :] + offs[None, :, :]                                # (M, K, 3)
            inside = np.all((q >= lo) & (q < hi), 2)
            qc = np.clip(q - lo, 0, hi - lo - 1)
            hit = sheet[qc[..., 0], qc[..., 1], qc[..., 2]] & inside
            cand = np.where(hit, norms[None, :], np.inf).min(1)
            out[ss] = np.minimum(out[ss], cand)
    return out


def count_runs(mask: np.ndarray) -> np.ndarray:
    """Number of runs of True along the last axis, per row."""
    m = mask.astype(np.int8)
    starts = (m[:, 1:] == 1) & (m[:, :-1] == 0)
    return starts.sum(1) + m[:, 0]


def sheet_counts(a: Winding, spacing: SpacingResult, surface, threshold: int = 128, step: float = 1.0,
                 margin: float = 3.0, pitch_cap: float = 4.0, fuse_frac: float = 0.5, node_tol: float = 2.0,
                 radius: float = 4.0) -> SheetCountResult:
    """Sheet runs along node(A) -> nearest(B) segments, for the nodes that spacing assessed.

    ``node_tol``: an endpoint counts as on a sheet when a predicted-sheet voxel lies within this distance (voxels).
    ``radius``: how far the per-node distance is measured exactly (capped at radius + 1 beyond)."""
    H, W = a.xyz.shape[:2]
    runs = np.full((H, W), -1, np.int8)
    start_on = np.zeros((H, W), bool)
    end_on = np.zeros((H, W), bool)
    verdict = np.full((H, W), S_UNASSESSED, np.int8)
    node_dist = np.full((H, W), np.nan, np.float32)
    nbr_dist = np.full((H, W), np.nan, np.float32)
    assess = (spacing.flag != UNASSESSED) & np.isfinite(spacing.distance)
    rc = np.stack(np.nonzero(assess), -1)
    if len(rc) == 0:
        return SheetCountResult(runs, start_on, end_on, verdict, node_dist, nbr_dist, node_tol)
    p = a.xyz[rc[:, 0], rc[:, 1]]
    q = spacing.nearest[rc[:, 0], rc[:, 1]]
    d = spacing.distance[rc[:, 0], rc[:, 1]]
    radius = max(radius, node_tol)
    dist = node_sheet_distance(surface, np.concatenate([p, q]), threshold, radius)
    dp, dq = dist[:len(p)], dist[len(p):]
    node_dist[rc[:, 0], rc[:, 1]] = dp
    nbr_dist[rc[:, 0], rc[:, 1]] = dq
    s_on, e_on = dp <= node_tol, dq <= node_tol
    start_on[rc[:, 0], rc[:, 1]] = s_on
    end_on[rc[:, 0], rc[:, 1]] = e_on
    # segment length is capped so a skipped-sheet segment is still sampled end to end but a stray huge distance is not
    margin = max(margin, node_tol + 1.0)
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
    kend = np.clip(np.rint((L + margin) / step).astype(int), 0, n - 1)
    # both endpoints on a sheet: is it the same run (merged), the next one (adjacent) or further (skipped)?
    # The window keeps the endpoint tolerance along the segment, or a run that starts a voxel past the neighbour
    # node would be dropped and an adjacent pair read as merged.
    w = int(np.ceil(node_tol / step))
    between = sheet.copy()
    between[:, :max(k0 - w, 0)] = False
    for i in range(len(rc)):
        between[i, kend[i] + w + 1:] = False
    r_between = count_runs(between)
    v = np.full(len(rc), S_UNASSESSED, np.int8)
    v[~s_on] = S_START_OFF
    v[s_on & ~e_on] = S_END_OFF
    both = s_on & e_on
    one_run = both & (r_between <= 1)
    # one unbroken run from node to neighbour: at a short distance the two windings sit on one sheet (a fit error);
    # at about a pitch the prediction itself fills the gap between two wraps (the model fused the sheets there)
    v[one_run & (d < fuse_frac * spacing.pitch)] = S_MERGED
    v[one_run & (d >= fuse_frac * spacing.pitch)] = S_FUSED
    v[both & (r_between == 2)] = S_ADJACENT
    v[both & (r_between >= 3)] = S_SKIPPED
    verdict[rc[:, 0], rc[:, 1]] = v
    return SheetCountResult(runs, start_on, end_on, verdict, node_dist, nbr_dist, node_tol)
