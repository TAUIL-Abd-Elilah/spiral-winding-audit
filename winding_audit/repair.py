"""Repair: put flagged nodes back on the sheet that lies between their neighbouring windings.

The post-hoc snap of the PHerc. 343 recipe moves every node along its normal to the nearest predicted sheet within
+-15 voxels. With a sheet pitch of 10-20 voxels the nearest sheet can be the next wrap's, so two windings end up on
one sheet; where no sheet is near, the node stays in the gap. The repair uses the neighbours as bounds. For a node of
winding k, the surface prediction is sampled along the node's normal, oriented from winding k-1 towards winding
k+1, and the sheet runs are listed:

    the node's own run      the run within ``node_tol`` voxels of the node
    the neighbours' runs    the runs at the positions of the nearest points of k-1 and k+1 on that line

A node whose own run is not a neighbour's run is kept as it is. A node whose run is also the run of a neighbour
that is about a pitch away (half a pitch or more) is kept too, as ``kept_shared_run``: the prediction does not
separate the two wraps there, so it cannot say which node is wrong, if any. A node that is off any sheet, or whose
run is a neighbour's run with the neighbour closer than half a pitch (merged), is moved to the centre of the run
nearest to it that lies strictly between the two neighbours' runs, if that run is within ``max_shift_frac``
pitches. If there is none (the neighbour, not this node, is on the wrong sheet; or there is no free sheet between
the neighbours), the node is left where it is and reported as unrepaired. A move that disagrees with its neighbourhood (a single node jumping more than
``jump_frac`` pitches while the nodes around it do not) is undone, so the repair cannot pull spikes out of the mesh.

Windings are repaired from the inside out, each against its already-repaired inner neighbour, so two windings on
one sheet are separated once, not both moved.

Skipped wraps (winding k+1 two pitches out, every later winding following it) are a topology error, not a placement
error: no node is on the wrong sheet locally, so the repair keeps them and the audit keeps reporting them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sheets import sample_volume
from .spacing import neighbour_spacing
from .tifxyz import Winding

R_KEPT, R_MOVED, R_UNREPAIRED, R_REJECTED, R_NOT_ASSESSED, R_SHARED = 0, 1, 2, 3, 4, 5
REPAIR_STATUS_NAMES = {R_KEPT: "kept", R_MOVED: "moved", R_SHARED: "kept_shared_run", R_UNREPAIRED: "unrepaired",
                       R_REJECTED: "move_rejected", R_NOT_ASSESSED: "not_assessed"}


@dataclass
class RepairResult:
    winding: Winding            # the repaired winding (same grid, same name)
    shift: np.ndarray           # (H, W) applied offset along the oriented normal, voxels (0 kept, nan no node)
    status: np.ndarray          # (H, W) int8 R_*
    pitch: float
    stats: dict = field(default_factory=dict)


def grid_normals(xyz: np.ndarray) -> np.ndarray:
    """Unit normals of a tifxyz grid (H, W, 3), NaN where undefined: cross product of the column and row tangents
    (central differences where both neighbours exist, one-sided otherwise)."""
    def tangent(axis):
        fwd = np.roll(xyz, -1, axis) - xyz
        bwd = xyz - np.roll(xyz, 1, axis)
        n = xyz.shape[axis]
        sl = [slice(None)] * 3
        sl[axis] = slice(n - 1, n)
        fwd[tuple(sl)] = np.nan
        sl[axis] = slice(0, 1)
        bwd[tuple(sl)] = np.nan
        f_ok, b_ok = np.isfinite(fwd).all(-1), np.isfinite(bwd).all(-1)
        t = np.full_like(xyz, np.nan)
        both = f_ok & b_ok
        t[both] = 0.5 * (fwd[both] + bwd[both])
        t[f_ok & ~b_ok] = fwd[f_ok & ~b_ok]
        t[b_ok & ~f_ok] = bwd[b_ok & ~f_ok]
        return t
    n = np.cross(tangent(1), tangent(0))
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.where(norm > 1e-9, norm, np.nan)
    return n


def edge_stretch(xyz: np.ndarray, grid_step: float) -> dict:
    """Lengths of the grid edges in grid steps: how much a winding is stretched (a fold or a jump shows as > 1.5)."""
    lens = []
    for axis in (0, 1):
        d = np.diff(xyz, axis=axis)
        l = np.linalg.norm(d, axis=-1)
        lens.append(l[np.isfinite(l)])
    l = np.concatenate(lens) / grid_step
    if len(l) == 0:
        return {"edges": 0, "frac_edges_over_1p5": None, "p99_edge": None}
    return {"edges": int(len(l)), "frac_edges_over_1p5": float((l > 1.5).mean()), "p99_edge": float(np.percentile(l, 99))}


def _run_id_at(runid: np.ndarray, centre_idx: np.ndarray, w: int) -> np.ndarray:
    """Per row, the run id nearest to column ``centre_idx`` within +-w columns (0 = no run there)."""
    N, M = runid.shape
    offs = np.arange(-w, w + 1)
    cols = centre_idx[:, None] + offs[None, :]
    inside = (cols >= 0) & (cols < M) & (centre_idx[:, None] >= 0)
    ids = np.where(inside, runid[np.arange(N)[:, None], np.clip(cols, 0, M - 1)], 0)
    cost = np.where(ids > 0, np.abs(offs)[None, :], np.iinfo(np.int32).max)
    return ids[np.arange(N), np.argmin(cost, 1)]


def repair_winding(a: Winding, prev: Winding, nxt: Winding, surface, threshold: int = 128, node_tol: float = 2.0,
                   max_shift_frac: float = 1.25, search_frac: float = 2.5, jump_frac: float = 0.4,
                   step: float = 1.0) -> RepairResult:
    H, W = a.xyz.shape[:2]
    sp_n = neighbour_spacing(a, nxt)
    sp_p = neighbour_spacing(a, prev)
    pitch = float(np.nanmedian([sp_n.pitch, sp_p.pitch]))
    status = np.full((H, W), R_NOT_ASSESSED, np.int8)
    shift = np.full((H, W), np.nan)
    shift[a.valid] = 0.0
    n = grid_normals(a.xyz)
    # the neighbours' positions are read by projecting their nearest points onto the normal; that holds even where
    # the spacing check left a node unassessed (its nearest point on the neighbour's border), as long as the nearest
    # point is not far to the side (the neighbour has no coverage there)
    side_max = max(2.0 * a.grid_step, 2.0 * float(np.nanmedian([sp_n.pitch, sp_p.pitch])))
    ok = a.valid & np.isfinite(n).all(-1) & np.isfinite(sp_n.distance) & np.isfinite(sp_p.distance)
    for q in (sp_n.nearest, sp_p.nearest):
        d = np.where(ok[..., None], q - a.xyz, 0.0)
        along = (d * np.nan_to_num(n)).sum(-1)
        side = np.linalg.norm(d - along[..., None] * np.nan_to_num(n), axis=-1)
        ok &= side <= side_max
    rc = np.stack(np.nonzero(ok), -1)
    out = Winding(a.name, a.path, a.xyz.copy(), dict(a.meta))
    if len(rc) == 0:
        return RepairResult(out, shift, status, pitch)
    p = a.xyz[ok]
    qn, qp = sp_n.nearest[ok], sp_p.nearest[ok]
    nv = n[ok]
    sgn = np.sign(((qn - qp) * nv).sum(1))                  # orient the normal from k-1 towards k+1, per node
    sgn[sgn == 0] = 1.0
    nv = nv * sgn[:, None]
    s_next = ((qn - p) * nv).sum(1)
    s_prev = ((qp - p) * nv).sum(1)
    S = search_frac * pitch
    t = np.arange(-np.ceil(S / step), np.ceil(S / step) + 1) * step
    M = len(t)
    i0 = int(np.argmin(np.abs(t)))
    vals = sample_volume(surface, (p[:, None, :] + t[None, :, None] * nv[:, None, :]).reshape(-1, 3)).reshape(-1, M)
    prof = vals >= threshold
    starts = prof & ~np.concatenate([np.zeros((len(p), 1), bool), prof[:, :-1]], 1)
    runid = np.where(prof, np.cumsum(starts, 1), 0).astype(np.int32)
    w = int(np.ceil(node_tol / step))
    idx = lambda s: np.where(np.abs(s) <= S, np.rint((s - t[0]) / step).astype(int), -1)
    own = _run_id_at(runid, np.full(len(p), i0), w)
    r_prev = _run_id_at(runid, idx(s_prev), w)
    r_next = _run_id_at(runid, idx(s_next), w)
    kept = (own > 0) & (own != r_prev) & (own != r_next)
    # on a run shared with a neighbour about a pitch away: the prediction joins two wraps (fused); not a placement
    # this repair can judge, so the node is kept
    half = 0.5 * pitch
    shared_far = ((own > 0) & ~kept & ~((own == r_prev) & (np.abs(s_prev) < half))
                  & ~((own == r_next) & (np.abs(s_next) < half)))
    st = np.full(len(p), R_KEPT, np.int8)
    st[shared_far] = R_SHARED
    sh = np.zeros(len(p))
    max_shift = max_shift_frac * pitch
    for i in np.nonzero(~kept & ~shared_far)[0]:
        ids = runid[i]
        if ids.max() == 0:
            st[i] = R_UNREPAIRED
            continue
        k = np.arange(1, ids.max() + 1)
        first = np.array([t[np.argmax(ids == j)] for j in k])
        last = np.array([t[M - 1 - np.argmax(ids[::-1] == j)] for j in k])
        centre = 0.5 * (first + last)
        lo = last[r_prev[i] - 1] if r_prev[i] > 0 else s_prev[i] + 0.5 * pitch
        hi = first[r_next[i] - 1] if r_next[i] > 0 else s_next[i] - 0.5 * pitch
        cand = (centre > lo) & (centre < hi) & (np.abs(centre) <= max_shift)
        if own[i] > 0:
            cand &= k != own[i]
        if not cand.any():
            st[i] = R_UNREPAIRED
            continue
        j = np.argmin(np.where(cand, np.abs(centre), np.inf))
        st[i] = R_MOVED
        sh[i] = centre[j]
    # a large move must agree with the moves around it, or it is a spike: undo it
    status[ok] = st
    shift[ok] = sh
    jump = jump_frac * pitch
    moved = status == R_MOVED
    big = moved & (np.abs(np.nan_to_num(shift)) > jump)
    if big.any():
        pad_s = np.pad(np.where(moved, shift, np.nan), 1, constant_values=np.nan)
        agree = np.zeros((H, W), int)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = pad_s[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
                agree += (np.abs(nb - shift) <= jump).astype(int)
        reject = big & (agree < 2)
        status[reject] = R_REJECTED
        shift[reject] = 0.0
    rc_m = np.nonzero(status == R_MOVED)
    n_or = np.zeros_like(a.xyz)
    n_or[ok] = nv
    out.xyz[rc_m] = a.xyz[rc_m] + shift[rc_m][:, None] * n_or[rc_m]
    v = a.valid
    stats = {"n": int(v.sum()), "assessed": int(ok.sum()), "pitch": pitch}
    for code, name in REPAIR_STATUS_NAMES.items():
        stats[name] = int((status[v] == code).sum())
    stats["median_abs_shift_moved"] = float(np.median(np.abs(shift[status == R_MOVED]))) if stats["moved"] else 0.0
    before, after = edge_stretch(a.xyz, a.grid_step), edge_stretch(out.xyz, a.grid_step)
    stats.update({f"before_{k}": x for k, x in before.items() if k != "edges"})
    stats.update({f"after_{k}": x for k, x in after.items() if k != "edges"})
    out.meta["repair"] = {"tool": "winding-audit repair", "node_tol": node_tol, "max_shift_frac": max_shift_frac,
                          "jump_frac": jump_frac, "threshold": threshold, "pitch": pitch,
                          "moved": stats["moved"], "unrepaired": stats["unrepaired"]}
    return RepairResult(out, shift, status, pitch, stats)


def repair_band(windings: list[Winding], surface, **kw) -> list[RepairResult | None]:
    """Repair every winding that has both neighbours, from the inside out (each against its repaired inner
    neighbour). The first and last winding are returned unchanged (None)."""
    out: list[RepairResult | None] = [None] * len(windings)
    current = list(windings)
    for i in range(1, len(windings) - 1):
        res = repair_winding(windings[i], current[i - 1], windings[i + 1], surface, **kw)
        out[i] = res
        current[i] = res.winding
    return out
