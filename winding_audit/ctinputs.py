"""Winding constraints from the raw CT, in the formats villa's spiral fitter reads.

``fit_spiral``'s default dense-spacing mode (``dense_spacing_mode: "winding_model"``) needs two inputs in the dataset:

* ``winding_inference/``: rays through the scroll with the sheet crossings along each ray and an integer level per
  crossing. The fitter samples pairs of crossings on one ray (adjacent, and 3 to 15 apart) and pulls the fitted
  winding difference between the two points towards the level difference.
* ``outer_shell/``: a tifxyz surface on the outside of the scroll. The fitter bins it by z and angle around the
  umbilicus and drops winding pairs that fall outside it.

The published neural store exists for PHerc. Paris 4 only. This module builds both inputs on the CPU from the masked
raw CT and the umbilicus, for any scroll:

* Rays run radially outward from the umbilicus in horizontal slices (every ``z_step`` slices, ``n_angles`` angles).
  Along each ray the CT is averaged in a small tube (``tube``: half-widths across the ray in the slice and in z),
  smoothed, and standardised over a sliding window; a crossing is a local maximum with prominence >= ``prominence``
  local standard deviations that is also brighter than air + ``floor_frac`` x (papyrus - air), from the 10th and
  90th percentiles inside the scroll (the rule calibrated on the PHerc. 343 pair that won First Letters).
* A peak count is not a winding count where sheets touch (two sheets read as one peak) or a sheet splits into plies
  (one sheet reads as two). Such a fault shows as a gap out of line with the local spacing (running median of
  ``pitch_window`` gaps, a full window at the ray's ends too; on a ray with fewer gaps, the slice's spacing at the
  same radius), or as two consecutive gaps whose sum is out of line (a fused pair midway between its neighbours, a
  wide ply split). The ray is cut there, and wherever the CT is missing or the ray leaves the volume. Each piece,
  after dropping ``edge_trim`` crossings at each end, is stored with levels 0, 1, 2, ...; pieces shorter than
  ``min_crossings`` are dropped, longer ones are stored as overlapping windows of ``window_crossings`` (villa draws
  a ray first, then a pair on it; windows bound how much more a crossing of a short piece weighs). The store trades
  coverage for precision; villa's store has no confidence field, which is why ambiguity is handled by cutting. The
  brightness floor is pooled over each slice and relaxed per ray only down to halfway from the ray's air level.
* The outer shell is the scroll's outside where the CT shows it: the mask edge of a masked CT (where its exact
  zeros begin), else the outermost crossing with the CT near air beyond it. It is filled in where the outside is not
  seen, and never lies inside the outermost stored crossing nearby: with the outer-shell loss off, villa uses the
  shell to drop winding pairs outside it, so a hole would only throw supervision away. villa also drops tracks
  outside the shell when a tracks store is used, so ``measured.tif`` marks where the outside was seen and the report
  gives the share.
* The CT may be a coarser pyramid level (ct_scale fit voxels per CT voxel); everything written is in the fit's
  frame.

Rays point outward from the umbilicus by construction, so levels grow with the ray parameter (villa PR #2001 is about
stores whose rays point inward). ``seed_winding`` is written as -1: the fitter does not read it, and a CT ray has no
seed winding.

Limits: a radial ray that crosses a fold meets one sheet more than once; that is not caught unless the gaps turn
irregular there. Pieces are short where the scroll is crushed. Check the store against an independent source before
fitting with it (``check_store`` checks the format only).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter, median_filter, uniform_filter1d
from scipy.signal import find_peaks

ARTIFACT_TYPE = "winding_inference_crossings"
FORMAT_VERSION = 1
_ARRAYS = ("ray_origin_zyx", "ray_step_zyx", "seed_winding", "crossing_t", "crossing_level", "crossing_offsets")


# ----------------------------------------------------------------------------------------------------- umbilicus

def load_umbilicus(path: str, coordinate_scale: float = 1.0):
    """z -> (y, x) from an umbilicus.json with ``control_points`` [{x, y, z}, ...] (the format villa's
    ``json_umbilicus_z_to_yx`` reads), linear between points and extrapolated beyond them like villa's."""
    with open(path) as f:
        data = json.load(f)
    pts = sorted(data["control_points"], key=lambda p: p["z"])
    z = np.array([p["z"] for p in pts], np.float64) * coordinate_scale
    yx = np.array([(p["y"], p["x"]) for p in pts], np.float64) * coordinate_scale

    def at(zq):
        zq = np.atleast_1d(np.asarray(zq, np.float64))
        out = np.empty((len(zq), 2))
        for k in range(2):
            out[:, k] = np.interp(zq, z, yx[:, k])
            lo, hi = zq < z[0], zq > z[-1]
            if len(z) > 1:
                out[lo, k] = yx[0, k] + (zq[lo] - z[0]) * (yx[1, k] - yx[0, k]) / (z[1] - z[0])
                out[hi, k] = yx[-1, k] + (zq[hi] - z[-1]) * (yx[-1, k] - yx[-2, k]) / (z[-1] - z[-2])
        return out

    return at


# ------------------------------------------------------------------------------------------------------- sampling

class ChunkLRU:
    """Decoded chunks kept between ``sample_points`` calls (least recently used dropped beyond ``capacity``), so the
    slices of one chunk row, cast one after another through the same wedge, read each chunk once."""

    def __init__(self, capacity: int = 160):
        self.capacity = int(capacity)
        self._d = OrderedDict()
        self.reads = 0

    def get(self, key, load):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        block = load()
        self.reads += 1
        self._d[key] = block
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)
        return block


def sample_points(arr, zyx: np.ndarray, cache: ChunkLRU | None = None) -> np.ndarray:
    """Nearest-voxel values of ``arr`` at integer (z, y, x) points, as float32. NaN outside the volume and, for a
    chunk cache with missing chunks (``_present``), inside chunks that are not there. Within one call every chunk is
    read once; with ``cache``, across calls too."""
    out = np.full(len(zyx), np.nan, np.float32)
    shape = np.array(arr.shape)
    ok = np.all((zyx >= 0) & (zyx < shape), 1)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return out
    chunks = np.array(getattr(arr, "chunks", None) or arr.shape)
    p = zyx[idx]
    k = p // chunks
    nk = (shape + chunks - 1) // chunks
    key = (k[:, 0] * nk[1] + k[:, 1]) * nk[2] + k[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    ends = np.r_[starts[1:], len(ks)]
    present = getattr(arr, "_present", None)
    for s, e in zip(starts, ends):
        sel = order[s:e]
        kk = k[sel[0]]
        tk = tuple(int(v) for v in kk)
        if present is not None and tk not in present:
            continue
        lo = kk * chunks
        hi = np.minimum(lo + chunks, shape)

        def load(lo=lo, hi=hi):
            return np.asarray(arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])

        block = cache.get(tk, load) if cache is not None else load()
        q = p[sel] - lo
        out[idx[sel]] = block[q[:, 0], q[:, 1], q[:, 2]]
    return out


def sample_rays(arr, z: np.ndarray, uyx: np.ndarray, theta: np.ndarray, r: np.ndarray, tube=(1, 1),
                cache: ChunkLRU | None = None) -> np.ndarray:
    """CT along horizontal rays (z_i, umbilicus uyx_i, angle theta_i) at radii ``r``, averaged over a tube of
    half-widths ``tube`` = (across the ray in the slice, in z) voxels; NaN where any tube sample is missing. All
    tube offsets go through one ``sample_points`` call."""
    a, b = int(tube[0]), int(tube[1])
    dy, dx = np.sin(theta), np.cos(theta)                   # outward direction in (y, x)
    ty, tx = dx, -dy                                        # across the ray, in the slice
    offs = [(u, v) for u in range(-a, a + 1) for v in range(-b, b + 1)]
    n, m = len(z), len(r)
    pts = np.empty((len(offs), n, m, 3), np.int64)
    for k, (u, v) in enumerate(offs):
        pts[k, :, :, 0] = np.rint(z + v)[:, None]
        pts[k, :, :, 1] = np.rint(uyx[:, 0, None] + r[None, :] * dy[:, None] + u * ty[:, None])
        pts[k, :, :, 2] = np.rint(uyx[:, 1, None] + r[None, :] * dx[:, None] + u * tx[:, None])
    vals = sample_points(arr, pts.reshape(-1, 3), cache).reshape(len(offs), n, m)
    return vals.mean(0)


# ------------------------------------------------------------------------------------------------------ crossings

@dataclass
class RayParams:
    """Lengths are in voxels of the fit's frame; ``generate`` converts them for a CT on a coarser grid."""
    step: float = 1.0            # sample spacing along the ray, in CT voxels
    r_min: float = 8.0           # first radius sampled (the umbilicus itself is not papyrus)
    r_max: float = 4000.0        # last radius sampled
    tube: tuple = (1, 1)         # tube half-widths (across the ray in the slice, in z), in CT voxels
    smooth: float = 1.0          # Gaussian smoothing along the ray
    window: float = 40.0         # standardisation window (about 2.5 sheet spacings at 8-9 um)
    prominence: float = 1.0      # local standard deviations
    floor_frac: float = 0.35     # brighter than air + floor_frac * (papyrus - air)
    min_sep: float = 3.0         # minimum distance between crossings
    pitch_window: int = 9        # gaps in the running median that sets the local spacing
    gap_lo: float = 0.6          # one gap below gap_lo x local spacing (a split ply) cuts the ray
    gap_hi: float = 1.5          # one gap above gap_hi x local spacing (a missed or fused sheet) cuts the ray
    pair_lo: float = 0.8         # two consecutive gaps summing below pair_lo x their local spacing cut both
    pair_hi: float = 1.25        # ... or above pair_hi x (a fault shared by two gaps)
    edge_trim: int = 1           # crossings dropped at each end of a piece
    min_crossings: int = 4       # shortest piece kept (villa's relative pairs need >= 4 for a 3-crossing span)
    window_crossings: int = 16   # pieces longer than this are stored as overlapping windows of this many crossings
    window_stride: int = 8       # (villa draws a ray first, then a pair on it; windows bound the imbalance)
    shell_clear: float = 32.0    # without a mask edge, the outermost crossing is the scroll's outside only if the CT
                                 # is present and dark (near air) for this far beyond it
    shell_margin: float = 4.0    # the shell lies at least this far outside the outermost crossing

    def in_ct_voxels(self, ct_scale: float) -> "RayParams":
        """The same parameters for a CT whose voxel is ``ct_scale`` fit voxels (step and tube stay in CT voxels)."""
        s = float(ct_scale)
        out = replace(self)
        for f in ("r_min", "r_max", "smooth", "window", "min_sep", "shell_clear", "shell_margin"):
            setattr(out, f, getattr(self, f) / s)
        return out


def smoothed(prof: np.ndarray, prm: RayParams):
    """Smoothed profile(s) along the last axis (NaN read as 0) and the valid mask."""
    valid = np.isfinite(prof)
    filled = np.where(valid, prof, 0.0).astype(np.float64)
    return gaussian_filter1d(filled, prm.smooth / prm.step, axis=-1), valid, filled


def slice_floor(prof: np.ndarray, prm: RayParams):
    """Brightness floor pooled over all rays of one slice: air + floor_frac x (papyrus - air), from the 10th and 90th
    percentiles of the smoothed CT inside the scroll. Each ray then uses min(its own floor, this one), but never less
    than halfway from its own air level to this one (``ray_floor``): on a ray that is mostly air inside the mask
    (flaked outer layers, voids) its own 90th percentile is still air and its own floor falls into the noise, while
    a fainter sector of the slice keeps a floor it can pass. None when the slice has too little CT inside."""
    sm, valid, filled = smoothed(prof, prm)
    inside = valid & (filled > 0.5)                      # masked volumes are exactly 0 outside the scroll
    if inside.sum() < 50:
        return None
    air, pap = np.percentile(sm[inside], [10, 90])
    return float(air + prm.floor_frac * (pap - air))


def ray_floor(air_ray: float, own: float, pooled: float | None) -> float:
    if pooled is None:
        return own
    return max(min(own, pooled), air_ray + 0.5 * (pooled - air_ray))


def local_spacing(gaps: np.ndarray, k: int) -> np.ndarray:
    """Running median of ``k`` gaps. At the ends the window stays k gaps wide (the first and last k gaps), so an end
    gap is never measured mostly against itself; a run shorter than k uses its overall median (``generate`` gives
    such short rays the slice's spacing at the same radius instead)."""
    n = len(gaps)
    k = max(1, int(k))
    k += (k + 1) % 2
    if n <= k:
        return np.full(n, float(np.median(gaps)))
    h = k // 2
    loc = median_filter(gaps, size=k, mode="nearest")
    loc[:h] = np.median(gaps[:k])
    loc[n - h:] = np.median(gaps[-k:])
    return loc


@dataclass
class Detection:
    pos: np.ndarray              # crossing positions along the ray
    gap_ok: np.ndarray           # per gap: no missing CT between the two crossings
    outer: float                 # the scroll's outside along this ray, NaN if not seen


_EMPTY = Detection(np.zeros(0), np.zeros(0, bool), np.nan)


def detect_crossings(prof: np.ndarray, t: np.ndarray, prm: RayParams, floor: float | None = None) -> Detection:
    """Crossings along one ray profile (NaN = no CT) and the scroll's outside along it.

    The outside is the mask edge where the CT is masked (exact zeros from the last inside sample to the end of the
    present CT, at least 4 voxels of them), else the outermost crossing + shell_margin where the CT beyond it is
    present and near air (below halfway from air to the floor) for shell_clear; never inside the outermost crossing
    + shell_margin. NaN when neither can be seen (missing CT or the volume's edge comes first)."""
    sm, valid, filled = smoothed(prof, prm)
    if valid.sum() < 10:
        return _EMPTY
    inside = valid & (filled > 0.5)
    if inside.sum() < 10:
        return _EMPTY
    air, pap = np.percentile(sm[inside], [10, 90])
    fl = ray_floor(air, air + prm.floor_frac * (pap - air), floor)
    win = max(5, int(round(prm.window / prm.step)))
    mu = uniform_filter1d(sm, win, mode="nearest")
    sd = np.sqrt(np.maximum(uniform_filter1d((sm - mu) ** 2, win, mode="nearest"), 1e-6))
    zsc = (sm - mu) / sd
    pk = find_peaks(zsc, prominence=prm.prominence, distance=max(1, int(round(prm.min_sep / prm.step))))[0]
    pk = pk[(sm[pk] >= fl) & valid[pk]]
    if len(pk) == 0:
        return _EMPTY
    i = np.clip(pk, 1, len(sm) - 2)
    y0, y1, y2 = sm[i - 1], sm[i], sm[i + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    pos = t[i] + np.clip(d, -0.5, 0.5) * prm.step
    last = pk[-1]
    miss = np.nonzero(~valid[last:])[0]
    end = last + (miss[0] if len(miss) else len(valid) - last)
    outer = np.nan
    ins = np.nonzero(inside[last:end])[0]
    li = last + ins[-1]                                  # last inside sample before missing CT / the volume's edge
    if (end - li - 1) * prm.step >= 4.0:                 # masked: exact zeros from there on
        outer = float(t[li]) + prm.step
    else:                                                # unmasked: the CT must stay near air for shell_clear
        dark = sm[last:end] < 0.5 * (air + fl)
        if dark.any():
            fb = last + int(np.argmax(dark))
            if (end - fb) * prm.step >= prm.shell_clear and dark[fb - last:].all():
                outer = float(pos[-1])
    if np.isfinite(outer):
        outer = max(outer, float(pos[-1]) + prm.shell_margin)
    nan_before = np.cumsum(~valid)
    gap_ok = nan_before[pk[1:]] == nan_before[pk[:-1]]
    return Detection(pos, gap_ok, outer)


def classify_gaps(pos: np.ndarray, gap_ok: np.ndarray, prm: RayParams, ref=None) -> np.ndarray:
    """Per gap, whether the two crossings are consecutive sheets as far as the CT can tell: the gap and every pair of
    consecutive gaps in line with the local spacing, and no missing CT between them. ``ref(midpoints)`` gives the
    slice's spacing at those radii; it replaces the ray's own median on rays of pitch_window gaps or fewer, where a
    fault makes up a large part of the sample."""
    gaps = np.diff(pos)
    if len(gaps) == 0:
        return np.zeros(0, bool)
    if len(gaps) <= prm.pitch_window and ref is not None:
        local = np.asarray(ref(0.5 * (pos[1:] + pos[:-1])), np.float64)
    else:
        local = local_spacing(gaps, min(prm.pitch_window, len(gaps)))
    regular = (gaps >= prm.gap_lo * local) & (gaps <= prm.gap_hi * local)
    if len(gaps) >= 2:
        two, lt = gaps[:-1] + gaps[1:], local[:-1] + local[1:]
        bad = (two < prm.pair_lo * lt) | (two > prm.pair_hi * lt)
        regular[:-1] &= ~bad
        regular[1:] &= ~bad
    return regular & gap_ok


def crossings_on_profile(prof: np.ndarray, t: np.ndarray, prm: RayParams, floor: float | None = None, ref=None):
    """``detect_crossings`` + ``classify_gaps``: (positions, regular per gap, outside or NaN)."""
    det = detect_crossings(prof, t, prm, floor)
    return det.pos, classify_gaps(det.pos, det.gap_ok, prm, ref), det.outer


def spacing_reference(dets, prm: RayParams):
    """The slice's spacing at a given radius: median of all its rays' gaps whose midpoints lie within
    +-0.5 x pitch_window x median gap of the query (the slice median where fewer than 5 do). None without gaps."""
    mids, gaps = [], []
    for d in dets:
        if len(d.pos) > 1:
            g = np.diff(d.pos)
            mids.append(0.5 * (d.pos[1:] + d.pos[:-1]))
            gaps.append(g)
    if not gaps:
        return None
    mids, gaps = np.concatenate(mids), np.concatenate(gaps)
    o = np.argsort(mids)
    mids, gaps = mids[o], gaps[o]
    med = float(np.median(gaps))
    half = 0.5 * prm.pitch_window * med

    def ref(q):
        q = np.atleast_1d(q)
        lo = np.searchsorted(mids, q - half)
        hi = np.searchsorted(mids, q + half, side="right")
        return np.array([np.median(gaps[a:b]) if b - a >= 5 else med for a, b in zip(lo, hi)])

    return ref


def pieces(pos: np.ndarray, regular: np.ndarray, prm: RayParams):
    """Runs of crossings joined by regular gaps, with ``edge_trim`` crossings dropped at each end; runs shorter than
    max(min_crossings, 2) are dropped. A list of position arrays."""
    out = []
    if len(pos) == 0:
        return out
    cut = np.nonzero(~regular)[0] + 1
    for run in np.split(pos, cut):
        if prm.edge_trim > 0:
            run = run[prm.edge_trim:len(run) - prm.edge_trim]
        if len(run) >= max(prm.min_crossings, 2):
            out.append(run)
    return out


def windows(run: np.ndarray, size: int, stride: int):
    """Overlapping windows of at most ``size`` crossings (every ``stride``, the last one ending at the run's end).
    villa draws a ray, then a pair on it: windows bound how much more a crossing of a short piece weighs than one of
    a long piece (about size / min_crossings), they do not make the weights equal."""
    if size <= 0 or len(run) <= size:
        return [run]
    starts = list(range(0, len(run) - size + 1, max(1, stride)))
    if starts[-1] != len(run) - size:
        starts.append(len(run) - size)
    return [run[s:s + size] for s in starts]


# --------------------------------------------------------------------------------------------------- generation

class Rays:
    """Stored rays kept as flat arrays (origin, step, crossing positions, offsets) rather than one Python object per
    ray. Iterating yields (origin_zyx, step_zyx, crossing_t) per ray."""

    def __init__(self):
        self._o, self._s, self._t, self._n = [], [], [], []
        self._final = None

    def add(self, origins, steps, ts):
        if not len(ts):
            return
        self._o.append(np.asarray(origins, np.float32).reshape(-1, 3))
        self._s.append(np.asarray(steps, np.float32).reshape(-1, 3))
        self._n.append(np.array([len(t) for t in ts], np.int64))
        self._t.append(np.concatenate(ts).astype(np.float32))
        self._final = None

    def arrays(self):
        if self._final is None:
            n = np.concatenate(self._n) if self._n else np.zeros(0, np.int64)
            off = np.zeros(len(n) + 1, np.int64)
            off[1:] = np.cumsum(n)
            self._final = (np.concatenate(self._o) if self._o else np.zeros((0, 3), np.float32),
                           np.concatenate(self._s) if self._s else np.zeros((0, 3), np.float32),
                           np.concatenate(self._t) if self._t else np.zeros(0, np.float32), off)
        return self._final

    @classmethod
    def from_list(cls, rays):
        r = cls()
        if len(rays):
            r.add([x[0] for x in rays], [x[1] for x in rays], [np.asarray(x[2], np.float32) for x in rays])
        return r

    def __len__(self):
        return len(self.arrays()[3]) - 1

    def __getitem__(self, i):
        o, s, t, off = self.arrays()
        return o[i], s[i], t[off[i]:off[i + 1]]

    def __iter__(self):
        o, s, t, off = self.arrays()
        for i in range(len(o)):
            yield o[i], s[i], t[off[i]:off[i + 1]]


def _slice_positions(zs_fit, S, cz, nz, tube_z):
    """CT slice per requested fit slice: rounded, then nudged so the tube's z layers stay in one chunk row (a tube
    reaching into the previous row would read that row's chunks a second time). Duplicates are dropped."""
    z_ct = np.rint(np.asarray(zs_fit, np.float64) / S).astype(np.int64)
    if nz > cz and tube_z > 0:
        m = z_ct % cz
        z_ct = np.where(m < tube_z, z_ct + (tube_z - m), z_ct)
        m = z_ct % cz
        z_ct = np.where(m > cz - 1 - tube_z, z_ct - (m - (cz - 1 - tube_z)), z_ct)
    z_ct = np.clip(z_ct, 0, nz - 1)
    _, first = np.unique(z_ct, return_index=True)
    return z_ct[np.sort(first)]


def generate(ct, umbilicus, z0: int, z1: int, z_step: int = 8, n_angles: int = 720, prm: RayParams | None = None,
             ct_scale: float = 1.0, max_points: int = 1_500_000, cache_chunks: int = 128,
             prof_budget_bytes: int = 512 << 20, progress=None):
    """Cast the rays for slices z0, z0 + z_step, ... < z1 (fit frame) and return
    (rays, outer, inner_bound, z_samples, thetas, stats).

    ``ct`` may be on a coarser grid than the fit: CT voxel c is fit position c x ``ct_scale`` (villa's convention for
    coarse volumes); ``umbilicus`` and every output are in the fit frame, and z_samples are the fit z of the CT slices
    actually read. rays: ``Rays`` (origin_zyx, step_zyx, crossing_t per ray); outer: (len(z_samples), n_angles)
    radius of the scroll's outside (NaN where not seen); inner_bound: the outermost stored crossing + shell_margin
    per ray (NaN where nothing was stored), a radius the shell must not fall below.

    The CT is read one chunk row at a time; within a row every slice of a wedge of rays goes through one sampling
    call, and decoded chunks stay in an LRU between neighbouring wedges, so each chunk is read about once."""
    prm = prm or RayParams()
    S = float(ct_scale)
    pc = prm.in_ct_voxels(S)
    shape = tuple(int(v) for v in ct.shape)
    cz = int((getattr(ct, "chunks", None) or shape)[0])
    z_ct = _slice_positions(np.arange(int(z0), int(z1), int(z_step)), S, cz, shape[0], int(pc.tube[1]))
    zs = z_ct * S
    thetas = np.arange(n_angles) * (2 * np.pi / n_angles)
    r = np.arange(pc.r_min, pc.r_max + 1e-9, pc.step)
    tube_n = (2 * pc.tube[0] + 1) * (2 * pc.tube[1] + 1)
    u_fit = umbilicus(zs.astype(np.float64))
    u_ct = u_fit / S
    rows = {}
    for i, zc in enumerate(z_ct):
        rows.setdefault(int(zc) // cz, []).append(i)
    rays = Rays()
    outer = np.full((len(zs), n_angles), np.nan)
    inner = np.full((len(zs), n_angles), np.nan)
    stats = dict(rays_cast=0, crossings=0, crossings_kept=0, pieces=0, stored_rays=0, gaps=0, regular_gaps=0,
                 slices_without_floor=0, chunk_reads=0)
    spacing, lengths, floors = [], [], []
    t0, done = time.time(), 0
    for row in sorted(rows):
        cache = ChunkLRU(cache_chunks)
        idx_all = rows[row]
        per_block = max(1, int(prof_budget_bytes // max(1, n_angles * len(r) * 4)))
        for b0 in range(0, len(idx_all), per_block):
            idx = idx_all[b0:b0 + per_block]
            prof = np.full((len(idx), n_angles, len(r)), np.nan, np.float32)
            per_batch = max(1, max_points // max(1, len(r) * tube_n * len(idx)))
            # a wedge must fit in the chunk cache, or neighbouring wedges re-read the chunks they share:
            # wedge chunks ~ phi * R^2 / 2 + 2 R (R = r_max in chunk widths) <= 3/4 of the cache
            c_yx = float((getattr(ct, "chunks", None) or shape)[1])
            R_c = pc.r_max / c_yx
            room = 0.75 * cache_chunks - 2.0 * R_c
            if room > 0 and R_c > 1:
                phi = 2.0 * room / (R_c * R_c)
                per_batch = min(per_batch, max(1, int(phi / (2 * np.pi / n_angles))))
            for s in range(0, n_angles, per_batch):
                th = thetas[s:s + per_batch]
                k = len(th)
                vals = sample_rays(ct, np.repeat(z_ct[idx], k), np.repeat(u_ct[idx], k, axis=0),
                                   np.tile(th, len(idx)), r, pc.tube, cache)
                prof[:, s:s + k] = vals.reshape(len(idx), k, len(r))
            for j, i in enumerate(idx):
                floor = slice_floor(prof[j], pc)
                if floor is None:
                    stats["slices_without_floor"] += 1
                else:
                    floors.append(floor)
                dets = [detect_crossings(prof[j, a], r, pc, floor) for a in range(n_angles)]
                ref = spacing_reference(dets, pc)
                origin = np.array([zs[i], u_fit[i, 0], u_fit[i, 1]], np.float32)
                o_list, s_list, t_list = [], [], []
                for a, (ang, det) in enumerate(zip(thetas, dets)):
                    regular = classify_gaps(det.pos, det.gap_ok, pc, ref)
                    outer[i, a] = det.outer * S
                    stats["rays_cast"] += 1
                    stats["crossings"] += len(det.pos)
                    stats["gaps"] += len(regular)
                    stats["regular_gaps"] += int(regular.sum())
                    if len(det.pos) > 1:
                        spacing.append(float(np.median(np.diff(det.pos))) * S)
                    step = np.array([0.0, np.sin(ang), np.cos(ang)], np.float32)
                    for run in pieces(det.pos, regular, pc):
                        stats["pieces"] += 1
                        stats["crossings_kept"] += len(run)
                        lengths.append(len(run))
                        inner[i, a] = np.fmax(inner[i, a], (run[-1] + pc.shell_margin) * S)
                        for w in windows(run, prm.window_crossings, prm.window_stride):
                            o_list.append(origin)
                            s_list.append(step)
                            t_list.append(w * S)
                rays.add(o_list, s_list, t_list)
                stats["stored_rays"] += len(t_list)
                done += 1
                if progress:
                    progress(done, len(zs), time.time() - t0)
            del prof
        stats["chunk_reads"] += cache.reads
    stats["median_spacing"] = float(np.median(spacing)) if spacing else None
    stats["median_floor"] = float(np.median(floors)) if floors else None
    stats["share_crossings_kept"] = stats["crossings_kept"] / max(1, stats["crossings"])
    stats["share_regular_gaps"] = stats["regular_gaps"] / max(1, stats["gaps"])
    ln = np.array(lengths)
    edges = [2, 4, 8, 16, 32, 64, 10 ** 9]
    stats["piece_lengths"] = {f"{lo}-{hi - 1}" if hi < 10 ** 9 else f"{lo}+": int(((ln >= lo) & (ln < hi)).sum())
                              for lo, hi in zip(edges[:-1], edges[1:])}
    stats["rays_with_outside_seen"] = int(np.isfinite(outer).sum())
    return rays, outer, inner, zs, thetas, stats


# -------------------------------------------------------------------------------------------------- store writer

def _canonical_digest(value) -> str:
    """The fingerprint rule of villa's winding_supervision / export_spiral_supervision."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save(directory: str, name: str, value: np.ndarray) -> dict:
    path = os.path.join(directory, f"{name}.npy")
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return {"file": f"{name}.npy", "shape": list(value.shape), "dtype": np.dtype(value.dtype).str,
            "bytes": os.path.getsize(path), "sha256": _file_digest(path)}


def _identity_view(manifest: dict) -> dict:
    view = copy.deepcopy(manifest)
    view.pop("fingerprint", None)
    for key in ("elapsed_seconds", "export_workers", "rays_per_task"):
        view.pop(key, None)
    for shard in view.get("shards", []):
        shard.pop("elapsed_seconds", None)
    return view


def write_store(out_dir: str, rays, provenance: dict | None = None, rays_per_shard: int = 250_000,
                validate: bool = True) -> dict:
    """Write ``rays`` (a ``Rays``, or a list of (origin_zyx, step_zyx, crossing_t)) as a villa
    ``winding_inference_crossings`` store (format version 1). Like villa's exporter it writes to
    ``<out_dir>.partial`` and renames, and refuses to overwrite an existing store. Levels are 0, 1, 2, ... along each
    ray. Rays with fewer than 2 crossings are refused before anything is written (villa's exporter never writes one,
    and its loader can fail on one)."""
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite a store")
    if not isinstance(rays, Rays):
        rays = Rays.from_list(rays)
    o_all, s_all, t_all, off_all = rays.arrays()
    n_all = np.diff(off_all)
    if validate and np.any(n_all < 2):
        raise ValueError(f"{int((n_all < 2).sum())} rays with fewer than 2 crossings; nothing written")
    partial = out_dir + ".partial"
    if os.path.exists(partial):
        shutil.rmtree(partial)
    os.makedirs(partial)
    t0 = time.time()
    shards, total_rays, total_crossings = [], 0, 0
    for k, s0 in enumerate(range(0, max(len(n_all), 1), rays_per_shard)):
        s1 = min(s0 + rays_per_shard, len(n_all))
        name = f"shard_{k}"
        d = os.path.join(partial, name)
        os.makedirs(d)
        n = s1 - s0
        lengths = n_all[s0:s1]
        offsets = np.zeros(n + 1, np.int64)
        offsets[1:] = np.cumsum(lengths)
        t = t_all[off_all[s0]:off_all[s1]] if n else np.zeros(0, np.float32)
        arrays = {
            "ray_origin_zyx": np.ascontiguousarray(o_all[s0:s1], np.float32).reshape(n, 3),
            "ray_step_zyx": np.ascontiguousarray(s_all[s0:s1], np.float32).reshape(n, 3),
            "seed_winding": np.full(n, -1, np.int16),
            "crossing_t": np.ascontiguousarray(t, np.float32),
            "crossing_level": (np.concatenate([np.arange(m, dtype=np.int16) for m in lengths])
                               if n else np.zeros(0, np.int16)),
            "crossing_offsets": offsets,
        }
        shards.append({"name": name, "arrays": {a_: _save(d, a_, arrays[a_]) for a_ in _ARRAYS},
                       "num_source_rays": n, "num_retained_rays": n, "num_crossings": int(offsets[-1])})
        total_rays += n
        total_crossings += int(offsets[-1])
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "format_version": FORMAT_VERSION,
        "coordinate_order": "zyx",
        "coordinate_space": "scale-0 voxels of the fit's volume (rays cast in the CT, scaled to the fit's frame)",
        "num_rays": total_rays,
        "num_crossings": total_crossings,
        "shards": shards,
        "source_generator": "spiral-winding-audit ct-inputs (radial rays from the umbilicus through the raw CT)",
        "provenance": json.loads(json.dumps(provenance or {})),
    }
    manifest["fingerprint"] = _canonical_digest(_identity_view(manifest))
    manifest["elapsed_seconds"] = time.time() - t0
    with open(os.path.join(partial, "manifest.json"), "w") as f:
        f.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if validate:
        check_store(partial)
    os.replace(partial, out_dir)
    return manifest


def check_store(path: str, verify_hashes: bool = True) -> dict:
    """Format check without torch, mirroring villa's loader: fingerprint, per-array checksum, shape and dtype,
    offsets, crossings ascending along each ray, levels strictly increasing (every sampled pair has a positive
    target). Returns a summary; raises ValueError on the first problem."""
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("artifact_type") != ARTIFACT_TYPE or int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("not a winding_inference_crossings store, format version 1")
    if manifest.get("coordinate_order") != "zyx":
        raise ValueError("coordinates must be zyx")
    if manifest.get("fingerprint") != _canonical_digest(_identity_view(manifest)):
        raise ValueError("manifest fingerprint mismatch")
    rays = crossings = 0
    lengths = []
    for shard in manifest["shards"]:
        root = os.path.join(path, shard["name"])
        a = {}
        for name in _ARRAYS:
            desc = shard["arrays"][name]
            p = os.path.join(root, desc["file"])
            if verify_hashes and _file_digest(p) != desc["sha256"]:
                raise ValueError(f"checksum mismatch: {p}")
            v = np.load(p, mmap_mode="r", allow_pickle=False)
            if list(v.shape) != list(desc["shape"]) or np.dtype(v.dtype).str != desc["dtype"]:
                raise ValueError(f"shape or dtype mismatch: {p}")
            a[name] = np.asarray(v)
        n = len(a["ray_origin_zyx"])
        off = a["crossing_offsets"]
        if len(off) != n + 1 or off[0] != 0 or off[-1] != len(a["crossing_t"]) or np.any(np.diff(off) < 0):
            raise ValueError(f"crossing offsets inconsistent in {shard['name']}")
        if np.any(np.diff(off) < 2):                     # villa's exporter never writes one; its loader can fail
            raise ValueError(f"a ray with fewer than 2 crossings in {shard['name']}")
        if len(a["seed_winding"]) != n or len(a["crossing_level"]) != len(a["crossing_t"]):
            raise ValueError(f"array lengths disagree in {shard['name']}")
        same_ray = np.ones(len(a["crossing_t"]), bool)
        starts = off[:-1]
        same_ray[starts[starts < len(same_ray)]] = False  # first crossing of each ray
        if np.any(np.diff(a["crossing_t"])[same_ray[1:]] <= 0):
            raise ValueError(f"crossings not ascending along a ray in {shard['name']}")
        if np.any(np.diff(a["crossing_level"].astype(np.int32))[same_ray[1:]] <= 0):
            raise ValueError(f"levels not increasing along a ray in {shard['name']}")
        rays += n
        crossings += len(a["crossing_t"])
        lengths.append(np.diff(off))
    if rays != manifest["num_rays"] or crossings != manifest["num_crossings"]:
        raise ValueError("ray or crossing counts disagree with the manifest")
    ln = np.concatenate(lengths) if lengths else np.zeros(0)
    return {"num_rays": rays, "num_crossings": crossings, "fingerprint": manifest["fingerprint"],
            "crossings_per_ray_median": float(np.median(ln)) if len(ln) else 0.0,
            "rays_with_relative_pairs": int((ln >= 4).sum())}


# -------------------------------------------------------------------------------------------------- outer shell

def _periodic(values: np.ndarray, src_ang: np.ndarray, dst_ang: np.ndarray):
    """Periodic linear interpolation of ``values`` at ``src_ang`` (NaN = unknown) onto ``dst_ang``; also whether
    each destination angle has a known source within 2.5 source steps."""
    ok = np.isfinite(values)
    if ok.sum() == 0:
        return np.full(len(dst_ang), np.nan), np.zeros(len(dst_ang), bool)
    two_pi = 2 * np.pi
    src = src_ang[ok]
    ext_a = np.r_[src - two_pi, src, src + two_pi]
    ext_v = np.r_[values[ok], values[ok], values[ok]]
    out = np.interp(dst_ang, ext_a, ext_v)
    step = two_pi / len(src_ang)
    near = np.min(np.abs(((dst_ang[:, None] - src[None, :]) + np.pi) % two_pi - np.pi), 1) <= 2.5 * step
    return out, near


def shell_grid(outer: np.ndarray, inner: np.ndarray, zs: np.ndarray, thetas: np.ndarray, umbilicus,
               shell_angles: int = 720, pad: int = 8, hold_cols: int = 3, hold_rows: int = 12):
    """The outer shell as a tifxyz grid without holes, for use as villa's pair filter.

    Rows: every integer z from zs[0] - pad (not below 0) to zs[-1] + pad (the nearest sampled slice is held beyond
    the band, which also keeps villa's shell confidence up at the band's edges). Columns: ``shell_angles`` angles at the centres of
    villa's theta bins (keep it equal to shell_num_theta_bins). The radius is the scroll's outside where it was seen
    (``outer``), interpolated in angle and z across the places where it was not, and never less than the outermost
    stored crossing nearby (``inner``, max-filtered over +-hold_cols columns and +-hold_rows rows, about three
    standard deviations of villa's table smoothing), so no stored pair falls outside it.

    Returns (zyx (H, W, 3) with -1 where nothing is known, measured (H, W) bool: the outside was seen there)."""
    two_pi = 2 * np.pi
    ang = (np.arange(shell_angles) + 0.5) * (two_pi / shell_angles)
    zrows = np.arange(max(0, int(zs[0]) - pad), int(zs[-1]) + pad + 1)      # tifxyz has no negative z
    n = len(zs)
    rad_s = np.full((n, shell_angles), np.nan)
    near_s = np.zeros((n, shell_angles), bool)
    low_s = np.full((n, shell_angles), -np.inf)
    col_of_ray = np.floor(thetas / two_pi * shell_angles).astype(int) % shell_angles
    for i in range(n):
        rad_s[i], near_s[i] = _periodic(outer[i], thetas, ang)
        ok = np.isfinite(inner[i])
        for dc in (-1, 0, 1):                         # a ray on a bin edge belongs to both neighbouring columns
            np.maximum.at(low_s[i], (col_of_ray[ok] + dc) % shell_angles, inner[i, ok])
    # z: interpolate between sampled slices, hold the nearest beyond them
    rad = np.full((len(zrows), shell_angles), np.nan)
    low = np.full((len(zrows), shell_angles), -np.inf)
    meas = np.zeros((len(zrows), shell_angles), bool)
    i0 = np.clip(np.searchsorted(zs, zrows, side="right") - 1, 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    inside = (zrows >= zs[0]) & (zrows <= zs[-1])
    exact = zs[i0] == zrows
    for j in range(shell_angles):
        ok = np.isfinite(rad_s[:, j])
        if ok.any():
            rad[:, j] = np.interp(zrows, zs[ok], rad_s[ok, j])
        known = np.isfinite(rad_s[:, j]) & near_s[:, j]
        meas[:, j] = known[i0] & (known[i1] | exact | ~inside)      # rows beyond the band hold the nearest slice
        lj = low_s[:, j]
        okl = np.isfinite(lj)
        if okl.any():
            low[:, j] = np.interp(zrows, zs[okl], lj[okl])
    low = maximum_filter(low, size=(2 * hold_rows + 1, 2 * hold_cols + 1), mode=("nearest", "wrap"))
    low[~np.isfinite(low)] = np.nan
    final = np.fmax(rad, low)
    uyx = umbilicus(zrows.astype(np.float64))
    zyx = np.full((len(zrows), shell_angles, 3), -1.0, np.float32)
    ok = np.isfinite(final)
    zz = np.broadcast_to(zrows[:, None], final.shape)
    yy = uyx[:, 0, None] + final * np.sin(ang)[None, :]
    xx = uyx[:, 1, None] + final * np.cos(ang)[None, :]
    zyx[ok] = np.stack([zz[ok], yy[ok], xx[ok]], -1)
    return zyx, meas & ok


def has_valid_quad(zyx: np.ndarray) -> bool:
    """villa's tifxyz loader needs at least one 2x2 block of valid vertices."""
    ok = zyx[..., 0] >= 0
    return bool((ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]).any()) if min(ok.shape) >= 2 else False


def write_shell(out_dir: str, zyx: np.ndarray, measured: np.ndarray | None = None, uuid: str = "ct_outer_shell",
                provenance: dict | None = None):
    """Write a tifxyz surface (x.tif, y.tif, z.tif float32 and meta.json; -1 marks missing vertices), plus
    measured.tif (1 where the scroll's outside was seen, 0 where the radius was filled in). villa reads only the
    coordinates."""
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    for k, name in enumerate(("z", "y", "x")):
        tifffile.imwrite(os.path.join(out_dir, f"{name}.tif"), np.ascontiguousarray(zyx[..., k], np.float32))
    ok = zyx[..., 0] >= 0
    if measured is not None:
        tifffile.imwrite(os.path.join(out_dir, "measured.tif"), measured.astype(np.uint8))
    if ok.any():
        p = zyx[ok]
        bbox = [p.min(0)[::-1].tolist(), p.max(0)[::-1].tolist()]
        dcol = np.linalg.norm(np.diff(zyx, axis=1), axis=-1)[ok[:, 1:] & ok[:, :-1]]
        spacing = float(np.median(dcol)) if len(dcol) else 1.0
    else:
        bbox, spacing = None, 1.0
    share = float(measured[ok].mean()) if (measured is not None and ok.any()) else None
    meta = {"format": "tifxyz", "type": "seg", "uuid": uuid, "scale": [1.0 / max(spacing, 1e-6), 1.0],
            "bbox": bbox, "measured_share": share,
            "source": "spiral-winding-audit ct-inputs: the scroll's outside where the CT shows it (mask edge, or "
                      "the outermost crossing with near-air CT beyond), filled in elsewhere and never inside the "
                      "outermost stored crossing. Made as the pair filter of fit_spiral's winding-model losses. "
                      "Where measured.tif is 0 it may lie inside the scroll: villa then also drops tracks beyond "
                      "it when a tracks store is used, and it is not fit for the outer-shell loss.",
            "provenance": json.loads(json.dumps(provenance or {}))}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def fit_overrides(z0: int, z1: int, dataset_dir: str | None = None) -> dict:
    """Config overrides (FIT_SPIRAL_CONFIG_OVERRIDES, or runners/run_single.py --config; in a VC3D session z_begin
    and z_end go in the run block) that make fit_spiral use the CT store: winding-model dense spacing and the
    shell as a pair filter only (no outer-shell pull). shell_outer_winding_idx still has to be set for the scroll:
    it bounds the dense-normal, minimum-spacing and regularisation losses whatever the shell loss weight. Verified
    patches are switched off when the dataset has none (villa requires the folder otherwise)."""
    ov = {"z_begin": int(z0), "z_end": int(z1), "dense_spacing_mode": "winding_model",
          "input_use_winding_inference": True, "input_use_outer_shell": True, "loss_weight_shell_outer": 0}
    if dataset_dir is not None and not os.path.isdir(os.path.join(dataset_dir, "verified_patches")):
        ov["input_use_verified_patches"] = False
    return ov


def outer_winding_estimate(outer: np.ndarray, spacing: float | None):
    """Rough count of windings from the umbilicus to the outside: median outside radius / median crossing spacing.
    A guide for shell_outer_winding_idx, not a measurement (CT peaks are not windings where sheets touch)."""
    if spacing is None or not np.isfinite(outer).any():
        return None
    return int(np.ceil(np.nanmedian(outer) / spacing))


def params_dict(prm: RayParams) -> dict:
    d = asdict(prm)
    d["tube"] = list(d["tube"])
    return d
