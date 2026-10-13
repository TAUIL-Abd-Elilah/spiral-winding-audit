"""Windings per sheet along radial rays: does the fit have one winding per sheet?

In one z-slice, rays are cast from the centre of the innermost winding's slice outline. Along each ray the
crossings of every winding (its slice outline, cut by the plane z = z0) and of every predicted sheet (runs of the
thresholded prediction, runs closer than ``merge`` voxels joined) are counted between the crossings of two chosen
windings. A fit with one winding per sheet gives a ratio of 1; a fit with too many turns gives more than 1, and a
nearest-sheet snap will then put pairs of windings on one sheet.

The count does not depend on the direction of the rays relative to the sheets, unlike a spacing measured in radius
(sheets that run at an angle to the rays are farther apart in radius than across).
"""
from __future__ import annotations

import numpy as np

from .tifxyz import Winding


def slice_outline(w: Winding, z0: float) -> np.ndarray:
    """(W, 3) points where each grid column of ``w`` crosses the plane z = z0 (NaN where it does not)."""
    X = w.xyz
    Z = X[..., 2]
    out = np.full((X.shape[1], 3), np.nan)
    ok = np.isfinite(X).all(-1)
    cross = ok[:-1] & ok[1:] & ((Z[:-1] - z0) * (Z[1:] - z0) <= 0) & (Z[:-1] != Z[1:])
    for c in np.nonzero(cross.any(0))[0]:
        r = np.nonzero(cross[:, c])[0][0]
        t = (z0 - Z[r, c]) / (Z[r + 1, c] - Z[r, c])
        out[c] = X[r, c] + t * (X[r + 1, c] - X[r, c])
    return out


def ray_crossings(outline: np.ndarray, cx: float, cy: float, thetas: np.ndarray) -> np.ndarray:
    """Radius at which an outline first crosses each ray from (cx, cy) at angle theta (NaN if it does not)."""
    p = outline[np.isfinite(outline).all(1)]
    out = np.full(len(thetas), np.nan)
    if len(p) < 3:
        return out
    phi = np.unwrap(np.arctan2(p[:, 1] - cy, p[:, 0] - cx))
    rho = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    for i, th in enumerate(thetas):
        d = phi - th
        d = d - 2 * np.pi * np.round(d / (2 * np.pi))
        s = np.nonzero((d[:-1] * d[1:] <= 0) & (np.abs(d[:-1] - d[1:]) < 1.0))[0]
        if len(s):
            j = s[0]
            t = d[j] / (d[j] - d[j + 1]) if d[j] != d[j + 1] else 0.0
            out[i] = rho[j] + t * (rho[j + 1] - rho[j])
    return out


def windings_per_sheet(windings: list[Winding], surface, z0: int, first: int, last: int, inner: int | None = None,
                       n_rays: int = 180, threshold: int = 128, merge: float = 4.0) -> dict:
    """Count windings and predicted sheets between the crossings of windings ``first`` and ``last`` (indices into
    ``windings``) along ``n_rays`` rays in slice z0. ``inner`` (default 0) is the winding whose outline centre is
    the ray origin."""
    inner = 0 if inner is None else inner
    outlines = [slice_outline(w, z0) for w in windings]
    c = outlines[inner]
    cx, cy = np.nanmean(c[:, 0]), np.nanmean(c[:, 1])
    thetas = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    R = np.array([ray_crossings(o, cx, cy, thetas) for o in outlines])
    sl = np.asarray(surface[int(z0)])
    n_w, n_s, ratio = [], [], []
    for i, th in enumerate(thetas):
        ra, rb = R[first, i], R[last, i]
        if not (np.isfinite(ra) and np.isfinite(rb)) or rb <= ra:
            continue
        rr = np.arange(ra - 3, rb + 4)
        x = np.rint(cx + rr * np.cos(th)).astype(int)
        y = np.rint(cy + rr * np.sin(th)).astype(int)
        inside = (x >= 0) & (y >= 0) & (x < sl.shape[1]) & (y < sl.shape[0])
        on = np.zeros(len(rr), bool)
        on[inside] = sl[y[inside], x[inside]] >= threshold
        d = np.diff(np.pad(on.astype(np.int8), (1, 1)))
        cen = (np.nonzero(d == 1)[0] + np.nonzero(d == -1)[0] - 1) / 2.0
        if len(cen) < 2:
            continue
        sheets = 1 + int(np.sum(np.diff(cen) >= merge))
        k = int(np.sum(np.isfinite(R[:, i]) & (R[:, i] >= ra - 3) & (R[:, i] <= rb + 3)))
        n_w.append(k)
        n_s.append(sheets)
        ratio.append((k - 1) / max(sheets - 1, 1))
    if not ratio:
        return {"rays": 0}
    ratio = np.array(ratio)
    gap = (R[last] - R[first]) / max(last - first, 1)
    return {"z": int(z0), "rays": len(ratio), "windings_crossed": float(np.median(n_w)),
            "sheets_crossed": float(np.median(n_s)), "windings_per_sheet": float(np.median(ratio)),
            "windings_per_sheet_p25": float(np.percentile(ratio, 25)),
            "windings_per_sheet_p75": float(np.percentile(ratio, 75)),
            "radius_per_winding": float(np.nanmedian(gap)),
            "first": windings[first].name, "last": windings[last].name}
