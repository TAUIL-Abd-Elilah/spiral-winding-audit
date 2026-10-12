"""A synthetic scroll for tests and demos: an Archimedean spiral of sheets in a cylindrical volume, plus one tifxyz
winding per sheet, with faults injected on request.

Sheet n (n = 0, 1, ...) lies at radius r = r0 + pitch * (n + theta / 2pi), theta in [0, 2pi). The volume marks voxels
within thickness/2 of a sheet with value 200 (a stand-in for the m7 probability). Windings sample the same sheets on a
regular (z, theta) grid. Faults:
  merged:  winding ``w`` is placed on sheet w+1 for theta in [t0, t1)  (two windings on one sheet there)
  skipped: winding ``w`` is placed on sheet w+1 for ALL theta, i.e. a wrap is missing between w-1 and w+1 only if w-1
           stays: implemented as winding ``w`` moved to sheet w+1 over [t0, t1) AND winding w+1 untouched, so the pair
           (w-1, w) is fine but (w, w+1) merges; to get a true skip we instead move winding ``w`` to sheet w-1 ... see
           inject_skip: winding ``w`` over [t0, t1) is placed on sheet w+1 while winding w+1 over the same range is
           placed on sheet w+2 (both shifted outward), so the pair (w-1, w) spans two pitches there.
  gap:     winding ``w`` over [t0, t1) is placed half a pitch out (in the gap between sheets: snap failed)
"""
from __future__ import annotations

import json
import os

import numpy as np
import tifffile
import zarr


def make_volume(shape_zyx=(24, 180, 180), r0=20.0, pitch=10.0, thickness=3.0, n_sheets=6, centre=None):
    Z, Y, X = shape_zyx
    cy, cx = centre or ((Y - 1) / 2, (X - 1) / 2)
    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float64)
    r = np.hypot(xx - cx, yy - cy)
    theta = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    u = (r - r0) / pitch - theta / (2 * np.pi)            # continuous winding coordinate
    n = np.rint(u)
    on = (np.abs(u - n) * pitch < thickness / 2) & (n >= 0) & (n < n_sheets)
    plane = np.where(on, 200, 0).astype(np.uint8)
    vol = np.broadcast_to(plane, (Z, Y, X)).copy()
    return vol, (cy, cx)


def make_windings(n_sheets=6, Z=24, r0=20.0, pitch=10.0, centre=(89.5, 89.5), n_theta=120, grid_step=2.0):
    """Winding n on sheet n: grid rows = z (every grid_step voxels), columns = theta."""
    cy, cx = centre
    zs = np.arange(0.5, Z - 0.5, grid_step)
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    out = {}
    for n in range(n_sheets):
        r = r0 + pitch * (n + thetas / (2 * np.pi))
        x = cx + r * np.cos(thetas)
        y = cy + r * np.sin(thetas)
        xyz = np.empty((len(zs), n_theta, 3))
        xyz[..., 0] = x[None, :]
        xyz[..., 1] = y[None, :]
        xyz[..., 2] = zs[:, None]
        out[n] = xyz
    return out, thetas


def _shift(xyz, thetas, t0, t1, d_pitch, pitch, centre):
    """Move the nodes with theta in [t0, t1) radially by d_pitch sheet pitches."""
    cy, cx = centre
    sel = (thetas >= t0) & (thetas < t1)
    ux, uy = np.cos(thetas[sel]), np.sin(thetas[sel])
    xyz = xyz.copy()
    xyz[:, sel, 0] += d_pitch * pitch * ux[None, :]
    xyz[:, sel, 1] += d_pitch * pitch * uy[None, :]
    return xyz


def inject_merge(windings, thetas, w, t0, t1, pitch, centre):
    windings[w] = _shift(windings[w], thetas, t0, t1, +1.0, pitch, centre)


def inject_skip(windings, thetas, w, t0, t1, pitch, centre):
    windings[w] = _shift(windings[w], thetas, t0, t1, +1.0, pitch, centre)
    windings[w + 1] = _shift(windings[w + 1], thetas, t0, t1, +1.0, pitch, centre)


def inject_gap(windings, thetas, w, t0, t1, pitch, centre):
    windings[w] = _shift(windings[w], thetas, t0, t1, +0.5, pitch, centre)


def write_dataset(out_dir, vol, windings, grid_step=2.0, tag="syn"):
    """Write <out_dir>/surface.zarr (group '0') and <out_dir>/windings/w<NNN>_<tag>/ tifxyz folders."""
    os.makedirs(out_dir, exist_ok=True)
    g = zarr.open_group(os.path.join(out_dir, "surface.zarr"), mode="w", zarr_format=2)
    arr = g.create_array("0", shape=vol.shape, chunks=(16, 64, 64), dtype="u1", fill_value=0)
    arr[:] = vol
    for n, xyz in windings.items():
        d = os.path.join(out_dir, "windings", f"w{n:03d}_{tag}")
        os.makedirs(d, exist_ok=True)
        for i, c in enumerate("xyz"):
            tifffile.imwrite(os.path.join(d, f"{c}.tif"), xyz[..., i].astype(np.float32))
        json.dump({"scale": [1.0 / grid_step, 1.0 / grid_step], "uuid": f"w{n:03d}_{tag}", "format": "tifxyz",
                   "type": "seg"}, open(os.path.join(d, "meta.json"), "w"), indent=2)
    return os.path.join(out_dir, "surface.zarr"), os.path.join(out_dir, "windings")
