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


def fuse_sheets(vol, w, t0, t1, r0=20.0, pitch=10.0, centre=None):
    """Fill the gap between sheet w and sheet w+1 for theta in [t0, t1): the prediction fuses two wraps there."""
    Z, Y, X = vol.shape
    cy, cx = centre or ((Y - 1) / 2, (X - 1) / 2)
    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float64)
    r = np.hypot(xx - cx, yy - cy)
    theta = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    u = (r - r0) / pitch - theta / (2 * np.pi)
    fill = (u >= w) & (u <= w + 1) & (theta >= t0) & (theta < t1)
    vol[:, fill] = 200
    return vol


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


def inject_offset(windings, thetas, w, t0, t1, pitch, centre, d_pitch=0.15):
    """Nodes slightly off the sheet centre (still within the +-1 voxel tolerance of the sheet edge)."""
    windings[w] = _shift(windings[w], thetas, t0, t1, d_pitch, pitch, centre)


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


def sheet_radii(theta, r0=12.0, pitch=10.0, n_sheets=8, touch=None, shifts=()):
    """Radius of each sheet at angle ``theta`` (array), (n_sheets, len(theta)), with the faults of ``ct_spiral``:
    touch = (k, t0, t1, frac) pushes sheet k frac pitches out over [t0, t1); each (k, t0, t1, dvox) in ``shifts``
    moves sheet k by dvox voxels over [t0, t1)."""
    theta = np.asarray(theta, np.float64)
    R = r0 + pitch * (np.arange(n_sheets)[:, None] + theta[None, :] / (2 * np.pi))
    if touch is not None:
        k, t0, t1, frac = touch
        sel = (theta >= t0) & (theta < t1)
        R[k, sel] += frac * pitch
    for k, t0, t1, dv in shifts:
        sel = (theta >= t0) & (theta < t1)
        R[k, sel] += dv
    return R


def ct_spiral(shape_zyx=(16, 224, 224), r0=12.0, pitch=10.0, n_sheets=8, sigma=1.3, air=45.0, amp=110.0,
              noise=5.0, touch=(4, 1.0, 1.6, 0.7), ply=(2, 3.0, 3.6, 2.5), seed=0, centre=None, shifts=(),
              sector_amp=None, outer_amp=None, masked=True):
    """A raw-CT stand-in: Archimedean spiral sheets as Gaussian ridges (sigma voxels) of ``amp`` above ``air``, with
    noise; when ``masked``, exactly 0 outside the scroll (half a pitch beyond the last sheet), else air outside too.

    touch = (k, t0, t1, frac): over theta in [t0, t1) sheet k sits frac pitches out, so it nearly touches sheet
    k+1 and the two read as one peak. ply = (k, t0, t1, half): over [t0, t1) sheet k is two plies at +-half
    voxels. shifts: (k, t0, t1, dvox) moves sheet k. sector_amp = (t0, t1, a): every sheet has amplitude a over
    [t0, t1). outer_amp = (k0, a): sheets k0 and beyond have amplitude a. Returns (uint8 volume, (cy, cx))."""
    Z, Y, X = shape_zyx
    cy, cx = centre or ((Y - 1) / 2, (X - 1) / 2)
    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float64)
    r = np.hypot(xx - cx, yy - cy)
    theta = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    R = sheet_radii(theta.ravel(), r0, pitch, n_sheets, touch, shifts).reshape(n_sheets, Y, X)
    plane = np.full((Y, X), air)
    for k in range(n_sheets):
        d = r - R[k]
        a_k = np.full((Y, X), float(amp))
        if outer_amp is not None and k >= outer_amp[0]:
            a_k[:] = outer_amp[1]
        if sector_amp is not None:
            a_k[(theta >= sector_amp[0]) & (theta < sector_amp[1])] = sector_amp[2]
        if ply is not None and k == ply[0]:
            sel = (theta >= ply[1]) & (theta < ply[2])
            h = ply[3]
            two = 0.8 * a_k * (np.exp(-(d - h) ** 2 / (2 * sigma ** 2)) + np.exp(-(d + h) ** 2 / (2 * sigma ** 2)))
            one = a_k * np.exp(-d ** 2 / (2 * sigma ** 2))
            plane += np.where(sel, two, one)
        else:
            plane += a_k * np.exp(-d ** 2 / (2 * sigma ** 2))
    outside = r > R[-1] + 0.5 * pitch
    rng = np.random.default_rng(seed)
    vol = plane[None] + rng.normal(0.0, noise, (Z, Y, X))
    vol = np.clip(vol, 1, 255)
    if masked:
        vol[:, outside] = 0
    return vol.astype(np.uint8), (cy, cx)


def true_sheet(zyx, centre, r0=12.0, pitch=10.0, n_sheets=8, touch=(4, 1.0, 1.6, 0.7), shifts=()):
    """For points (N, 3) zyx of ``ct_spiral``: index of the nearest sheet and the distance to it (voxels)."""
    cy, cx = centre
    zyx = np.asarray(zyx, np.float64)
    dy, dx = zyx[:, 1] - cy, zyx[:, 2] - cx
    r = np.hypot(dx, dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    R = sheet_radii(theta, r0, pitch, n_sheets, touch, shifts)
    d = np.abs(R - r[None, :])
    k = np.argmin(d, 0)
    return k, d[k, np.arange(len(k))]


def write_chunk_cache(vol, out_dir, chunk=32, skip_empty=False):
    """Write ``vol`` as a raw chunk cache (``<z>_<y>_<x>.bin``, each chunk padded to chunk^3), the layout
    ``volumes.ChunkCacheArray`` reads. A missing chunk reads as unknown there, so all-zero chunks are written
    unless ``skip_empty``."""
    os.makedirs(out_dir, exist_ok=True)
    Z, Y, X = vol.shape
    for kz in range(0, Z, chunk):
        for ky in range(0, Y, chunk):
            for kx in range(0, X, chunk):
                blk = vol[kz:kz + chunk, ky:ky + chunk, kx:kx + chunk]
                if skip_empty and not blk.any():
                    continue
                pad = np.zeros((chunk, chunk, chunk), vol.dtype)
                pad[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
                pad.tofile(os.path.join(out_dir, f"{kz // chunk}_{ky // chunk}_{kx // chunk}.bin"))
    return out_dir


def write_umbilicus(path, centre, z0=0, z1=16):
    cy, cx = centre
    json.dump({"control_points": [{"x": cx, "y": cy, "z": z, "score": 100} for z in (z0, z1)]}, open(path, "w"))
    return path
