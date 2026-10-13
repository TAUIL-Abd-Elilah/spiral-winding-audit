"""CT layers left between consecutive windings, along radial rays in one z-slice of the raw L0 CT.

For every ray and every consecutive pair of winding crossings (r_k, r_k+1): the CT layers (smoothed, detrended
profile peaks, prominence 1 local sd) strictly between the two crossings and more than ``far`` voxels from both are
"uncovered" (no render of either winding is centred within ``far`` voxels of them). Reports, over all gaps: the share
of gaps with at least one uncovered layer, uncovered layers per gap, the share of stacked gaps (< 2 vox), and the
gap length. Windings come either as separate tifxyz folders (``--windings dir --tag t --first --last``) or as one
mesh spanning two turns (``--mesh``, split into turns by unwrapped angle). CT chunks come from the route24 cache;
with ``--fetch`` missing chunks on the rays are downloaded through route24/zfetch (cap ROUTE24_CAP_GB)."""
import sys, os, json, argparse
import numpy as np, psutil
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
sys.path.insert(0, 'D:/Competition/Vesuvius progress prizes/spiral-winding-audit')
from winding_audit.tifxyz import load_winding
from winding_audit.density import slice_outline, ray_crossings

psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
ap = argparse.ArgumentParser()
ap.add_argument('--volume', required=True, help='e.g. PHerc0343/volumes/2025...-masked.zarr')
ap.add_argument('--umbilicus', required=True)
ap.add_argument('--z', type=float, required=True)
ap.add_argument('--windings'); ap.add_argument('--tag'); ap.add_argument('--first', type=int); ap.add_argument('--last', type=int)
ap.add_argument('--mesh')
ap.add_argument('--rays', type=int, default=72)
ap.add_argument('--far', type=float, default=8.0)
ap.add_argument('--fetch', action='store_true')
ap.add_argument('--floor-frac', type=float, default=0.35)
ap.add_argument('--out', required=True)
a = ap.parse_args()
CACHE = 'D:/VesuviusCTCache/route24'
CDIR = os.path.join(CACHE, a.volume.replace('/', '__'), 'L0')
CH = 128
if a.fetch:
    sys.path.insert(0, 'D:/Competition/Vesuvius progress prizes/_fl/route24')
    import zfetch
    M = zfetch.meta(a.volume, 0)
    assert tuple(M['chunks']) == (CH, CH, CH), M['chunks']
_cache = {}


def chunk(k):
    if k in _cache:
        return _cache[k]
    if len(_cache) > 48:
        _cache.clear()
    f = os.path.join(CDIR, f'{k[0]}_{k[1]}_{k[2]}.bin')
    blk = None
    if not os.path.exists(f) and a.fetch:
        try:
            raw = zfetch.chunk(a.volume, 0, k, M)
            blk = np.asarray(raw).reshape(CH, CH, CH) if raw is not None and np.asarray(raw).size == CH ** 3 else None
        except Exception as e:
            print('fetch failed', k, e, flush=True)
    if blk is None and os.path.exists(f):
        b = np.fromfile(f, np.uint8)
        blk = b.reshape(CH, CH, CH) if b.size == CH ** 3 else None
    _cache[k] = blk
    return blk


def ct_at(xyz):
    zyx = np.rint(xyz[:, ::-1]).astype(np.int64)
    keys = zyx // CH
    out = np.full(len(xyz), np.nan, np.float32)
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    for i, k in enumerate(uk):
        blk = chunk(tuple(int(v) for v in k))
        if blk is None:
            continue
        sel = np.nonzero(inv == i)[0]
        q = zyx[sel] - k * CH
        out[sel] = blk[q[:, 0], q[:, 1], q[:, 2]]
    return out


um = json.load(open(a.umbilicus))
um = um['control_points'] if isinstance(um, dict) and 'control_points' in um else um
if isinstance(um, list) and isinstance(um[0], dict):
    zs = np.array([p['z'] for p in um], float); xs_ = np.array([p['x'] for p in um], float); ys_ = np.array([p['y'] for p in um], float)
else:
    arr = np.asarray(um, float)
    zs, ys_, xs_ = arr[:, 0], arr[:, 1], arr[:, 2]
o = np.argsort(zs)
cx = float(np.interp(a.z, zs[o], xs_[o])); cy = float(np.interp(a.z, zs[o], ys_[o]))
thetas = np.linspace(0, 2 * np.pi, a.rays, endpoint=False)
outlines = []
if a.mesh:
    ol = slice_outline(load_winding(a.mesh), a.z)
    ol = ol[np.isfinite(ol).all(1)]
    phi = np.unwrap(np.arctan2(ol[:, 1] - cy, ol[:, 0] - cx))
    phi = phi - phi[0] if phi[-1] > phi[0] else phi[0] - phi
    turn = np.floor(phi / (2 * np.pi)).astype(int)
    for t in sorted(set(turn.tolist())):
        if (turn == t).sum() > 10:
            outlines.append(ol[turn == t])
else:
    for k in range(a.first, a.last + 1):
        p = f'{a.windings}/w{k:03d}_{a.tag}'
        if os.path.isdir(p):
            o_ = slice_outline(load_winding(p), a.z)
            outlines.append(o_[np.isfinite(o_).all(1)])
R = np.array([ray_crossings(ol, cx, cy, thetas) for ol in outlines])
print('centre', round(cx), round(cy), '| turns/windings', len(outlines), '| finite crossings per ray', np.isfinite(R).sum(0)[:12], flush=True)
step = 0.5
gaps = []
dth = thetas[1] - thetas[0]
slope = np.nanmedian(np.abs((np.roll(R, -1, 1) - np.roll(R, 1, 1)) / (2 * dth)) / np.maximum(R, 1), 0)
for i, th in enumerate(thetas):
    col = np.sort(R[np.isfinite(R[:, i]), i])
    if len(col) < 2:
        continue
    rr = np.arange(col[0] - 20, col[-1] + 20, step)
    d = np.array([np.cos(th), np.sin(th)]); tv = np.array([-d[1], d[0]])
    base = np.stack([cx + rr * d[0], cy + rr * d[1], np.full(len(rr), a.z)], 1)
    acc = np.zeros(len(rr)); bad = np.zeros(len(rr), bool)
    for tt in range(-3, 4):
        for dz in (-2, -1, 0, 1, 2):
            v = ct_at(base + np.array([tt * tv[0], tt * tv[1], dz]))
            bad |= ~np.isfinite(v); acc += np.nan_to_num(v)
    ct = acc / 35.0
    sm = gaussian_filter1d(ct, 1.0 / step)
    mu = uniform_filter1d(sm, int(40 / step)); sd = np.sqrt(np.maximum(uniform_filter1d((sm - mu) ** 2, int(40 / step)), 1e-6))
    zc = (sm - mu) / sd
    air, pap = np.percentile(sm, [10, 90])
    floor = air + a.floor_frac * (pap - air)          # a layer must be brighter than the air between layers
    ix = find_peaks(zc, prominence=1.0, distance=int(5 / step))[0]
    pk = rr[ix[sm[ix] >= floor]]
    for r0, r1 in zip(col[:-1], col[1:]):
        seg = (rr >= r0) & (rr <= r1)
        if bad[seg].any():
            continue
        inside = pk[(pk > r0) & (pk < r1)]
        unc = inside[(inside - r0 > a.far) & (r1 - inside > a.far)]
        gaps.append(dict(theta=float(np.degrees(th)), r0=float(r0), gap=float(r1 - r0), layers_between=int(len(inside)),
                         uncovered=int(len(unc)), slope=float(slope[i])))
def stats(G):
    g = np.array([x['gap'] for x in G]); u = np.array([x['uncovered'] for x in G]); lb = np.array([x['layers_between'] for x in G])
    if not len(g):
        return None
    ok = g >= 2
    return dict(gaps=len(G), median_gap=float(np.median(g)), stacked_lt2=float(np.mean(g < 2)),
                uncovered_per_gap=float(u.mean()), ct_peaks_per_10vox=float(lb[ok].sum() / g[ok].sum() * 10) if ok.any() else None)


square = [x for x in gaps if np.isfinite(x['slope']) and x['slope'] <= 0.3]
g = np.array([x['gap'] for x in gaps]); u = np.array([x['uncovered'] for x in gaps])
summ = dict(all_rays=stats(gaps), square_on_rays=stats(square), z=a.z, gaps=len(gaps), median_gap=float(np.median(g)) if len(g) else None,
            share_gaps_with_uncovered_layer=float(np.mean(u > 0)) if len(u) else None,
            uncovered_layers_per_gap=float(np.mean(u)) if len(u) else None,
            share_stacked_gaps_lt2=float(np.mean(g < 2)) if len(g) else None,
            share_gaps_over_20=float(np.mean(g > 20)) if len(g) else None)
if a.fetch:
    summ['downloaded_GB'] = zfetch._bytes['n'] / 1e9
print(json.dumps(summ, indent=1))
json.dump(dict(summary=summ, gaps=gaps), open(a.out, 'w'), indent=1)
