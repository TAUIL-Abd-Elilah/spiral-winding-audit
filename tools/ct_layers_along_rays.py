"""Windings per CT layer (not per m7 layer) along radial rays in one z-slice, from the raw L0 CT (route24 chunk
cache, read-only; rays that touch a missing chunk are cut to their longest complete stretch).

Along each ray from the umbilicus, between the crossings of two fitted windings (raw fit, before snap):
  windings   crossings of the fitted windings' slice outlines
  m7         runs of the prediction >= 128, joined when closer than 4 vox (as `winding-audit density`)
  CT period  spectral peak (1/40..1/6 per vox) of the detrended CT profile in 128-vox windows, and the first
             autocorrelation peak; layers = path length / period
  CT peaks   local maxima of the smoothed, detrended profile with prominence >= p (in local std units) that are
             brighter than air (above air + 0.35 (papyrus - air), 10th / 90th percentile of the ray): without that
             floor, noise in wide air gaps counts as layers
and, per m7 run, the number of CT peaks inside it (2+ = the prediction joins touching sheets) and CT peaks that
fall between m7 runs (sheets the prediction misses)."""
import sys, os, json, argparse
import numpy as np, psutil
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
sys.path.insert(0, 'D:/Competition/Vesuvius progress prizes/spiral-winding-audit')
from winding_audit.tifxyz import load_winding
from winding_audit.sheets import open_surface, sample_volume
from winding_audit.density import slice_outline, ray_crossings

psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
ap = argparse.ArgumentParser()
ap.add_argument('--scroll', default='PHerc0175A')
ap.add_argument('--cache', required=True)
ap.add_argument('--meshes', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--z', type=float, required=True)
ap.add_argument('--first', type=int, default=40)
ap.add_argument('--last', type=int, default=115)
ap.add_argument('--rays', type=int, default=36)
ap.add_argument('--out', required=True)
ap.add_argument('--floor-frac', type=float, default=0.35,
                help='a CT peak counts as a layer only if brighter than air + this share of (papyrus - air), '
                     'from the 10th / 90th percentiles of the ray')
a = ap.parse_args()
B = f'D:/VesuviusCTCache/spiral/{a.scroll}'
CH = 128
_cache = {}


def chunk(k):
    if k in _cache:
        return _cache[k]
    f = os.path.join(a.cache, f'{k[0]}_{k[1]}_{k[2]}.bin')
    blk = None
    if os.path.exists(f):
        b = np.fromfile(f, np.uint8)
        if b.size == CH ** 3:
            blk = b.reshape(CH, CH, CH)
    if len(_cache) > 64:
        _cache.clear()
    _cache[k] = blk
    return blk


def ct_at(xyz):
    """(N, 3) float xyz -> (N,) float, NaN where the chunk is missing."""
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


um = json.load(open(f'{B}/dataset/umbilicus.json'))['control_points']
zs = np.array([p['z'] for p in um], float)
o = np.argsort(zs)
cx = float(np.interp(a.z, zs[o], np.array([p['x'] for p in um], float)[o]))
cy = float(np.interp(a.z, zs[o], np.array([p['y'] for p in um], float)[o]))
names = [f'w{k:03d}_{a.tag}' for k in range(a.first, a.last + 1)]
thetas = np.linspace(0, 2 * np.pi, a.rays, endpoint=False)
R = []
for nm in names:
    w = load_winding(f'{a.meshes}/{nm}')
    R.append(ray_crossings(slice_outline(w, a.z), cx, cy, thetas))
    del w
R = np.array(R)
S = open_surface(f'{B}/m7.zarr', '0')
step = 0.5
rows, prof_dump = [], {}
for i, th in enumerate(thetas):
    ra, rb = R[0, i], R[-1, i]
    if not (np.isfinite(ra) and np.isfinite(rb)) or rb <= ra + 100:
        continue
    rr = np.arange(ra - 8, rb + 8, step)
    d = np.array([np.cos(th), np.sin(th)])
    tvec = np.array([-d[1], d[0]])
    base = np.stack([cx + rr * d[0], cy + rr * d[1], np.full(len(rr), a.z)], 1)
    acc = np.zeros(len(rr))
    bad = np.zeros(len(rr), bool)
    for tt in range(-3, 4):
        for dz in (-2, -1, 0, 1, 2):
            v = ct_at(base + np.array([tt * tvec[0], tt * tvec[1], dz]))
            bad |= ~np.isfinite(v)
            acc += np.nan_to_num(v)
    ct = acc / 35.0
    if bad.any():
        good = ~bad
        e = np.diff(np.r_[0, good.astype(int), 0])
        s0, s1 = np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]
        if len(s0) == 0:
            continue
        j = np.argmax(s1 - s0)
        lo, hi = s0[j], s1[j]
        if (hi - lo) * step < 300:
            continue
        rr, ct, base = rr[lo:hi], ct[lo:hi], base[lo:hi]
    m7 = sample_volume(S, base) >= 128
    L = len(rr) * step
    sm = gaussian_filter1d(ct, 1.0 / step)
    mu = uniform_filter1d(sm, int(40 / step))
    sd = np.sqrt(np.maximum(uniform_filter1d((sm - mu) ** 2, int(40 / step)), 1e-6))
    zc = (sm - mu) / sd
    W = int(128 / step)
    per = []
    for s in range(0, len(zc) - W + 1, W // 2):
        seg = zc[s:s + W] - zc[s:s + W].mean()
        F = np.abs(np.fft.rfft(seg * np.hanning(W), n=4096))
        f = np.fft.rfftfreq(4096, d=step)
        band = (f >= 1 / 40) & (f <= 1 / 6)
        per.append(1 / f[band][np.argmax(F[band])])
    p = zc - zc.mean()
    ac = np.correlate(p, p, 'full')[len(p) - 1:]
    ac = ac / ac[0]
    lags = np.arange(len(ac)) * step
    pk_ac = [lags[j] for j in range(1, len(ac) - 1)
             if 6 <= lags[j] <= 45 and ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1]]
    air, pap = np.percentile(sm, [10, 90])
    floor = air + a.floor_frac * (pap - air)          # a layer must be brighter than the air between layers
    peaks = {}
    for pr in (0.5, 1.0, 1.5):
        ix = find_peaks(zc, prominence=pr, distance=int(5 / step))[0]
        peaks[pr] = ix[sm[ix] >= floor]
    dd = np.diff(np.pad(m7.astype(np.int8), (1, 1)))
    s0, s1 = np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]
    runs = []
    for u, v in zip(s0, s1):
        if runs and (u - runs[-1][1]) * step < 4:
            runs[-1][1] = v
        else:
            runs.append([u, v])
    pk = peaks[1.0]
    inside = np.zeros(len(pk), bool)
    per_run = []
    for u, v in runs:
        m = (pk >= u - 2) & (pk < v + 2)
        inside |= m
        per_run.append(int(m.sum()))
    per_run = np.array(per_run)
    k_w = int(np.sum(np.isfinite(R[:, i]) & (R[:, i] >= rr[0]) & (R[:, i] <= rr[-1])))
    row = dict(theta_deg=float(np.degrees(th)), r0=float(rr[0]), r1=float(rr[-1]), length=float(L), windings=k_w,
               m7_runs=len(runs),
               m7_run_width_med=float(np.median([(v - u) * step for u, v in runs])) if runs else None,
               ct_period_spec=float(np.median(per)), ct_period_ac=(float(pk_ac[0]) if pk_ac else None),
               ct_peaks={str(k): int(len(v)) for k, v in peaks.items()},
               m7_runs_with_0_1_2plus_ct_peaks=[int((per_run == 0).sum()), int((per_run == 1).sum()),
                                                int((per_run >= 2).sum())],
               ct_peaks_outside_m7=int((~inside).sum()))
    row['windings_per_m7'] = (k_w - 1) / max(len(runs) - 1, 1)
    row['windings_per_ct_spec'] = k_w / (L / row['ct_period_spec'])
    row['windings_per_ct_peaks'] = {k: k_w / max(v, 1) for k, v in row['ct_peaks'].items()}
    rows.append(row)
    if i % max(1, a.rays // 6) == 0:
        prof_dump[str(i)] = dict(r=rr.tolist(), ct=ct.tolist(), m7=m7.astype(int).tolist(),
                                 windings=[float(x) for x in R[:, i] if np.isfinite(x)])
    print(f"theta {row['theta_deg']:5.1f}: L {L:6.0f} | windings {k_w} | m7 runs {len(runs)} "
          f"(width {row['m7_run_width_med']}) | CT period spec {row['ct_period_spec']:.1f} ac {row['ct_period_ac']} | "
          f"CT peaks p0.5/1/1.5 {list(row['ct_peaks'].values())} | m7 runs w/ 0/1/2+ CT peaks "
          f"{row['m7_runs_with_0_1_2plus_ct_peaks']} | CT peaks outside m7 {row['ct_peaks_outside_m7']}", flush=True)


def med(k):
    return float(np.median([r[k] for r in rows]))


summ = dict(scroll=a.scroll, z=a.z, tag=a.tag, first=a.first, last=a.last, rays=len(rows),
            windings_per_m7=med('windings_per_m7'), windings_per_ct_spec=med('windings_per_ct_spec'),
            windings_per_ct_peaks={k: float(np.median([r['windings_per_ct_peaks'][k] for r in rows]))
                                   for k in ('0.5', '1.0', '1.5')},
            ct_period_spec=med('ct_period_spec'),
            m7_spacing=float(np.median([r['length'] / max(r['m7_runs'] - 1, 1) for r in rows])),
            winding_spacing=float(np.median([r['length'] / max(r['windings'] - 1, 1) for r in rows])),
            m7_runs_with_0_1_2plus=np.sum([r['m7_runs_with_0_1_2plus_ct_peaks'] for r in rows], 0).tolist(),
            ct_peaks_outside_m7=int(np.sum([r['ct_peaks_outside_m7'] for r in rows])),
            ct_peaks_total_p1=int(np.sum([r['ct_peaks']['1.0'] for r in rows])))
print(json.dumps(summ, indent=1))
json.dump(dict(summary=summ, rays=rows, profiles=prof_dump), open(a.out, 'w'))
