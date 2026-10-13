"""Polar-unwrapped raw L0 CT strip in one z-slice (radius down, arc length across), with the m7 prediction and the
raw fitted windings, so that sheets can be followed over a long stretch of arc. Read-only on all inputs."""
import sys, os, json, argparse
import numpy as np, psutil
sys.path.insert(0, 'D:/Competition/Vesuvius progress prizes/spiral-winding-audit')
from winding_audit.tifxyz import load_winding
from winding_audit.sheets import open_surface, sample_volume
from winding_audit.density import slice_outline, ray_crossings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
ap = argparse.ArgumentParser()
ap.add_argument('--scroll', default='PHerc0175A')
ap.add_argument('--cache', required=True)
ap.add_argument('--meshes', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--z', type=int, required=True)
ap.add_argument('--theta', type=float, nargs=2, required=True, help='degrees from, to')
ap.add_argument('--r', type=float, nargs=2, required=True)
ap.add_argument('--windings', type=int, nargs=2, default=[30, 125])
ap.add_argument('--out', required=True)
ap.add_argument('--npz')
a = ap.parse_args()
B = f'D:/VesuviusCTCache/spiral/{a.scroll}'
CH = 128
_cache = {}


def ct_at(xyz):
    zyx = np.rint(xyz[:, ::-1]).astype(np.int64)
    keys = zyx // CH
    out = np.full(len(xyz), np.nan, np.float32)
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    for i, k in enumerate(uk):
        k = tuple(int(v) for v in k)
        if k not in _cache:
            f = os.path.join(a.cache, f'{k[0]}_{k[1]}_{k[2]}.bin')
            blk = None
            if os.path.exists(f):
                b = np.fromfile(f, np.uint8)
                blk = b.reshape(CH, CH, CH)[a.z % CH].copy() if b.size == CH ** 3 else None
            _cache[k] = blk
        blk = _cache[k]
        if blk is None:
            continue
        sel = np.nonzero(inv == i)[0]
        q = zyx[sel] - np.array(k) * CH
        out[sel] = blk[q[:, 1], q[:, 2]]
    return out


um = json.load(open(f'{B}/dataset/umbilicus.json'))['control_points']
zs = np.array([p['z'] for p in um], float)
o = np.argsort(zs)
cx = float(np.interp(a.z, zs[o], np.array([p['x'] for p in um], float)[o]))
cy = float(np.interp(a.z, zs[o], np.array([p['y'] for p in um], float)[o]))
rm = 0.5 * (a.r[0] + a.r[1])
t0, t1 = np.radians(a.theta)
n_t = int((t1 - t0) * rm)                       # ~1 vox of arc per column at mid radius
th = np.linspace(t0, t1, n_t)
rr = np.arange(a.r[0], a.r[1], 1.0)
TT, RR = np.meshgrid(th, rr)
pts = np.stack([cx + RR * np.cos(TT), cy + RR * np.sin(TT), np.full(RR.shape, a.z)], -1).reshape(-1, 3)
ct = ct_at(pts).reshape(RR.shape)
S = open_surface(f'{B}/m7.zarr', '0')
m7 = sample_volume(S, pts).reshape(RR.shape)
cross = {}
for k in range(a.windings[0], a.windings[1] + 1):
    p = f'{a.meshes}/w{k:03d}_{a.tag}'
    if os.path.isdir(p):
        cross[k] = ray_crossings(slice_outline(load_winding(p), a.z), cx, cy, th)
if a.npz:
    np.savez_compressed(a.npz, ct=ct, m7=m7, th=th, rr=rr, cross=np.array([cross[k] for k in sorted(cross)]),
                        ks=np.array(sorted(cross)))
lo, hi = np.nanpercentile(ct, [1, 99.5])
H, W = ct.shape
fig, ax = plt.subplots(2, 1, figsize=(W / 70, 2 * H / 70 + 1), dpi=100)
for A in ax:
    A.imshow(ct, cmap='gray', vmin=lo, vmax=hi, aspect='auto', interpolation='nearest',
             extent=[0, W, rr[-1], rr[0]])
ov = np.zeros(ct.shape + (4,))
ov[m7 >= 128] = (0.1, 0.85, 1.0, 0.45)
ax[1].imshow(ov, aspect='auto', interpolation='nearest', extent=[0, W, rr[-1], rr[0]])
for k, c in cross.items():
    ax[1].plot(np.arange(W), c, '-', color='#ff9f1c', lw=0.7)
ax[0].set_title(f'{a.scroll} z {a.z}: raw L0 CT unrolled, angle {a.theta[0]:.0f}-{a.theta[1]:.0f} deg '
                f'(across, ~1 vox of arc per pixel at r {rm:.0f}), radius {a.r[0]:.0f}-{a.r[1]:.0f} (down)', fontsize=9)
ax[1].set_title('cyan: m7 >= 128; orange: raw fitted windings (before snap)', fontsize=9)
for A in ax:
    A.set_ylim(rr[-1], rr[0])
    A.set_ylabel('radius (vox)')
fig.tight_layout()
fig.savefig(a.out)
print('saved', a.out, ct.shape)
