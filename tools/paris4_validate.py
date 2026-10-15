"""G1: the ct-inputs crossing rule on PHerc. Paris 4, against the team's neural winding store and the human
relative-winding annotations, inside a few CT boxes (chunks fetched once into a local cache, deleted afterwards).

  python tools/paris4_validate.py select   --store <shard dir> --relative <relative_windings.json> --out <dir>
  python tools/paris4_validate.py fetch    --out <dir> [--workers 8]
  python tools/paris4_validate.py evaluate --store <shard dir> --relative <relative_windings.json> --out <dir>

The neural store's coordinates are scale-0 voxels of the 9.6 um volume, i.e. level 2 of the 2.4 um scan (checked in
`evaluate` by the position agreement of crossings). Rays: origin + t * step, t = 0..383, unit steps along the fit's
normals. Human points: p = [x, y, z] with a winding number per collection."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from winding_audit import ctinputs as ci  # noqa: E402
from winding_audit.volumes import ChunkCacheArray  # noqa: E402

BASE = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/"
        "20260411134726-2.400um-0.2m-78keV-masked.zarr/2")
SHAPE = (18946, 8174, 8174)
C = 128
RAY_LEN = 384


def load_store(d):
    o = np.load(os.path.join(d, "ray_origin_zyx.npy"))
    s = np.load(os.path.join(d, "ray_step_zyx.npy"))
    t = np.load(os.path.join(d, "crossing_t.npy"))
    lv = np.load(os.path.join(d, "crossing_level.npy"))
    off = np.load(os.path.join(d, "crossing_offsets.npy"))
    return o, s, t, lv, off


def load_segments(path):
    """Consecutive points of each human collection (sorted by point id): (a_zyx, b_zyx, winding difference)."""
    rw = json.load(open(path))["collections"]
    segs = []
    for cid, c in rw.items():
        pts = sorted(((int(k), v) for k, v in c["points"].items()), key=lambda kv: kv[0])
        for (_, p), (_, q) in zip(pts[:-1], pts[1:]):
            if "wind_a" not in p or "wind_a" not in q:
                continue
            a = np.array(p["p"][::-1], float)
            b = np.array(q["p"][::-1], float)
            segs.append((a, b, float(q["wind_a"] - p["wind_a"]), cid))
    return segs


def box_of(kz, ky, kx, n):
    lo = np.array([kz * C, ky * C, kx * C], float)
    hi = lo + np.array([C, n * C, n * C], float)
    return lo, hi


def inside(p, lo, hi, m=2.0):
    return np.all((p >= lo + m) & (p <= hi - 1 - m), axis=-1)


def cmd_select(a):
    o, s, *_ = load_store(a.store)
    e = o + (RAY_LEN - 1) * s
    segs = load_segments(a.relative)
    sa = np.array([x[0] for x in segs])
    sb = np.array([x[1] for x in segs])
    n = a.box
    cands = []
    zs = np.unique(np.floor(np.r_[sa[:, 0], o[:, 0]] / C).astype(int))
    for kz in zs:
        rz = (o[:, 0] >= kz * C + 2) & (e[:, 0] >= kz * C + 2) & (o[:, 0] <= kz * C + C - 3) & (e[:, 0] <= kz * C + C - 3)
        hz = (sa[:, 0] >= kz * C + 2) & (sb[:, 0] >= kz * C + 2) & (sa[:, 0] <= kz * C + C - 3) & (sb[:, 0] <= kz * C + C - 3)
        if hz.sum() == 0 or rz.sum() == 0:
            continue
        oz, ez, saz, sbz = o[rz], e[rz], sa[hz], sb[hz]
        for ky in range(0, SHAPE[1] // C - n + 1, 2):
            for kx in range(0, SHAPE[2] // C - n + 1, 2):
                lo, hi = box_of(kz, ky, kx, n)
                nh = int((inside(saz, lo, hi) & inside(sbz, lo, hi)).sum())
                if nh < 5:
                    continue
                nr = int((inside(oz, lo, hi) & inside(ez, lo, hi)).sum())
                if nr >= 50 and nh >= 5:
                    cands.append((nh * 20 + nr, int(kz), ky, kx, nr, nh))
    cands.sort(reverse=True)
    um = json.load(open(a.umbilicus))["control_points"]
    uz = np.array([p["z"] for p in um], float)
    uy = np.array([p["y"] for p in um], float)
    ux = np.array([p["x"] for p in um], float)
    picked = []
    for sc, kz, ky, kx, nr, nh in cands:
        lo, hi = box_of(kz, ky, kx, n)
        c = (lo + hi) / 2
        rad = float(np.hypot(c[1] - np.interp(c[0], uz, uy), c[2] - np.interp(c[0], uz, ux)))
        # spread: no two boxes in the same chunk row or overlapping in yx; radius classes inner/middle/outer
        if any(p["kz"] == kz or (abs(p["ky"] - ky) < n and abs(p["kx"] - kx) < n and abs(p["kz"] - kz) < 2)
               for p in picked):
            continue
        picked.append(dict(kz=kz, ky=ky, kx=kx, n=n, rays=nr, segments=nh, radius=round(rad)))
        if len(picked) >= a.boxes:
            break
    os.makedirs(a.out, exist_ok=True)
    json.dump(picked, open(os.path.join(a.out, "boxes.json"), "w"), indent=1)
    print(json.dumps(picked, indent=1))
    print("chunks to fetch:", sum(p["n"] * p["n"] for p in picked), f"(~{sum(p['n'] * p['n'] for p in picked) * 2} MiB)")


def _fetch_one(key, cache):
    f = os.path.join(cache, f"{key[0]}_{key[1]}_{key[2]}.bin")
    if os.path.exists(f) and os.path.getsize(f) == C ** 3:
        return 0
    url = f"{BASE}/{key[0]}/{key[1]}/{key[2]}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                b = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                b = b""                                    # outside the mask: the fill value, 0
                break
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"could not fetch {url}")
    arr = np.zeros(C ** 3, np.uint8) if not b else np.frombuffer(b, np.uint8)
    if arr.size != C ** 3:
        # an edge chunk is stored full-size in zarr v2; anything else is an error
        raise RuntimeError(f"{url}: {arr.size} bytes")
    tmp = f + ".part"
    arr.tofile(tmp)
    os.replace(tmp, f)
    return len(b)


def cmd_fetch(a):
    boxes = json.load(open(os.path.join(a.out, "boxes.json")))
    cache = os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    keys = [(b["kz"], b["ky"] + i, b["kx"] + j) for b in boxes for i in range(b["n"]) for j in range(b["n"])]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        got = list(ex.map(lambda k: _fetch_one(k, cache), keys))
    print(f"{len(keys)} chunks, {sum(got) / 2 ** 20:.0f} MiB downloaded in {time.time() - t0:.0f} s, "
          f"{sum(1 for g in got if g == 0)} cached or outside the mask")


def _perp(step):
    a = np.array([1.0, 0, 0]) if abs(step[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(step, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(step, e1)
    return e1, e2


def sample_lines(arr, origins, steps, t, cache, tube=1):
    """CT along lines origin + t * step, averaged over a (2 tube + 1)^2 tube across each line."""
    offs = [(u, v) for u in range(-tube, tube + 1) for v in range(-tube, tube + 1)]
    pts = np.empty((len(offs), len(origins), len(t), 3), np.int64)
    for i, (o, s) in enumerate(zip(origins, steps)):
        e1, e2 = _perp(s)
        base = o[None, :] + t[:, None] * s[None, :]
        for k, (u, v) in enumerate(offs):
            pts[k, i] = np.rint(base + u * e1[None, :] + v * e2[None, :])
    vals = ci.sample_points(arr, pts.reshape(-1, 3), cache).reshape(len(offs), len(origins), len(t))
    return vals.mean(0)


def _match(src, dst, tol):
    """For each src position, the index of the nearest dst position within tol, else -1."""
    if len(dst) == 0:
        return np.full(len(src), -1)
    j = np.clip(np.searchsorted(dst, src), 1, len(dst) - 1)
    jj = np.where(np.abs(dst[j - 1] - src) <= np.abs(dst[j] - src), j - 1, j)
    return np.where(np.abs(dst[jj] - src) <= tol, jj, -1)


def cmd_evaluate(a):
    o, s, ct_t, lv, off = load_store(a.store)
    e = o + (RAY_LEN - 1) * s
    segs = load_segments(a.relative)
    boxes = json.load(open(os.path.join(a.out, "boxes.json")))
    arr = ChunkCacheArray(os.path.join(a.out, "cache"), SHAPE, chunk=C)
    prm = ci.RayParams()
    tol = a.tol
    t = np.arange(RAY_LEN, dtype=np.float64)
    res = {"params": ci.params_dict(prm), "match_tolerance_vox": tol, "boxes": []}
    tot = dict(adj=0, adj_ok=0, rel=0, rel_err=[], neural=0, neural_found=0, ours=0, ours_matched=0,
               stored=0, stored_matched=0, seg=0, seg_ok=0, seg_piece=0, seg_piece_ok=0)
    for b in boxes:
        lo, hi = box_of(b["kz"], b["ky"], b["kx"], b["n"])
        sel = np.nonzero(inside(o, lo, hi) & inside(e, lo, hi))[0]
        cache = ci.ChunkLRU(4 * b["n"] * b["n"])
        prof = sample_lines(arr, o[sel].astype(np.float64), s[sel].astype(np.float64), t, cache)
        floor = ci.slice_floor(prof, prm)
        dets = [ci.detect_crossings(prof[i], t, prm, floor) for i in range(len(sel))]
        ref = ci.spacing_reference(dets, prm)
        bx = dict(adj=0, adj_ok=0, rel=0, rel_err=[], neural=0, neural_found=0, ours=0, ours_matched=0,
                  stored=0, stored_matched=0)
        for i, r in enumerate(sel):
            nt = ct_t[off[r]:off[r + 1]].astype(np.float64)
            nl = lv[off[r]:off[r + 1]].astype(np.int64)
            det = dets[i]
            reg = ci.classify_gaps(det.pos, det.gap_ok, prm, ref)
            m_ours = _match(det.pos, nt, tol)
            m_neur = _match(nt, det.pos, tol) if len(det.pos) else np.full(len(nt), -1)
            bx["neural"] += len(nt)
            bx["neural_found"] += int((m_neur >= 0).sum())
            bx["ours"] += len(det.pos)
            bx["ours_matched"] += int((m_ours >= 0).sum())
            for run in ci.pieces(det.pos, reg, prm):
                mi = _match(run, nt, tol)
                bx["stored"] += len(run)
                bx["stored_matched"] += int((mi >= 0).sum())
                for k in range(len(run) - 1):
                    if mi[k] >= 0 and mi[k + 1] >= 0:
                        bx["adj"] += 1
                        bx["adj_ok"] += int(nl[mi[k + 1]] - nl[mi[k]] == 1)
                for k in range(len(run)):
                    for d in range(3, 16):
                        if k + d < len(run) and mi[k] >= 0 and mi[k + d] >= 0:
                            bx["rel"] += 1
                            bx["rel_err"].append(int(nl[mi[k + d]] - nl[mi[k]] - d))
        # human segments inside this box
        hs = [x for x in segs if inside(x[0][None], lo, hi)[0] and inside(x[1][None], lo, hi)[0]]
        hres = []
        for pa, pb, dw, cid in hs:
            L = float(np.linalg.norm(pb - pa))
            if L < 3 or dw == 0:
                continue
            st = (pb - pa) / L
            ts = np.arange(-20.0, L + 20.0)
            pr = sample_lines(arr, pa[None], st[None], ts, cache)[0]
            det = ci.detect_crossings(pr, ts, prm, floor)
            reg = ci.classify_gaps(det.pos, det.gap_ok, prm, ref)
            sp = float(np.median(np.diff(det.pos))) if len(det.pos) > 2 else 20.0
            dlt = 0.35 * sp
            between = int(((det.pos > dlt) & (det.pos < L - dlt)).sum())
            ok = between == abs(dw) - 1
            # stored levels: a piece holding a crossing at each end point
            piece_ok = None
            for run in ci.pieces(det.pos, reg, prm):
                ia = np.nonzero(np.abs(run) <= dlt)[0]
                ib = np.nonzero(np.abs(run - L) <= dlt)[0]
                if len(ia) and len(ib):
                    piece_ok = int(ib[0] - ia[0]) == abs(dw)
            hres.append(dict(collection=cid, length=round(L, 1), dw=dw, between=between, ok=ok, piece_ok=piece_ok))
        seg_n = len(hres)
        seg_ok = sum(h["ok"] for h in hres)
        seg_p = [h for h in hres if h["piece_ok"] is not None]
        bx_out = {k: v for k, v in bx.items() if k != "rel_err"}
        err = np.array(bx["rel_err"]) if bx["rel_err"] else np.zeros(0)
        bx_out.update(dict(box=b, rays=len(sel), floor=floor,
                           adjacent_agreement=bx["adj_ok"] / max(1, bx["adj"]),
                           relative_exact=float(np.mean(err == 0)) if len(err) else None,
                           relative_abs_err_p95=float(np.percentile(np.abs(err), 95)) if len(err) else None,
                           neural_recall=bx["neural_found"] / max(1, bx["neural"]),
                           position_precision=bx["ours_matched"] / max(1, bx["ours"]),
                           stored_share_of_neural=bx["stored_matched"] / max(1, bx["neural"]),
                           human_segments=seg_n, human_count_exact=seg_ok / max(1, seg_n),
                           human_in_one_piece=len(seg_p),
                           human_piece_exact=(sum(h["piece_ok"] for h in seg_p) / len(seg_p)) if seg_p else None,
                           chunk_reads=cache.reads))
        res["boxes"].append(bx_out)
        for k in ("adj", "adj_ok", "rel", "neural", "neural_found", "ours", "ours_matched", "stored", "stored_matched"):
            tot[k] += bx[k]
        tot["rel_err"] += bx["rel_err"]
        tot["seg"] += seg_n
        tot["seg_ok"] += seg_ok
        tot["seg_piece"] += len(seg_p)
        tot["seg_piece_ok"] += sum(h["piece_ok"] for h in seg_p)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in bx_out.items() if k != "box"}),
              flush=True)
    err = np.array(tot["rel_err"]) if tot["rel_err"] else np.zeros(0)
    summ = dict(rays=sum(x["rays"] for x in res["boxes"]),
                adjacent_pairs=tot["adj"], adjacent_agreement=tot["adj_ok"] / max(1, tot["adj"]),
                relative_pairs=tot["rel"], relative_exact=float(np.mean(err == 0)) if len(err) else None,
                relative_abs_err_median=float(np.median(np.abs(err))) if len(err) else None,
                relative_abs_err_p95=float(np.percentile(np.abs(err), 95)) if len(err) else None,
                neural_recall=tot["neural_found"] / max(1, tot["neural"]),
                position_precision=tot["ours_matched"] / max(1, tot["ours"]),
                stored_share_of_neural=tot["stored_matched"] / max(1, tot["neural"]),
                human_segments=tot["seg"], human_count_exact=tot["seg_ok"] / max(1, tot["seg"]),
                human_in_one_piece=tot["seg_piece"],
                human_piece_exact=tot["seg_piece_ok"] / max(1, tot["seg_piece"]) if tot["seg_piece"] else None)
    summ["passes_bar"] = bool(summ["adjacent_agreement"] >= 0.85 and summ["relative_abs_err_median"] == 0
                              and (summ["relative_abs_err_p95"] or 0) <= 1)
    res["summary"] = summ
    json.dump(res, open(os.path.join(a.out, "g1_results.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "fetch", "evaluate"])
    ap.add_argument("--store"); ap.add_argument("--relative"); ap.add_argument("--umbilicus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=int, default=6, help="box side in chunks (yx)")
    ap.add_argument("--boxes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args(argv)
    {"select": cmd_select, "fetch": cmd_fetch, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
