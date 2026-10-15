"""G1 follow-up: can the crossing rule pass on held-out Paris 4 boxes if its parameters are chosen on other boxes?

Profiles along the team's rays and the human segments are sampled once per box; then every parameter set in a small
grid is scored on the training boxes, the best (adjacent agreement >= 0.85 and at least 3 % of neural crossings
stored, then highest exact share of 3-15 pairs) is applied unchanged to the held-out boxes. Writes tune_results.json.

  python tools/paris4_tune.py --store <shard dir> --relative <relative_windings.json> --out <dir> [--train 0 2 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paris4_validate as pv  # noqa: E402
from winding_audit import ctinputs as ci  # noqa: E402
from winding_audit.volumes import ChunkCacheArray  # noqa: E402

GRID = dict(smooth=[1.0, 2.0, 3.0], prominence=[1.0, 1.5, 2.0], min_sep=[3.0, 6.0], floor_frac=[0.35, 0.5],
            gap=[(0.6, 1.5, 0.8, 1.25), (0.7, 1.35, 0.85, 1.15)], edge_trim=[1, 2])


def profiles(a):
    o, s, ct_t, lv, off = pv.load_store(a.store)
    e = o + (pv.RAY_LEN - 1) * s
    segs = pv.load_segments(a.relative)
    boxes = json.load(open(os.path.join(a.out, "boxes.json")))
    arr = ChunkCacheArray(os.path.join(a.out, "cache"), pv.SHAPE, chunk=pv.C)
    t = np.arange(pv.RAY_LEN, dtype=np.float64)
    data = []
    for b in boxes:
        lo, hi = pv.box_of(b["kz"], b["ky"], b["kx"], b["n"])
        sel = np.nonzero(pv.inside(o, lo, hi) & pv.inside(e, lo, hi))[0]
        cache = ci.ChunkLRU(4 * b["n"] * b["n"])
        prof = pv.sample_lines(arr, o[sel].astype(float), s[sel].astype(float), t, cache)
        neural = [(ct_t[off[r]:off[r + 1]].astype(float), lv[off[r]:off[r + 1]].astype(int)) for r in sel]
        human = []
        for pa, pb, dw, cid in segs:
            if not (pv.inside(pa[None], lo, hi)[0] and pv.inside(pb[None], lo, hi)[0]):
                continue
            L = float(np.linalg.norm(pb - pa))
            if L < 3 or dw == 0:
                continue
            ts = np.arange(-20.0, L + 20.0)
            human.append((pv.sample_lines(arr, pa[None], ((pb - pa) / L)[None], ts, cache)[0], ts, L, dw))
        data.append(dict(box=b, t=t, prof=prof, neural=neural, human=human))
    return data


def score(d, prm, tol=5.0):
    floor = ci.slice_floor(d["prof"], prm)
    dets = [ci.detect_crossings(p, d["t"], prm, floor) for p in d["prof"]]
    ref = ci.spacing_reference(dets, prm)
    adj = adj_ok = 0
    errs = []
    neural_n = stored_m = 0
    for det, (nt, nl) in zip(dets, d["neural"]):
        reg = ci.classify_gaps(det.pos, det.gap_ok, prm, ref)
        neural_n += len(nt)
        for run in ci.pieces(det.pos, reg, prm):
            mi = pv._match(run, nt, tol)
            stored_m += int((mi >= 0).sum())
            for k in range(len(run) - 1):
                if mi[k] >= 0 and mi[k + 1] >= 0:
                    adj += 1
                    adj_ok += int(nl[mi[k + 1]] - nl[mi[k]] == 1)
            for k in range(len(run)):
                for dd in range(3, 16):
                    if k + dd < len(run) and mi[k] >= 0 and mi[k + dd] >= 0:
                        errs.append(int(nl[mi[k + dd]] - nl[mi[k]] - dd))
    h_ok = 0
    for pr, ts, L, dw in d["human"]:
        det = ci.detect_crossings(pr, ts, prm, floor)
        sp = float(np.median(np.diff(det.pos))) if len(det.pos) > 2 else 20.0
        dl = 0.35 * sp
        h_ok += int(((det.pos > dl) & (det.pos < L - dl)).sum() == abs(dw) - 1)
    return dict(adj=adj, adj_ok=adj_ok, errs=errs, neural=neural_n, stored=stored_m, human=len(d["human"]),
                human_ok=h_ok)


def summarise(parts):
    adj = sum(p["adj"] for p in parts)
    errs = np.array(sum((p["errs"] for p in parts), []))
    neural = sum(p["neural"] for p in parts)
    hum = sum(p["human"] for p in parts)
    return dict(adjacent_pairs=adj, adjacent_agreement=sum(p["adj_ok"] for p in parts) / max(1, adj),
                relative_pairs=int(len(errs)),
                relative_exact=float(np.mean(errs == 0)) if len(errs) else None,
                relative_abs_err_median=float(np.median(np.abs(errs))) if len(errs) else None,
                relative_abs_err_p95=float(np.percentile(np.abs(errs), 95)) if len(errs) else None,
                stored_share_of_neural=sum(p["stored"] for p in parts) / max(1, neural),
                human_count_exact=sum(p["human_ok"] for p in parts) / max(1, hum))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True); ap.add_argument("--relative", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, nargs="+", default=[0, 2, 4])
    a = ap.parse_args(argv)
    data = profiles(a)
    train = [data[i] for i in a.train]
    test = [d for i, d in enumerate(data) if i not in a.train]
    rows = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        kw = dict(zip(keys, combo))
        g = kw.pop("gap")
        prm = ci.RayParams(gap_lo=g[0], gap_hi=g[1], pair_lo=g[2], pair_hi=g[3], **kw)
        tr = summarise([score(d, prm) for d in train])
        rows.append((dict(kw, gap=list(g)), prm, tr))
    ok = [r for r in rows if r[2]["adjacent_agreement"] >= 0.85 and r[2]["stored_share_of_neural"] >= 0.03
          and r[2]["relative_exact"] is not None]
    best = max(ok, key=lambda r: (r[2]["relative_exact"], r[2]["adjacent_agreement"])) if ok else \
        max(rows, key=lambda r: (r[2]["adjacent_agreement"], r[2]["relative_exact"] or 0))
    te = summarise([score(d, best[1]) for d in test])
    te["passes_bar"] = bool(te["adjacent_agreement"] >= 0.85 and te["relative_abs_err_median"] == 0
                            and (te["relative_abs_err_p95"] or 9) <= 1)
    out = dict(train_boxes=a.train, grid=GRID, n_candidates=len(rows), n_meeting_train_constraints=len(ok),
               chosen=best[0], train=best[2], heldout=te,
               top10_train=[dict(params=r[0], **r[2]) for r in sorted(rows, key=lambda r: -(r[2]["relative_exact"] or 0))[:10]])
    json.dump(out, open(os.path.join(a.out, "tune_results.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("n_candidates", "n_meeting_train_constraints", "chosen", "train", "heldout")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
