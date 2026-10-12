"""winding-audit: check the windings of a spiral fit against each other and the surface prediction.

  winding-audit audit <windings dir> --surface <prediction zarr> [--group 0] --out <dir> [--voxel-um 8.64]
                      [--merge-frac 0.3] [--skip-frac 1.7] [--threshold 128] [--node-tol 2.0] [--radius 4.0]
                      [--subdir tifxyz/flatten.tifxyz] [--sections z1 z2 ..] [--no-sheets] [--grid-figures]
  winding-audit repair <windings dir> --surface <prediction zarr> [--group 0] --out <dir> [--subdir ...]
                       [--node-tol 2.0] [--max-shift 1.25] [--jump 0.4] [--audit --voxel-um 8.64 --sections ..]
  winding-audit synthetic <out dir> [--merge W T0 T1] [--skip W T0 T1] [--gap W T0 T1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import report as rep
from .figures import winding_grid_figure
from .sheets import open_surface, sheet_counts
from .spacing import neighbour_spacing
from .tifxyz import list_windings, load_winding, save_winding


def cmd_audit(a):
    t0 = time.time()
    ws = list_windings(a.windings, subdir=a.subdir)
    if len(ws) < 2:
        sys.exit(f"need at least two windings in {a.windings}, found {len(ws)}")
    surface = None if a.no_sheets else open_surface(a.surface, a.group)
    rows, flags, verdicts = [], {}, {}
    maps_dir, rows_dir = _ensure(a.out, "maps"), _ensure(a.out, "rows")
    for w1, w2 in zip(ws[:-1], ws[1:]):
        if w2.index is not None and w1.index is not None and w2.index != w1.index + 1:
            print(f"warning: {w1.name} -> {w2.name} are not consecutive windings", flush=True)
        npz, rowf = os.path.join(maps_dir, f"{w1.name}.npz"), os.path.join(rows_dir, f"{w1.name}.json")
        if os.path.exists(npz) and os.path.exists(rowf):          # resume: this pair was finished earlier
            rows.append(json.load(open(rowf)))
            m = np.load(npz)
            flags[w1.name] = m["flag"]
            if m["verdict"].size:
                verdicts[w1.name] = m["verdict"]
            continue
        sp = neighbour_spacing(w1, w2, a.merge_frac, a.skip_frac)
        sc = (_retry(lambda: sheet_counts(w1, sp, surface, a.threshold, node_tol=a.node_tol, radius=a.radius))
              if surface is not None else None)
        row = rep.pair_row(w1, w2, sp, sc, a.voxel_um)
        rows.append(row)
        flags[w1.name] = sp.flag
        if sc is not None:
            verdicts[w1.name] = sc.verdict
        np.savez_compressed(npz, distance=sp.distance, flag=sp.flag, runs=(sc.runs if sc is not None else np.array([])),
                            verdict=(sc.verdict if sc is not None else np.array([])),
                            start_on_sheet=(sc.start_on_sheet if sc is not None else np.array([])),
                            node_dist=(sc.node_dist if sc is not None else np.array([])),
                            neighbour_dist=(sc.neighbour_dist if sc is not None else np.array([])))
        json.dump(row, open(rowf, "w"))
        if a.grid_figures:
            winding_grid_figure(w1.name, sp, sc, os.path.join(a.out, "windings", f"{w1.name}.png"), w1.grid_step,
                                a.voxel_um)
        print(f"{w1.name:>16s} -> {w2.name:<16s} pitch {sp.pitch:5.1f} vox  spacing merged {row['frac_merged']:.3f} "
              f"skipped {row['frac_skipped']:.3f}" + (f" | off-sheet {row['frac_node_off_sheet']:.3f} | sheets: "
              f"adjacent {row['frac_sheet_adjacent']:.3f} merged {row['frac_sheet_merged']:.3f} "
              f"fused {row['frac_sheet_fused']:.3f} skipped {row['frac_sheet_skipped']:.3f}" if sc is not None else ""),
              flush=True)
    rep.write_rows(rows, a.out)
    summary = rep.summarize(rows)
    summary["seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("summary:", json.dumps(summary))
    if a.sections:
        name = os.path.basename(os.path.normpath(a.windings))
        for z in a.sections:
            png = rep.cross_section(ws, flags, z, os.path.join(a.out, "sections", f"z{int(z):06d}_spacing.png"),
                                    surface, title=name)
            print("wrote", png)
            if verdicts:
                png = rep.cross_section(ws, verdicts, z, os.path.join(a.out, "sections", f"z{int(z):06d}_sheets.png"),
                                        surface, title=name, by_verdict=True)
                print("wrote", png)


def _ensure(*parts):
    p = os.path.join(*parts)
    os.makedirs(p, exist_ok=True)
    return p


def _retry(fn, tries: int = 20, wait: float = 60.0):
    """Run ``fn``; on MemoryError (a shared machine running out of RAM while a chunk is decoded) wait and retry."""
    for i in range(tries):
        try:
            return fn()
        except MemoryError:
            if i == tries - 1:
                raise
            print(f"memory error, waiting {wait:.0f} s before retry {i + 2}/{tries}", flush=True)
            time.sleep(wait)


def cmd_repair(a):
    """Repair a band from the inside out; resumable (a winding already written is loaded, not redone)."""
    import csv

    from .repair import REPAIR_STATUS_NAMES, repair_winding

    t0 = time.time()
    ws = list_windings(a.windings, subdir=a.subdir)
    if len(ws) < 3:
        sys.exit(f"need at least three windings in {a.windings}, found {len(ws)}")
    surface = open_surface(a.surface, a.group)
    wdir, mdir = _ensure(a.out, "windings"), _ensure(a.out, "repair_maps")
    rows, current = [], list(ws)
    for i, w in enumerate(ws):
        od = os.path.join(wdir, w.name)
        rowf = os.path.join(mdir, f"{w.name}.json")
        if i in (0, len(ws) - 1):                       # one neighbour only: written unchanged
            if not os.path.exists(os.path.join(od, "x.tif")):
                save_winding(w, od)
            continue
        if os.path.exists(rowf) and os.path.exists(os.path.join(od, "x.tif")):
            current[i] = load_winding(od, w.name)
            rows.append(json.load(open(rowf)))
            continue
        res = _retry(lambda: repair_winding(w, current[i - 1], ws[i + 1], surface, a.threshold, a.node_tol,
                                            a.max_shift, jump_frac=a.jump))
        current[i] = res.winding
        save_winding(res.winding, od)
        np.savez_compressed(os.path.join(mdir, f"{w.name}.npz"), shift=res.shift, status=res.status)
        row = {"winding": w.name, **res.stats}
        json.dump(row, open(rowf, "w"))
        rows.append(row)
        print(f"{w.name:>16s} pitch {res.pitch:5.1f}  " + "  ".join(
            f"{k} {row[k]}" for k in REPAIR_STATUS_NAMES.values()) +
            f"  median shift {row['median_abs_shift_moved']:.1f} vox  edges>1.5: "
            f"{row['before_frac_edges_over_1p5']:.3f} -> {row['after_frac_edges_over_1p5']:.3f}", flush=True)
    if rows:
        with open(os.path.join(a.out, "repair.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        tot = {k: sum(r[k] for r in rows) for k in ["n", "assessed", *REPAIR_STATUS_NAMES.values()]}
        tot["seconds"] = round(time.time() - t0, 1)
        json.dump(tot, open(os.path.join(a.out, "repair_summary.json"), "w"), indent=1)
        print("repair:", json.dumps(tot), flush=True)
    if a.audit:
        aa = argparse.Namespace(**vars(a))
        aa.windings, aa.subdir, aa.out = wdir, "", os.path.join(a.out, "audit")
        aa.merge_frac, aa.skip_frac, aa.no_sheets, aa.grid_figures = 0.3, 1.7, False, a.grid_figures
        cmd_audit(aa)


def cmd_synthetic(a):
    from . import synthetic as syn
    vol, centre = syn.make_volume(n_sheets=a.sheets, pitch=a.pitch)
    ws, thetas = syn.make_windings(n_sheets=a.sheets, pitch=a.pitch, centre=centre)
    for w, t0, t1 in a.merge or []:
        syn.inject_merge(ws, thetas, int(w), float(t0), float(t1), a.pitch, centre)
    for w, t0, t1 in a.skip or []:
        syn.inject_skip(ws, thetas, int(w), float(t0), float(t1), a.pitch, centre)
    for w, t0, t1 in a.gap or []:
        syn.inject_gap(ws, thetas, int(w), float(t0), float(t1), a.pitch, centre)
    s, d = syn.write_dataset(a.out, vol, ws)
    print("wrote", s, "and", d)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="winding-audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="audit a folder of tifxyz windings")
    p.add_argument("windings"); p.add_argument("--surface"); p.add_argument("--group", default="0")
    p.add_argument("--out", required=True); p.add_argument("--voxel-um", type=float)
    p.add_argument("--merge-frac", type=float, default=0.3); p.add_argument("--skip-frac", type=float, default=1.7)
    p.add_argument("--threshold", type=int, default=128); p.add_argument("--sections", type=float, nargs="*")
    p.add_argument("--subdir", default="",
                   help="tifxyz path inside each winding folder, e.g. tifxyz/flatten.tifxyz for flatten output")
    p.add_argument("--node-tol", type=float, default=2.0,
                   help="a node is on a sheet when a predicted-sheet voxel lies within this distance (voxels)")
    p.add_argument("--radius", type=float, default=4.0, help="node-to-sheet distance is exact up to this (voxels)")
    p.add_argument("--no-sheets", action="store_true", help="spacing only (no surface prediction)")
    p.add_argument("--grid-figures", action="store_true", help="one PNG per winding: flag, distance and sheet maps")
    p.set_defaults(func=cmd_audit)
    r = sub.add_parser("repair", help="move flagged nodes onto the sheet between their neighbours")
    r.add_argument("windings"); r.add_argument("--surface", required=True); r.add_argument("--group", default="0")
    r.add_argument("--out", required=True); r.add_argument("--subdir", default="")
    r.add_argument("--threshold", type=int, default=128); r.add_argument("--node-tol", type=float, default=2.0)
    r.add_argument("--radius", type=float, default=4.0)
    r.add_argument("--max-shift", type=float, default=1.25, help="largest move, in sheet pitches")
    r.add_argument("--jump", type=float, default=0.4,
                   help="a move larger than this (pitches) must agree with two neighbouring moves")
    r.add_argument("--audit", action="store_true", help="audit the repaired band into <out>/audit")
    r.add_argument("--voxel-um", type=float); r.add_argument("--sections", type=float, nargs="*")
    r.add_argument("--grid-figures", action="store_true")
    r.set_defaults(func=cmd_repair)
    s = sub.add_parser("synthetic", help="write a synthetic spiral scroll with optional faults")
    s.add_argument("out"); s.add_argument("--sheets", type=int, default=6); s.add_argument("--pitch", type=float, default=10.0)
    s.add_argument("--merge", nargs=3, action="append", metavar=("W", "T0", "T1"))
    s.add_argument("--skip", nargs=3, action="append", metavar=("W", "T0", "T1"))
    s.add_argument("--gap", nargs=3, action="append", metavar=("W", "T0", "T1"))
    s.set_defaults(func=cmd_synthetic)
    a = ap.parse_args(argv)
    if a.cmd == "audit" and not a.no_sheets and not a.surface:
        ap.error("--surface is required unless --no-sheets")
    a.func(a)


if __name__ == "__main__":
    main()
