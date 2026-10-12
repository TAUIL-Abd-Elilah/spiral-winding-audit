"""winding-audit: check the windings of a spiral fit against each other and the surface prediction.

  winding-audit audit <windings dir> --surface <prediction zarr> [--group 0] --out <dir> [--voxel-um 8.64]
                      [--merge-frac 0.3] [--skip-frac 1.7] [--threshold 128] [--sections z1 z2 ..] [--no-sheets]
  winding-audit synthetic <out dir> [--merge W T0 T1] [--skip W T0 T1] [--gap W T0 T1]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import report as rep
from .sheets import open_surface, sheet_counts
from .spacing import neighbour_spacing
from .tifxyz import list_windings


def cmd_audit(a):
    t0 = time.time()
    ws = list_windings(a.windings)
    if len(ws) < 2:
        sys.exit(f"need at least two windings in {a.windings}, found {len(ws)}")
    surface = None if a.no_sheets else open_surface(a.surface, a.group)
    rows, flags = [], {}
    for w1, w2 in zip(ws[:-1], ws[1:]):
        if w2.index is not None and w1.index is not None and w2.index != w1.index + 1:
            print(f"warning: {w1.name} -> {w2.name} are not consecutive windings", flush=True)
        sp = neighbour_spacing(w1, w2, a.merge_frac, a.skip_frac)
        sc = sheet_counts(w1, sp, surface, a.threshold) if surface is not None else None
        row = rep.pair_row(w1, w2, sp, sc, a.voxel_um)
        rows.append(row)
        flags[w1.name] = sp.flag
        np.savez_compressed(os.path.join(_ensure(a.out, "maps"), f"{w1.name}.npz"), distance=sp.distance,
                            flag=sp.flag, runs=(sc.runs if sc is not None else np.array([])),
                            start_on_sheet=(sc.start_on_sheet if sc is not None else np.array([])))
        print(f"{w1.name:>16s} -> {w2.name:<16s} pitch {sp.pitch:5.1f} vox  merged {row['frac_merged']:.3f}  "
              f"skipped {row['frac_skipped']:.3f}" + (f"  runs2 {row['frac_runs2']:.3f}  runs1 {row['frac_runs1']:.3f}  "
              f"runs3+ {row['frac_runs3plus']:.3f}" if sc is not None else ""), flush=True)
    rep.write_rows(rows, a.out)
    summary = rep.summarize(rows)
    summary["seconds"] = round(time.time() - t0, 1)
    import json
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("summary:", json.dumps(summary))
    if a.sections:
        for z in a.sections:
            png = rep.cross_section(ws, flags, z, os.path.join(a.out, "sections", f"z{int(z):06d}.png"), surface,
                                    title=os.path.basename(os.path.normpath(a.windings)))
            print("wrote", png)


def _ensure(*parts):
    p = os.path.join(*parts)
    os.makedirs(p, exist_ok=True)
    return p


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
    p.add_argument("--no-sheets", action="store_true", help="spacing only (no surface prediction)")
    p.set_defaults(func=cmd_audit)
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
