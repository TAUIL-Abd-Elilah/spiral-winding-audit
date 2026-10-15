"""winding-audit: check the windings of a spiral fit against each other and the surface prediction.

  winding-audit audit <windings dir> --surface <prediction zarr> [--group 0] --out <dir> [--voxel-um 8.64]
                      [--merge-frac 0.3] [--skip-frac 1.7] [--threshold 128] [--node-tol 2.0] [--radius 4.0]
                      [--subdir tifxyz/flatten.tifxyz] [--sections z1 z2 ..] [--no-sheets] [--grid-figures]
  winding-audit repair <windings dir> --surface <prediction zarr> [--group 0] --out <dir> [--subdir ...]
                       [--node-tol 2.0] [--max-shift 1.25] [--jump 0.4] [--audit --voxel-um 8.64 --sections ..]
                       [--ct <raw CT zarr or chunk folder>] [--ct-prominence 1.0]
  winding-audit pieces <windings dir> --surface <prediction zarr> [--group 0] --out <dir> [--subdir ...]
                       [--voxel-um 8.64] [--min-area-cm2 0.05] [--figures]
  winding-audit density <windings dir> --surface <prediction zarr> [--group 0] --z Z [--z Z2 ..]
                        [--first w040] [--last w115] [--subdir ...]
  winding-audit synthetic <out dir> [--merge W T0 T1] [--skip W T0 T1] [--gap W T0 T1]
  winding-audit ct-inputs --ct <raw CT zarr or chunk folder> [--ct-group 0 --ct-scale 1] [--shape Z Y X]
                          [--umbilicus <umbilicus.json>] --z0 Z0 --z1 Z1 --out <dataset dir> [--z-step 8]
                          [--angles 720] [--r-min 8] [--r-max R] [--prominence 1.0] [--floor-frac 0.35]
                          [--gap-lo 0.6] [--gap-hi 1.5] [--min-crossings 4] [--edge-trim 1] [--tube 1 1]
  winding-audit check-store <winding_inference dir>
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
from .sheets import open_surface, prediction_pitch, sheet_counts
from .spacing import neighbour_spacing
from .tifxyz import Winding, list_windings, load_winding, save_winding


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
        # the pitch comes from the prediction when there is one: a pair on one sheet cannot shrink it
        pitch = _retry(lambda: prediction_pitch(w1, surface, a.threshold)) if surface is not None else None
        sp = neighbour_spacing(w1, w2, a.merge_frac, a.skip_frac,
                               pitch=pitch if pitch is not None and np.isfinite(pitch) else None)
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
        print(f"{w1.name:>16s} -> {w2.name:<16s} pitch {sp.pitch:5.1f} vox, pair spacing "
              f"{row['pair_median_vox']:5.1f}  spacing merged {row['frac_merged']:.3f} "
              f"skipped {row['frac_skipped']:.3f}" + (f" | off-sheet {row['frac_node_off_sheet']:.3f} | sheets: "
              f"adjacent {row['frac_sheet_adjacent']:.3f} merged {row['frac_sheet_merged']:.3f} "
              f"fused {row['frac_sheet_fused']:.3f} skipped {row['frac_sheet_skipped']:.3f}" if sc is not None else ""),
              flush=True)
    rep.add_local_pitch(rows, a.voxel_um)
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
    ct = None
    if a.ct:
        from .volumes import open_volume
        ct = open_volume(a.ct, a.ct_group, like=surface)
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
                                            a.max_shift, jump_frac=a.jump, ct=ct, ct_prominence=a.ct_prominence))
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


def cmd_pieces(a):
    """Single-sheet, deduplicated pieces per winding; resumable (a finished winding's labels are loaded)."""
    import csv

    import tifffile

    from .pieces import winding_pieces

    t0 = time.time()
    ws = list_windings(a.windings, subdir=a.subdir)
    surface = open_surface(a.surface, a.group)
    wdir, ldir = _ensure(a.out, "windings"), _ensure(a.out, "labels")
    rows, prev_kept = [], None
    for i, w in enumerate(ws):
        lab_f, row_f = os.path.join(ldir, f"{w.name}.tif"), os.path.join(ldir, f"{w.name}.json")
        if os.path.exists(lab_f) and os.path.exists(row_f):
            prev_kept = tifffile.imread(lab_f) > 0
            rows.append(json.load(open(row_f)))
            continue
        node_cm2 = (w.grid_step * a.voxel_um * 1e-4) ** 2 if a.voxel_um else None
        min_nodes = max(1, int(round(a.min_area_cm2 / node_cm2))) if node_cm2 else a.min_nodes
        res = _retry(lambda: winding_pieces(w, ws[i - 1] if i else None, ws[i + 1] if i + 1 < len(ws) else None,
                                            surface, inner_kept=prev_kept, threshold=a.threshold,
                                            node_tol=a.node_tol, min_nodes=min_nodes))
        prev_kept = res.kept
        tifffile.imwrite(lab_f, res.label)
        masked = Winding(w.name, w.path, np.where(res.kept[..., None], w.xyz, np.nan), dict(w.meta))
        masked.meta["pieces"] = {"tool": "winding-audit pieces", "pieces": res.stats["pieces"],
                                 "kept_nodes": res.stats["kept"], "node_tol": a.node_tol, "pitch": res.pitch}
        save_winding(masked, os.path.join(wdir, w.name))
        row = {"winding": w.name, **res.stats}
        if node_cm2:
            sizes = np.bincount(res.label.ravel())[1:]
            row.update({"area_cm2": res.stats["n"] * node_cm2, "kept_cm2": res.stats["kept"] * node_cm2,
                        "duplicate_cm2": res.stats["duplicate"] * node_cm2,
                        "largest_piece_cm2": res.stats["largest"] * node_cm2,
                        "pieces_over_1cm2": int((sizes * node_cm2 >= 1.0).sum())})
        json.dump(row, open(row_f, "w"))
        rows.append(row)
        if a.figures:
            _pieces_figure(w, res, os.path.join(a.out, "figures", f"{w.name}.png"))
        print(f"{w.name:>16s} kept {row['kept'] / max(row['n'], 1):.2f} of {row['n']} nodes in {row['pieces']} pieces"
              f" (largest {row['largest']}), duplicate {row['duplicate'] / max(row['n'], 1):.2f}, on-sheet "
              f"{row['on_sheet'] / max(row['n'], 1):.2f}", flush=True)
    if rows:
        with open(os.path.join(a.out, "pieces.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        keys = ["n", "on_sheet", "duplicate", "kept", "pieces"] + (
            ["area_cm2", "kept_cm2", "duplicate_cm2", "pieces_over_1cm2"] if "area_cm2" in rows[0] else [])
        tot = {k: sum(r[k] for r in rows) for k in keys}
        tot["windings"] = len(rows)
        tot["seconds"] = round(time.time() - t0, 1)
        json.dump(tot, open(os.path.join(a.out, "pieces_summary.json"), "w"), indent=1)
        print("pieces:", json.dumps(tot), flush=True)


def _pieces_figure(w, res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.full(res.label.shape, np.nan)
    img[w.valid] = -1.0                                  # not kept
    img[res.duplicate & w.valid] = -2.0                  # duplicate of the inner neighbour
    k = res.label > 0
    img[k] = (res.label[k] - 1) % 20
    fig, ax = plt.subplots(figsize=(min(18, 4 + w.xyz.shape[1] / 40), 3.2), dpi=110)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#bcbd22", "#8c564b", "#e377c2",
                           "#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5", "#9edae5", "#dbdb8d", "#c49c94", "#f7b6d2",
                           "#393b79", "#637939", "#8c6d31", "#7b4173"])    # no reds: red is the duplicate colour
    ax.imshow(np.where(img >= 0, img, np.nan), cmap=cmap, vmin=-0.5, vmax=19.5, interpolation="nearest",
              aspect="auto")
    ax.imshow(np.where(img == -1, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.5, interpolation="nearest",
              aspect="auto")
    ax.imshow(np.where(img == -2, 1.0, np.nan), cmap="Reds", vmin=0, vmax=1.5, interpolation="nearest",
              aspect="auto")
    ax.set_title(f"{w.name}: single-sheet pieces (colours), duplicate of the inner winding (red), "
                 f"off a sheet or too small (grey); {res.stats['pieces']} pieces", fontsize=8)
    ax.set_xlabel("around the turn (grid columns)")
    ax.set_ylabel("along the axis (rows)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def cmd_density(a):
    """Windings per predicted sheet along radial rays, in one or more slices."""
    from .density import windings_per_sheet

    ws = list_windings(a.windings, subdir=a.subdir)
    names = [w.name for w in ws]
    pick = lambda tag, default: next((i for i, n in enumerate(names) if n.startswith(tag)), default) if tag else default
    first, last = pick(a.first, min(10, len(ws) - 2)), pick(a.last, len(ws) - 1)
    surface = open_surface(a.surface, a.group)
    out = []
    for z in a.z:
        r = windings_per_sheet(ws, surface, int(z), first, last, n_rays=a.rays, threshold=a.threshold)
        out.append(r)
        if r.get("rays"):
            print(f"z {r['z']}: {r['rays']} rays, {r['first']} to {r['last']}: {r['windings_crossed']:.0f} windings, "
                  f"{r['sheets_crossed']:.0f} predicted sheets -> {r['windings_per_sheet']:.2f} windings per sheet "
                  f"(IQR {r['windings_per_sheet_p25']:.2f}-{r['windings_per_sheet_p75']:.2f}); "
                  f"{r['radius_per_winding']:.1f} voxels of radius per winding", flush=True)
        else:
            print(f"z {int(z)}: no usable rays", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


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


def _scroll_radius(ct, umb, z0, z1, S, coarse):
    """Largest distance (CT voxels) from the umbilicus to the scroll in the band, from data that costs little to
    read: the present chunks of a chunk cache, or a coarse level of an OME-Zarr. None when neither is available."""
    zc0, zc1 = z0 / S, z1 / S
    if hasattr(ct, "_present"):
        c = ct.chunks[0]
        ks = np.array([k for k in ct._present if zc0 // c <= k[0] <= (zc1 - 1) // c])
        if len(ks) == 0:
            return None
        u = umb(np.array([z0, z1 - 1], np.float64)) / S
        best = 0.0
        for dy in (0, c):
            for dx in (0, c):
                y, x = ks[:, 1] * c + dy, ks[:, 2] * c + dx
                best = max(best, float(np.max(np.hypot(y[:, None] - u[None, :, 0], x[:, None] - u[None, :, 1]))))
        return best
    if not coarse:
        return None
    full = coarse[0]
    for _key, arr in coarse[1:]:
        f = full.shape[1] / arr.shape[1]
        if f < 2 or arr.shape[1] * arr.shape[2] * max(1, int((zc1 - zc0) / f)) > 64_000_000:
            continue
        z_lo, z_hi = int(zc0 // f), int(np.ceil(zc1 / f))
        slab = np.asarray(arr[z_lo:max(z_hi, z_lo + 1)])
        best = None
        for i in range(slab.shape[0]):                  # each coarse slice against the umbilicus at that slice
            nz = np.nonzero(slab[i] > 0)
            if len(nz[0]) == 0:
                continue
            u = umb(np.array([(z_lo + i + 0.5) * f * S], np.float64))[0] / S
            d = float(np.max(np.hypot(nz[0] * f - u[0], nz[1] * f - u[1])))
            best = d if best is None else max(best, d)
        return None if best is None else best + 2 * f
    return None


def _villa_path(root, p):
    from pathlib import Path
    q = Path(os.path.expandvars(os.path.expanduser(str(p).strip())))
    if not q.is_absolute():
        q = Path(root) / q
    return os.path.normcase(str(q.resolve(strict=False)))


def cmd_ct_inputs(a):
    from . import ctinputs as ci
    from .volumes import ChunkCacheArray, is_chunk_cache
    store_dir, shell_dir = os.path.join(a.out, "winding_inference"), os.path.join(a.out, "outer_shell")
    for d in (store_dir, shell_dir):
        if os.path.exists(d):
            sys.exit(f"{d} exists; remove it or choose another --out")
    if a.window_crossings and (a.window_crossings < max(a.min_crossings, 2)
                               or not 1 <= a.window_stride <= a.window_crossings):
        sys.exit("--window-crossings must be 0 or at least --min-crossings (and at least 2), with "
                 "1 <= --window-stride <= --window-crossings")
    if a.min_crossings < 2:
        sys.exit("--min-crossings must be at least 2")
    # the dataset's own spec: villa scales its umbilicus by it, and may read the inputs from other paths
    spec_path = os.path.join(a.out, "spiral-scroll.json")
    spec = json.load(open(spec_path)) if os.path.exists(spec_path) else {}
    paths = spec.get("paths") or {}
    villa_umb = _villa_path(a.out, paths.get("umbilicus")) if paths.get("umbilicus") else \
        _villa_path(a.out, "umbilicus.json")
    if a.umbilicus is None:
        if not os.path.exists(villa_umb):
            sys.exit(f"give --umbilicus (no {villa_umb})")
        a.umbilicus = villa_umb
    elif (spec or os.path.exists(villa_umb)) and _villa_path(".", a.umbilicus) != villa_umb:
        sys.exit(f"--umbilicus {a.umbilicus} is not the umbilicus villa will read for this dataset ({villa_umb}); "
                 "the rays must start from the fit's own umbilicus")
    um_scale = a.umbilicus_scale
    if spec:
        spec_scale = float((spec.get("umbilicus") or {}).get("coordinate_scale", 1.0))
        if um_scale is not None and abs(um_scale - spec_scale) > 1e-9:
            sys.exit(f"--umbilicus-scale {um_scale} disagrees with {spec_path} (umbilicus.coordinate_scale "
                     f"{spec_scale}), which villa applies")
        um_scale = spec_scale
        for key, want in (("winding_inference", store_dir), ("outer_shell", shell_dir)):
            p = paths.get(key)
            if p and _villa_path(a.out, p) != _villa_path(".", want):
                sys.exit(f"{spec_path} points villa's {key} at {p}; write there or remove the override")
    um_scale = 1.0 if um_scale is None else um_scale
    if a.ct_group != "0" and a.ct_scale is None:
        sys.exit("--ct-group other than 0: give --ct-scale (fit voxels per CT voxel, e.g. 2 for level 1 of a scan "
                 "fitted at level 0) so everything is written in the fit's frame")
    S = 1.0 if a.ct_scale is None else float(a.ct_scale)
    coarse = None
    if is_chunk_cache(a.ct):
        if not a.shape:
            sys.exit("--ct is a chunk folder: give the volume's --shape Z Y X")
        ct = ChunkCacheArray(a.ct, a.shape)
    else:
        ct = open_surface(a.ct, a.ct_group)
        try:
            import zarr
            g = zarr.open(a.ct, mode="r")
            if not hasattr(g, "shape"):
                keys = sorted((k for k in g.array_keys() if k.isdigit()), key=int)
                coarse = [ct] + [(k, g[k]) for k in keys if int(k) > int(a.ct_group)]
        except Exception:
            coarse = None
    umb = ci.load_umbilicus(a.umbilicus, um_scale)
    shape = np.array(ct.shape)
    uyx = umb(np.array([a.z0, a.z1 - 1], np.float64)) / S
    corners = np.array([[0, 0], [0, shape[2]], [shape[1], 0], [shape[1], shape[2]]], np.float64)
    corner = float(np.max(np.hypot(*(corners[None] - uyx[:, None]).transpose(2, 0, 1)))) * S
    r_max, r_note = a.r_max, "given"
    if r_max is None:
        rad = _scroll_radius(ct, umb, a.z0, a.z1, S, coarse)
        if rad is None:
            r_max, r_note = corner, "farthest volume corner (no cheap scroll extent)"
        else:
            r_max = min(corner, rad * S + a.shell_clear + a.shell_margin + 2 * S)
            r_note = "scroll extent + shell clearance"
    prm = ci.RayParams(r_min=a.r_min, r_max=r_max, tube=tuple(a.tube), prominence=a.prominence,
                       floor_frac=a.floor_frac, window=a.window, gap_lo=a.gap_lo, gap_hi=a.gap_hi,
                       edge_trim=a.edge_trim, min_crossings=a.min_crossings, shell_clear=a.shell_clear,
                       shell_margin=a.shell_margin, window_crossings=a.window_crossings,
                       window_stride=a.window_stride)
    t0 = time.time()

    def progress(i, n, el):
        print(f"slice {i}/{n}  {el:7.0f} s", flush=True)

    rays, outer, inner, zs, thetas, stats = ci.generate(ct, umb, a.z0, a.z1, a.z_step, a.angles, prm, ct_scale=S,
                                                        progress=progress)
    zyx, measured = ci.shell_grid(outer, inner, zs, thetas, umb, a.shell_angles)
    if not rays:
        sys.exit("no ray kept a run of regular crossings; nothing written")
    if not ci.has_valid_quad(zyx):
        sys.exit("no outer shell could be built (no crossing stored and the outside not seen); nothing written")
    prov = {"ct": os.path.abspath(a.ct), "ct_group": a.ct_group, "ct_scale": S,
            "umbilicus": os.path.abspath(a.umbilicus), "umbilicus_scale": um_scale, "z0": a.z0, "z1": a.z1,
            "z_step": a.z_step, "angles": a.angles, "r_max_source": r_note, "params": ci.params_dict(prm)}
    manifest = ci.write_store(store_dir, rays, provenance=prov)
    meta = ci.write_shell(shell_dir, zyx, measured, provenance=prov)
    summary = ci.check_store(store_dir)
    ok = zyx[..., 0] >= 0
    est = ci.outer_winding_estimate(outer, stats["median_spacing"])
    warnings = []
    if meta["measured_share"] is not None and meta["measured_share"] < 0.9:
        warnings.append(f"the scroll's outside was seen on {meta['measured_share']:.0%} of the shell; elsewhere it is "
                        "filled in and may lie inside the scroll: fine as villa's pair filter, but villa also drops "
                        "tracks outside the shell when a tracks store is used, and it is not fit for the shell loss")
    if stats["slices_without_floor"]:
        warnings.append(f"{stats['slices_without_floor']} slices had too little CT inside the scroll for a floor")
    report = dict(provenance=prov, stats=stats, store=summary, shell_rows=int(zyx.shape[0]),
                  shell_share_valid=float(ok.mean()), shell_measured_share=meta["measured_share"],
                  outer_winding_estimate=est, warnings=warnings, seconds=time.time() - t0)
    json.dump(report, open(os.path.join(a.out, "ct_inputs_report.json"), "w"), indent=1)
    json.dump(ci.fit_overrides(a.z0, a.z1, a.out), open(os.path.join(a.out, "fit_overrides.json"), "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("stats", "store", "shell_measured_share", "seconds")}, indent=1))
    for w in warnings:
        print("warning:", w)
    guide = f" (rough guide from the CT: about {est} windings to the outside)." if est else "."
    print(f"wrote {store_dir} ({manifest['num_rays']} rays, {manifest['num_crossings']} crossings) and {shell_dir}. "
          "fit_overrides.json is for FIT_SPIRAL_CONFIG_OVERRIDES (in a VC3D session z_begin/z_end go in the run "
          "block). Set shell_outer_winding_idx for this scroll whatever the shell loss weight: it bounds the dense "
          "and regularisation losses" + guide)


def cmd_check_store(a):
    from . import ctinputs as ci
    print(json.dumps(ci.check_store(a.store), indent=1))


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
    r.add_argument("--ct", help="raw CT (OME-Zarr, or a folder of <z>_<y>_<x>.bin 128^3 chunks on the prediction's "
                                "grid): keep nodes on CT layers the prediction misses, and move stacked nodes to a "
                                "free CT layer between their neighbours")
    r.add_argument("--ct-group", default="0")
    r.add_argument("--ct-prominence", type=float, default=1.0,
                   help="a CT layer is a local maximum this many local standard deviations above its surroundings")
    r.add_argument("--audit", action="store_true", help="audit the repaired band into <out>/audit")
    r.add_argument("--voxel-um", type=float); r.add_argument("--sections", type=float, nargs="*")
    r.add_argument("--grid-figures", action="store_true")
    r.set_defaults(func=cmd_repair)
    q = sub.add_parser("pieces", help="single-sheet, deduplicated pieces of each winding (a mask per winding)")
    q.add_argument("windings"); q.add_argument("--surface", required=True); q.add_argument("--group", default="0")
    q.add_argument("--out", required=True); q.add_argument("--subdir", default="")
    q.add_argument("--threshold", type=int, default=128); q.add_argument("--node-tol", type=float, default=2.0)
    q.add_argument("--voxel-um", type=float)
    q.add_argument("--min-area-cm2", type=float, default=0.05, help="smallest piece kept (needs --voxel-um)")
    q.add_argument("--min-nodes", type=int, default=50, help="smallest piece kept without --voxel-um")
    q.add_argument("--figures", action="store_true", help="one PNG per winding with the pieces")
    q.set_defaults(func=cmd_pieces)
    e = sub.add_parser("density", help="windings per predicted sheet along radial rays in a slice")
    e.add_argument("windings"); e.add_argument("--surface", required=True); e.add_argument("--group", default="0")
    e.add_argument("--subdir", default=""); e.add_argument("--z", type=float, nargs="+", required=True)
    e.add_argument("--first", help="name prefix of the inner winding of the count (default: the 11th)")
    e.add_argument("--last", help="name prefix of the outer winding of the count (default: the last)")
    e.add_argument("--rays", type=int, default=180); e.add_argument("--threshold", type=int, default=128)
    e.add_argument("--out", help="write the result as JSON")
    e.set_defaults(func=cmd_density)
    s = sub.add_parser("synthetic", help="write a synthetic spiral scroll with optional faults")
    s.add_argument("out"); s.add_argument("--sheets", type=int, default=6); s.add_argument("--pitch", type=float, default=10.0)
    s.add_argument("--merge", nargs=3, action="append", metavar=("W", "T0", "T1"))
    s.add_argument("--skip", nargs=3, action="append", metavar=("W", "T0", "T1"))
    s.add_argument("--gap", nargs=3, action="append", metavar=("W", "T0", "T1"))
    s.set_defaults(func=cmd_synthetic)
    c = sub.add_parser("ct-inputs", help="winding_inference/ and outer_shell/ for fit_spiral from the raw CT")
    c.add_argument("--ct", required=True, help="raw masked CT: OME-Zarr (group or array), or a folder of "
                                               "<z>_<y>_<x>.bin 128^3 chunks (then --shape)")
    c.add_argument("--ct-group", default="0", help="OME-Zarr level to read (default 0, the fit's grid)")
    c.add_argument("--ct-scale", type=float, default=None,
                   help="fit voxels per CT voxel, required for a --ct-group other than 0 (2 for level 1); CT voxel c "
                        "is taken at fit position c x scale, villa's convention for coarse volumes")
    c.add_argument("--shape", type=int, nargs=3)
    c.add_argument("--umbilicus", default=None,
                   help="umbilicus.json with control_points (villa's format); default: the one villa reads for "
                        "the dataset in --out (paths.umbilicus in its spiral-scroll.json, else umbilicus.json)")
    c.add_argument("--umbilicus-scale", type=float, default=None,
                   help="villa's umbilicus.coordinate_scale (default: from <out>/spiral-scroll.json, else 1)")
    c.add_argument("--z0", type=int, required=True); c.add_argument("--z1", type=int, required=True)
    c.add_argument("--z-step", type=int, default=8); c.add_argument("--angles", type=int, default=720)
    c.add_argument("--r-min", type=float, default=8.0)
    c.add_argument("--r-max", type=float, default=None,
                   help="default: the scroll's extent (present chunks, or a coarse OME-Zarr level) plus clearance")
    c.add_argument("--tube", type=int, nargs=2, default=[1, 1],
                   help="tube half-widths across the ray in the slice and in z (voxels)")
    c.add_argument("--prominence", type=float, default=1.0); c.add_argument("--floor-frac", type=float, default=0.35)
    c.add_argument("--window", type=float, default=40.0)
    c.add_argument("--gap-lo", type=float, default=0.6); c.add_argument("--gap-hi", type=float, default=1.5)
    c.add_argument("--edge-trim", type=int, default=1); c.add_argument("--min-crossings", type=int, default=4)
    c.add_argument("--window-crossings", type=int, default=16, help="longest stored ray, in crossings")
    c.add_argument("--window-stride", type=int, default=8)
    c.add_argument("--shell-clear", type=float, default=32.0); c.add_argument("--shell-margin", type=float, default=4.0)
    c.add_argument("--shell-angles", type=int, default=720)
    c.add_argument("--out", required=True, help="dataset folder; winding_inference/ and outer_shell/ go in it")
    c.set_defaults(func=cmd_ct_inputs)
    k = sub.add_parser("check-store", help="format check of a winding_inference store (no torch)")
    k.add_argument("store"); k.set_defaults(func=cmd_check_store)
    a = ap.parse_args(argv)
    if a.cmd == "audit" and not a.no_sheets and not a.surface:
        ap.error("--surface is required unless --no-sheets")
    a.func(a)


if __name__ == "__main__":
    main()
