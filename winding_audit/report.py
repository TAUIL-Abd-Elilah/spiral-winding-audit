"""Audit report: per-pair summary rows, JSON/CSV, and cross-section figures."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from .sheets import SheetCountResult
from .spacing import MERGED, OK, SKIPPED, SpacingResult, UNASSESSED
from .tifxyz import Winding


COLLAPSED, STRETCHED = 0.75, 1.25    # pair pitch / local pitch outside these: the pair is collapsed or stretched


def pair_row(a: Winding, b: Winding, sp: SpacingResult, sc: SheetCountResult | None, voxel_um: float | None) -> dict:
    row = {"winding": a.name, "next": b.name, "n_nodes": int(a.valid.sum()), **sp.fractions()}
    row["pitch_vox"] = sp.pitch
    if sc is not None:
        h = sc.histogram()
        row.update({f"sheet_{k}": v for k, v in h.items()})
        n, both = max(h["n"], 1), max(h["both_on_sheet"], 1)
        row["node_tol"] = sc.node_tol
        row["frac_node_off_sheet"] = h["node_off_sheet"] / n            # of this winding's assessed nodes
        row["frac_neighbour_off_sheet"] = h["neighbour_off_sheet"] / n
        for k in h:                                                     # frac_node_within_1 / _2 / _3 voxels
            if k.startswith("node_within_"):
                row[f"frac_{k}"] = h[k] / n
        row["frac_sheet_adjacent"] = h["adjacent"] / both                # of nodes with both ends on a sheet
        row["frac_sheet_merged"] = h["merged"] / both
        row["frac_sheet_fused"] = h["fused"] / both
        row["frac_sheet_skipped"] = h["skipped"] / both
    step = a.grid_step
    row["area_cm2"] = float(a.valid.sum()) * step * step * ((voxel_um or 1.0) * 1e-4) ** 2 if voxel_um else None
    row["pitch_um"] = sp.pitch * voxel_um if voxel_um else None
    return row


def add_local_pitch(rows: list[dict], voxel_um: float | None = None, half_window: int = 3):
    """Local pitch = the mean pair pitch over +-half_window neighbouring pairs (the band's radial span per winding
    there). A pair whose two windings sit on one sheet has a small median spacing of its own and hides from its own
    pitch; against the local pitch it shows (ratio well below 1, its neighbour pair above 1)."""
    def pv(r):
        if r.get("pitch_vox") is not None:
            return r["pitch_vox"]
        return r["pitch_um"] / voxel_um if r.get("pitch_um") is not None and voxel_um else np.nan
    p = np.array([pv(r) for r in rows], float)
    for i, r in enumerate(rows):
        w = p[max(0, i - half_window):i + half_window + 1]
        local = float(np.nanmean(w)) if np.isfinite(w).any() else None
        r["pitch_vox"] = None if not np.isfinite(p[i]) else float(p[i])
        r["local_pitch_vox"] = local
        r["pitch_ratio"] = float(p[i] / local) if local and np.isfinite(p[i]) else None
    return rows


def write_rows(rows: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(out_dir, "audit.json"), "w"), indent=1)
    if rows:
        keys = list(rows[0].keys())
        with open(os.path.join(out_dir, "audit.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = sum(r["n_assessed"] for r in rows)
    wsum = lambda k: sum(r[k] * r["n_assessed"] for r in rows if r.get(k) is not None) / max(n, 1)
    out = {"pairs": len(rows), "nodes_assessed": n, "frac_merged": wsum("frac_merged"),
           "frac_skipped": wsum("frac_skipped")}
    if "frac_sheet_adjacent" in rows[0]:
        nb = sum(r["sheet_both_on_sheet"] for r in rows)
        bsum = lambda k: sum(r[k] * r["sheet_both_on_sheet"] for r in rows) / max(nb, 1)
        out.update({"node_tol": rows[0].get("node_tol"), "frac_node_off_sheet": wsum("frac_node_off_sheet"),
                    "frac_neighbour_off_sheet": wsum("frac_neighbour_off_sheet"),
                    **{k: wsum(k) for k in rows[0] if k.startswith("frac_node_within_")},
                    "nodes_both_on_sheet": nb, "frac_sheet_adjacent": bsum("frac_sheet_adjacent"),
                    "frac_sheet_merged": bsum("frac_sheet_merged"), "frac_sheet_fused": bsum("frac_sheet_fused"),
                    "frac_sheet_skipped": bsum("frac_sheet_skipped")})
    if rows[0].get("area_cm2") is not None:
        out["area_cm2"] = sum(r["area_cm2"] for r in rows)
    ratio = [r for r in rows if r.get("pitch_ratio") is not None]
    if ratio:
        col = [r for r in ratio if r["pitch_ratio"] < COLLAPSED]
        out["pairs_collapsed"] = len(col)          # the two windings much closer than the local pitch
        out["pairs_stretched"] = sum(r["pitch_ratio"] > STRETCHED for r in ratio)
        if rows[0].get("area_cm2") is not None:
            out["area_cm2_collapsed"] = sum(r["area_cm2"] for r in col)
    return out


SPACING_COLORS = {OK: "#2ecc71", MERGED: "#e74c3c", SKIPPED: "#f39c12", UNASSESSED: "#7f8c8d"}
# sheet verdicts: adjacent, merged, skipped, node off sheet, neighbour off sheet, unassessed, fused
VERDICT_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12", 3: "#8e44ad", 4: "#d7bde2", 5: "#7f8c8d", 6: "#3498db"}


def cross_section(windings: list[Winding], flags: dict[str, np.ndarray], z0: float, out_png: str,
                  surface=None, half_z: float | None = None, title: str = "", by_verdict: bool = False):
    """All windings' nodes within half_z of z0 over the surface-prediction slice, coloured by spacing flag or, with
    ``by_verdict`` (``flags`` then holds verdict maps), by the sheet verdict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = VERDICT_COLORS if by_verdict else SPACING_COLORS
    legend = ("green adjacent, red same sheet, blue gap filled by the prediction, orange skipped, "
              "purple node off any sheet, lilac neighbour off sheet, grey unassessed" if by_verdict else
              "green ok, red merged, orange skipped, grey unassessed")
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    sel = []
    for w in windings:
        hz = half_z if half_z is not None else w.grid_step / 2
        v = w.valid & (np.abs(w.xyz[..., 2] - z0) <= hz)
        if v.any():
            sel.append((w, v))
    if surface is not None and sel:
        # read only the windings' bounding box of this slice (a full slice of a 192^3-chunked store is thousands of
        # chunks), subsampled for display
        zi = int(round(z0))
        if 0 <= zi < surface.shape[0]:
            pts = np.concatenate([w.xyz[v][:, :2] for w, v in sel])
            x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - 64, 0)
            x1 = min(int(np.ceil(pts[:, 0].max())) + 64, surface.shape[2])
            y1 = min(int(np.ceil(pts[:, 1].max())) + 64, surface.shape[1])
            step = max(1, int(np.ceil(max(x1 - x0, y1 - y0) / 2000)))
            img = np.asarray(surface[zi, y0:y1, x0:x1])[::step, ::step]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                      extent=[x0, x0 + img.shape[1] * step, y0 + img.shape[0] * step, y0])
    for w, v in sel:
        f = flags.get(w.name)
        c = [colors[int(k)] for k in (f[v] if f is not None else np.full(v.sum(), UNASSESSED))]
        ax.scatter(w.xyz[v, 0], w.xyz[v, 1], s=3, c=c, linewidths=0)
    ax.set_title(f"{title} z={z0:.0f}\n{legend}", fontsize=8, wrap=True)
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
