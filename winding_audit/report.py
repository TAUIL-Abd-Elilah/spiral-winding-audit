"""Audit report: per-pair summary rows, JSON/CSV, and cross-section figures."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from .sheets import SheetCountResult
from .spacing import MERGED, OK, SKIPPED, SpacingResult, UNASSESSED
from .tifxyz import Winding


def pair_row(a: Winding, b: Winding, sp: SpacingResult, sc: SheetCountResult | None, voxel_um: float | None) -> dict:
    row = {"winding": a.name, "next": b.name, "n_nodes": int(a.valid.sum()), **sp.fractions()}
    if sc is not None:
        h = sc.histogram()
        row.update({f"sheet_{k}": v for k, v in h.items()})
        row["frac_runs2"] = h["runs2"] / max(h["n"], 1)
        row["frac_runs1"] = h["runs1"] / max(h["n"], 1)
        row["frac_runs3plus"] = h["runs3plus"] / max(h["n"], 1)
        row["frac_start_on_sheet"] = h["start_on_sheet"] / max(h["n"], 1)
    step = a.grid_step
    row["area_cm2"] = float(a.valid.sum()) * step * step * ((voxel_um or 1.0) * 1e-4) ** 2 if voxel_um else None
    row["pitch_um"] = sp.pitch * voxel_um if voxel_um else None
    return row


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
    if "frac_runs2" in rows[0]:
        out.update({"frac_runs2": wsum("frac_runs2"), "frac_runs1": wsum("frac_runs1"),
                    "frac_runs3plus": wsum("frac_runs3plus"), "frac_start_on_sheet": wsum("frac_start_on_sheet")})
    if rows[0].get("area_cm2") is not None:
        out["area_cm2"] = sum(r["area_cm2"] for r in rows)
    return out


def cross_section(windings: list[Winding], flags: dict[str, np.ndarray], z0: float, out_png: str,
                  surface=None, half_z: float | None = None, title: str = ""):
    """All windings' nodes within half_z of z0 over the surface-prediction slice, coloured by flag."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {OK: "#2ecc71", MERGED: "#e74c3c", SKIPPED: "#f39c12", UNASSESSED: "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    if surface is not None:
        zi = int(round(z0))
        if 0 <= zi < surface.shape[0]:
            ax.imshow(np.asarray(surface[zi]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for w in windings:
        hz = half_z if half_z is not None else w.grid_step / 2
        v = w.valid & (np.abs(w.xyz[..., 2] - z0) <= hz)
        if not v.any():
            continue
        f = flags.get(w.name)
        c = [colors[int(k)] for k in (f[v] if f is not None else np.full(v.sum(), UNASSESSED))]
        ax.scatter(w.xyz[v, 0], w.xyz[v, 1], s=3, c=c, linewidths=0)
    ax.set_title(f"{title} z={z0:.0f}: green ok, red merged, orange skipped, grey unassessed")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
