"""Per-winding figures: the flag map on the winding's own grid (rows = along the scroll axis, columns = around the
turn), the distance map, and the sheet-count map. These show *where* on a winding the problems are; the cross-sections
in report.py show them in the volume."""
from __future__ import annotations

import os

import numpy as np

from .sheets import SheetCountResult
from .spacing import MERGED, OK, SKIPPED, SpacingResult, UNASSESSED


def winding_grid_figure(name: str, sp: SpacingResult, sc: SheetCountResult | None, out_png: str, grid_step: float,
                        voxel_um: float | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    H, W = sp.flag.shape
    rows = 4 if sc is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(min(18, 4 + W / 40), 2.2 * rows + 1), dpi=110, sharex=True)
    axes = np.atleast_1d(axes)
    unit = "mm" if voxel_um else "grid steps"
    scale = (grid_step * voxel_um / 1000.0) if voxel_um else 1.0
    extent = [0, W * scale, H * scale, 0]

    flag = sp.flag.astype(float)
    flag[~np.isfinite(sp.distance)] = np.nan
    cmap = ListedColormap(["#2ecc71", "#e74c3c", "#f39c12", "#bdc3c7"])
    axes[0].imshow(flag, cmap=cmap, vmin=-0.5, vmax=3.5, extent=extent, interpolation="nearest", aspect="auto")
    axes[0].set_title(f"{name}: green ok, red merged, orange skipped, grey unassessed  (pitch {sp.pitch:.1f} vox)")
    axes[0].set_ylabel(f"along axis ({unit})")

    d = sp.distance / sp.pitch
    im = axes[1].imshow(d, cmap="coolwarm", vmin=0, vmax=2.5, extent=extent, interpolation="nearest", aspect="auto")
    axes[1].set_title("distance to the next winding, in sheet pitches (1 = one wrap)")
    axes[1].set_ylabel(f"along axis ({unit})")
    fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.01)

    if sc is not None:
        v = sc.verdict.astype(float)
        v[sc.verdict == 5] = np.nan
        v[sc.verdict == 6] = 5                      # draw 'fused' as the sixth colour
        vmap = ListedColormap(["#2ecc71", "#e74c3c", "#f39c12", "#8e44ad", "#d7bde2", "#3498db"])
        axes[2].imshow(v, cmap=vmap, vmin=-0.5, vmax=5.5, extent=extent, interpolation="nearest", aspect="auto")
        axes[2].set_title("surface prediction: green adjacent, red same sheet (fit), blue gap filled by the prediction "
                          "(fused), orange a sheet skipped, purple node off any sheet, lilac neighbour off sheet")
        axes[2].set_ylabel(f"along axis ({unit})")
        im = axes[3].imshow(sc.node_dist, cmap="magma_r", vmin=0, vmax=sc.node_dist[np.isfinite(sc.node_dist)].max()
                            if np.isfinite(sc.node_dist).any() else 1, extent=extent, interpolation="nearest",
                            aspect="auto")
        axes[3].set_title(f"distance from the node to the nearest predicted-sheet voxel (voxels; on sheet when "
                          f"<= {sc.node_tol:g})")
        axes[3].set_ylabel(f"along axis ({unit})")
        fig.colorbar(im, ax=axes[3], fraction=0.02, pad=0.01)
    axes[-1].set_xlabel(f"around the turn ({unit})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
