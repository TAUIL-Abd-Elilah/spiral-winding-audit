"""The repair must move merged and off-sheet nodes onto the sheet between their neighbours, and leave clean nodes,
fused-prediction regions and skipped wraps alone."""
import numpy as np
import pytest

from winding_audit import synthetic as syn
from winding_audit.repair import R_KEPT, R_MOVED, R_NOT_ASSESSED, repair_band
from winding_audit.sheets import S_ADJACENT, S_MERGED, S_START_OFF, open_surface, sheet_counts
from winding_audit.spacing import neighbour_spacing
from winding_audit.tifxyz import list_windings

PITCH = 10.0


@pytest.fixture(scope="module")
def repaired(tmp_path_factory):
    out = tmp_path_factory.mktemp("rep")
    vol, centre = syn.make_volume(n_sheets=6, pitch=PITCH)
    syn.fuse_sheets(vol, w=4, t0=0.3, t1=1.3, pitch=PITCH, centre=centre)
    ws, thetas = syn.make_windings(n_sheets=6, pitch=PITCH, centre=centre)
    syn.inject_merge(ws, thetas, w=1, t0=1.0, t1=2.0, pitch=PITCH, centre=centre)
    syn.inject_skip(ws, thetas, w=4, t0=3.5, t1=4.5, pitch=PITCH, centre=centre)
    syn.inject_gap(ws, thetas, w=2, t0=5.0, t1=5.8, pitch=PITCH, centre=centre)
    syn.inject_offset(ws, thetas, w=1, t0=4.0, t1=4.8, pitch=PITCH, centre=centre)
    s, d = syn.write_dataset(str(out), vol, ws)
    surface = open_surface(s, "0")
    windings = list_windings(d)
    return surface, windings, thetas, repair_band(windings, surface)


def _cols(thetas, t0, t1, margin=0.15):
    return (thetas >= t0 + margin) & (thetas < t1 - margin)


def _assessed(st):
    """The band's top and bottom rows have no neighbour above or below to compare with: not assessed."""
    return st[st != R_NOT_ASSESSED]


def _faults(thetas):
    """Every injected fault plus its blur at the edges."""
    m = np.zeros_like(thetas, bool)
    for t0, t1 in ((0.7, 2.3), (3.2, 4.8), (3.7, 5.1), (4.7, 6.1), (0.0, 1.6)):
        m |= (thetas >= t0) & (thetas < t1)
    return m


def test_merged_region_moves_one_pitch_inward(repaired):
    _, ws, thetas, res = repaired
    r = res[1]
    inside = _cols(thetas, 1.0, 2.0)
    st, sh = r.status[:, inside], r.shift[:, inside]
    assert (_assessed(st) == R_MOVED).mean() > 0.9, np.bincount(st.ravel(), minlength=5)
    assert np.all(np.abs(sh[st == R_MOVED] + PITCH) < 2.0)     # back to its own sheet, one pitch in


def test_off_sheet_region_moves_back(repaired):
    _, ws, thetas, res = repaired
    r = res[2]
    inside = _cols(thetas, 5.0, 5.8)
    st, sh = r.status[:, inside], r.shift[:, inside]
    assert (_assessed(st) == R_MOVED).mean() > 0.9, np.bincount(st.ravel(), minlength=5)
    assert np.all(np.abs(sh[st == R_MOVED] + 0.5 * PITCH) < 2.0)


def test_clean_nodes_untouched(repaired):
    _, ws, thetas, res = repaired
    clean = ~_faults(thetas)
    for k in (1, 2, 3):
        r = res[k]
        assert (_assessed(r.status[:, clean]) == R_KEPT).mean() > 0.97
        assert np.array_equal(r.winding.xyz[:, clean], ws[k].xyz[:, clean])
    # a node 1.5 voxels off its sheet centre is still on its sheet: kept
    assert (_assessed(res[1].status[:, _cols(thetas, 4.0, 4.8)]) == R_KEPT).mean() > 0.9


def test_fused_and_skipped_regions_are_left_alone(repaired):
    _, ws, thetas, res = repaired
    r = res[4]
    for t0, t1 in ((0.3, 1.3), (3.5, 4.5)):
        c = _cols(thetas, t0, t1)
        assert (r.status[:, c] == R_MOVED).mean() < 0.05
        assert np.allclose(r.winding.xyz[:, c], ws[4].xyz[:, c])
    assert res[0] is None and res[-1] is None                  # edge windings have one neighbour: not repaired


def test_audit_after_repair_is_clean(repaired):
    surface, ws, thetas, res = repaired
    new = [ws[0]] + [r.winding for r in res[1:-1]] + [ws[-1]]
    inside = _cols(thetas, 1.0, 2.0)
    for k in (0, 1):
        sp = neighbour_spacing(new[k], new[k + 1])
        v = sheet_counts(new[k], sp, surface).verdict[:, inside]
        v = v[v != 5]
        assert (v == S_ADJACENT).mean() > 0.9, (k, np.bincount(v, minlength=7))
        assert (v == S_MERGED).mean() < 0.05
    gap = _cols(thetas, 5.0, 5.8)
    sp = neighbour_spacing(new[2], new[3])
    v = sheet_counts(new[2], sp, surface).verdict[:, gap]
    v = v[v != 5]
    assert (v == S_START_OFF).mean() < 0.05


def test_cli_repair_runs_and_resumes(tmp_path):
    from winding_audit.cli import main
    vol, centre = syn.make_volume(n_sheets=5, pitch=PITCH)
    ws, thetas = syn.make_windings(n_sheets=5, pitch=PITCH, centre=centre)
    syn.inject_merge(ws, thetas, w=2, t0=1.0, t1=2.0, pitch=PITCH, centre=centre)
    s, d = syn.write_dataset(str(tmp_path / "data"), vol, ws)
    out = tmp_path / "rep"
    args = ["repair", d, "--surface", s, "--out", str(out), "--audit", "--voxel-um", "8.64"]
    main(args)
    assert (out / "repair.csv").exists() and (out / "audit" / "summary.json").exists()
    assert sorted(p.name for p in (out / "windings").iterdir()) == [f"w{k:03d}_syn" for k in range(5)]
    import json
    first = json.load(open(out / "repair_summary.json"))
    assert first["moved"] > 0
    main(args)                                                   # second run loads what was written
    assert json.load(open(out / "repair_summary.json"))["moved"] == first["moved"]
