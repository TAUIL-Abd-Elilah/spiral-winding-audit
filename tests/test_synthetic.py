"""The audit must flag an injected merger and an injected skipped wrap, and nothing on clean pairs."""
import numpy as np
import pytest

from winding_audit import synthetic as syn
from winding_audit.sheets import S_ADJACENT, S_FUSED, S_MERGED, S_SKIPPED, S_START_OFF, open_surface, sheet_counts
from winding_audit.spacing import MERGED, OK, SKIPPED, neighbour_spacing
from winding_audit.tifxyz import list_windings

PITCH = 10.0


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    vol, centre = syn.make_volume(n_sheets=6, pitch=PITCH)
    syn.fuse_sheets(vol, w=4, t0=0.3, t1=1.3, pitch=PITCH, centre=centre)                # prediction fills gap w4..w5
    ws, thetas = syn.make_windings(n_sheets=6, pitch=PITCH, centre=centre)
    syn.inject_merge(ws, thetas, w=1, t0=1.0, t1=2.0, pitch=PITCH, centre=centre)     # w1 on sheet 2 for theta 1..2
    syn.inject_skip(ws, thetas, w=4, t0=3.5, t1=4.5, pitch=PITCH, centre=centre)      # w4, w5 shifted out: w3->w4 spans 2
    syn.inject_gap(ws, thetas, w=2, t0=5.0, t1=5.8, pitch=PITCH, centre=centre)       # w2 half a pitch out: off any sheet
    syn.inject_offset(ws, thetas, w=1, t0=4.0, t1=4.8, pitch=PITCH, centre=centre)    # w1 1.5 vox out: at its sheet's edge
    s, d = syn.write_dataset(str(out), vol, ws)
    windings = list_windings(d)
    return open_surface(s, "0"), windings, thetas, {"surface": s, "windings": d}


def _theta_cols(thetas, t0, t1):
    return (thetas >= t0) & (thetas < t1)


def _interior(thetas, t0, t1, margin=0.3):
    """Columns well inside an injected fault: at its edges the nearest-neighbour distance blurs over a few columns."""
    return (thetas >= t0 + margin) & (thetas < t1 - margin)


def _assessed(flag):
    return flag[flag != 3]


def test_clean_pair_is_clean(dataset):
    surface, ws, thetas, _ = dataset
    sp = neighbour_spacing(ws[0], ws[1])          # w0 -> w1 is untouched except where w1 moved outward (merge region)
    cols = ~_theta_cols(thetas, 0.7, 2.3)         # the fault plus its blur at both edges
    f = _assessed(sp.flag[:, cols])
    assert abs(sp.pitch - PITCH) < 1.0
    assert (f == OK).mean() > 0.97
    assert (sp.flag[:, cols] == 3).mean() < 0.25  # only the band's top and bottom rows may be unassessed
    sc = sheet_counts(ws[0], sp, surface)
    v = sc.verdict[:, cols]
    v = v[v != 5]
    assert (v == S_ADJACENT).mean() > 0.97


def test_merger_is_flagged(dataset):
    surface, ws, thetas, _ = dataset
    sp = neighbour_spacing(ws[1], ws[2])          # w1 sits on sheet 2 for theta in [1, 2): merged with w2 there
    inside = _interior(thetas, 1.0, 2.0)
    outside = ~_theta_cols(thetas, 0.7, 2.3)
    assert (_assessed(sp.flag[:, inside]) == MERGED).mean() > 0.9
    assert (_assessed(sp.flag[:, outside]) == MERGED).mean() < 0.02
    sc = sheet_counts(ws[1], sp, surface)
    v_in = sc.verdict[:, inside]
    assert (v_in[v_in != 5] == S_MERGED).mean() > 0.9


def test_skip_is_flagged(dataset):
    surface, ws, thetas, _ = dataset
    sp = neighbour_spacing(ws[3], ws[4])          # w4 moved to sheet 5 for theta in [3.5, 4.5): w3 -> w4 spans 2 pitches
    inside = _interior(thetas, 3.5, 4.5)
    outside = ~_theta_cols(thetas, 3.2, 4.8)
    assert (_assessed(sp.flag[:, inside]) == SKIPPED).mean() > 0.9
    assert (_assessed(sp.flag[:, outside]) == SKIPPED).mean() < 0.02
    sc = sheet_counts(ws[3], sp, surface)
    v_in = sc.verdict[:, inside]
    assert (v_in[v_in != 5] == S_SKIPPED).mean() > 0.9


def test_neighbour_at_sheet_edge_is_still_adjacent(dataset):
    """w1 sits at the edge of its sheet for theta in [4, 4.8): the run may start one voxel past the node, which must
    not turn w0 -> w1 into 'merged' (the bug this test guards)."""
    surface, ws, thetas, _ = dataset
    sp = neighbour_spacing(ws[0], ws[1])
    inside = _interior(thetas, 4.0, 4.8, margin=0.15)
    sc = sheet_counts(ws[0], sp, surface)
    v_in = sc.verdict[:, inside]
    v_in = v_in[v_in != 5]
    assert (v_in == S_ADJACENT).mean() > 0.9, np.bincount(v_in, minlength=6)


def test_fused_prediction_is_separated_from_merger(dataset):
    """Between w4 and w5 the prediction fills the gap for theta in [0.3, 1.3): the windings are a pitch apart (spacing
    ok) but one unbroken run joins them -> 'fused', not 'merged'."""
    surface, ws, thetas, _ = dataset
    sp = neighbour_spacing(ws[4], ws[5])
    inside = _interior(thetas, 0.3, 1.3, margin=0.15)
    assert (_assessed(sp.flag[:, inside]) == OK).mean() > 0.9
    sc = sheet_counts(ws[4], sp, surface)
    v_in = sc.verdict[:, inside]
    v_in = v_in[v_in != 5]
    assert (v_in == S_FUSED).mean() > 0.9, np.bincount(v_in, minlength=7)
    assert (v_in == S_MERGED).mean() < 0.05


def test_node_off_sheet_is_reported(dataset):
    surface, ws, thetas, _ = dataset
    sp = neighbour_spacing(ws[2], ws[3])          # w2 sits half a pitch out for theta in [5, 5.8): not on any sheet
    inside = _interior(thetas, 5.0, 5.8, margin=0.15)
    outside = ~_theta_cols(thetas, 4.7, 6.1)
    sc = sheet_counts(ws[2], sp, surface)
    v_in = sc.verdict[:, inside]
    assert (v_in[v_in != 5] == S_START_OFF).mean() > 0.9
    v_out = sc.verdict[:, outside]
    assert (v_out[v_out != 5] == S_START_OFF).mean() < 0.02
    assert (_assessed(sp.flag[:, inside]) == OK).mean() > 0.9      # spacing alone cannot see this fault


def test_cli_audit_runs(dataset, tmp_path):
    from winding_audit.cli import main
    _, _, _, paths = dataset
    out = tmp_path / "audit"
    main(["audit", paths["windings"], "--surface", paths["surface"], "--group", "0", "--out", str(out),
          "--voxel-um", "8.64", "--sections", "12", "--grid-figures"])
    assert (out / "audit.csv").exists() and (out / "summary.json").exists()
    assert (out / "sections" / "z000012.png").exists()
    assert (out / "windings" / "w000_syn.png").exists()
