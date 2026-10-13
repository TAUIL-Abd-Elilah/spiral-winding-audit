"""Pieces must stay on one sheet, be cut where a winding leaves its sheet, and keep a shared sheet only once."""
import numpy as np
import pytest

from winding_audit import synthetic as syn
from winding_audit.pieces import band_pieces
from winding_audit.sheets import open_surface
from winding_audit.tifxyz import list_windings

PITCH = 10.0


def _band(tmp_path_factory, name, faults):
    out = tmp_path_factory.mktemp(name)
    vol, centre = syn.make_volume(n_sheets=6, pitch=PITCH)
    ws, thetas = syn.make_windings(n_sheets=6, pitch=PITCH, centre=centre)
    for f in faults:
        f(ws, thetas, centre)
    s, d = syn.write_dataset(str(out), vol, ws)
    surface = open_surface(s, "0")
    windings = list_windings(d)
    return windings, thetas, band_pieces(windings, surface, min_nodes=10)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return _band(tmp_path_factory, "clean", [])


@pytest.fixture(scope="module")
def faulty(tmp_path_factory):
    return _band(tmp_path_factory, "faulty", [
        lambda ws, th, c: syn.inject_merge(ws, th, w=1, t0=1.0, t1=2.0, pitch=PITCH, centre=c),
        lambda ws, th, c: syn.inject_gap(ws, th, w=3, t0=5.0, t1=5.8, pitch=PITCH, centre=c)])


def _cols(thetas, t0, t1, margin=0.15):
    return (thetas >= t0 + margin) & (thetas < t1 - margin)


def test_clean_band_is_one_piece_per_winding(clean):
    windings, _, res = clean
    for w, r in zip(windings, res):
        assert r.stats["pieces"] == 1, r.stats
        assert r.stats["duplicate"] == 0
        assert r.kept.sum() >= 0.97 * w.valid.sum()


def test_merged_sector_is_cut_and_kept_once(faulty):
    windings, thetas, res = faulty
    inside = _cols(thetas, 1.0, 2.0)
    before, after = _cols(thetas, 0.0, 1.0), _cols(thetas, 2.0, 3.0)
    l1 = res[1].label
    # w1 sits on sheet 2 in the merged sector: kept there, but as its own piece
    assert (l1[:, inside] > 0).mean() > 0.9
    sector = set(np.unique(l1[:, inside])) - {0}
    rest = set(np.unique(l1[:, before])) | set(np.unique(l1[:, after]))
    assert sector and not (sector & rest)
    # w2 covers the same sheet there: dropped as a duplicate, kept elsewhere
    assert (res[2].duplicate[:, inside]).mean() > 0.9
    assert (res[2].label[:, inside] > 0).mean() < 0.05
    assert (res[2].label[:, after] > 0).mean() > 0.9


def test_off_sheet_sector_is_dropped_and_splits_the_winding(faulty):
    windings, thetas, res = faulty
    r = res[3]
    assert (r.label[:, _cols(thetas, 5.0, 5.8)] > 0).mean() < 0.05
    assert r.stats["pieces"] == 2


def test_cli_pieces_runs_and_resumes(tmp_path):
    import json
    from winding_audit.cli import main
    vol, centre = syn.make_volume(n_sheets=5, pitch=PITCH)
    ws, thetas = syn.make_windings(n_sheets=5, pitch=PITCH, centre=centre)
    syn.inject_merge(ws, thetas, w=1, t0=1.0, t1=2.0, pitch=PITCH, centre=centre)
    s, d = syn.write_dataset(str(tmp_path / "data"), vol, ws)
    out = tmp_path / "pieces"
    args = ["pieces", d, "--surface", s, "--out", str(out), "--min-nodes", "10", "--figures"]
    main(args)
    first = json.load(open(out / "pieces_summary.json"))
    assert first["duplicate"] > 0 and first["kept"] < first["n"]
    assert (out / "windings" / "w002_syn" / "x.tif").exists() and (out / "figures" / "w001_syn.png").exists()
    main(args)
    assert json.load(open(out / "pieces_summary.json"))["kept"] == first["kept"]
