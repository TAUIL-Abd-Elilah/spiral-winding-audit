"""Windings per sheet along radial rays: 1 for a fit with one winding per sheet, about 2 for one with two."""
import numpy as np

from winding_audit import synthetic as syn
from winding_audit.density import windings_per_sheet
from winding_audit.sheets import open_surface
from winding_audit.tifxyz import Winding, list_windings


def test_one_winding_per_sheet(tmp_path):
    vol, centre = syn.make_volume(n_sheets=6, pitch=10.0)
    ws, _ = syn.make_windings(n_sheets=6, pitch=10.0, centre=centre)
    s, d = syn.write_dataset(str(tmp_path), vol, ws)
    r = windings_per_sheet(list_windings(d), open_surface(s, "0"), z0=12, first=1, last=5)
    assert r["rays"] > 100 and abs(r["windings_per_sheet"] - 1.0) < 0.1, r


def test_two_windings_per_sheet(tmp_path):
    vol, centre = syn.make_volume(n_sheets=6, pitch=10.0)
    # a fit with half the pitch: 11 windings over the radius of 6 sheets
    ws, _ = syn.make_windings(n_sheets=11, pitch=5.0, centre=centre, r0=20.0)
    s, d = syn.write_dataset(str(tmp_path), vol, ws)
    r = windings_per_sheet(list_windings(d), open_surface(s, "0"), z0=12, first=2, last=10)
    assert r["rays"] > 100 and 1.7 < r["windings_per_sheet"] < 2.3, r
