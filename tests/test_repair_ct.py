"""With the raw CT, the repair must separate two windings stacked on one predicted surface when the prediction misses
the layer between them, and keep a node that sits on a CT layer the prediction misses."""
import os

import numpy as np
import pytest
import zarr
from scipy.ndimage import gaussian_filter

from winding_audit import synthetic as syn
from winding_audit.repair import R_KEPT, R_KEPT_CT, R_MOVED_CT, R_NOT_ASSESSED, R_UNREPAIRED, repair_band
from winding_audit.sheets import open_surface
from winding_audit.tifxyz import list_windings

PITCH = 10.0
STACK = (1.0, 2.0)       # winding 3 snapped onto sheet 2; the prediction has no sheet 3 there
MISSED = (4.0, 5.0)      # winding 1 on sheet 1; the prediction has no sheet 1 there


def _drop(vol, n, t0, t1, centre, r0=20.0, pitch=PITCH):
    Z, Y, X = vol.shape
    cy, cx = centre
    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float64)
    theta = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    u = (np.hypot(xx - cx, yy - cy) - r0) / pitch - theta / (2 * np.pi)
    vol[:, (np.abs(u - n) < 0.35) & (theta >= t0) & (theta < t1)] = 0


@pytest.fixture(scope="module")
def bands(tmp_path_factory):
    out = tmp_path_factory.mktemp("repct")
    full, centre = syn.make_volume(n_sheets=6, pitch=PITCH)
    pred = full.copy()
    _drop(pred, 3, *STACK, centre)
    _drop(pred, 1, *MISSED, centre)
    rng = np.random.default_rng(0)
    ct = gaussian_filter(60.0 + 0.45 * full.astype(np.float64), 1.0) + rng.normal(0, 4.0, full.shape)
    ws, thetas = syn.make_windings(n_sheets=6, pitch=PITCH, centre=centre)
    ws[3] = syn._shift(ws[3], thetas, *STACK, -1.0, PITCH, centre)
    s, d = syn.write_dataset(str(out), pred, ws)
    g = zarr.open_group(os.path.join(str(out), "ct.zarr"), mode="w", zarr_format=2)
    arr = g.create_array("0", shape=ct.shape, chunks=(16, 64, 64), dtype="u1", fill_value=0)
    arr[:] = np.clip(ct, 0, 255).astype(np.uint8)
    surface, ctv = open_surface(s, "0"), open_surface(os.path.join(str(out), "ct.zarr"), "0")
    windings = list_windings(d)
    return thetas, windings, repair_band(windings, surface), repair_band(windings, surface, ct=ctv)


def _cols(thetas, t0, t1, margin=0.15):
    return (thetas >= t0 + margin) & (thetas < t1 - margin)


def _assessed(st):
    return st[st != R_NOT_ASSESSED]


def test_stacked_winding_moves_to_the_ct_layer_between_its_neighbours(bands):
    thetas, ws, plain, with_ct = bands
    cols = _cols(thetas, *STACK)
    assert (_assessed(plain[3].status[:, cols]) == R_UNREPAIRED).mean() > 0.9
    st, sh = with_ct[3].status[:, cols], with_ct[3].shift[:, cols]
    assert (_assessed(st) == R_MOVED_CT).mean() > 0.9, np.bincount(st.ravel(), minlength=8)
    assert np.all(np.abs(sh[st == R_MOVED_CT] - PITCH) < 2.0)        # one pitch out, onto sheet 3


def test_node_on_a_layer_the_prediction_misses_is_kept(bands):
    thetas, ws, plain, with_ct = bands
    cols = _cols(thetas, *MISSED)
    st = with_ct[1].status[:, cols]
    assert (_assessed(st) == R_KEPT_CT).mean() > 0.9, np.bincount(st.ravel(), minlength=8)
    assert np.array_equal(with_ct[1].winding.xyz[:, cols], ws[1].xyz[:, cols])


def test_ct_changes_nothing_where_the_prediction_is_complete(bands):
    thetas, ws, plain, with_ct = bands
    clean = ~(_cols(thetas, STACK[0] - 0.4, STACK[1] + 0.4, 0) | _cols(thetas, MISSED[0] - 0.4, MISSED[1] + 0.4, 0))
    for k in (1, 2, 3, 4):
        st = with_ct[k].status[:, clean]
        assert (_assessed(st) == R_KEPT).mean() > 0.97, (k, np.bincount(st.ravel(), minlength=8))
        assert np.array_equal(with_ct[k].winding.xyz[:, clean], ws[k].xyz[:, clean])
