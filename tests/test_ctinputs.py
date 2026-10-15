"""CT winding-crossing inputs for villa's spiral fitter, on a synthetic masked CT with two faults planted: two sheets
that touch (one CT peak for two windings) and one sheet split into two plies (two peaks for one winding). Further
tests cover the defects an adversarial review found in the first version (each named in its test).

Every stored pair must carry the true winding difference; villa's own loader, pair sampler, outer-shell map and
store validator are used when a villa checkout is available (VILLA_SPIRAL_DIR = path to villa/spiral-fitting)."""
import json
import os
import shutil
import sys

import numpy as np
import pytest

from winding_audit import ctinputs as ci
from winding_audit import synthetic as syn
from winding_audit.volumes import ChunkCacheArray

PITCH, R0, NS = 10.0, 12.0, 8
TOUCH, PLY = (4, 1.0, 1.6, 0.7), (2, 3.0, 3.6, 2.5)
SHAPE = (16, 224, 224)
INNER_HOLE = (0, 3, 4)      # y 96-127, x 128-159 at chunk 32: inside the scroll, on the rays near theta = 0
OUTER_HOLE = (0, 6, 3)      # y 192-223, x 96-127: the outermost sheets and the air beyond, near theta = pi/2


def _prm(**kw):
    # the synthetic volume leaves ~20 voxels of air outside the scroll along the axes, so a smaller clearance
    d = dict(r_min=4.0, r_max=120.0, window=30.0, shell_clear=12.0)
    d.update(kw)
    return ci.RayParams(**d)


def _run(ct, umb, prm=None, z0=2, z1=14, z_step=4, n_angles=180, **kw):
    return ci.generate(ct, umb, z0, z1, z_step=z_step, n_angles=n_angles, prm=prm or _prm(), **kw)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("ctin")
    ct, centre = syn.ct_spiral(SHAPE, R0, PITCH, NS, touch=TOUCH, ply=PLY)
    um = syn.write_umbilicus(str(out / "umbilicus.json"), centre, 0, SHAPE[0])
    umb = ci.load_umbilicus(um)
    rays, outer, inner, zs, thetas, stats = _run(ct, umb)
    store = str(out / "winding_inference")
    ci.write_store(store, rays, provenance={"test": True})
    zyx, measured = ci.shell_grid(outer, inner, zs, thetas, umb, shell_angles=720)
    shell = str(out / "outer_shell")
    ci.write_shell(shell, zyx, measured)
    return dict(out=out, ct=ct, centre=centre, um=um, umb=umb, rays=rays, outer=outer, inner=inner, zs=zs,
                thetas=thetas, stats=stats, store=store, shell=shell, shell_zyx=zyx, measured=measured)


@pytest.fixture(scope="module")
def holed(built, tmp_path_factory):
    """The same CT as a chunk cache (chunk 32) with one chunk missing inside the scroll and one beyond it."""
    cache = syn.write_chunk_cache(built["ct"], str(tmp_path_factory.mktemp("cache") / "c"), chunk=32)
    for h in (INNER_HOLE, OUTER_HOLE):
        os.remove(os.path.join(cache, f"{h[0]}_{h[1]}_{h[2]}.bin"))
    arr = ChunkCacheArray(cache, built["ct"].shape, chunk=32)
    rays, outer, inner, zs, thetas, stats = _run(arr, built["umb"])
    zyx, measured = ci.shell_grid(outer, inner, zs, thetas, built["umb"], shell_angles=720)
    return dict(cache=cache, arr=arr, rays=rays, outer=outer, inner=inner, zs=zs, thetas=thetas, stats=stats,
                shell_zyx=zyx, measured=measured)


def _all_points(rays):
    pts, ray_id, lvl = [], [], []
    for i, (o, s, t) in enumerate(rays):
        pts.append(o[None, :] + t[:, None] * s[None, :])
        ray_id.append(np.full(len(t), i))
        lvl.append(np.arange(len(t)))
    return np.concatenate(pts), np.concatenate(ray_id), np.concatenate(lvl)


def _levels_right(rays, centre, touch=TOUCH, n_sheets=NS, r0=R0, pitch=PITCH):
    pts, ray_id, lvl = _all_points(rays)
    k, d = syn.true_sheet(pts, centre, r0, pitch, n_sheets, touch)
    for i in np.unique(ray_id):
        sel = ray_id == i
        if not np.array_equal(k[sel] - k[sel][0], lvl[sel] - lvl[sel][0]):
            return False, (i, k[sel].tolist())
    return True, d


# ------------------------------------------------------------------------------------------- core properties

def test_every_stored_level_difference_is_the_true_winding_difference(built):
    pts, _, _ = _all_points(built["rays"])
    ok, info = _levels_right(built["rays"], built["centre"])
    assert ok, f"ray {info[0]}: sheets {info[1]}"
    k, d = syn.true_sheet(pts, built["centre"], R0, PITCH, NS, TOUCH)
    cy, cx = built["centre"]
    th = np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 2] - cx), 2 * np.pi)
    on_ply = (th >= PLY[1]) & (th < PLY[2]) & (k == PLY[0])
    # crossings sit on the sheets; on the split sheet a crossing may sit on one ply (PLY[3] voxels off centre)
    assert np.median(d) < 0.5 and np.percentile(d[~on_ply], 99) < 2.0
    assert np.all(d[on_ply] < PLY[3] + 0.6)


def test_faults_cut_rays_and_clean_regions_are_covered(built):
    pts, ray_id, _ = _all_points(built["rays"])
    cy, cx = built["centre"]
    th = np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 2] - cx), 2 * np.pi)
    k, _ = syn.true_sheet(pts, built["centre"], R0, PITCH, NS, TOUCH)
    # no stored crossing on the fused pair (one peak for two windings is ambiguous), and never both plies of the
    # split sheet on one ray (that would add a winding)
    in_touch = (th >= TOUCH[1]) & (th < TOUCH[2]) & np.isin(k, [TOUCH[0], TOUCH[0] + 1])
    assert in_touch.sum() == 0
    in_ply = (th >= PLY[1]) & (th < PLY[2]) & (k == PLY[0])
    assert np.bincount(ray_id[in_ply], minlength=len(built["rays"])).max(initial=0) <= 1
    # coverage away from the faults
    clean = ~(((th >= TOUCH[1] - 0.1) & (th < TOUCH[2] + 0.1)) | ((th >= PLY[1] - 0.1) & (th < PLY[2] + 0.1)))
    per_ray = np.bincount(ray_id[clean], minlength=len(built["rays"]))
    st = built["stats"]
    assert st["rays_cast"] == 3 * 180 and st["pieces"] > 0
    expected_clean_rays = 3 * 180 * (1 - (TOUCH[2] - TOUCH[1] + PLY[2] - PLY[1] + 0.4) / (2 * np.pi))
    assert (per_ray >= NS - 3).sum() >= 0.75 * expected_clean_rays
    assert abs(st["median_spacing"] - PITCH) < 1.0


def test_store_format_check_and_tamper_detection(built, tmp_path):
    summary = ci.check_store(built["store"])
    assert summary["num_rays"] == len(built["rays"])
    assert summary["num_crossings"] == sum(len(r[2]) for r in built["rays"])
    with pytest.raises(FileExistsError):
        ci.write_store(built["store"], built["rays"])
    cp = str(tmp_path / "copy")
    shutil.copytree(built["store"], cp)
    lv = os.path.join(cp, "shard_0", "crossing_level.npy")
    a = np.load(lv)
    a[1] += 1
    np.save(lv, a)
    with pytest.raises(ValueError, match="checksum"):
        ci.check_store(cp)
    cp2 = str(tmp_path / "copy2")
    shutil.copytree(built["store"], cp2)
    m = json.load(open(os.path.join(cp2, "manifest.json")))
    m["num_rays"] += 1
    json.dump(m, open(os.path.join(cp2, "manifest.json"), "w"))
    with pytest.raises(ValueError, match="fingerprint"):
        ci.check_store(cp2)


def test_outer_shell_follows_the_scroll_outside(built):
    zyx, meas = built["shell_zyx"], built["measured"]
    ok = zyx[..., 0] >= 0
    pad = 8
    first = max(0, built["zs"][0] - pad)                                  # rows never go below z = 0
    assert zyx.shape[:2] == (built["zs"][-1] + pad + 1 - first, 720) and zyx[0, 0, 0] == first
    assert ok.all()                                                       # no holes, band edges padded
    assert meas.mean() > 0.97                                             # the outside was seen nearly everywhere
    cy, cx = built["centre"]
    p = zyx[meas]
    r = np.hypot(p[:, 1] - cy, p[:, 2] - cx)
    th = np.mod(np.arctan2(p[:, 1] - cy, p[:, 2] - cx), 2 * np.pi)
    r_last = syn.sheet_radii(th, R0, PITCH, NS, TOUCH)[-1]
    assert np.median(np.abs(r - (r_last + _prm().shell_margin))) < 1.0
    # columns sit at the centres of villa's 720 theta bins (review: on the edges, float rounding emptied 14 %)
    col = np.mod(np.arctan2(zyx[0, :, 1] - cy, zyx[0, :, 2] - cx), 2 * np.pi) / (2 * np.pi) * 720
    assert np.allclose(col - np.floor(col), 0.5, atol=1e-3)


def test_missing_chunk_never_inside_a_stored_ray(built, holed):
    for o, s_, t in holed["rays"]:
        seg = np.arange(t[0], t[-1] + 0.5, 0.5)
        q = np.rint(o[None, :] + seg[:, None] * s_[None, :]).astype(np.int64) // 32
        for h in (INNER_HOLE, OUTER_HOLE):
            assert not np.any(np.all(q == np.array(h), 1)), f"a stored ray spans the missing chunk {h}"
    ang = (holed["thetas"] + np.pi) % (2 * np.pi) - np.pi
    outer = holed["outer"]
    assert np.isfinite(outer[:, np.abs(ang) < 0.12]).mean() > 0.9          # an inner hole hides nothing outside
    assert np.isnan(outer[:, np.abs(ang - np.pi / 2) < 0.12]).mean() > 0.9  # missing CT beyond: not seen
    ok, info = _levels_right(holed["rays"], built["centre"])
    assert ok, info


def test_umbilicus_extrapolates_like_villa(tmp_path):
    p = str(tmp_path / "u.json")
    json.dump({"control_points": [{"x": 10, "y": 20, "z": 0}, {"x": 30, "y": 20, "z": 10},
                                  {"x": 30, "y": 40, "z": 20}]}, open(p, "w"))
    f = ci.load_umbilicus(p, coordinate_scale=2.0)
    got = f(np.array([0.0, 10.0, 20.0, 40.0, 50.0, -10.0]))
    assert np.allclose(got[1], [40.0, 40.0])
    assert np.allclose(got[3], [80.0, 60.0])
    assert np.allclose(got[4], [100.0, 60.0])
    assert np.allclose(got[5], [40.0, 0.0])


# ---------------------------------------------------------------------------- review findings, one test each

def _profile(positions, faint=(), length=200, noise=5.0, seed=0, sigma=1.3):
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    prof = 45.0 + rng.normal(0, noise, length)
    for i, p in enumerate(positions):
        if i not in faint:
            prof += 110.0 * np.exp(-(t - p) ** 2 / (2 * sigma ** 2))
    return prof, t


def _pieces_true(positions, prm, faint=(), seeds=range(40), same=None):
    """For noisy 1D profiles: True if every stored piece steps by exactly one sheet per crossing. ``same`` maps a
    position index to the sheet it belongs to (two plies of one sheet, a fused pair)."""
    pos_arr = np.asarray(positions, float)
    for seed in seeds:
        prof, t = _profile(positions, faint, seed=seed)
        pos, reg, _ = ci.crossings_on_profile(prof, t, prm)
        for run in ci.pieces(pos, reg, prm):
            idx = np.argmin(np.abs(run[:, None] - pos_arr[None, :]), 1)
            if same is not None:
                idx = np.array([same.get(int(i), int(i)) for i in idx])
            if not np.all(np.diff(idx) == 1):
                return False, (seed, run.round(1).tolist(), idx.tolist())
    return True, None


def test_end_gaps_are_judged_against_a_full_window():
    # review: a running median padded with 'nearest' measured the last gap against itself, so two missed sheets at
    # a ray's end (sheets 11 and 13 of 15) were stored as consecutive windings on 200 of 200 noisy rays
    positions = 20 + 10.0 * np.arange(15)
    ok, info = _pieces_true(positions, ci.RayParams(window=30.0), faint=(11, 13))
    assert ok, info
    ok, info = _pieces_true(positions, ci.RayParams(window=30.0), faint=(1, 3))     # the inner end too
    assert ok, info


def test_faults_shared_by_two_gaps_cut_the_ray():
    prm = ci.RayParams(window=30.0)
    # review: a fused pair midway between its neighbours makes two gaps of ~1.27 local spacings, each in bounds;
    # sheets 7 and 8 read as one peak, so a piece that contains it and either neighbour has a wrong count
    fused = [5, 16, 27, 38, 49, 60, 71, 83.3, 86.7, 99, 110, 121, 132, 143, 154]
    sheet = {i: i for i in range(len(fused))}
    for seed in range(40):
        prof, t = _profile(fused, seed=seed)
        pos, reg, _ = ci.crossings_on_profile(prof, t, prm)
        for run in ci.pieces(pos, reg, prm):
            near = np.min(np.abs(run[:, None] - np.array([83.3, 86.7])[None, :]), 1) < 2.5
            assert not near.any(), (seed, run.round(1).tolist())
            idx = np.array([sheet[int(np.argmin(np.abs(np.array(fused) - p)))] for p in run])
            assert np.all(np.diff(idx) == 1), (seed, run.round(1).tolist())
    # review: a ply split of +-3.5 at pitch 10 gives gaps 6.5 / 7 / 6.5, all above gap_lo
    ply = [20, 30, 36.5, 43.5, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    same = {2: 2, 3: 2}
    same.update({i: i - 1 for i in range(4, len(ply))})
    ok, info = _pieces_true(ply, prm, same=same)
    assert ok, info


def test_pooled_floor_ignores_air_in_a_flaked_sector(tmp_path):
    # review: a per-ray floor fell to the air level where the outer layers had flaked off inside the mask, and noise
    # maxima were stored as windings
    n = 10
    ct, centre = syn.ct_spiral((16, 256, 256), R0, PITCH, n, touch=None, ply=None, noise=8.0)
    cy, cx = centre
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float64)
    r = np.hypot(xx - cx, yy - cy)
    th = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    R = syn.sheet_radii(th.ravel(), R0, PITCH, n).reshape(n, 256, 256)
    flaked = (th >= 0.5) & (th < 1.5) & (r > R[3] + 0.5 * PITCH) & (ct[0] > 0)
    rng = np.random.default_rng(1)
    ct[:, flaked] = np.clip(45 + rng.normal(0, 8.0, (16, int(flaked.sum()))), 1, 255).astype(np.uint8)
    umb = ci.load_umbilicus(syn.write_umbilicus(str(tmp_path / "u.json"), centre, 0, 16))
    rays, *_ = _run(ct, umb, prm=_prm(r_max=150.0))
    pts, _, _ = _all_points(rays)
    k, d = syn.true_sheet(pts, centre, R0, PITCH, n, None)
    tp = np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 2] - cx), 2 * np.pi)
    in_sector = (tp >= 0.55) & (tp < 1.45)
    assert np.all(k[in_sector] <= 3), "a crossing stored in the air of the flaked sector"
    assert np.percentile(d, 99) < 2.0
    ok, info = _levels_right(rays, centre, touch=None, n_sheets=n)
    assert ok, info


def test_short_rays_are_never_written_and_rejected_on_check(tmp_path):
    # review: --min-crossings 0 stored empty rays, check_store accepted them, villa's loader then failed
    prm = ci.RayParams(min_crossings=0, edge_trim=2)
    assert all(len(p) >= 2 for p in ci.pieces(np.array([1.0, 11, 21]), np.array([True, True]), prm))
    o, s = np.zeros(3, np.float32), np.array([0, 0, 1], np.float32)
    store = str(tmp_path / "s")
    ci.write_store(store, [(o, s, np.array([5.0, 15.0], np.float32)), (o, s, np.array([7.0], np.float32))])
    with pytest.raises(ValueError, match="fewer than 2"):
        ci.check_store(store)


def test_long_runs_are_stored_as_bounded_windows(built):
    # review: villa draws a ray, then a pair on it, so unbounded pieces gave short pieces many times the weight
    ws = ci.windows(np.arange(40.0), 16, 8)
    assert [len(w) for w in ws] == [16, 16, 16, 16] and ws[0][0] == 0 and ws[-1][-1] == 39
    rays, *_ = _run(built["ct"], built["umb"], prm=_prm(window_crossings=5, window_stride=3))
    assert max(len(r[2]) for r in rays) <= 5
    ok, info = _levels_right(rays, built["centre"])
    assert ok, info


def test_shell_has_no_holes_and_holds_every_stored_crossing(built, holed):
    # review: where the outside was not seen the shell had holes, and villa dropped every pair in those sectors
    zyx, meas = holed["shell_zyx"], holed["measured"]
    assert (zyx[..., 0] >= 0).all()
    assert meas.mean() < 0.999                                   # the outer hole really hid the outside somewhere
    cy, cx = built["centre"]
    pad = 8
    pts, _, _ = _all_points(holed["rays"])
    th = np.mod(np.arctan2(pts[:, 1] - cy, pts[:, 2] - cx), 2 * np.pi)
    col = np.floor(th / (2 * np.pi) * 720).astype(int) % 720
    row = np.rint(pts[:, 0]).astype(int) - max(0, holed["zs"][0] - pad)
    sh = zyx[row, col]
    r_shell = np.hypot(sh[:, 1] - cy, sh[:, 2] - cx)
    r_pt = np.hypot(pts[:, 1] - cy, pts[:, 2] - cx)
    assert np.all(r_pt <= r_shell + 1e-3)


def test_single_slice_band_gives_a_loadable_shell(built):
    # review: a band with one sampled slice wrote a one-row shell that villa's loader rejects
    rays, outer, inner, zs, thetas, _ = _run(built["ct"], built["umb"], z0=4, z1=8, z_step=8)
    assert len(zs) == 1
    zyx, _ = ci.shell_grid(outer, inner, zs, thetas, built["umb"])
    assert zyx.shape[0] == 4 + 8 + 1 and ci.has_valid_quad(zyx)          # rows 0 .. 12 around the slice at z 4


def test_truncated_chunk_reads_as_missing_not_air(built, tmp_path):
    # review: a truncated chunk file counted as present and read as air, which moved the shell inside the scroll
    cache = syn.write_chunk_cache(built["ct"], str(tmp_path / "c"), chunk=32)
    f = os.path.join(cache, f"{OUTER_HOLE[0]}_{OUTER_HOLE[1]}_{OUTER_HOLE[2]}.bin")
    with open(f, "r+b") as fh:
        fh.truncate(1000)
    arr = ChunkCacheArray(cache, built["ct"].shape, chunk=32)
    assert OUTER_HOLE not in arr._present
    _, outer, *_ = _run(arr, built["umb"])
    ang = (np.arange(180) * 2 * np.pi / 180 + np.pi) % (2 * np.pi) - np.pi
    assert np.isnan(outer[:, np.abs(ang - np.pi / 2) < 0.12]).mean() > 0.9


def test_coarse_level_is_written_in_the_fit_frame(built):
    # review: a coarser pyramid level wrote its own voxel grid while the manifest said scale-0
    ct = built["ct"].astype(np.float32).reshape(8, 2, 112, 2, 112, 2)
    ct2 = np.where(ct.min((1, 3, 5)) == 0, 0, ct.mean((1, 3, 5))).astype(np.uint8)
    rays, outer, *_ = _run(ct2, built["umb"], ct_scale=2.0)
    assert len(rays) > 100
    ok, info = _levels_right(rays, built["centre"])
    assert ok, info
    pts, _, _ = _all_points(rays)
    _, d = syn.true_sheet(pts, built["centre"], R0, PITCH, NS, TOUCH)
    assert np.median(d) < 1.0                               # in fit voxels, not the CT level's half-size grid
    assert np.nanmedian(outer) > 60                         # the outside radius is in fit voxels too


def test_chunks_are_read_about_once(built):
    # review: every chunk was re-read for each slice of its chunk row (16x on a real band)
    cache = syn.write_chunk_cache(built["ct"], str(built["out"] / "c_reads"), chunk=32)
    arr = ChunkCacheArray(cache, built["ct"].shape, chunk=32)
    *_, stats = _run(arr, built["umb"])
    assert 0 < stats["chunk_reads"] <= len(arr._present)


def test_overrides_switch_off_verified_patches_only_when_absent(tmp_path):
    # review: villa requires verified_patches/ unless it is switched off
    ov = ci.fit_overrides(2, 14, str(tmp_path))
    assert ov["input_use_verified_patches"] is False and ov["dense_spacing_mode"] == "winding_model"
    os.makedirs(tmp_path / "verified_patches")
    assert "input_use_verified_patches" not in ci.fit_overrides(2, 14, str(tmp_path))


# ---------------------------------------------------------------------------------------------------- CLI

def _cli_common(um):
    return ["--umbilicus", um, "--z0", "2", "--z1", "14", "--z-step", "4", "--angles", "180",
            "--r-min", "4", "--window", "30", "--shell-clear", "12"]


def _zarr_ct(ct, path):
    import zarr
    g = zarr.open_group(path, mode="w", zarr_format=2)
    g.create_array("0", shape=ct.shape, chunks=(16, 64, 64), dtype="u1", fill_value=0)[:] = ct
    return path


def test_cli_ct_inputs_from_a_chunk_folder_and_from_zarr(built, tmp_path):
    from winding_audit.cli import main
    cache = syn.write_chunk_cache(built["ct"], str(tmp_path / "cache"), chunk=128)
    zpath = _zarr_ct(built["ct"], str(tmp_path / "ct.zarr"))
    common = _cli_common(built["um"]) + ["--r-max", "120"]
    for name, src in (("cache", ["--ct", cache, "--shape", *map(str, built["ct"].shape)]),
                      ("zarr", ["--ct", zpath])):
        out = str(tmp_path / f"ds_{name}")
        main(["ct-inputs", *src, *common, "--out", out])
        rep = json.load(open(os.path.join(out, "ct_inputs_report.json")))
        assert rep["store"]["num_rays"] > 0 and rep["shell_share_valid"] == 1.0
        ov = json.load(open(os.path.join(out, "fit_overrides.json")))
        assert ov["dense_spacing_mode"] == "winding_model" and ov["loss_weight_shell_outer"] == 0
        assert ov["input_use_verified_patches"] is False
        assert os.path.exists(os.path.join(out, "outer_shell", "measured.tif"))
        with pytest.raises(SystemExit):                       # never overwrites a store
            main(["ct-inputs", *src, *common, "--out", out])
    # the same CT read two ways gives the same arrays (the manifests differ only in the recorded source path)
    sha = {n: {k: v["sha256"] for k, v in json.load(open(os.path.join(str(tmp_path / f"ds_{n}"),
                                                                       "winding_inference", "manifest.json")))
               ["shards"][0]["arrays"].items()} for n in ("cache", "zarr")}
    assert sha["cache"] == sha["zarr"]
    main(["check-store", os.path.join(str(tmp_path / "ds_zarr"), "winding_inference")])


def _read_store(store):
    m = json.load(open(os.path.join(store, "manifest.json")))
    rays = []
    for sh in m["shards"]:
        d = os.path.join(store, sh["name"])
        off = np.load(os.path.join(d, "crossing_offsets.npy"))
        o = np.load(os.path.join(d, "ray_origin_zyx.npy"))
        s = np.load(os.path.join(d, "ray_step_zyx.npy"))
        t = np.load(os.path.join(d, "crossing_t.npy"))
        rays += [(o[i], s[i], t[off[i]:off[i + 1]]) for i in range(len(o))]
    return rays


def test_cli_reads_the_dataset_spec_and_refuses_conflicts(built, tmp_path):
    # review: ct-inputs ignored the dataset's spiral-scroll.json (umbilicus coordinate_scale, path overrides)
    from winding_audit.cli import main
    cy, cx = built["centre"]
    half = str(tmp_path / "umbilicus_half.json")              # an umbilicus stored at half the fit's resolution
    json.dump({"control_points": [{"x": cx / 2, "y": cy / 2, "z": z / 2} for z in (0, 16)]}, open(half, "w"))
    spec = {"schema_version": 1, "name": "syn", "voxel_size_um": 8.64, "spiral_outward_sense": "CW",
            "umbilicus": {"coordinate_scale": 2.0}}
    zpath = _zarr_ct(built["ct"], str(tmp_path / "ct.zarr"))
    common = ["--ct", zpath, "--r-max", "120"] + _cli_common(half)

    ds = tmp_path / "ds"
    os.makedirs(ds)
    json.dump(spec, open(ds / "spiral-scroll.json", "w"))
    with pytest.raises(SystemExit):                            # an explicit scale that villa would not apply
        main(["ct-inputs", *common, "--umbilicus-scale", "1", "--out", str(ds)])
    main(["ct-inputs", *common, "--out", str(ds)])
    assert json.load(open(ds / "ct_inputs_report.json"))["provenance"]["umbilicus_scale"] == 2.0
    ok, info = _levels_right(_read_store(str(ds / "winding_inference")), built["centre"])
    assert ok, info

    ds2 = tmp_path / "ds2"
    os.makedirs(ds2)
    json.dump(dict(spec, paths={"winding_inference": "published/winding_inference"}),
              open(ds2 / "spiral-scroll.json", "w"))
    with pytest.raises(SystemExit):                            # villa would read the store from elsewhere
        main(["ct-inputs", *common, "--out", str(ds2)])
    with pytest.raises(SystemExit):                            # a coarser level needs its scale
        main(["ct-inputs", *common, "--ct-group", "1", "--out", str(tmp_path / "ds3")])


def test_auto_r_max_comes_from_the_present_chunks(tmp_path):
    # review: the default ray length ran to the volume's farthest corner, mostly outside the scroll
    from winding_audit import cli
    ct, centre = syn.ct_spiral((16, 320, 320), R0, PITCH, NS)
    cache = syn.write_chunk_cache(ct, str(tmp_path / "c"), chunk=32, skip_empty=True)
    arr = ChunkCacheArray(cache, ct.shape, chunk=32)
    umb = ci.load_umbilicus(syn.write_umbilicus(str(tmp_path / "u.json"), centre, 0, 16))
    rad = cli._scroll_radius(arr, umb, 2, 14, 1.0, None)
    corner = float(np.hypot(*centre))
    outside = R0 + PITCH * NS + 0.5 * PITCH
    assert outside <= rad <= outside + 2 * 32 * np.sqrt(2) and rad < 0.8 * corner


# ------------------------------------------------------------------------------------- with villa's own code

def _villa():
    d = os.environ.get("VILLA_SPIRAL_DIR")
    if not d or not os.path.isdir(d):
        pytest.skip("set VILLA_SPIRAL_DIR to villa/spiral-fitting to run the loader round trip")
    pytest.importorskip("torch")
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def test_villa_loader_samples_the_true_targets(built):
    _villa()
    import torch
    import winding_supervision as ws
    store = ws.load_winding_inference_store(built["store"], "cpu", verify=True, z_range=(0, SHAPE[0]))
    assert store.fingerprint["num_rays"] == len(built["rays"])
    g = torch.Generator().manual_seed(0)
    rel = store.sample_relative(4000, 3, 15, generator=g)
    adj = store.sample_adjacent(4000, generator=g)
    assert len(rel["target"]) == 4000 and len(adj["target"]) == 4000
    for sample in (rel, adj):
        pts = sample["points"].numpy().reshape(-1, 3)
        k, _ = syn.true_sheet(pts, built["centre"], R0, PITCH, NS, TOUCH)
        k = k.reshape(-1, 2)
        assert np.array_equal(sample["target"].numpy().astype(int), k[:, 1] - k[:, 0])
        assert (sample["target"].numpy() > 0).all()
    assert (adj["target"].numpy() == 1).all()
    m = json.load(open(os.path.join(built["store"], "manifest.json")))
    assert ws._canonical_digest(ci._identity_view(m)) == m["fingerprint"]


@pytest.mark.parametrize("which", ["complete", "holed"])
def test_villa_shell_map_keeps_the_stored_pairs(built, holed, tmp_path, which):
    _villa()
    import torch
    import fit_spiral
    import tifxyz
    from umbilicus import json_umbilicus_z_to_yx
    src = built if which == "complete" else holed
    shell = str(tmp_path / "shell")
    ci.write_shell(shell, src["shell_zyx"], src["measured"])
    patch = tifxyz.load_tifxyz(shell)
    cfg = {"shell_table_smooth_sigma_z": 4.0, "shell_table_smooth_sigma_theta": 1.0, "shell_min_confidence": 0.25}
    # the fitter's own table range: z_begin - 160 .. z_end + 160
    smap = fit_spiral.ShellPolarMap(patch, json_umbilicus_z_to_yx(built["um"]), z_min=2 - 160, z_max=14 + 160,
                                    num_theta_bins=720, device="cpu", config=cfg)
    pts, _, _ = _all_points(src["rays"])
    shell_r, r, conf, valid = smap.lookup(torch.from_numpy(pts.astype(np.float32)))
    keep = valid & (r <= shell_r)
    assert keep.float().mean().item() > 0.995


def test_villa_export_validator_accepts_the_store(built):
    d = _villa()
    exp = os.path.join(os.path.dirname(d), "vesuvius", "src", "vesuvius", "neural_tracing", "winding_models",
                       "export_spiral_supervision.py")
    if not os.path.exists(exp):
        pytest.skip("villa's export_spiral_supervision.py not found next to spiral-fitting")
    import importlib.util
    spec = importlib.util.spec_from_file_location("villa_export_spiral_supervision", exp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.validate_spiral_supervision(built["store"])["num_rays"] == len(built["rays"])
