# winding-audit

Checks the windings of a spiral fit against each other and against the surface prediction, and says where two
windings sit on the same sheet, where a wrap was skipped, and where a node is not on a sheet at all.

**Status: 0.3 (12 October 2026): audit, plus a first repair.** The audit measures, per node, how far each winding
is from the next one's surface and from the predicted sheets. The repair moves a node that is off any sheet, or
on a neighbouring winding's sheet, onto the sheet between its two neighbours. Numbers in this README are
measurements on real fits and will change.

## Why

Erwin Nieuwlaar's PHerc. 343 First Letters recipe (villa `fit_spiral` → snap onto the surface prediction → lasagna
flatten → render → ink) is now run by several groups. Its windings are accepted by eye. Nothing checks that winding
k and winding k+1 lie on different sheets everywhere, or that a winding does not drift onto its neighbour's sheet. A
winding on the wrong sheet renders the same papyrus as its neighbour, or the gap between wraps, and the ink models
then read nothing new.

On our own PHerc. 0175A fit (120 windings, band z 6900–7900, 853 cm² after flattening), 21 % of the nodes lie within
0.3 sheet pitch of the next winding's surface, and 41 of 119 winding pairs are collapsed onto one sheet over most of
their area. 15 % of the nodes are on no predicted sheet. Erwin's winning pair shows the same defects, at similar
rates, mostly outside its letters. This tool makes them visible per node, per winding and per band, and repairs the
part that a re-snap can repair.

## What it measures

1. **Neighbour spacing.** Distance from every node of winding k to the *surface* of winding k+1: the nearest node
   of k+1 at a similar angle around the turn is found first (a KD-tree with the angular position as a fourth axis,
   so the spiral seam, where the end of one winding meets the start of the next, is not read as a merger), then the
   node is projected onto the tangent plane of k+1 at that node. Measuring to the nearest *node* instead misses
   windings that lie on top of each other: on a 20-voxel grid the nearest node can be 10 or more voxels to the
   side. The sheet pitch is read from the prediction itself, as the median spacing of consecutive predicted sheets
   along the node normals, so a pair that collapses onto one sheet cannot shrink its own pitch. `merged` = closer
   than 0.3 × pitch; `skipped` = farther than 1.7 × pitch. A far node whose nearest point is on the neighbour's own
   boundary is `unassessed` (the true nearest point may be outside the neighbour's coverage).
2. **Sheets between.** First each end is placed: the exact distance from the node, and from its nearest point on
   winding k+1, to the nearest voxel of the surface prediction (m7, uint8; binary in the published level-0 stores,
   thresholded at 128). A node more than `--node-tol` voxels (default 2) from any predicted sheet is `node_off_sheet`,
   a finding on its own: on Erwin Nieuwlaar's PHerc. 343 windings such nodes sit in a CT trough and render no papyrus
   (see the positive control below). When both ends are on a sheet, the prediction is sampled every voxel along the
   straight segment between them and the runs of sheet voxels give the verdict: 2 = adjacent
   wraps, 3+ = a wrap was skipped, 1 = one unbroken run joins them, which is `merged` (both windings on one
   sheet, a fit error) when the neighbour is closer than half a pitch and `fused` (the prediction fills the gap
   between two wraps, a model error in a compressed region) when it is about a pitch away. This separates fit
   errors from prediction errors, which the spacing check alone cannot do.
3. **Collapsed pairs.** Each pair's pitch is compared with the local pitch, the mean pair pitch over three pairs
   either side. Two windings that sit on one sheet all the way round make their own median spacing small, so
   against their own pitch they look normal; against the local pitch the pair shows as `collapsed` (ratio below
   0.75), usually next to a `stretched` pair (above 1.25).
4. **Report.** `audit.csv` / `audit.json` with one row per winding pair (pitch, local pitch and their ratio, spacing fractions merged / skipped /
   unassessed, fraction of nodes off any sheet and within 1 / 2 / 3 voxels of one, and among nodes with both ends on
   a sheet the fractions adjacent / merged / fused / skipped, plus area in cm² when `--voxel-um` is given),
   `summary.json` for the band, per-pair maps (`maps/<winding>.npz`: distance, flag, verdict, runs, node-to-sheet
   distance on the winding grid), `--grid-figures` (one PNG per winding with the four maps) and `--sections`
   cross-section PNGs at the requested z (all windings over the prediction slice, coloured by spacing flag and by
   sheet verdict). The audit is resumable: pairs already written are skipped on a re-run.

5. **Repair** (`winding-audit repair`). The PHerc. 343 snap moves each node along its normal to the *nearest*
   predicted sheet within ±15 voxels; with a pitch of 10–20 voxels that can be the next wrap's sheet. The repair
   uses the neighbours as bounds. Along the node's normal, oriented from winding k−1 to winding k+1, the prediction
   is sampled over ±2.5 pitches and its sheet runs are listed. A node on its own run (within 2 voxels) that is not a
   neighbour's run is kept, bit for bit. A node on a run it shares with a neighbour half a pitch or more away is
   kept too (`kept_shared_run`): the prediction joins the two wraps there and cannot say which node, if any, is
   wrong. A node off any sheet, or sharing a run with a neighbour closer than half a pitch, moves to the centre of
   the nearest run strictly between the two neighbours' runs, at most 1.25 pitches away. If there is none, the node
   stays and is reported `unrepaired`: either the neighbour is the misplaced one (it is moved when its own turn
   comes), or there is no free sheet between the neighbours. A move larger
   than 0.4 pitch must agree with at least two neighbouring moves, or it is undone, so the repair cannot pull a
   spike out of the mesh. Windings are repaired from the inside out, each against its already-repaired inner
   neighbour, so two windings on one sheet are separated once. Skipped wraps are left alone: every winding after
   the skip follows it, so no node is misplaced locally; it is a topology error and stays in the audit.

## Results (12 October 2026, v0.3)

Three bands fitted with the PHerc. 343 recipe (villa `fit_spiral` at `0fb38c45`, 20k steps, snapped onto the
published m7 prediction at level 0, then lasagna-flattened), 120 windings each, audited against the same
prediction after the snap and after the flatten (the flatten output is what gets rendered). Erwin Nieuwlaar's
released pair is the control. "Off any sheet" and "within 0.3 pitch" are fractions of all assessed nodes; the four
sheet verdicts are fractions of the nodes with both ends on a predicted sheet. A pair is collapsed when its median
spacing is under 0.75 of the prediction's sheet pitch.

<!-- results-table -->
| band | stage | area | off any sheet | within 0.3 pitch of the next winding | adjacent | same sheet | gap filled | wrap skipped | collapsed pairs |
|---|---|---|---|---|---|---|---|---|---|
| PHerc. 343, w047→w048 (Erwin Nieuwlaar, control) | flatten | 7 cm² | 28 % | 22 % | 61 % | 32 % | 4 % | 2 % | 0 of 1 |
| PHerc. 0175A, z 6900–7900 | snap | 645 cm² | 14 % | 21 % | 71 % | 26 % | 1 % | 2 % | 42 of 119 (179 cm²) |
| PHerc. 0175A, z 6900–7900 | flatten | 853 cm² | 15 % | 21 % | 71 % | 25 % | 1 % | 2 % | 41 of 119 (232 cm²) |
| PHerc. 0175A, z 6900–7900 | flatten + repair | 853 cm² | 13 % | 20 % | 73 % | 24 % | 1 % | 2 % | 40 of 119 (218 cm²) |
| PHerc. 0813, z 12200–13200 | flatten | 1090 cm² | 34 % | 17 % | 60 % | 27 % | 6 % | 7 % | 29 of 119 (298 cm²) |
| PHerc. 0800, z 5300–6300 | flatten | 1037 cm² | 28 % | 24 % | 52 % | 37 % | 8 % | 3 % | 87 of 119 (722 cm²) |
<!-- /results-table -->

- **The main defect is collapsed windings: two consecutive windings on one sheet.** On PHerc. 0175A, 21 % of the
  flattened nodes are within 0.3 pitch of the next winding's surface, and 41 of 119 pairs (232 cm²) are collapsed
  over most of their area. Exact point-to-triangle distances to the neighbouring mesh confirm it (w030→w031: 55 % of
  nodes within 2 voxels; w070→w071: 2 %). Where a pair collapses, every sheet around it still holds one winding, so
  the band has more windings than sheets there (next point). Erwin's winning pair has the same defect over 21.5 % of w047, mostly outside the
  letters (14 % inside the letter area, 24 % elsewhere).
- **The collapse starts in the fit, and the snap makes it.** Before the snap, the PHerc. 0175A windings do not touch
  (2.4 % of nodes within 0.3 pitch of the next winding), but the inner ones are 8–10 voxels apart against a sheet
  pitch of 15 in the prediction: about 1.5 windings per sheet. The snap then moves every winding to its nearest
  sheet, and pairs of windings land on the same one (21 %), next to gaps of about 17 voxels. Comparing the fit's
  winding spacing with the prediction's sheet spacing before snapping shows where the fitted spiral has too many
  turns.
- **The other two bands are worse.** PHerc. 0813: 29 of 119 pairs collapsed (298 cm² of 1,090) and 34 % of nodes
  off any sheet. PHerc. 0800: 87 of 119 pairs collapsed (722 cm² of 1,037), only 52 % adjacent wraps. Its fit is
  too dense across the whole band, not only in the core: at four sampled windings (w020, w050, w080, w110), before
  the snap the windings are 9.5–12 voxels apart against a predicted sheet pitch of 16.5–17.5. A winding that runs between two sheets is snapped to one or the
  other depending on its local offset, so it can switch sheets around the turn.
- **15 % of the flattened PHerc. 0175A nodes are on no predicted sheet.** On the control, such nodes sit in a CT trough: a render
  there samples the gap between wraps.
- **The repair moves 3.1 % of the nodes onto a sheet between their neighbours, and the CT confirms the moves.** On
  three PHerc. 0175A windings that have 66-layer renders, the 811 moved nodes inside the renders go from a mean CT
  value of 83 at their old position to 104 at the new one (nodes kept on their own sheet: 99). Over the band, nodes
  off any sheet drop from 15.2 % to 12.7 % and adjacent wraps rise from 71.4 % to 73.2 %. Collapsed windings stay:
  with both neighbours on the sheets around it, a winding has no free sheet to move to. They need a re-fit, not a
  re-snap.

![PHerc. 0175A slice z 7400: windings w106–w115 over the prediction; w110 and w111 run on one sheet](docs/pherc0175a_w110_section.png)

![PHerc. 0175A w110 → w111 on the winding grid: spacing flags, distance in pitches, sheet verdict, node-to-sheet distance](docs/pherc0175a_w110_flat.png)

![Repair checked against the CT: mean depth profile of kept nodes and of moved nodes at their old and new positions](docs/pherc0175a_repair_ct.png)

![PHerc. 0175A slice z 7400, all flattened windings over the prediction, coloured by sheet verdict](docs/pherc0175a_z7400.png)

## Positive control: Erwin Nieuwlaar's PHerc. 343 windings

The two windings of the awarded PHerc. 343 First Letters image (w047 and w048, z 9500–11000, released by Erwin
Nieuwlaar as lasagna flatten output on a 20-voxel grid) were audited against the m7 prediction he snapped to
(`20250521140437-surface-20260413222639-surface-m7-L0-th0.2.zarr`, level 0). The CT under each node was read from a
66-layer render of his concatenated mesh.

![Erwin's PHerc. 343 windings: CT depth profile by distance to the prediction, and audit verdicts in the letter area](docs/pherc0343_control.png)

- **The off-sheet measure tracks the CT.** Nodes on a predicted-sheet voxel have a sharp papyrus peak 2.5 voxels
  outward, inside the ink window. Nodes 2–6 voxels from the prediction have a weak peak at the window's edge. Nodes
  with no predicted sheet within 6 voxels sit in a CT trough about 20 grey levels darker: a render there samples the
  gap between wraps.
- **The area where the text was read scores higher.** Inside the letter area of w047 (letters 6–18), 77 % of the
  nodes with both ends on a sheet are adjacent wraps, 19 % share a sheet with w048, and 17 % of nodes are off any
  sheet. On the rest of w047 the figures are 56 %, 37 % and 32 %. 16 of the 18 letter centres lie on a
  predicted-sheet voxel.
- **A winning pair is not clean everywhere.** 21.5 % of w047's nodes are within 0.3 pitch (4.7 voxels) of w048's
  surface: 14 % in the letter area, 24 % elsewhere. Exact point-to-triangle distances to the w048 mesh give the same
  figure (20.6 %). Over the whole pair, 28 % of nodes are off any sheet; of the rest, 61 % are adjacent wraps, 32 %
  share a sheet with w048, 4 % have the prediction filling the gap and 2 % skip a wrap. The audit is a map of where
  a winding can be trusted, not a pass or fail for the whole winding.

## Use

```
pip install -e .            # numpy, scipy, zarr>=3, tifffile, matplotlib
winding-audit audit <dir with w000_<tag>/ ... tifxyz folders> --surface <m7 prediction zarr root> --group 0 \
    --out runs/pherc0175a_band1 --voxel-um 8.64 --sections 7000 7400 7800
```

```
winding-audit repair <band dir> --surface <m7 zarr root> --group 0 --out runs/pherc0175a_band1_repaired \
    --audit --voxel-um 8.64
# repaired tifxyz in <out>/windings, per-node shift/status maps, repair.csv, and the audit of the result in <out>/audit
```

`--surface` is the surface prediction at the same resolution level as the tifxyz coordinates (level 0 voxels for the
spiral fitter's output). `--no-sheets` runs the spacing check only. `--subdir tifxyz/flatten.tifxyz` reads lasagna
flatten output, where each winding folder holds its tifxyz one level down. `--node-tol` sets how far (voxels) a
node may be from a predicted sheet and still count as on it; the per-node distance is in the maps either way.
The synthetic scroll used by the tests:

```
winding-audit synthetic demo --merge 1 1.0 2.0 --skip 4 3.5 4.5
winding-audit audit demo/windings --surface demo/surface.zarr --out demo/audit --sections 12
pytest tests
```

## Inputs and limits

- Input windings are tifxyz folders as written by villa's spiral fitter / snap (x.tif, y.tif, z.tif, meta.json;
  -1 = no node), one folder per winding named `w<NNN>…`. Consecutive winding numbers are assumed to be consecutive
  wraps.
- The pitch comes from the prediction's sheet spacing (median over 2,000 sampled nodes per winding). Missed sheets
  in the prediction make it slightly long (on PHerc. 0175A, 14–17 voxels against 12–13 for clean winding pairs).
  Without a prediction (`--no-sheets`) the pair's own median spacing is used and collapsed pairs are judged against
  the mean of the neighbouring pairs.
- The neighbour's surface is its tangent plane at the nearest node, not the exact mesh. Checked against exact
  point-to-triangle distances on PHerc. 0175A: within 1 voxel for about 90 % of nodes.
- Collapsed windings are reported, not repaired: separating them means moving every later winding by one sheet in
  that sector, which is a re-fit.
- Agreement with the surface prediction is agreement with a model, not with the papyrus. The CT checks in this
  README cover the control pair and three rendered PHerc. 0175A windings only.

## Credits

Built on villa's spiral fitter and tifxyz format (ScrollPrize), the m7 surface predictions (Vesuvius Challenge open
data), and Erwin Nieuwlaar's PHerc. 343 recipe (snap step). Developed with AI assistance under my direction.

MIT license.
