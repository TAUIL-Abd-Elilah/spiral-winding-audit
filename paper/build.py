"""Build paper/paper.pdf: fill the result tables of paper.html from docs/results and print it with headless Chrome.

    python paper/build.py [--chrome PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"


def pc(x: float) -> str:
    return f"{100 * x:.0f}&nbsp;%"


def results_table() -> str:
    rows = [("PHerc. 343 w047&rarr;w048 (control)", "flatten", "pherc0343_control"),
            ("PHerc. 0175A z 6900&ndash;7900", "fit (before snap)", "pherc0175a_z6900-7900_fit"),
            ("PHerc. 0175A z 6900&ndash;7900", "snap", "pherc0175a_z6900-7900_snap"),
            ("PHerc. 0175A z 6900&ndash;7900", "flatten", "pherc0175a_z6900-7900_flat"),
            ("PHerc. 0175A z 6900&ndash;7900", "flatten + repair", "pherc0175a_z6900-7900_flat_repaired"),
            ("PHerc. 0813 z 12200&ndash;13200", "snap", "pherc0813_z12200-13200_snap"),
            ("PHerc. 0813 z 12200&ndash;13200", "flatten", "pherc0813_z12200-13200_flat"),
            ("PHerc. 0800 z 5300&ndash;6300", "snap", "pherc0800_z5300-6300_snap"),
            ("PHerc. 0800 z 5300&ndash;6300", "flatten", "pherc0800_z5300-6300_flat")]
    out = ['<table><tr><th>band</th><th>stage</th><th class="n">area</th><th class="n">off any sheet</th>'
           '<th class="n">within 0.3 pitch of next</th><th class="n">adjacent</th><th class="n">same sheet</th>'
           '<th class="n">gap filled</th><th class="n">skipped</th><th class="n">close pairs</th></tr>']
    for name, stage, d in rows:
        f = RES / d / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        out.append(f"<tr><td>{name}</td><td>{stage}</td><td class='n'>{s['area_cm2']:.0f}&nbsp;cm²</td>"
                   f"<td class='n'>{pc(s['frac_node_off_sheet'])}</td><td class='n'>{pc(s['frac_merged'])}</td>"
                   f"<td class='n'>{pc(s['frac_sheet_adjacent'])}</td><td class='n'>{pc(s['frac_sheet_merged'])}</td>"
                   f"<td class='n'>{pc(s['frac_sheet_fused'])}</td><td class='n'>{pc(s['frac_sheet_skipped'])}</td>"
                   f"<td class='n'>{s.get('pairs_collapsed', 0)} / {s['pairs']}</td></tr>")
    out.append("</table>")
    out.append('<p class="note"><b>Table 1.</b> Audit of each band. "Off any sheet" and "within 0.3 pitch" are fractions '
               'of all assessed nodes; the four verdicts are fractions of nodes with both ends on a sheet; close pairs have '
               'a median spacing under 0.75 of the prediction\'s sheet pitch. Before the snap the nodes are not yet on '
               'sheets, so the fit row\'s off-sheet figure is expected to be high.</p>')
    return "\n".join(out)


def density_table() -> str:
    rows = [("PHerc. 0800 z 6300&ndash;7800", "pherc0800_band3_fit"), ("PHerc. 0800 z 5300&ndash;6300", "pherc0800_band2_fit"),
            ("PHerc. 0175A z 6900&ndash;7900", "pherc0175a_fit"), ("PHerc. 0813 z 12200&ndash;13200", "pherc0813_fit")]
    out = ['<table><tr><th>fit, w040 to w115</th><th>slice</th><th class="n">windings crossed</th>'
           '<th class="n">sheets crossed</th><th class="n">windings per sheet</th><th class="n">radius per winding</th></tr>']
    for name, d in rows:
        f = RES / "density" / f"{d}.json"
        if not f.exists():
            continue
        for r in json.load(open(f)):
            if not r.get("rays"):
                continue
            out.append(f"<tr><td>{name}</td><td>z {r['z']}</td><td class='n'>{r['windings_crossed']:.0f}</td>"
                       f"<td class='n'>{r['sheets_crossed']:.0f}</td><td class='n'>{r['windings_per_sheet']:.2f}</td>"
                       f"<td class='n'>{r['radius_per_winding']:.1f}&nbsp;vox</td></tr>")
    out.append("</table>")
    out.append('<p class="note"><b>Table 2.</b> Windings and predicted sheets crossed by 180 radial rays between the '
               'crossings of w040 and w115, before any snap (medians over rays).</p>')
    return "\n".join(out)


def pieces_table() -> str:
    rows = [("PHerc. 343 w047 + w048 (control)", "pherc0343_control"), ("PHerc. 0175A, flatten", "pherc0175a_z6900-7900_flat"),
            ("PHerc. 0175A, flatten + repair", "pherc0175a_z6900-7900_flat_repaired"),
            ("PHerc. 0813, flatten", "pherc0813_z12200-13200_flat"), ("PHerc. 0800, flatten", "pherc0800_z5300-6300_flat")]
    out = ['<table><tr><th>band</th><th class="n">area</th><th class="n">on a sheet</th><th class="n">duplicate</th>'
           '<th class="n">kept: single sheet, read once</th><th class="n">pieces (&ge;&nbsp;1&nbsp;cm²)</th></tr>']
    for name, d in rows:
        f = RES / d / "pieces_summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        out.append(f"<tr><td>{name}</td><td class='n'>{s['area_cm2']:.0f}&nbsp;cm²</td><td class='n'>{pc(s['on_sheet'] / s['n'])}</td>"
                   f"<td class='n'>{s['duplicate_cm2']:.0f}&nbsp;cm² ({pc(s['duplicate'] / s['n'])})</td>"
                   f"<td class='n'><b>{s['kept_cm2']:.0f}&nbsp;cm² ({pc(s['kept'] / s['n'])})</b></td>"
                   f"<td class='n'>{s['pieces']} ({s['pieces_over_1cm2']})</td></tr>")
    out.append("</table>")
    out.append('<p class="note"><b>Table 3.</b> Single-sheet pieces of each flattened band (all 120 windings).</p>')
    return "\n".join(out)


def find_chrome(explicit: str | None) -> str:
    for c in [explicit, os.environ.get("CHROME"), r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", shutil.which("chromium"),
              shutil.which("google-chrome"), shutil.which("chrome")]:
        if c and Path(c).exists():
            return c
    raise SystemExit("no Chrome/Edge found; pass --chrome PATH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome")
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    src = (ROOT / "paper" / "paper.html").read_text(encoding="utf-8")
    html = (src.replace("<!-- RESULTS-TABLE -->", results_table())
               .replace("<!-- DENSITY-TABLE -->", density_table())
               .replace("<!-- PIECES-TABLE -->", pieces_table()))
    built = ROOT / "paper" / "paper_built.html"
    built.write_text(html, encoding="utf-8")
    if a.html_only:
        print("wrote", built)
        return
    pdf = ROOT / "paper" / "paper.pdf"
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([find_chrome(a.chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", built.as_uri()],
                       check=True, timeout=180, capture_output=True)
    print("wrote", pdf, f"{pdf.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
