"""AHP AUC on the 277 sample points that have a value in every factor layer and in the AHP map.

It also lists the sample points that are left out, and how many of the 277 fall in each block
cross-validation fold.

Input: the CSV exported from SintangPoint_Joined after Extract Multi Values to Points
(FinalAhp.tif -> field AHP_WS). Run:  python ahp_auc_all_points.py all_points.csv
Needs only the Python that comes with ArcGIS Pro.
"""
import csv, sys

FACTORS = ["DEM", "DIST", "DD", "FA", "LULC", "RAIN", "Slope_1", "SOIL"]   # edit if your names differ

path = sys.argv[1] if len(sys.argv) > 1 else "all_points.csv"
text = open(path, newline="", encoding="utf-8-sig").read().splitlines()
delim = max([",", ";", "\t"], key=text[0].count)
rows = list(csv.DictReader(text, delimiter=delim))

def num(s):
    s = (s or "").strip().replace(",", ".")
    try:
        v = float(s)
        return None if v <= -9999 else v
    except ValueError:
        return None

def auc(points):
    """Share of flood/non-flood pairs where the flood point scores higher (ties count half)."""
    f = [v for y, v in points if y == 1]; n = [v for y, v in points if y == 0]
    wins = sum((a > b) + 0.5 * (a == b) for a in f for b in n)
    return wins / (len(f) * len(n)), len(f), len(n)

cv_pts, left_out, folds = [], [], {}
for r in rows:
    y, a = int(float(r["LABEL"])), num(r["AHP_WS"])
    missing = [k for k in FACTORS if num(r[k]) is None]
    if a is None or missing:           # the AHP needs its map value, the forest all eight factors
        left_out.append((r.get("PT_ID", "?"), "flood" if y == 1 else "non-flood",
                         ", ".join((["AHP map"] if a is None else []) + missing)))
        continue
    cv_pts.append((y, a))
    if "FOLD" in r:
        folds[r["FOLD"]] = folds.get(r["FOLD"], 0) + 1

print(f"Sample points in the file: {len(rows)}")
print(f"Left out ({len(left_out)}), with the layers that have no value:")
for pt, kind, what in left_out:
    print(f"  {pt:7s} {kind:10s} {what}")

a, nf, nn = auc(cv_pts)
print(f"Points used: {nf + nn} ({nf} flood, {nn} non-flood), AHP AUC = {a:.3f}")
if folds:
    print("Points per fold: " + ", ".join(f"{k}: {folds[k]}" for k in sorted(folds)))
