"""AHP AUC on all sample points and on the 277 block cross-validation points.

It also lists the sample points that are left out, and how many of the rest fall in each fold.

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

all_pts, cv_pts, left_out, folds = [], [], [], {}
for r in rows:
    y, a = int(float(r["LABEL"])), num(r["AHP_WS"])
    missing = [k for k in FACTORS if num(r[k]) is None]
    if a is None or missing:
        left_out.append((r.get("PT_ID", "?"), "flood" if y == 1 else "non-flood",
                         ", ".join((["AHP map"] if a is None else []) + missing)))
    if a is None:
        continue                       # no AHP value: the AHP cannot be scored here
    all_pts.append((y, a))
    if not missing:                    # the forest needs all eight factor values
        cv_pts.append((y, a))
        if "FOLD" in r:
            folds[r["FOLD"]] = folds.get(r["FOLD"], 0) + 1

print(f"Sample points in the file: {len(rows)}")
print(f"Left out of the block cross-validation ({len(left_out)}), with the layers that have no value:")
for pt, kind, what in left_out:
    print(f"  {pt:7s} {kind:10s} {what}")

for name, pts in [("All sample points", all_pts), ("Block cross-validation points", cv_pts)]:
    a, nf, nn = auc(pts)
    print(f"{name}: {nf + nn} points ({nf} flood, {nn} non-flood), AHP AUC = {a:.3f}")
if folds:
    print("Block cross-validation points per fold: " + ", ".join(f"{k}: {folds[k]}" for k in sorted(folds)))
