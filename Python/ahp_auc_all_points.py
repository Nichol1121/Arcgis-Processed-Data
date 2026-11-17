"""AHP AUC on all sample points and on the block cross-validation points.

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

all_pts, cv_pts = [], []
for r in rows:
    y, a = int(float(r["LABEL"])), num(r["AHP_WS"])
    if a is None:
        continue                       # no AHP value (river cell or outside the map)
    all_pts.append((y, a))
    if all(num(r[k]) is not None for k in FACTORS):
        cv_pts.append((y, a))          # the forest can only use points with all eight factor values

for name, pts in [("All sample points", all_pts), ("Block cross-validation points", cv_pts)]:
    a, nf, nn = auc(pts)
    print(f"{name}: {nf + nn} points ({nf} flood, {nn} non-flood), AHP AUC = {a:.3f}")
