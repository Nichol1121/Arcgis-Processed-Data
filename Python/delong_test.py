"""Paired DeLong test: are the RF and AHP AUCs different on the same testing points?

Input: a CSV exported from ArcGIS with one row per testing point and three columns:
  LABEL (1 = flood, 0 = non-flood), RF (flood probability), AHP (AHP index).
Run:   python delong_test.py testing_values.csv
Needs only numpy (included in ArcGIS Pro's Python). Edit the names below if your fields differ.
"""
import csv, math, sys
import numpy as np

LABEL, RF, AHP = "LABEL", "RandomFore", "AHP_WS"

def midrank(x):
    order = np.argsort(x, kind="mergesort"); xs = x[order]
    n = len(x); r = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        r[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n); out[order] = r
    return out

def delong(y, scores):
    """AUCs and their covariance matrix (DeLong et al., 1988; fast form of Sun and Xu, 2014)."""
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([midrank(pos[i]) for i in range(k)])
    ty = np.array([midrank(neg[i]) for i in range(k)])
    tz = np.array([midrank(np.concatenate([pos[i], neg[i]])) for i in range(k)])
    auc = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return auc, cov, m, n

path = sys.argv[1] if len(sys.argv) > 1 else "testing_values.csv"
with open(path, newline="", encoding="utf-8-sig") as f:
    text = f.read()
header = text.splitlines()[0]
delim = max([",", ";", "\t"], key=header.count)          # ArcGIS may use ; when decimals are commas
reader = csv.DictReader(text.splitlines(), delimiter=delim)
cols = {c.strip().lower(): c for c in reader.fieldnames}

def find(name, alts):
    for c in [name] + alts:
        if c.lower() in cols:
            return cols[c.lower()]
    sys.exit(f"No '{name}' column found. The columns in your CSV are:\n  " + ", ".join(reader.fieldnames)
             + f"\nEdit the line  LABEL, RF, AHP = ...  at the top of this script to use your column names.")

kl = find(LABEL, ["label_1", "flood", "class", "presence"])
kr = find(RF, ["rf"])
ka = find(AHP, ["finalahp1"])
print(f"Using columns: label = {kl}, RF = {kr}, AHP = {ka}")
rows = []
for r in reader:
    try:
        row = (int(float(r[kl].replace(",", "."))), float(r[kr].replace(",", ".")), float(r[ka].replace(",", ".")))
        if row[1] > -9999 and row[2] > -9999:  # -9999 = NoData (the point on the river cell)
            rows.append(row)
    except (ValueError, TypeError, AttributeError):
        pass  # a point with an empty value (e.g. on a river cell) is skipped
a = np.array(rows)
y, s = a[:, 0].astype(int), a[:, 1:].T
auc, cov, m, n = delong(y, s)
se = np.sqrt(np.diag(cov))
diff = auc[0] - auc[1]
sd = math.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
z = diff / sd
p = math.erfc(abs(z) / math.sqrt(2))
r = cov[0, 1] / (se[0] * se[1])
print(f"Points used: {m + n} ({m} flood, {n} non-flood)")
print(f"RF  AUC {auc[0]:.3f}  95% CI {auc[0]-1.96*se[0]:.2f}-{auc[0]+1.96*se[0]:.2f}")
print(f"AHP AUC {auc[1]:.3f}  95% CI {auc[1]-1.96*se[1]:.2f}-{auc[1]+1.96*se[1]:.2f}")
print(f"Correlation between the two AUC estimates: {r:.2f}")
print(f"Difference {diff:.3f}  95% CI {diff-1.96*sd:.3f} to {diff+1.96*sd:.3f}")
print(f"Paired DeLong test: z = {z:.2f}, p = {p:.4f}")
