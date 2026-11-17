"""Python re-implementation of the random forest: permutation importance (Table IV, 'RF permutation'),
out-of-bag error and test AUC, and the alternative settings in Section III-C.

Input: a CSV with one row per sample point: LABEL (1 = flood, 0 = non-flood),
SET ('train' or 'test') and the eight factor scores (1-9). A table exported from ArcGIS Pro also
works: it may use ';' between fields, SET_ for SET and Slope_1 for SLOPE.
Run:   python rf_permutation_importance.py all_points.csv
all_points.csv holds the factor scores that ArcGIS Pro extracted at the sample points, the same values
the ArcGIS forest was trained on. Like ArcGIS Pro, the script skips points with a NoData factor value.
Works in the ArcGIS Pro Python environment (it includes scikit-learn) or any Python with
numpy, pandas and scikit-learn. The forest draws its bootstrap samples by row, so the row order
and the scikit-learn version change the numbers slightly; the ranking of the factors stays the same.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

FACTORS = ["FA", "DEM", "DD", "DIST", "LULC", "RAIN", "SLOPE", "SOIL"]
# other names the same field may have in a table exported from ArcGIS Pro
ALIASES = {"SET": ["SET_"], "SLOPE": ["Slope_1"], "LABEL": []}

path = sys.argv[1] if len(sys.argv) > 1 else "all_points.csv"
first = open(path, encoding="utf-8-sig").readline()
sep = max([",", ";", "\t"], key=first.count)
df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", decimal="," if sep == ";" else ".")
upper = {c.strip().upper(): c for c in df.columns}

def col(name):
    for n in [name] + ALIASES.get(name, []):
        if n.upper() in upper:
            return upper[n.upper()]
    sys.exit(f"No field '{name}' in {path}. Fields found: {', '.join(df.columns)}")

df = df.rename(columns={col(n): n for n in FACTORS + ["SET", "LABEL"]})
for f in FACTORS:                      # empty cells and -9999 are NoData; ArcGIS Pro skips such points
    df[f] = pd.to_numeric(df[f], errors="coerce").where(lambda v: v > -9999)
skipped = df[FACTORS].isna().any(axis=1)
if skipped.any():
    print(f"Skipped {skipped.sum()} points with a NoData factor value")
df = df[~skipped]
df["SET"] = df["SET"].astype(str).str.strip().str.lower()
train, test = df[df.SET == "train"], df[df.SET == "test"]
if len(train) == 0 or len(test) == 0:
    sys.exit(f"{path} has {len(train)} training and {len(test)} testing points; it needs both. "
             "Use all_points.csv, or another table of all sample points.")

# 1. Train the forest with the same settings as the ArcGIS forest
rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=1,
                            max_depth=None, criterion="gini", oob_score=True, random_state=0)
rf.fit(train[FACTORS], train.LABEL)
auc = roc_auc_score(test.LABEL, rf.predict_proba(test[FACTORS])[:, 1])
print(f"Training points {len(train)}, testing points {len(test)}")
print(f"Test AUC {auc:.3f}, out-of-bag error {100 * (1 - rf.oob_score_):.1f}%\n")

# 2. Shuffle one factor at a time on the testing points (20 times) and measure the AUC drop
pi = permutation_importance(rf, test[FACTORS], test.LABEL, scoring="roc_auc",
                            n_repeats=20, random_state=0)
drop = pi.importances_mean

# 3. Share of the total drop; a factor whose shuffling does not lower the AUC gets 0
pos = np.clip(drop, 0, None)
share = 100 * pos / pos.sum()
print(f"{'Factor':8s} {'AUC drop':>9s} {'Share (%)':>10s}")
for f, d, s in sorted(zip(FACTORS, drop, share), key=lambda t: -t[2]):
    print(f"{f:8s} {d:9.4f} {s:10.1f}")

# 4. Alternative settings (Section III-C): change one setting at a time and report the test AUC
print("\nTest AUC with other settings:")
for name, change in [("chosen settings", {}), ("500 trees", {"n_estimators": 500}),
                     ("4 predictors per split", {"max_features": 4}), ("minimum leaf size 5", {"min_samples_leaf": 5})]:
    kw = dict(n_estimators=100, max_features="sqrt", min_samples_leaf=1, max_depth=None, criterion="gini", random_state=0)
    kw.update(change)
    m = RandomForestClassifier(**kw).fit(train[FACTORS], train.LABEL)
    print(f"  {name:24s} {roc_auc_score(test.LABEL, m.predict_proba(test[FACTORS])[:, 1]):.3f}")
