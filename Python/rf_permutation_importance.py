"""Python re-implementation of the random forest: permutation importance (Table IV, 'RF permutation'),
out-of-bag error and test AUC, and the alternative settings in Section III-C.

Input: a CSV with one row per sample point: LABEL (1 = flood, 0 = non-flood),
SET ('train' or 'test') and the eight factor scores (1-9).
Run:   python rf_permutation_importance.py sintang_points_scores.csv
The rows are in the order the model used them (training rows first), so the numbers match the paper.
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

FACTORS = ["FA", "DEM", "DD", "DIST", "LULC", "RAIN", "SLOPE", "SOIL"]  # edit to match your field names

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sintang_points_scores.csv")
train, test = df[df.SET == "train"], df[df.SET == "test"]

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
