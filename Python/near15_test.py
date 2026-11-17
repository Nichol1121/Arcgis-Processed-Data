"""Test with non-flood points placed only within 15 km of a reported flood (Section IV-C).

Input: near15_points_scores.csv, a second sample of 142 flood and 142 non-flood points in which every
non-flood point lies within 15 km of a reported flood, split 70/30 like the main sample.
Run:   python near15_test.py near15_points_scores.csv
Needs numpy, pandas and scikit-learn (all included in ArcGIS Pro's Python).
"""
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

FACTORS = ["FA", "DEM", "DD", "DIST", "LULC", "RAIN", "SLOPE", "SOIL"]
AHP_WEIGHTS = {"FA": 29.39, "DEM": 19.06, "DD": 14.14, "DIST": 13.60, "LULC": 9.31, "RAIN": 6.71, "SLOPE": 4.52, "SOIL": 3.26}

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "near15_points_scores.csv")
train, test = df[df.SET == "train"], df[df.SET == "test"]

# AHP index = weighted sum of the eight scores (same weights as the AHP map)
ahp = sum(test[f] * w for f, w in AHP_WEIGHTS.items())

# random forest with the same settings as in the paper
rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=1,
                            max_depth=None, criterion="gini", random_state=0)
rf.fit(train[FACTORS], train.LABEL)
prob = rf.predict_proba(test[FACTORS])[:, 1]

print(f"Training points {len(train)}, testing points {len(test)}")
print(f"Random forest AUC {roc_auc_score(test.LABEL, prob):.3f}")
print(f"AHP AUC           {roc_auc_score(test.LABEL, ahp):.3f}")
