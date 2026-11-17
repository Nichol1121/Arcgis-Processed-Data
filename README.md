# Flood susceptibility in Sintang district (AHP and random forest)

Data and code for the paper **"Flood Susceptibility Assessment Using AHP and Random Forest Machine Learning in Sintang District, West Kalimantan Province, Indonesia"** (A. N. Riandi and E. Irwansyah, ICDSG 2026).

## Contents

| Path | What it is |
|---|---|
| `DEM.tif`, `Distance.tif`, `Drainage.tif`, `FlowAclumation.tif`, `LULC.tif`, `Rainfall.tif`, `Slope.tif`, `Soil.tif` | The eight conditioning factors, scored 1–9 on the 790 m grid (Table II of the paper) |
| `results/FinalAhp.tif` | AHP flood susceptibility index (Weighted Sum of the eight scored layers) |
| `results/FinalAHPNamed.tif` | AHP map in five classes (Fig. 4) |
| `results/RandomForest.tif` | Random forest flood probability |
| `results/RandomForestNamed1.tif` | Random forest map in five classes (Fig. 5) |
| `sintang_unique_flood_points.xlsx` | Groundsource flood locations in Sintang district |
| `Sintang_district_boundary_shapefile.rar` | District boundary |
| `Python/` | Python scripts and the point tables they read (below) |

## Python scripts

The maps, the ROC curves (ArcSDM) and the block cross-validation were made in ArcGIS Pro 3.3. The scripts in `Python/` compute the numbers that ArcGIS Pro does not report:

| Script | Input | What it gives | Where in the paper |
|---|---|---|---|
| `delong_test.py` | `testing_values.csv` | AUC and 95% CI of both maps on the 85 testing points, and the paired DeLong test | Abstract; Sections III-E and IV-C; Fig. 6 |
| `ahp_auc_all_points.py` | `all_points.csv` | AHP AUC on all sample points and on the block cross-validation points | Section IV-C; Fig. 7 |
| `rf_permutation_importance.py` | `sintang_points_scores.csv` | Permutation importance, and the test AUC with other forest settings | Table IV; Sections III-C and IV-B |
| `near15_test.py` | `near15_points_scores.csv` | AUCs with non-flood points placed only within 15 km of a reported flood | Section IV-C |

### What you need

The Python that comes with ArcGIS Pro already has everything the scripts use (numpy, pandas and scikit-learn). With another Python 3 installation, first run:

```
pip install numpy pandas scikit-learn
```

### How to run

1. Download the `Python` folder (or the whole repository) to your computer, for example to `C:\GIS\FinalProject\Python`.
2. Open **Python Command Prompt**: Windows Start menu → **ArcGIS** → **Python Command Prompt**.
3. Go to the folder:
   ```
   cd C:\GIS\FinalProject\Python
   ```
4. Run a script by typing `python`, the script name and its input file, as in the examples below.

**Paired DeLong test**

```
python delong_test.py testing_values.csv
```

Expected output:

```
Using columns: label = LABEL, RF = RandomFore, AHP = AHP_WS
Points used: 85 (42 flood, 43 non-flood)
RF  AUC 0.931  95% CI 0.88-0.98
AHP AUC 0.802  95% CI 0.71-0.90
Correlation between the two AUC estimates: 0.64
Difference 0.129  95% CI 0.057 to 0.201
Paired DeLong test: z = 3.49, p = 0.0005
```

**AHP AUC on all sample points**

```
python ahp_auc_all_points.py all_points.csv
```

Expected output:

```
All sample points: 281 points (141 flood, 140 non-flood), AHP AUC = 0.822
Block cross-validation points: 277 points (139 flood, 138 non-flood), AHP AUC = 0.828
```

**Permutation importance**

```
python rf_permutation_importance.py sintang_points_scores.csv
```

Expected output (shortened):

```
Training points 198, testing points 86
Test AUC 0.928, out-of-bag error 19.2%

Factor    AUC drop  Share (%)
DEM         0.2305       81.1
RAIN        0.0250        8.8
DIST        0.0170        6.0
...
Test AUC with other settings:
  chosen settings          0.928
  500 trees                0.926
  4 predictors per split   0.922
  minimum leaf size 5      0.945
```

The "Share (%)" column is the "RF permutation" column of Table IV. The out-of-bag error printed here belongs to the Python forest. The paper reports the out-of-bag error of the ArcGIS Pro forest (18%), taken from the tool's messages.

**Test with non-flood points within 15 km of a reported flood**

```
python near15_test.py near15_points_scores.csv
```

Expected output:

```
Training points 198, testing points 86
Random forest AUC 0.804
AHP AUC           0.755
```

### Input files

- `testing_values.csv` and `all_points.csv` come straight from ArcGIS Pro:
  1. **Extract Multi Values to Points** on the testing points (or on all sample points), with `results/FinalAhp.tif` saved as field `AHP_WS` and `results/RandomForest.tif` saved as field `RandomFore`, and **Bilinear interpolation** left off.
  2. **Export Table** to a `.csv` file in a normal folder (not inside a `.gdb`).
- `sintang_points_scores.csv` and `near15_points_scores.csv` hold the factor scores (1–9) of each sample point, its label (`LABEL` 1 = flood, 0 = non-flood) and whether it is a training or a testing point (`SET`). Their rows are in the order the model used them, so the scripts give exactly the numbers above.

### Notes

- The scripts read files separated by `,` or `;` and decimals written with `.` or `,`, as ArcGIS Pro exports them in different languages.
- A value of -9999 means NoData, for example the testing point on a river cell. Such points are skipped.
- To use other column names, edit the line near the top of a script that lists them (for example `LABEL, RF, AHP = ...` in `delong_test.py`).
- The random forest results were produced with scikit-learn 1.9.1. Another version of scikit-learn can change them slightly, but not the order of the factors.
