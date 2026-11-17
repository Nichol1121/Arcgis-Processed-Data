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
| `ahp_auc_all_points.py` | `all_points.csv` | Which sample points make up the 277 points, and the AHP AUC on them | Section IV-C; Fig. 7 |
| `rf_permutation_importance.py` | `all_points.csv` | Permutation importance, and the test AUC with other forest settings | Table IV; Sections III-C and IV-B |

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

**AHP AUC on the 277 points**

```
python ahp_auc_all_points.py all_points.csv
```

Expected output:

```
Sample points in the file: 284
Left out (7), with the layers that have no value:
  F121    flood      LULC
  F113    flood      LULC
  F18     flood      AHP map, LULC
  NF002   non-flood  LULC
  NF041   non-flood  AHP map, DEM, LULC, Slope_1
  NF065   non-flood  LULC
  NF105   non-flood  AHP map
Points used: 277 (139 flood, 138 non-flood), AHP AUC = 0.828
Points per fold: 1: 34, 2: 93, 3: 50, 4: 55, 5: 45
```

See [Block cross-validation and the 277 points](#block-cross-validation-and-the-277-points) for what these numbers mean.

**Permutation importance**

```
python rf_permutation_importance.py all_points.csv
```

Expected output:

```
Skipped 6 points with a NoData factor value
Training points 195, testing points 83
Test AUC 0.920, out-of-bag error 16.4%

Factor    AUC drop  Share (%)
DEM         0.1971       77.0
RAIN        0.0213        8.3
SOIL        0.0201        7.8
DIST        0.0137        5.4
DD          0.0037        1.4
FA         -0.0097        0.0
LULC       -0.0006        0.0
SLOPE      -0.0002        0.0

Test AUC with other settings:
  chosen settings          0.920
  500 trees                0.919
  4 predictors per split   0.919
  minimum leaf size 5      0.931
```

The script uses the factor scores that ArcGIS Pro extracted at the sample points, the same values the ArcGIS forest was trained on. Like ArcGIS Pro, it skips the 6 points that have a NoData factor value (see [the 277 points](#block-cross-validation-and-the-277-points)), so it trains on 195 points and tests on 83. The "Share (%)" column is the "RF permutation" column of Table IV. The out-of-bag error printed here belongs to the Python forest. The paper reports the out-of-bag error of the ArcGIS Pro forest (18%), taken from the tool's messages.

### Input files

- `testing_values.csv` and `all_points.csv` come straight from ArcGIS Pro:
  1. **Extract Multi Values to Points** on the testing points (or on all 284 sample points), with `results/FinalAhp.tif` saved as field `AHP_WS` and `results/RandomForest.tif` saved as field `RandomFore`, and **Bilinear interpolation** left off.
  2. **Export Table** to a `.csv` file in a normal folder (not inside a `.gdb`).
- `all_points.csv` also holds, for each of the 284 sample points, the eight factor scores (1–9) extracted from the factor rasters, its label (`LABEL` 1 = flood, 0 = non-flood), whether it is a training or a testing point (`SET_`), and its `GRID_ID` and `FOLD`.

### Block cross-validation and the 277 points

A random 70/30 split puts many testing points close to training points, which can make the forest look more accurate than it is. The spatial block cross-validation (Section III-E) keeps points from the same 20 km block together.

**How the 277 points are found**

The sample has 284 points (142 flood, 142 non-flood). Seven are left out because a layer has no value (NoData) there:

| Point | Layers with no value |
|---|---|
| F18 | AHP map, LULC |
| NF041 | AHP map, elevation, LULC, slope |
| NF105 | AHP map |
| F113, F121, NF002, NF065 | LULC |

Six of the seven have no LULC value, because permanent water is NoData in the LULC layer. The forest cannot be trained on, or predict, a point without all eight factor values (ArcGIS Pro skips such records). The AHP cannot be scored without a value in the AHP map. Both maps are therefore compared on the 277 points that have both: 139 flood and 138 non-flood.

`ahp_auc_all_points.py` lists the seven points and counts the rest.

**How the 277 points are used (ArcGIS Pro)**

1. **Generate Tessellation**: square cells of 400 km² (20 km × 20 km) over the district. Each cell gets a `GRID_ID`, for example `E-7`.
2. **Spatial Join**: join the grid to the sample points, so each point gets the `GRID_ID` of its cell.
3. Give each grid cell a fold number from 1 to 5 at random, in a field `FOLD` (**Add Field**, then **Calculate Field**). All points in the same cell share a fold, so a block never has points on both the training and the testing side. `all_points.csv` holds the `GRID_ID` and `FOLD` of every point.
4. Run **Forest-based and Boosted Classification and Regression** five times with the same settings as the main forest (Section III-C). In run *k*, train on the points with `FOLD <> k` and predict to the points with `FOLD = k` (Prediction Type: **Predict to features**).
5. **Merge** the five prediction outputs. Each point now has a flood probability from a forest that was not trained on any point in its block. The AUC of these probabilities on the 277 points is the forest's block cross-validation AUC: 0.891.
6. Score the AHP on the same 277 points with `ahp_auc_all_points.py`: 0.828. The AHP is not trained on any points, so the folds do not change its result.

The folds differ in size (34 to 93 points) because the points are not spread evenly. Fold 2 is the largest because it holds cell `E-7` around Sintang City.

### Notes

- The scripts read files separated by `,` or `;` and decimals written with `.` or `,`, as ArcGIS Pro exports them in different languages.
- A value of -9999 means NoData, for example the testing point on a river cell. Such points are skipped.
- To use other column names, edit the line near the top of a script that lists them (for example `LABEL, RF, AHP = ...` in `delong_test.py`).
- The random forest results were produced with scikit-learn 1.9.1. Another version of scikit-learn can change them slightly, but not the order of the factors.
