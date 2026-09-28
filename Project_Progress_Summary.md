# Ames Housing Project: Progress Summary

**Last Updated:** 2026-09-28

---

## 1. Project Framing & Pivot
* Successfully pivoted the project scope to the **Ames Housing Dataset (Dataset #8)**.
* Defined the core business problem: Subjective, inconsistent manual real estate appraisals leading to mispricing risks.
* Established a Dual-Lens solution architecture:
  * **Primary Lens:** Predictive Engine for `SalePrice` (Regression).
  * **Secondary Lens:** Market Segmentation to group properties into structural tiers (Clustering).

---

## 2. Environment Setup
* Created an isolated Python virtual environment (`.venv`).
* Cleaned the project directory, removed legacy project files, and strictly structured the folders (`data/raw/`, `notebooks/`, `src/`).
* Installed necessary regression and analysis libraries (`pandas`, `scikit-learn`, `xgboost`, `optuna`).

---

## 3. Data Ingestion & Diagnostic First-Pass
* Loaded the primary dataset (`train.csv`) containing 1,460 houses.
* Executed an initial ML diagnostic suite to extract:
  * Total rows, columns, and duplicate checks.
  * A Missing Data Severity Report (identifying columns like `Alley` with >90% missing data).
  * Separation of features by type (numerical vs. categorical).
  * Visual distribution of the target variable (`SalePrice`), identifying a right-skew (skewness = 1.74) that mathematically justifies a Log Transformation.

---

## 4. Preventing Data Leakage (The Vault)
* Dropped useless identifiers (the `Id` column) to prevent algorithm memorization.
* Separated features (`X`) from the target variable (`y`).
* Executed an 80/20 Train/Test Split → **X_train: 1,168 rows | X_test: 292 rows**.
* **Methodology Note:** A Random Split was utilized because `SalePrice` is a continuous variable, meaning there are no discrete classes to stratify.
* The Test set (`X_test`, `y_test`) is now strictly locked away to prevent data leakage during the deep EDA and Preprocessing phases.

---

## 5. ✅ Deep Exploratory Data Analysis (EDA) — COMPLETE
> Full findings documented in [`EDA_Report.md`](EDA_Report.md)

* **Phase 1 – Univariate Analysis:**
  * Confirmed `SalePrice` right-skewness (skew = **1.74**). Generated histograms for all 36 numerical features, identifying discrete vs. continuous features and "fake number" columns (`MSSubClass`, `MoSold`, `YrSold`).
* **Phase 2 – Missingness Mechanism Diagnostics:**
  * Audited 19 columns with missing data. Classified them as MNAR (structural absence — e.g., `PoolQC` at 99.49% missing = no pool) vs. MCAR (random omission — e.g., `LotFrontage` at 18.58%). Proved the "Garage Cluster" pattern (all 5 Garage columns missing at exactly 5.48%).
* **Phase 3 – Anomaly & Outlier Detection:**
  * IQR analysis across 17 continuous features. Identified `GrLivArea > 4000` as physically anomalous distressed-sale data points requiring surgical removal.
* **Phase 4 – Target Correlation Analysis:**
  * Ranked all 36 numerical features by Pearson correlation with `SalePrice`. Top predictors: `OverallQual` (0.786), `GrLivArea` (0.696), `GarageCars` (0.641). Near-zero predictors flagged for elimination.
* **Phase 5 – Multicollinearity Analysis:**
  * Feature-to-feature heatmap identified high-redundancy pairs (e.g., `GarageCars` ↔ `GarageArea` ~0.88).
* **Phase 6 – High-Cardinality & Sparse Category Audit:**
  * Custom scanner identified 63+ rare categories (<10 houses) across 25 columns and zero-variance columns (`Utilities`, `PoolQC`, `MiscFeature`).

---

## 6. ✅ Data Preprocessing Pipeline — COMPLETE
> Full pipeline documented in [`Preprocessing_Report.md`](Preprocessing_Report.md)

All steps executed in [`02_Data_Preprocessing.ipynb`](notebooks/02_Data_Preprocessing.ipynb) with strict data leakage prevention throughout.

| Step | Action | Result |
|---|---|---|
| 1 | Drop `GrLivArea > 4000` outlier rows | 1,168 → **1,166 rows** |
| 2 | Log-transform `SalePrice` (`np.log1p`) | Skewed target → approximately normal |
| 3 | MNAR categorical imputation (15 cols → `"None"`) | Structural absence correctly encoded |
| 4 | MNAR numerical imputation (10 cols → `0`) | Structural absence = zero measurement |
| 5 | `LotFrontage` — Neighborhood-grouped median imputation | Context-accurate fill; 0 remaining nulls |
| 6 | `Electrical` — Mode imputation (`'SBrkr'`) | 0 remaining nulls in entire dataset |
| 7 | Drop zero-variance columns (`Utilities`, `PoolQC`, `MiscFeature`) | Noise removed |
| 8 | Rare category grouping → `"Other"` (63 categories, 25 features) | Cross-validation crash risk eliminated |
| 9 | Feature engineering: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `GarageAge` | Domain knowledge injected |
| 9.5 | Data type correction: `MSSubClass`, `MoSold`, `YrSold` → `str` | Fake numbers protected |
| 10 | Yeo-Johnson skewness correction (18 features, threshold > 0.75) | All features approximately normal |
| 11 | Ordinal encoding (16 quality/condition rating columns) | Hierarchy preserved mathematically |
| 12 | One-Hot encoding (all remaining nominal text columns) | 79 → **214 features** |
| 13 | StandardScaler (fit on `X_train` only) | Mean=0, Std=1 across all features |
| 14 | Final validation + export to `data/processed/` | ✅ 0 nulls, shapes aligned |

**Final Dataset Dimensions:**
* `X_train_scaled`: **1,166 rows × 214 features**
* `X_test_scaled`: **292 rows × 214 features**

---

## 7. 🔜 Next Steps

- [ ] **Notebook 03:** Market Segmentation & Clustering (K-Means / DBSCAN on engineered features)
- [ ] **Notebook 04:** Feature Engineering & Selection (SHAP, RFE, Lasso-based elimination)
- [ ] **Notebook 05:** Pricing Engine Modeling (Ridge, Lasso, XGBoost, LightGBM — with Optuna HPO)

---

## Project File Index

| File | Description | Status |
|---|---|---|
| `notebooks/01_Exploratory_Data_Analysis.ipynb` | Full EDA notebook | ✅ Complete |
| `notebooks/02_Data_Preprocessing.ipynb` | Full preprocessing pipeline | ✅ Complete |
| `notebooks/03_Market_Segmentation_Clustering.ipynb` | Clustering notebook | 🔜 Next |
| `notebooks/04_Feature_Engineering_and_Selection.ipynb` | Feature selection | 🔜 Upcoming |
| `notebooks/05_Pricing_Engine_Modeling.ipynb` | Modeling & evaluation | 🔜 Upcoming |
| `EDA_Report.md` | Comprehensive EDA findings report | ✅ Generated |
| `Preprocessing_Report.md` | Comprehensive preprocessing pipeline report | ✅ Generated |
| `data/processed/X_train_scaled.csv` | Cleaned, encoded, scaled training features | ✅ Ready |
| `data/processed/X_test_scaled.csv` | Cleaned, encoded, scaled test features | ✅ Ready |
| `data/processed/y_train.csv` | Log-transformed training target | ✅ Ready |
| `data/processed/y_test.csv` | Log-transformed test target | ✅ Ready |
