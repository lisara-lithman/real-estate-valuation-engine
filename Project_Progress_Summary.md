# Ames Housing Project: Progress Summary

**Last Updated:** 2026-10-03

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

## 7. ✅ Feature Selection & Dimensionality Reduction — COMPLETE
> Full rationale documented in [`Feature_Selection_Plan.md`](Feature_Selection_Plan.md)

All steps executed in [`03_Feature_Selection.ipynb`](notebooks/03_Feature_Selection.ipynb) to safely reduce the 214 preprocessed features into optimal, model-specific datasets.

* **Method 1 – Filter Methods:** Removed exact duplicate columns and used VIF (Variance Inflation Factor) to flag/remove severe multicollinearity.
* **Method 2 – Embedded Methods:** 
  * Used **LassoCV (L1 Regularization)** to force useless linear features to exactly zero.
  * Used **Random Forest Importance** to rank non-linear signals.
* **Method 3 – Wrapper Methods:**
  * Executed **RFECV** (Recursive Feature Elimination with Cross-Validation) to mathematically prove the optimal number of features (~100).
* **Final Model-to-Feature Mapping (The Rationale):**
  * `X_strict` (~60-80 features) → **OLS:** Cleaned of all multicollinearity and noise, as OLS is mathematically fragile.
  * `X_mid` (~100 features) → **Ridge & SVR:** Balances L2 regularization strengths while removing pure noise to avoid the curse of dimensionality.
  * `X_full` (~150 features) → **Random Forest, LightGBM, K-Means:** Tree models are immune to collinearity and need the maximum amount of signal to find non-linear interactions. Bottom 60 garbage features were pruned to prevent "split dilution" and micro-overfitting.

**Final Datasets saved to:** `data/final_for_modeling/`

---

## 8. 🔜 Next Steps

- [ ] **Notebook 04:** Pricing Engine Modeling (OLS, Ridge, SVR, RF, LightGBM — with Optuna HPO)
- [ ] **Notebook 05:** Market Segmentation & Clustering (Secondary Lens)
- [ ] **Final UI Feature Selection:** Extract SHAP values to select the top 10-15 human-readable features for the seller UI.

---

## Project File Index

| File | Description | Status |
|---|---|---|
| `notebooks/01_Exploratory_Data_Analysis.ipynb` | Full EDA notebook | ✅ Complete |
| `notebooks/02_Data_Preprocessing.ipynb` | Full preprocessing pipeline | ✅ Complete |
| `notebooks/03_Feature_Selection.ipynb` | Advanced Feature Selection pipeline | ✅ Complete |
| `notebooks/04_Pricing_Engine_Modeling.ipynb` | Regression Modeling & evaluation | 🔜 Next |
| `notebooks/05_Market_Segmentation_Clustering.ipynb` | Clustering notebook | 🔜 Upcoming |
| `EDA_Report.md` | Comprehensive EDA findings report | ✅ Generated |
| `Preprocessing_Report.md` | Comprehensive preprocessing pipeline report | ✅ Generated |
| `Feature_Selection_Plan.md` | Rationale & Model Mapping logic | ✅ Generated |
| `data/final_for_modeling/X_train_*.csv` | Cleaned, selected training features (strict/mid/full) | ✅ Ready |
| `data/final_for_modeling/X_test_*.csv` | Cleaned, selected test features (strict/mid/full) | ✅ Ready |
---

## Phase 4: Modeling (Notebook 04a - OLS Baseline)
**Finalized:** 2026-10-03 | **Status:** Completed

### 1. The OLS Baseline (`X_strict`)
We trained a standard Ordinary Least Squares (OLS) model on the `X_strict` dataset (68 features filtered via Lasso and VIF). 
* **Metrics:** Train R²: 0.904 | Test R²: 0.854 | Test MAE: ~$20,727 (11.6% Error).
* **The Good (Interpretability):** The coefficients were structurally sound and business-ready. The model learned that an increase in `OverallQual` adds ~$16,400, and a larger `BsmtFinSF1` adds ~$22,500.
* **The Bad (Accuracy Ceiling):** A $20k average error is too high for a production valuation engine. Furthermore, a massive gap between MAE ($20k) and RMSE ($33k) proved that OLS (which can only draw straight lines) is making highly expensive mistakes on luxury outlier homes.

### 2. The Multicollinearity Experiment (`X_all`)
To prove the necessity of Feature Selection, we bypassed `X_strict` and fed OLS the raw, un-filtered 214-feature dataset. 
* **The Result:** The model suffered a mathematical meltdown. OLS attempted to balance overlapping features (like `TotalBath` vs `FullBath`) by assigning them wildly inflated, opposing coefficients (e.g., penalizing a house -$128,000 for its TotalBath score, while crediting +$95,000 for FullBath). 
* **The Verdict:** While predictive metrics (R²) seemed okay, the model became a "black box of garbage." Stakeholders cannot trust a model whose pricing logic is structurally broken.

### 3. Key Takeaway & Pivot
This notebook perfectly demonstrated the **Interpretability vs. Accuracy Trade-off**:
1. To keep OLS structurally sound (interpretable), we had to delete 146 features (`X_strict`). 
2. But deleting 146 features meant throwing away real market nuance, hitting a hard ceiling on predictive accuracy.
3. **Next Step:** We need an algorithm that can use the larger datasets (`X_mid` / `X_full`) to gain accuracy, but has a mathematical defense mechanism against the "Exploding Bathroom" multicollinearity problem. We pivot to **Ridge Regression (L2 Regularization)** in Notebook 04b.

* **Note on Dataset Sizes:** Updated `Feature_Selection_Plan.md` to reflect that `X_mid` (Lasso) actually contains 172 features, and `X_strict` (RFECV) contains 68 features, correcting a naming convention error from Phase 3.
