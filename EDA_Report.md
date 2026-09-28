# 📊 Exploratory Data Analysis — Comprehensive Findings Report
### Ames Housing Dataset | Machine Learning Project
**Generated:** 2026-09-28 | **Notebook:** `01_Exploratory_Data_Analysis.ipynb`

---

## Executive Summary

This report documents all findings from the Exploratory Data Analysis (EDA) phase conducted on the **Ames Housing Dataset**. The dataset contains residential property sale records from Ames, Iowa, with the primary objective of predicting `SalePrice` using a supervised regression model, and secondarily segmenting the market via clustering.

The EDA was conducted **exclusively on the training split** (1,168 observations, 79 features) to preserve the statistical virginity of the test set and prevent any form of data leakage. All insights derived from this phase directly informed the preprocessing pipeline decisions documented in the companion report.

---

## Dataset Overview

| Property | Value |
|---|---|
| **Raw Dataset Size** | 1,460 houses × 80 columns |
| **After Train/Test Split (80/20)** | |
| — Training Set (`X_train`) | 1,168 rows × 79 features |
| — Test Set (`X_test`) | 292 rows × 79 features |
| **Target Variable** | `SalePrice` (continuous, USD) |
| **Split Method** | Random (no stratification — continuous target) |
| **Data Leakage Prevention** | Test set locked immediately after split; all EDA calculations performed on `X_train` only |

> [!IMPORTANT]
> The `Id` column was dropped prior to analysis to prevent the model from memorizing row identifiers as a proxy for price. The 80/20 random split was chosen because `SalePrice` is a continuous variable — stratification is mathematically undefined for regression targets.

---

## Phase 1: Univariate Analysis — Distributions & Skewness

### 1.1 Target Variable: `SalePrice`

**Skewness Score: `1.74`**

The target variable is **severely right-skewed**, exhibiting a long tail toward high-value luxury properties. This is one of the most critical findings of the EDA, with direct consequences for model training.

| Metric | Value |
|---|---|
| Skewness | **1.74** (severe — threshold for concern is > 0.75) |
| Distribution Shape | Long right tail; majority of homes clustered in lower-to-mid price range |

**Why this matters for ML:**
- Regression models (particularly linear ones) optimize their loss function (MSE/RMSE) on the **raw dollar scale**
- A \$100,000 prediction error on a \$700,000 mansion generates **4× the penalty** of the same error on a \$175,000 home
- This causes the model to allocate disproportionate capacity learning to price luxury outliers at the expense of the far more common ordinary homes
- The model's overall error metric will be dominated by rare, expensive cases

**Preprocessing Decision:** Apply `np.log1p()` to `SalePrice`, compressing the luxury tail and converting the skewed distribution into an approximately normal bell curve. All models train on the log-transformed target. At inference time, predictions are reversed using `np.expm1()` back to real dollar values.

---

### 1.2 Numerical Feature Distributions

Analysis of all **36 numerical features** in `X_train` via a grid of histograms revealed two distinct categories of numerical data:

**Category A: Continuous Features (Severely Right-Skewed)**

These features exhibit the classic "spike near zero, long tail to the right" pattern typical of area and dollar measurements in real estate:

| Feature | Nature |
|---|---|
| `LotArea` | Lot size — majority of homes on small lots, with extreme outlier estates |
| `GrLivArea` | Above-ground living area — clear right skew |
| `BsmtFinSF2` | Secondary finished basement area — heavily zero-inflated |
| `EnclosedPorch`, `ScreenPorch`, `3SsnPorch` | Porch sizes — most homes have 0; a few have large ones |
| `MiscVal`, `PoolArea` | Extremely rare, near-zero for 99%+ of homes |

**Category B: Discrete / Count Features (Disguised as Numbers)**

These columns contain small integer values (0, 1, 2, 3) and represent **counts or ordinal ranks**, not continuous measurements. Applying mathematical transformations to them would be meaningless:

| Feature | Nature |
|---|---|
| `FullBath`, `HalfBath` | Bathroom counts — typically 1–3 |
| `Fireplaces` | Fireplace count — 0, 1, 2, or 3 |
| `BedroomAbvGr` | Bedroom count |
| `OverallQual`, `OverallCond` | Already encoded quality ratings (1–10) |
| `GarageCars` | Garage capacity (0–4 cars) |

**Category C: "Fake Numbers" (Categorical Masquerading as Numerical)**

A critical discovery was that several integer columns are **classification codes**, not mathematical quantities:

| Feature | True Nature |
|---|---|
| `MSSubClass` | Dwelling type code (e.g., `20` = 1-Story 1946+, `60` = 2-Story 1946+) |
| `MoSold` | Month of sale (1–12, but no mathematical relationship between months) |
| `YrSold` | Year of sale (2006–2010, more of a market epoch than a continuous number) |

> [!WARNING]
> Leaving `MSSubClass`, `MoSold`, and `YrSold` as integers would cause the automated skewness scanner to wrongly apply Yeo-Johnson transformation to them, producing nonsensical results. These must be cast to strings before the skewness step.

---

## Phase 2: Missingness Mechanism Diagnostics

A full audit of null values in `X_train` revealed **19 columns** with missing data. The key analytical insight was distinguishing between two fundamentally different types of missingness:

### 2.1 Missing Not At Random (MNAR) — Structural Absence

Columns with **>40% missing data** are not cases of a surveyor forgetting to record a measurement. They represent the **physical non-existence** of a feature in that house. Evidence: Garage-related columns (`GarageType`, `GarageYrBlt`, `GarageFinish`, `GarageQual`, `GarageCond`) all share **exactly the same 5.48% missing rate** — mathematically proving they go missing together because the house simply has no garage.

| Column | Missing % | Structural Meaning |
|---|---|---|
| `PoolQC` | **99.49%** | No Pool |
| `MiscFeature` | **96.06%** | No Miscellaneous Feature |
| `Alley` | **93.66%** | No Alley Access |
| `Fence` | **80.05%** | No Fence |
| `MasVnrType` | **58.48%** | No Masonry Veneer |
| `FireplaceQu` | **46.83%** | No Fireplace |
| `GarageType` | **5.48%** | No Garage |
| `GarageYrBlt` | **5.48%** | No Garage |
| `GarageFinish` | **5.48%** | No Garage |
| `GarageQual` | **5.48%** | No Garage |
| `GarageCond` | **5.48%** | No Garage |
| `BsmtFinType1` | **2.40%** | No Basement |
| `BsmtFinType2` | **2.40%** | No Basement |
| `BsmtExposure` | **2.40%** | No Basement |
| `BsmtCond` | **2.40%** | No Basement |
| `BsmtQual` | **2.40%** | No Basement |

**The "Garage Cluster" Pattern:** The identical 5.48% missing rate across all 5 Garage columns is the definitive mathematical proof of structural missingness. If `GarageType` is missing, all other garage columns are missing too — because the building simply has no garage.

**Critical ML Implication:** Imputing MNAR columns with median/mean values would be **mathematically incorrect and destructive**. If we filled `GarageArea` for a house without a garage with the median (~500 sq ft), we would fabricate a phantom garage in the model's reality, teaching it to falsely add hundreds of square feet of garage space to the property price calculation.

### 2.2 Missing Completely At Random (MCAR)

| Column | Missing % | Reason |
|---|---|---|
| `LotFrontage` | **18.58%** | Genuine surveyor omission |
| `MasVnrArea` | **0.51%** | Sporadic recording error |
| `Electrical` | **0.09%** | Single missing entry |

`LotFrontage` requires special treatment. Its distribution is **right-skewed** (Mean: 70.03 ft, Median: 70.00 ft), and lot sizes vary dramatically by neighborhood. A single global median would be inaccurate (a dense urban area has much smaller lot frontages than a suburban sprawl).

---

## Phase 3: Anomaly & Outlier Detection

### 3.1 Box Plot Visual Analysis

Visual box plots were generated for all 36 numerical features. The **whisker analysis** identified features where data points extend far beyond the expected range.

### 3.2 Automated IQR Outlier Detection

Using the **Interquartile Range (IQR) method** (Lower bound = Q1 - 1.5×IQR, Upper bound = Q3 + 1.5×IQR), outliers were detected across 17 continuous features:

| Feature | Outlier Count | Severity |
|---|---|---|
| `EnclosedPorch` | **160** | 🔴 Very High |
| `BsmtFinSF2` | **131** | 🔴 Very High |
| `ScreenPorch` | **97** | 🟠 High |
| `MasVnrArea` | **77** | 🟠 High |
| `LotFrontage` | **68** | 🟠 High |
| `OpenPorchSF` | **60** | 🟡 Medium-High |
| `LotArea` | **54** | 🟡 Medium-High |
| `TotalBsmtSF` | **47** | 🟡 Medium |
| `WoodDeckSF` | **30** | 🟡 Medium |
| `GrLivArea` | **23** | 🟡 Medium |
| `BsmtUnfSF` | **21** | 🟢 Lower |
| `GarageArea` | **18** | 🟢 Lower |
| `1stFlrSF` | **13** | 🟢 Lower |
| `YearBuilt` | **5** | 🟢 Low |
| `BsmtFinSF1` | **5** | 🟢 Low |
| `2ndFlrSF` | **1** | 🟢 Negligible |
| `GarageYrBlt` | **1** | 🟢 Negligible |

### 3.3 Critical Outlier: `GrLivArea`

The `GrLivArea` (Above-Grade Living Area) analysis exposed **physically anomalous data points** — houses with over 4,000 sq ft of above-ground living space that sold for inexplicably low prices. These are highly likely **foreclosures, distressed sales, or institutional transactions** that do not reflect standard market pricing dynamics.

> [!CAUTION]
> These points are not just statistical outliers — they represent **broken pricing rules**. If a 4,500 sq ft house sold for \$150,000 (due to foreclosure), a regression model trained on this data will learn: "large houses = low price." This is a catastrophically false rule that would corrupt predictions for all normal large homes.

**Decision:** Surgically drop only the specific rows where `GrLivArea > 4000`. This is a **manual, one-time intervention** applied only to `X_train`. We intentionally do **not** drop all IQR outliers, as many represent legitimate luxury properties whose natural high prices the model should learn to predict.

---

## Phase 4: Feature-to-Target Correlation Analysis

Pearson correlation coefficients were computed between all numerical features and `SalePrice` to identify the most predictive signals.

### 4.1 Strong Positive Correlators (The "Big Hitters")

| Feature | Correlation | Interpretation |
|---|---|---|
| `OverallQual` | **0.786** | Overall material & finish quality — strongest predictor |
| `GrLivArea` | **0.696** | Above-grade living area sq ft |
| `GarageCars` | **0.641** | Garage capacity (car count) |
| `GarageArea` | **0.624** | Garage size in sq ft |
| `TotalBsmtSF` | **0.598** | Total basement sq ft |
| `1stFlrSF` | **0.588** | First floor sq ft |
| `FullBath` | **0.553** | Number of full bathrooms |
| `TotRmsAbvGrd` | **0.520** | Total rooms above grade |
| `YearBuilt` | **0.517** | Construction year (newer = pricier) |
| `YearRemodAdd` | **0.509** | Remodel year |
| `GarageYrBlt` | **0.480** | Garage construction year |
| `MasVnrArea` | **0.459** | Masonry veneer area |
| `Fireplaces` | **0.458** | Number of fireplaces |

### 4.2 Moderate Positive Correlators

| Feature | Correlation |
|---|---|
| `BsmtFinSF1` | 0.359 |
| `LotFrontage` | 0.330 |
| `WoodDeckSF` | 0.330 |
| `2ndFlrSF` | 0.314 |
| `OpenPorchSF` | 0.300 |
| `HalfBath` | 0.280 |
| `LotArea` | 0.266 |
| `BsmtFullBath` | 0.226 |
| `BsmtUnfSF` | 0.222 |

### 4.3 Weak / Near-Zero Correlators (Noise Candidates)

| Feature | Correlation | Note |
|---|---|---|
| `BedroomAbvGr` | 0.156 | Weak — count doesn't drive price like quality does |
| `ScreenPorch` | 0.119 | Very weak |
| `PoolArea` | 0.116 | Very weak — pools are rare |
| `3SsnPorch` | 0.052 | Near-zero |
| `MoSold` | 0.042 | No seasonal price effect |
| `BsmtFinSF2` | -0.006 | Essentially zero |
| `YrSold` | -0.009 | No significant time trend |
| `LowQualFinSF` | -0.011 | Slightly negative |
| `MiscVal` | -0.020 | Negative — miscellaneous features don't add value |
| `BsmtHalfBath` | -0.048 | Slightly negative |
| `OverallCond` | **-0.074** | Counter-intuitive — older-style homes in good condition aren't pricier |
| `MSSubClass` | -0.088 | Encoding artifact — must treat as categorical |
| `KitchenAbvGr` | -0.143 | Negative — extra kitchens in multi-family homes lower per-unit price |
| `EnclosedPorch` | **-0.150** | Enclosed porches associated with older, lower-value homes |

> [!NOTE]
> The negative correlation of `OverallCond` (-0.074) vs. the strong positive of `OverallQual` (+0.786) is a subtle but important distinction. **Quality** (materials and craftsmanship) drives price strongly. **Condition** (current upkeep state) shows almost no independent effect — likely because `YearRemodAdd` already captures renovation recency. Features near zero (|r| < 0.05) are primary candidates for elimination during feature selection.

---

## Phase 5: Multicollinearity Analysis (Heatmap)

A feature-to-feature correlation heatmap was computed across all 36 numerical features to identify redundant variable pairs that could destabilize linear models and dilute feature importance in tree-based models.

### 5.1 Highly Collinear Feature Pairs (Redundancy Risk)

| Feature Pair | Correlation | Problem |
|---|---|---|
| `GarageCars` ↔ `GarageArea` | ~0.88 | Near-perfect twins — both measure garage size |
| `TotalBsmtSF` ↔ `1stFlrSF` | ~0.80 | First floor area ≈ basement area for single-story homes |
| `GrLivArea` ↔ `TotRmsAbvGrd` | ~0.83 | More rooms = more area — correlated by definition |
| `YearBuilt` ↔ `GarageYrBlt` | ~0.83 | Garage typically built with the house |

**Why Multicollinearity is Dangerous:**
1. **Linear Models:** When two features are highly correlated, their regression coefficients become numerically unstable. A tiny change in the data can flip their signs and magnitudes wildly.
2. **Tree-Based Models (XGBoost, LightGBM):** The algorithm arbitrarily splits importance between the two twins, understating each one's true importance and making feature importance rankings misleading.
3. **Solution:** During feature selection (Notebook 04), one feature from each correlated pair should be dropped or consolidated through feature engineering (e.g., combining `GarageCars` and `GarageArea` into a single engineered signal).

---

## Phase 6: High-Cardinality & Sparse Category Audit

A custom rare-category scanner was built to flag any text category appearing in **fewer than 10 houses** across all categorical columns.

### 6.1 Rare Category Report

| Feature | Rare Category | Count | Risk |
|---|---|---|---|
| `MSZoning` | `C (all)` | 4 | Cross-validation crash risk |
| `Street` | `Grvl` | 4 | Cross-validation crash risk |
| `LotShape` | `IR3` | 8 | Sparse learning |
| `Utilities` | `NoSeWa` | **1** | 🔴 Zero-variance risk |
| `LotConfig` | `FR3` | 3 | Cross-validation crash risk |
| `LandSlope` | `Sev` | 9 | Near-crash risk |
| `Neighborhood` | `Veenker` (9), `NPkVill` (7), `Blueste` (1) | 1–9 | Unreliable pricing estimates |
| `Condition1` | `PosA` (8), `RRNn` (5), `RRNe` (1) | 1–8 | Sparse |
| `Condition2` | 7 rare categories | 1–3 each | 🔴 Extremely sparse |
| `HouseStyle` | `2.5Fin` | 7 | Sparse |
| `RoofStyle` | `Gambrel` (9), `Mansard` (5), `Shed` (2) | 2–9 | Sparse |
| `RoofMatl` | `Tar&Grv` (9), `WdShngl` (4), `WdShake` (3), `Metal` (1), `ClyTile` (1) | 1–9 | 🔴 Extreme crash risk |

### 6.2 The Machine Learning Danger of Sparse Categories

**Crash Risk:** During K-Fold Cross-Validation, the dataset is split into training and validation folds. If a rare category (e.g., `RoofMatl = "Metal"`, appearing only once) ends up only in a validation fold — the model has never seen it during training. When One-Hot Encoding then asks the model to process an unknown category, it **crashes with a `ValueError`**.

**Statistical Unreliability:** If `RoofMatl = "Metal"` appears only once, and that one house happened to be a foreclosure, the model learns: *"Metal roofs → extremely cheap."* With only one data point, this is pure overfitting noise, not a statistical signal.

**Zero-Variance Column (`Utilities`):** The `Utilities` column contains 1,459 out of 1,460 houses with the value `"AllPub"`. This column provides essentially zero discriminative power — almost every house is identical on this dimension. Keeping it wastes model capacity and memory.

### 6.3 Key Structural Insight

The `Utilities` column's near-zero variance makes it the most extreme case — it must be dropped entirely. For all other rare categories, the "Other bucket" strategy consolidates them into a meaningful group with sufficient sample size for stable learning.

---

## Summary of EDA Findings & Preprocessing Directives

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINDING                     → PREPROCESSING ACTION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  SalePrice skew = 1.74       → Log-transform target variable               │
│  36 numerical features       → Detect and transform skewed ones (>0.75)    │
│  Fake number columns         → Cast MSSubClass, MoSold, YrSold to str      │
│  MNAR missingness (>40%)     → Fill with "None" / 0, not median            │
│  MCAR missingness (random)   → Neighborhood-median (LotFrontage) / Mode    │
│  GrLivArea > 4000 anomaly    → Drop rows from X_train only                 │
│  Near-zero correlators       → Flag for feature selection phase             │
│  GarageCars ↔ GarageArea     → Feature engineering to consolidate          │
│  Rare categories (<10 houses)→ Group into "Other" bucket                   │
│  Utilities (zero variance)   → Drop entire column                          │
│  Raw calendar years          → Engineer HouseAge, RemodAge, GarageAge      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways for Modeling

1. **`OverallQual` (r = 0.786)** is the single most powerful predictor of sale price. Any model that doesn't heavily utilize this feature is leaving significant predictive power on the table.

2. **Size matters more than count.** `GrLivArea` (r = 0.696) outperforms `BedroomAbvGr` (r = 0.156). Buyers care more about total space than room count.

3. **Newness drives price.** `YearBuilt` (r = 0.517) and `YearRemodAdd` (r = 0.509) are both strong predictors — newer and recently renovated homes command significant premiums. These should be engineered into `HouseAge` and `RemodAge` for cleaner interpretation.

4. **Garage is a major value driver.** Both `GarageCars` (r = 0.641) and `GarageArea` (r = 0.624) rank in the top 5 predictors — but their near-perfect collinearity means only one signal is needed, or they should be combined.

5. **Calendar-based features are nearly useless.** `YrSold` (r = -0.009) and `MoSold` (r = 0.042) show virtually no predictive power, confirming the market was largely stable across the 2006–2010 data collection window.

---

*Report generated from: [01_Exploratory_Data_Analysis.ipynb](file:///Users/lisara/Desktop/Machine_learning_Project/notebooks/01_Exploratory_Data_Analysis.ipynb)*
