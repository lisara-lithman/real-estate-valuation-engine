# 🔧 Data Preprocessing Pipeline — Comprehensive Report
### Ames Housing Dataset | Machine Learning Project
**Generated:** 2026-09-28 | **Notebook:** `02_Data_Preprocessing.ipynb`

---

## Executive Summary

This report provides a complete, step-by-step account of every transformation applied to the Ames Housing dataset during the preprocessing phase. For each step, we document **what was done**, **why it was necessary**, **how it was implemented**, and **the measurable result**.

The pipeline was designed with one overriding constraint: **zero data leakage**. Every statistical measure (medians, modes, variance thresholds, skewness values, scaling parameters) was calculated exclusively from `X_train` and then applied to both `X_train` and `X_test`. The test set was treated as a sealed vault throughout.

### Pipeline at a Glance

| Step | Action | Why |
|---|---|---|
| 1 | Drop extreme outlier rows | Prevent broken pricing rules |
| 2 | Log-transform `SalePrice` | Normalize skewed target |
| 3 | MNAR categorical imputation | Structural absence ≠ missing data |
| 4 | MNAR numerical imputation | Numeric counterpart of absence |
| 5 | MCAR: Grouped median imputation | Neighborhood-accurate LotFrontage |
| 6 | MCAR: Mode imputation | Clean up random single omissions |
| 7 | Drop zero-variance columns | Remove noise |
| 8 | Group rare categories → "Other" | Prevent CV crashes |
| 9 | Feature engineering | Inject domain knowledge |
| 9.5 | Data type correction | Protect fake numbers |
| 10 | Yeo-Johnson skewness correction | Normalize skewed features |
| 11 | Ordinal encoding | Preserve quality hierarchy |
| 12 | One-Hot encoding | Convert nominal text to numbers |
| 13 | Feature scaling (StandardScaler) | Equalize feature magnitudes |
| 14 | Final validation & export | Verify pipeline integrity |

---

## Step 1: Drop Extreme Outlier Rows

### What Was Done
Rows where `GrLivArea > 4000` sq ft were permanently removed from `X_train` (and the corresponding rows from `y_train`).

```python
outlier_mask = X_train['GrLivArea'] > 4000
X_train = X_train[~outlier_mask]
y_train = y_train[~outlier_mask]
```

**Result:** 2 rows removed. `X_train` reduced from 1,168 → **1,166 rows**.

### Why This Step Was Necessary
The EDA Phase 3 box plot and IQR analysis identified a specific pattern: houses with extremely large above-grade living areas (>4,000 sq ft) that sold for disproportionately low prices. These are almost certainly **distressed sales** — foreclosures, estate auctions, or institutional transactions — that do not represent normal market dynamics.

**The risk of keeping them:** A regression model learns by finding patterns. If it sees a 4,500 sq ft house that sold for \$150,000 (foreclosure price), it encodes a false rule: *"Large houses are cheap."* This would cause systematic undervaluation of all normal large homes in the test set. The impact is not limited to rare cases — it corrupts the model's understanding of the `GrLivArea` feature for all houses.

### Why NOT All IQR Outliers Were Dropped
The IQR analysis flagged 23 outliers in `GrLivArea`. We dropped only the **physically anomalous** ones (>4,000 sq ft with misaligned prices). The remaining outliers represent legitimate luxury properties — 3,500 sq ft mansions that sold for appropriately high prices. Dropping those would **remove valid market data** and prevent the model from learning the luxury segment. Surgical precision is better than blunt removal.

> [!IMPORTANT]
> This step was applied **only to `X_train`**. `X_test` was never modified by row deletions. Deleting rows from the test set would constitute evaluation fraud.

---

## Step 2: Target Variable Transformation — Log Transform of `SalePrice`

### What Was Done
```python
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)
```
All models were trained against `y_train_log`. At evaluation time, predictions were reversed: `np.expm1(predicted_log_price)` to recover real dollar values.

**`np.log1p(x)` vs `np.log(x)`:** `log1p` is numerically safe because it handles `x = 0` without producing `-∞`. For `SalePrice`, all values are positive, but this is best practice.

### Why This Step Was Necessary
The EDA confirmed `SalePrice` has a skewness of **1.74**. On the raw dollar scale:

| Problem | Raw Scale | Log Scale |
|---|---|---|
| Error penalty distribution | Errors on \$700K homes penalized 4× more | All errors treated proportionally |
| Distribution shape | Right-skewed long tail | Approximately normal bell curve |
| Model focus | Overfit to luxury outliers | Balanced across all price ranges |
| Residual behavior | Heteroscedastic (errors grow with price) | Homoscedastic (stable errors) |

**The mathematical justification:** MSE = mean of (prediction - actual)². On the raw scale, a \$100K error at \$700K = 10,000,000,000 penalty. The same \$100K error at \$200K = 10,000,000,000 penalty too — but on log scale, a proportional error at \$700K (say 14.3%) = same penalty as a proportional error at \$200K (also 14.3%). The model learns in percentage terms rather than absolute dollar terms, which is economically meaningful.

---

## Step 3: MNAR Missing Value Strategy — Structural Imputation

### 3.1 Categorical MNAR: Fill with `"None"`

**What Was Done:** 15 categorical columns where `NaN` represents the structural absence of a physical feature were filled with the string `"None"` in both `X_train` and `X_test`.

```python
mnar_categorical = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'MasVnrType'
]
for col in mnar_categorical:
    X_train[col] = X_train[col].fillna('None')
    X_test[col]  = X_test[col].fillna('None')
```

**Why `"None"` (a string) instead of `0` or `np.nan`?**
- These columns are **categorical text columns** (e.g., `PoolQC` contains `"Ex"`, `"Gd"`, `"TA"`, `"Fa"`, `"NA"`)
- `"None"` creates a **new, meaningful category** that the model can learn from: *"This house has no pool"* vs *"This house has an excellent pool"*
- Keeping `NaN` would cause encoding steps to crash or silently exclude these rows
- `0` would be incompatible with text columns

### 3.2 Numerical MNAR: Fill with `0`

**What Was Done:** 10 numerical columns (the numerical counterparts of MNAR categorical columns) were filled with `0`:

```python
mnar_numerical = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]
for col in mnar_numerical:
    X_train[col] = X_train[col].fillna(0)
    X_test[col]  = X_test[col].fillna(0)
```

**Result:** 0 remaining nulls in all MNAR numerical columns.

**Why `0` instead of median?**
- `GarageArea = 0` is mathematically correct — the house has zero square feet of garage space
- `GarageCars = 0` is correct — the house can park zero cars
- `TotalBsmtSF = 0` is correct — the house has zero basement square footage

**The `GarageYrBlt = 0` caveat:** Filling a calendar year with `0` creates a logical problem (year 0 AD). This temporary placeholder was explicitly noted and resolved in **Step 9 (Feature Engineering)** where `GarageYrBlt` is converted into `GarageAge` and then dropped.

---

## Step 4: MCAR Missing Value Strategy — Smart Imputation

### 4.1 `LotFrontage`: Neighborhood-Grouped Median Imputation

**What Was Done:**
```python
# Learn median per Neighborhood ONLY from X_train
neighborhood_medians = X_train.groupby('Neighborhood')['LotFrontage'].median()

# Apply to X_train
X_train['LotFrontage'] = X_train.apply(
    lambda row: neighborhood_medians[row['Neighborhood']]
    if pd.isnull(row['LotFrontage']) else row['LotFrontage'], axis=1
)

# Apply the SAME rules to X_test (no recalculation)
X_test['LotFrontage'] = X_test.apply(
    lambda row: neighborhood_medians[row['Neighborhood']]
    if pd.isnull(row['LotFrontage']) else row['LotFrontage'], axis=1
)
```

**Result:** `LotFrontage` remaining nulls — Train: **0**, Test: **0**.

**Why not a simple global median?**

`LotFrontage` was missing for 217 training samples (18.58%). The distribution is right-skewed (Mean: 70.03 ft, Median: 70.00 ft), making median safer than mean. However, a global median of 70 ft would be inaccurate because:

| Neighborhood Type | Typical Lot Frontage |
|---|---|
| Dense urban (e.g., IDOTRR) | ~20–40 ft |
| Standard residential (e.g., CollgCr) | ~65–75 ft |
| Suburban/estate (e.g., NoRidge) | ~90–120 ft |

Imputing a dense urban house's missing `LotFrontage` with the global 70 ft median would artificially inflate its lot — teaching the model that this neighborhood has larger lots than it really does. By imputing within each neighborhood, we provide a **contextually accurate and statistically justified** fill value.

**Leakage Prevention:** The neighborhood medians were computed from `X_train` only. When applied to `X_test`, we used the training-derived lookup table — not recalculated from test data.

### 4.2 `Electrical`: Mode Imputation

**What Was Done:**
```python
electrical_mode = X_train['Electrical'].mode()[0]  # Returns: 'SBrkr'
X_train['Electrical'] = X_train['Electrical'].fillna(electrical_mode)
X_test['Electrical']  = X_test['Electrical'].fillna(electrical_mode)
```

**Result:** Mode = `'SBrkr'` (Standard Circuit Breakers). 1 missing value resolved.

**Why mode?** With only 1 missing value in a categorical column, any sophisticated imputation is overkill. The Mode (most frequent value) is a statistically sound, simple, and unbiased estimator for a single random omission.

### 4.3 Validation Checkpoint

```
=== FINAL MISSING VALUE CHECK ===
Total missing values in X_train: 0
Total missing values in X_test:  0

SUCCESS! 100% of missing values successfully managed.
```

---

## Step 5: Drop Zero-Variance & Redundant Columns

**Columns Dropped:**

| Column | Reason for Dropping |
|---|---|
| `Utilities` | 1,459 out of 1,460 houses = `"AllPub"`. Near-zero variance — teaches the model nothing, as almost all houses are identical. |
| `PoolQC` | After MNAR fill, 99%+ of houses = `"None"`. Effectively zero variance. |
| `MiscFeature` | After MNAR fill, 96%+ of houses = `"None"`. Effectively zero variance. |

**Why zero-variance columns are harmful:**
- They occupy column slots in the feature matrix (memory waste)
- During One-Hot Encoding, they create dummy variables that are nearly all-zero (more memory, no signal)
- Some ML algorithms experience numerical instability with near-constant features
- Keeping them violates the principle of parsimony — simpler models with fewer noise features generalize better

---

## Step 6: Rare Category Grouping — "The Other Bucket"

### What Was Done

A custom scanner identified all text categories appearing in **fewer than 10 training houses**. These rare categories were relabeled as `"Other"`:

```python
for col in categorical_features:
    counts = X_train[col].value_counts()
    rare_categories = counts[counts < 10].index.tolist()
    X_train[col] = X_train[col].replace(rare_categories, 'Other')
    X_test[col]  = X_test[col].replace(rare_categories, 'Other')
```

**Result:** **63 rare categories** across **25 features** were consolidated into `"Other"` buckets.

### Why This Step Is Critical

**Cross-Validation Crash Prevention:** During K-Fold CV, the training data is split into k folds. If a category like `RoofMatl = "Metal"` (1 house) lands only in a validation fold, the training folds have never seen it. When `pd.get_dummies()` (One-Hot Encoding) runs on the validation fold and encounters an unknown column, it raises a `ValueError` and crashes the entire CV loop.

**Statistical Reliability:** With 1 sample, a model cannot learn a reliable average price for that category — it will memorize the single sample (pure overfitting). The "Other" bucket, containing many rare properties combined, has sufficient volume for the model to learn a stable, generalizable average price.

**Test Set Robustness:** The same rare-to-Other mapping was applied to `X_test` using the rules derived from `X_train`. If the test set contains a category that was rare in training, it's relabeled to "Other" — ensuring the model always receives a known category at inference time.

---

## Step 7: Feature Engineering — Injecting Domain Knowledge

### 7.1 New Features Created

```python
def engineer_features(df):
    df = df.copy()
    df['TotalSF']   = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['HouseAge']  = df['YrSold'] - df['YearBuilt']
    df['RemodAge']  = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
    df.loc[df['GarageYrBlt'] == 0, 'GarageAge'] = 0  # Fix: no garage = age 0
    df = df.drop(columns=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'])
    return df

X_train = engineer_features(X_train)
X_test  = engineer_features(X_test)
```

**Result:** X_train shape: **1,166 × 76** (added 5 features, dropped 3 calendar years, dropped 3 zero-variance, net change: -1 vs raw).

### 7.2 Rationale for Each Engineered Feature

| Feature | Formula | Rationale |
|---|---|---|
| **`TotalSF`** | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Total livable square footage is the single most intuitive real estate value driver. A model cannot automatically discover this sum — it needs to be given it explicitly. Often becomes the #1 feature importance. |
| **`TotalBath`** | `FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath` | Half bathrooms are worth half as much as full bathrooms. This creates a single, weighted composite score that eliminates 4 separate correlated bathroom columns. |
| **`HouseAge`** | `YrSold - YearBuilt` | Buyers think in terms of "the house is 15 years old," not "it was built in 2007." Converting to age makes the feature directly interpretable by the model in the same terms buyers use. |
| **`RemodAge`** | `YrSold - YearRemodAdd` | Recent renovations increase perceived value. A house remodeled 2 years ago commands more than the same house remodeled 30 years ago. This converts the raw year to a renovation recency score. |
| **`GarageAge`** | `YrSold - GarageYrBlt` | Resolves the `GarageYrBlt = 0` (no garage) math bug — a house with no garage gets `GarageAge = 0`, which correctly codes "no garage age relationship." |

### 7.3 Data Type Correction ("Fake Numbers")

```python
fake_numbers = ['MSSubClass', 'MoSold', 'YrSold']
for col in fake_numbers:
    X_train[col] = X_train[col].astype(str)
    X_test[col]  = X_test[col].astype(str)
```

**Why:** `MSSubClass = 20` doesn't mean anything is twice as large as `MSSubClass = 10`. They are housing **type codes**, not measurements. Leaving them as integers causes the skewness scanner to transform them with Yeo-Johnson, and the scaler to normalize them — both of which would produce numerically meaningful-looking but semantically nonsensical values. Casting to string ensures they're processed by the categorical encoding pipeline instead.

Similarly, `MoSold` (months 1–12) and `YrSold` (years 2006–2010) are time labels, not quantities on a meaningful numerical scale.

---

## Step 8: Numerical Feature Skewness Correction (Yeo-Johnson)

### What Was Done

**Discovery Phase:**
```python
skewness = X_train.select_dtypes(include=['int64', 'float64']).apply(lambda x: x.skew())
skewed_features = skewness[abs(skewness) > 0.75].index.tolist()
# Result: 18 highly skewed features
```

**Transformation Phase:**
```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=False)
pt.fit(X_train[skewed_features])  # FIT ONLY on X_train
X_train[skewed_features] = pt.transform(X_train[skewed_features])
X_test[skewed_features]  = pt.transform(X_test[skewed_features])
```

**18 Features Transformed:**

| Feature | Why Skewed |
|---|---|
| `LotFrontage` | Most homes have standard frontage; a few have massive frontages |
| `LotArea` | Most are small lots; a few estate properties are vastly larger |
| `MasVnrArea` | Most homes have 0 sq ft veneer; a few have elaborate facades |
| `BsmtFinSF2` | Most homes have no secondary finished basement |
| `BsmtUnfSF` | Distribution concentrated near zero for some types |
| `1stFlrSF` | Right tail from large first floors |
| `2ndFlrSF` | Many single-story homes have 0; 2-story homes cluster differently |
| `LowQualFinSF` | Nearly all homes have 0; rare exceptions have very large values |
| `GrLivArea` | Luxury homes create a right tail |
| `BsmtHalfBath` | Most homes: 0; a few: 1 or 2 |
| `KitchenAbvGr` | Most: 1; multi-family homes: 2+ |
| `WoodDeckSF` | Most homes: 0; a few have large decks |
| `OpenPorchSF` | Most: 0 or small; a few very large |
| `EnclosedPorch` | Nearly all 0; rare homes have enclosed porches |
| `3SsnPorch` | Extremely rare; almost all 0 |
| `ScreenPorch` | Almost all 0 |
| `MiscVal` | Extremely rare miscellaneous value |
| `TotalSF` | Engineered feature inherits skewness from components |

### Why Yeo-Johnson Instead of Log Transform

| Method | Handles Zeros? | Handles Negatives? |
|---|---|---|
| `np.log(x)` | ❌ (produces -∞) | ❌ (undefined) |
| `np.log1p(x)` | ✅ | ❌ |
| **Yeo-Johnson** | ✅ | ✅ |

Many of our skewed features (like `2ndFlrSF`, `WoodDeckSF`) contain **zeros** (houses with no second floor, no deck). Applying `log` or `log1p` would map these to 0 or undefined values. Yeo-Johnson handles all cases cleanly.

**Why `standardize=False`:** We set `standardize=False` because StandardScaler (Step 13) handles normalization. Doing it twice would be redundant and could introduce numerical precision issues.

**Leakage Prevention:** `pt.fit()` was called only on `X_train`. The learned transformation parameters (λ values per feature) were stored in `pt`. When `pt.transform(X_test)` was called, the same λ values were applied — no information from `X_test` was ever used to derive transformation parameters.

---

## Step 9: Categorical Encoding — Converting Text to Numbers

### 9.1 Ordinal Encoding (Preserving Hierarchy)

**What Was Done:** 16 columns with inherent ordering were manually mapped to integers:

```python
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'GarageQual', 'GarageCond', 'FireplaceQu']

bsmt_exp_map   = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
bsmt_fin_map   = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
garage_fin_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
shape_map      = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
slope_map      = {'Sev': 1, 'Mod': 2, 'Gtl': 3}
func_map       = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
```

**Full Ordinal Encoding Mapping Table:**

| Column | Encoding |
|---|---|
| `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `GarageQual`, `GarageCond`, `FireplaceQu` | None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5 |
| `BsmtExposure` | None=0, No=1, Mn=2, Av=3, Gd=4 |
| `BsmtFinType1`, `BsmtFinType2` | None=0, Unf=1, LwQ=2, Rec=3, BLQ=4, ALQ=5, GLQ=6 |
| `GarageFinish` | None=0, Unf=1, RFn=2, Fin=3 |
| `LotShape` | IR3=1, IR2=2, IR1=3, Reg=4 |
| `LandSlope` | Sev=1, Mod=2, Gtl=3 |
| `Functional` | Sal=1, Sev=2, Maj2=3, Maj1=4, Mod=5, Min2=6, Min1=7, Typ=8 |

**Why Ordinal and not One-Hot for these?**
Consider `KitchenQual`. Using One-Hot Encoding would create 5 binary columns: `KitchenQual_Po`, `KitchenQual_Fa`, `KitchenQual_TA`, `KitchenQual_Gd`, `KitchenQual_Ex`. This destroys the mathematical relationship: the model cannot know that `Gd > TA` unless it discovers this empirically. Ordinal encoding (1, 2, 3, 4, 5) directly encodes this relationship — `5 > 4 > 3` is mathematically true in the same way `Ex > Gd > TA` is semantically true.

### 9.2 One-Hot Encoding (Nominal Categories)

**What Was Done:**
```python
X_train = pd.get_dummies(X_train)
X_test  = pd.get_dummies(X_test)

# Alignment: Ensure both DataFrames have identical column sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
```

**Result:** `X_train` shape after encoding: **1,166 × 214**

**Why the alignment step is critical:** After One-Hot Encoding, if a category existed in `X_train` but not in `X_test` (e.g., a neighborhood that appeared 8 times in train but 0 times in test), `X_test` would be missing that dummy column. The `.align()` call uses a left join (keeping all `X_train` columns) and fills any missing `X_test` columns with `0`. Without this, the model would crash when it receives a feature vector of a different length than what it was trained on.

**Why not use `drop='first'` to avoid the Dummy Variable Trap?**
The Dummy Variable Trap (perfect multicollinearity) is only a hard problem for pure Ordinary Least Squares. Ridge, Lasso, XGBoost, and LightGBM — the models planned for this project — handle it via regularization. Dropping the first category is therefore optional here. The alignment strategy of keeping all columns is safer for consistency between train and test.

---

## Step 10: Feature Scaling — StandardScaler

### What Was Done

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # Learn mean and std ONLY from X_train

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
```

**Mathematical Proof of Success (on `TotalSF` feature):**
| | Mean | Std Dev |
|---|---|---|
| **BEFORE Scaling** | 52.69 sq ft | 6.47 |
| **AFTER Scaling** | **-0.00** | **1.00** |

Every feature in `X_train_scaled` and `X_test_scaled` now has Mean ≈ 0 and Standard Deviation = 1.

### Why Feature Scaling Is Necessary

**The Scale Disparity Problem:** Without scaling:
- `TotalSF` ranges from ~0 to ~10,000+ (Yeo-Johnson transformed) 
- `OverallQual` ranges from 1 to 10
- `HasPool` is a binary 0 or 1

Linear models (Ridge, Lasso) minimize the cost function by adjusting feature weights (coefficients). If `TotalSF` operates on a scale 1,000× larger than `HasPool`, the gradient descent algorithm naively assumes `TotalSF` is 1,000× more important just because its numerical range is larger. The resulting coefficients are not interpretable and the model underweights small-scale features.

**Why StandardScaler over MinMaxScaler?**

| Scaler | Formula | Behavior |
|---|---|---|
| `StandardScaler` | `(x - mean) / std` | Preserves outlier information; not bounded |
| `MinMaxScaler` | `(x - min) / (max - min)` | Compresses to [0,1]; extreme outliers compress everything else |

After Yeo-Johnson transformation, our features still have residual variability. StandardScaler is more robust for post-transformation data because it doesn't set hard bounds — a single large outlier won't compress the entire feature range.

**Tree-Based Model Note:** Gradient-boosted trees (XGBoost, LightGBM) are **invariant to feature scaling** — they make decisions based on rank ordering, not absolute values. The scaling step is primarily for the linear baseline models (Ridge, Lasso) planned for benchmarking.

> [!CAUTION]
> `scaler.fit()` was called **only on `X_train`**. This is the most commonly violated rule in ML pipelines. If `fit()` were called on both datasets combined (or on `X_test` separately), the scaler would learn different means and standard deviations for each set — constituting direct data leakage. The test set's statistics (its mean, its variance) would influence the normalization of both datasets.

---

## Step 11: Final Validation Check & Export

### Validation Results

```python
assert X_train.isnull().sum().sum() == 0   # ✅ PASSED
assert X_test.isnull().sum().sum() == 0    # ✅ PASSED
assert X_train.shape[1] == X_test.shape[1] # ✅ PASSED
```

### Exported Files

```
data/processed/
├── X_train_scaled.csv   (1,166 rows × 214 features)
├── X_test_scaled.csv    (292 rows × 214 features)
├── y_train.csv          (1,166 log-transformed SalePrice values)
└── y_test.csv           (292 log-transformed SalePrice values)
```

---

## Pipeline Summary: Before vs. After

| Property | Raw Data | After Preprocessing |
|---|---|---|
| **X_train rows** | 1,168 | **1,166** (-2 outlier rows) |
| **X_train columns** | 79 | **214** (after encoding) |
| **Missing values** | 19 columns with NaNs | **Zero — 0 missing values** |
| **Target distribution** | Right-skewed (skew = 1.74) | Log-normal (approximately normal) |
| **Categorical columns** | 43 text columns | **0** — all converted to numbers |
| **Feature scale** | Wildly varied (1 to 200,000+) | All standardized (mean=0, std=1) |
| **Skewed features** | 18 features with skew > 0.75 | All corrected via Yeo-Johnson |
| **New features added** | — | TotalSF, TotalBath, HouseAge, RemodAge, GarageAge |

---

## Data Leakage Prevention: The Golden Rules Observed

Throughout the entire pipeline, the following rules were strictly enforced:

| Rule | Where Applied |
|---|---|
| Train/Test split occurs **first**, before any analysis | Step 0 (notebook setup) |
| All statistical measures (medians, modes, means) derived from **`X_train` only** | Steps 3, 4, 5, 6 |
| Neighborhood medians for `LotFrontage` calculated from `X_train`, applied to `X_test` | Step 4.1 |
| Electrical mode calculated from `X_train`, applied to `X_test` | Step 4.2 |
| Rare category thresholds identified in `X_train`, mapping applied to `X_test` | Step 6 |
| Yeo-Johnson `PowerTransformer.fit()` called only on `X_train` | Step 8 |
| `StandardScaler.fit()` called only on `X_train` | Step 10 |
| Row deletions (outlier removal) applied only to `X_train`, **never to `X_test`** | Step 1 |

---

## Next Steps

The preprocessed datasets (`X_train_scaled`, `X_test_scaled`, `y_train_log`, `y_test_log`) are now ready for:

```
Notebook 03: Market Segmentation Clustering
Notebook 04: Feature Engineering and Selection (SHAP, RFE, Lasso)
Notebook 05: Pricing Engine Modeling (Ridge, Lasso, XGBoost, LightGBM)
```

---

*Report generated from: [02_Data_Preprocessing.ipynb](file:///Users/lisara/Desktop/Machine_learning_Project/notebooks/02_Data_Preprocessing.ipynb) | [Preprocessing_Plan.md](file:///Users/lisara/Desktop/Machine_learning_Project/Preprocessing_Plan.md)*
