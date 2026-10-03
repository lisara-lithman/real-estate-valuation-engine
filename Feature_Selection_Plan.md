# Feature Selection Plan — Final
### Ames Housing Dataset | Machine Learning Project
**Finalized:** 2026-10-02 | **Implemented In:** `Notebook 04`

---

## Starting Point

| Dataset | Shape |
|---|---|
| `X_train_scaled` | 1,166 × 214 — all selection work runs on this |
| `X_test_scaled` | 292 × 214 — sealed until final evaluation |

> [!CAUTION]
> Every selection decision is derived from `X_train_scaled` only.
> The test set is evaluated **once** at the very end of Notebook 05.

---

## Dual-Lens Output Strategy

```
Feature Selection (Notebook 04)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  LENS 1 — Regression                    LENS 2 — Clustering
  Three model-specific sets              Reuse X_full directly
  (X_strict / X_mid / X_full)           (same features that define price
                                          also define market tiers)
```

---

## Method 1 — Filter Methods

Filter methods evaluate features **independently of any model**, using only statistical
properties of the data. They are fast and model-agnostic.

### 1.1 Remove Exact Duplicate Columns

```python
from itertools import combinations

duplicate_cols = []
for c1, c2 in combinations(X_train_scaled.columns, 2):
    if X_train_scaled[c1].equals(X_train_scaled[c2]):
        duplicate_cols.append(c2)

X_train_scaled.drop(columns=list(set(duplicate_cols)), inplace=True)
X_test_scaled.drop(columns=list(set(duplicate_cols)), inplace=True)
```

### 1.2 Correlation-Based Filtering

**Feature-to-target correlation** — rank features by their Pearson correlation with `log(SalePrice)`:

```python
correlations = X_train_scaled.corrwith(y_train).abs().sort_values(ascending=False)
```

Features with near-zero correlation on **both** Pearson and Spearman are flagged as weak candidates.

> [!WARNING]
> Correlation is used as a **diagnostic signal only** — not as an automatic deletion rule.
> A feature with weak linear correlation can still be highly useful to Random Forest or LightGBM
> through non-linear interactions.

**Feature-to-feature correlation** — flag highly redundant pairs (|r| ≥ 0.90):

```python
corr = X_train_scaled.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr_pairs = upper.stack()[upper.stack() >= 0.90].sort_values(ascending=False)
```

Known pairs from EDA:

| Pair | Correlation |
|---|---|
| `GarageArea` ↔ `GarageCars` | ~0.88 |
| `GrLivArea` ↔ `TotRmsAbvGrd` | ~0.83 |
| `TotalBsmtSF` ↔ `1stFlrSF` | ~0.80 |

### 1.3 VIF (Variance Inflation Factor)

Measures multicollinearity — how much each feature can be explained by the other features.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

num_cols = [col for col in X_train_scaled.columns
            if X_train_scaled[col].nunique() > 2]
X_num = X_train_scaled[num_cols]

vif_data = pd.DataFrame({
    'feature': num_cols,
    'VIF': [variance_inflation_factor(X_num.values, i)
            for i in range(X_num.shape[1])]
}).sort_values('VIF', ascending=False)
```

| VIF | Interpretation | Action |
|---|---|---|
| 1–4 | No multicollinearity | Keep |
| 5–10 | Moderate | Monitor |
| > 10 | Severe | Remove the less informative feature from the pair |

> [!IMPORTANT]
> Do not remove **all** high-VIF features just to please OLS. The project narrative requires:
> OLS breaks under multicollinearity → Ridge's L2 regularization stabilizes it.
> Only remove features that measure the exact same physical thing as another feature.

**Filter Method Output:** Flagged redundant pairs + VIF report → feeds `X_strict` construction.

---

## Method 2 — Embedded Methods

Embedded methods perform feature selection **during model training** — the model's own learning
process determines which features to keep or discard.

### 2.1 Lasso Regression (L1 Regularization)

Lasso applies an L1 penalty that mathematically forces the coefficients of useless features
to exactly **zero**, effectively removing them from the model.

```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 100),
    cv=5, max_iter=10_000, random_state=42
)
lasso_cv.fit(X_train_scaled, y_train)

lasso_coefs = pd.Series(lasso_cv.coef_, index=X_train_scaled.columns)
lasso_selected = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)
lasso_zeroed   = lasso_coefs[lasso_coefs == 0].index.tolist()

print(f"Features kept: {len(lasso_selected)}")
print(f"Features zeroed out: {len(lasso_zeroed)}")
```

The non-zero Lasso features form the **linear-model candidate set** (`X_strict` base).

### 2.2 Random Forest Feature Importance

Random Forest internally calculates how much each feature reduces prediction error across
all its trees — a built-in embedded selection signal.

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
rf.fit(X_train_scaled, y_train)

rf_importance = pd.DataFrame({
    'feature':    X_train_scaled.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

The top features by RF importance form the **tree-model candidate set** (`X_full` base).

**Embedded Method Output:** Lasso coefficient ranking + RF importance ranking.

---

## Method 3 — Wrapper Methods

Wrapper methods evaluate **subsets of features** by actually training a model on each subset
and measuring performance. More accurate than filter methods but computationally heavier.

### Recursive Feature Elimination (RFE)

RFE works by:
1. Training a model on all features
2. Removing the least important feature
3. Retraining on the remaining features
4. Repeating until the target number of features is reached

```python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge

# RFECV automatically finds the optimal number of features via cross-validation
rfecv = RFECV(
    estimator=Ridge(alpha=1.0),
    step=5,          # Remove 5 features per iteration
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
rfecv.fit(X_train_scaled, y_train)

rfe_selected = X_train_scaled.columns[rfecv.support_].tolist()
print(f"Optimal number of features: {rfecv.n_features_}")
```

**Why RFECV over plain RFE:** RFECV uses cross-validation to automatically find the optimal feature
count — no need to manually guess a number.

**Wrapper Method Output:** The CV-optimal feature subset → feeds `X_mid` construction.

---

## Final Step — Combine All Signals and Produce Output Datasets

Aggregate the signals from all three method categories:

| Feature | Filter Score | Lasso (Embedded) | RF Importance (Embedded) | RFE Selected (Wrapper) | Final Decision |
|---|---|---|---|---|---|
| `OverallQual` | High | Non-zero | Top 5 | ✅ | **Keep — all sets** |
| `TotalSF` | High | Non-zero | Top 5 | ✅ | **Keep — all sets** |
| `BsmtHalfBath` | Low | Zero | Bottom | ❌ | **Drop** |

Build three final feature sets:

| Dataset | Size | Built From | Used By |
|---|---|---|---|
| `X_strict` | ~60–80 | Lasso non-zero + VIF-cleaned | OLS |
| `X_mid` | ~100 | RFECV optimal set | Ridge, SVR-RBF |
| `X_full` | ~150 | Top RF importance features | Random Forest, LightGBM, **K-Means** |

```python
# Apply column selection to test set — no statistics from test set
X_test_strict = X_test_scaled[strict_cols]
X_test_mid    = X_test_scaled[mid_cols]
X_test_full   = X_test_scaled[full_cols]
```

---

## Summary

```
Filter Methods   →  Correlation analysis + VIF
                    (fast, model-independent diagnostic)
                              │
Embedded Methods →  Lasso (L1 zeroes weak features)
                    + Random Forest importance
                    (selection happens during training)
                              │
Wrapper Methods  →  RFECV (recursive elimination with CV)
                    (most accurate — evaluates actual subsets)
                              │
                              ▼
                 X_strict / X_mid / X_full
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
      OLS (strict)   Ridge + SVR (mid)   RF + LGBM + K-Means (full)
```

---

## Notebook 04 Outputs

| File | Description |
|---|---|
| `X_train_strict.csv` / `X_test_strict.csv` | ~60–80 features — OLS |
| `X_train_mid.csv` / `X_test_mid.csv` | ~100 features — Ridge, SVR |
| `X_train_full.csv` / `X_test_full.csv` | ~150 features — RF, LightGBM, K-Means |
| `vif_report.csv` | VIF scores for all numerical features |
| `lasso_coefficients.csv` | Lasso feature coefficients and zero/non-zero status |
| `rf_importance.csv` | Random Forest feature importance ranking |
| `rfecv_results.csv` | RFE CV scores per feature count |

---

*Plan designed for: [04_Feature_Engineering_and_Selection.ipynb](file:///Users/lisara/Desktop/Machine_learning_Project/notebooks/04_Feature_Engineering_and_Selection.ipynb)*
*Informed by: [EDA_Report.md](file:///Users/lisara/Desktop/Machine_learning_Project/EDA_Report.md) | [Preprocessing_Report.md](file:///Users/lisara/Desktop/Machine_learning_Project/Preprocessing_Report.md) | [Model_Lineup.md](file:///Users/lisara/Desktop/Machine_learning_Project/Model_Lineup.md)*

---

## Comprehensive Rationale: Model-to-Feature Mapping

We assigned specific feature sets to specific models based on a single core principle: **The mathematical vulnerabilities of each algorithm.** Different machine learning algorithms process data fundamentally differently. Some are incredibly fragile and break if you give them redundant data, while others are robust and thrive on having as much information as possible.

### 1. Ordinary Least Squares (OLS) → `X_strict`
* **How it was built:** Lasso to delete weak linear features + actively deleting features with high VIF (Variance Inflation Factor).
* **Mathematical Rationale:** OLS is highly sensitive to multicollinearity. If you give OLS two features that measure the exact same thing (like `GarageCars` and `GarageArea`), the underlying matrix mathematics break down, resulting in wildly unstable coefficients. OLS needs the absolute cleanest, most strictly independent feature set to survive.

### 2. Ridge Regression & SVR → `X_mid`
* **How it was built:** RFECV (Recursive Feature Elimination with Cross Validation) iteratively tested subsets of features to find the mathematical middle-ground with the lowest CV error.
* **Mathematical Rationale:**
  * **Ridge:** Uses an L2 penalty, which prevents the math from breaking when features are correlated. It doesn't need the severe VIF cleanup of OLS, but it still gets confused by pure noise.
  * **SVR (Support Vector Regressor):** Works by measuring geometric distances between houses in space. Too many features (dimensions) leads to the *Curse of Dimensionality*, making distances meaningless.
  * **Conclusion:** Both models need a carefully balanced, scientifically optimized set of features. Not too strict, not too broad.

### 3. Random Forest, LightGBM, & K-Means → `X_full`
* **How it was built:** Ranked features by Random Forest Importance and kept the top ~150, discarding the bottom noise.
* **Mathematical Rationale:**
  * Tree-based models are immune to multicollinearity (they just split on whichever correlated feature works best) and thrive on extra features to find hidden non-linear interactions.
  * **Why not use all 214 features?** 
    1. **The 'Dilution' Problem (Random Forest):** RF looks at a random subset of features at every split. If 30% of your features are pure noise, there is a high probability that at some splits, all randomly chosen features are garbage, forcing a bad split.
    2. **Micro-Overfitting (LightGBM):** Gradient boosting tries to correct tiny errors. If you leave pure noise in, it will use that noise to perfectly memorize an outlier in the training data.
    3. **Empirical Proof:** Our 5-Fold Cross Validation showed that performance degrades (or flatlines with higher complexity) when including the bottom 60+ garbage features. By the Principle of Parsimony, we choose the simpler, more stable 150-feature model.
