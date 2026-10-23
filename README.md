# 🏠 Ames Housing: Dual-Lens Automated Valuation & Market Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20LightGBM%20%7C%20CatBoost-orange.svg)]()
[![UI](https://img.shields.io/badge/Interface-Gradio-red.svg)]()

---

## 🚨 1. The Business Problem

In residential real estate valuation, legacy appraisal processes rely on subjective, manual inspections. This introduces critical operational vulnerabilities:
1. **Appraisal Inconsistency:** Human appraisers introduce variance and cognitive bias.
2. **Pricing Latency & Mispricing Risk:** Inaccurate valuations lead to costly days-on-market delays or lost equity.
3. **Overcapitalization Blindspots:** Sellers frequently invest in renovations (e.g., \$15,000 luxury kitchen remodels) without knowing whether their neighborhood price ceiling will support a profitable return.

---

## 💡 2. The Solution: Dual-Lens Intelligence System

This project implements an end-to-end Machine Learning pipeline structured around two synergistic analytical lenses:

```
                                  AMES REAL ESTATE ENGINE
                                             │
                   ┌─────────────────────────┴─────────────────────────┐
                   ▼                                                   ▼
       PRIMARY LENS (Predictive)                          SECONDARY LENS (Diagnostic)
      5-Model Stacking Ensemble                           Market Segmentation Clustered
  (Ridge, SVR, RF, LGBM, CatBoost)                        (K-Means, DBSCAN, Hierarchical)
                   │                                                   │
                   ▼                                                   ▼
       $14,419 Test MAE | R² 0.9140                         3 Structural Market Tiers
      Automated Risk & Confidence Engine                    (Starter, Mid-Market, Luxury)
                   │                                                   │
                   └─────────────────────────┬─────────────────────────┘
                                             ▼
                                  WEB APPLICATION (app.py)
                           Tab 1: Valuation & Risk Engine
                           Tab 2: Custom Property Appraisal
                           Tab 3: Renovation ROI Advisor
```

* **Primary Lens (Predictive Valuation):** A mathematically diverse 5-model stacked ensemble that predicts property values with commercial-grade accuracy ($R^2 \approx 0.914$, Test MAE $\approx \$14,419$, $74.7\%$ of predictions within $10\%$ tolerance). Paired with a **Business Risk Engine** that audits inter-model variance to assign Green, Yellow, or Red automated listing tiers.
* **Secondary Lens (Market Segmentation):** Unsupervised clustering that groups homes into 3 structural tiers (**Starter Home**, **Mid-Market Suburban**, **Luxury Estate**) using physical characteristics (`TotalSF`, `OverallQual`, `YearBuilt`, `LotArea`, `GarageCars`), following an outlier isolation discovery narrative using DBSCAN and Hierarchical Ward linkage.
* **The Synergy (Renovation ROI Advisor):** Unites both lenses. Simulates a fixed \$15,000 renovation and calculates dynamic return: budget starter tiers reject the upgrade (Net Loss), while mid-market homes capture substantial equity boost (Net Profit).

---

## 📊 3. Master Model Leaderboard

Each algorithm was trained on its mathematically optimal feature representation to avoid multicollinearity, split dilution, or distance distortion:

| Rank | Model | Feature Dataset | Test $R^2$ | Test RMSE | Test MAE | Within 10% Error |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 🥇 | **5-Model Blended Ensemble (Meta-Model)** | **Optimal Multi-Lens Set** | **0.9140** | **$25,680** | **$14,419** | **74.7%** |
| 🥈 | **Tuned LightGBM** | `X_full` (150 Features) | 0.9165 | $25,301 | $15,629 | 71.6% |
| 🥉 | **Tuned CatBoost** | `X_clean_text` (76 Text Features) | 0.9057 | $26,896 | $15,247 | 74.3% |
| 4 | **Tuned Random Forest** | `X_full` (150 Features) | 0.8883 | $29,272 | $16,823 | 70.5% |
| 5 | **Tuned SVR (RBF Kernel)** | `X_strict` (68 Features) | 0.8125 | $37,926 | $18,830 | 69.5% |
| 6 | **Tuned Ridge Regression** | `X_scaled` (214 Features) | 0.8588 | $32,913 | $20,090 | 57.5% |
| 7 | **OLS Baseline** | `X_strict` (68 Features) | 0.8572 | $33,098 | $20,550 | 54.1% |

---

## 🗂️ 4. Project Pipeline & Notebook Directory

The pipeline is organized in a strict Directed Acyclic Graph (DAG) with zero data leakage:

```
notebooks/
├── 01_Exploratory_Data_Analysis.ipynb
│   └── 6-phase diagnostic suite: skewness discovery, missingness audit, outlier identification.
├── 02_Data_Preprocessing.ipynb
│   └── Leak-free grouped median imputation, Yeo-Johnson power transform, ordinal-safe encoding, standard scaling.
├── 03_Feature_Selection.ipynb
│   └── Multi-stage selection: Filter (VIF/Pearson), Embedded (Lasso/RF Importance), Wrapper (RFECV).
├── 04_Model_Training/
│   ├── 04a_Model_OLS_Baseline.ipynb (Benchmark linear baseline on X_strict)
│   ├── 04b_Model_Ridge.ipynb (L2 regularization & polynomial expansion on X_scaled)
│   ├── 04c_Model_SVR.ipynb (Kernel distance regression on X_strict)
│   ├── 04d_Model_RandomForest.ipynb (Ensemble bagging trees on X_full)
│   ├── 04e_Model_LightGBM.ipynb (Sequential gradient boosting on X_full)
│   ├── 04g_Model_CatBoost.ipynb (Categorical boosting on X_clean_text)
│   └── 04f_Model_Stacking.ipynb (Optimal SLSQP meta-model blending)
├── 05_Business_Risk_Engine.ipynb
│   └── Automated governance: variance audit, outlier veto rules, Green/Yellow/Red tiers.
└── 06_Market_Segmentation/
    ├── 06_1_Clustering_Data_Prep.ipynb (Physical feature extraction and scaling)
    ├── 06_2_KMeans_Deep_Dive.ipynb (K=3 vs K=4 analysis)
    ├── 06_3_DBSCAN_Deep_Dive.ipynb (Density outlier isolation; cleans 31 anomalies)
    └── 06_4_Hierarchical_Clustering.ipynb (Ward linkage dendrogram cut at Y=33 -> 3 market tiers)
```

---

## 🛡️ 5. Preprocessing & Data Integrity Safeguards

1. **The Vault Split:** 80/20 train-test split executed before transformations to strictly prohibit target and feature leakage.
2. **Leak-Free Imputation:** `LotFrontage` imputed strictly with neighborhood medians learned from `X_train` with global median fallback.
3. **Ordinal Isolation:** Ordinal quality and condition scales (`Po < Fa < TA < Gd < Ex`) are isolated from rare category grouping to preserve mathematical hierarchy without generating `NaN` values.
4. **Target Preservation:** Power transformations applied exclusively to physical numerical features; `SalePrice` modeled cleanly across pipelines.

---

## 🖥️ 6. Running the Application

### 1. Environment Activation
```bash
source .venv/bin/activate
```

### 2. Launch the Gradio Dashboard
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### Dashboard Features:
* **Tab 1: Valuation & Confidence Engine:** Select any test database house to view ensemble valuation, actual target, error delta, uncertainty metrics, and the automated risk tier.
* **Tab 2: Custom Property Appraisal:** Input custom structural features (Neighborhood, Quality, Living Area, Basement SF, Garage) to generate live pricing and risk classifications.
* **Tab 3: Renovation ROI Advisor:** Select a property to simulate a standard \$15,000 kitchen upgrade. The system applies the market tier classifier and prediction models to provide automated business recommendations:
  * **Starter Home:** `"Do not renovate — Net Loss"` (Market ceiling cap).
  * **Mid-Market / Luxury:** `"Proceed — Net Profit"` (Added equity exceeds cost).
