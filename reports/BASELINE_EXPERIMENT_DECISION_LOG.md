# Historical Decision Log & Baseline Experiment Record
**Project:** Ames Housing Valuation & Market Segmentation Engine  
**Document Purpose:** Permanent archival of the Stage 1 Baseline Model Lineup, Residual Diagnostics, and the Architectural Decision Log transitioning to the Stage 2 (Arm's-Length Normal Sales) Champion Pipeline.

---

## 1. Executive Summary & Purpose of This Log
In accordance with professional Machine Learning lifecycle standards (Data-Centric AI and CRISP-DM), this document preserves the empirical baseline of our initial modeling phase before applying the production domain filter (`SaleCondition == 'Normal'`). This ensures complete transparency, reproducibility, and auditability for academic defense and final report documentation.

---

## 2. Stage 1: Full Population Baseline Architecture

### 2.1 Baseline Dataset Scope
* **Total Transactions:** 1,460 homes in raw dataset.
* **Initial Preprocessing Split:** 80/20 train/test random split.
  * **Training Set:** 1,166 homes (after removing Dean De Cock's 2 massive partial outliers: `GrLivArea > 4000` & `SalePrice < $300k`).
  * **Holdout Test Set:** 292 homes (strictly locked away, 0 data leakage).
* **Population Inclusions:** All transaction types (Normal, Partial/Unfinished, Family transfers, Abnormal/Foreclosure, Allocations).

### 2.2 Stage 1 Master Leaderboard (Evaluated on 292 Holdout Test Homes)

| Model Name | Feature Input Space | Train $R^2$ | Test $R^2$ | $R^2$ Gap | Train MAE | Test MAE | Test RMSE | Test MdAPE | Test $\le 10\%$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5-Model Stacking Ensemble** | **Blended (Multi-Space)** | **0.9824** | **0.9140** | **0.0683** | **$6,217** | **$14,419.35** | **$25,679.57** | **5.18%** | **74.7%** |
| **Tuned LightGBM** | Selected Full (150) | 0.9875 | 0.9165 | 0.0710 | $4,850 | $15,629.12 | $25,301.39 | 6.03% | 71.6% |
| **Tuned CatBoost** | Clean Text (Categorical) | 0.9790 | 0.9057 | 0.0733 | $6,120 | $15,247.18 | $26,895.77 | 5.64% | 74.3% |
| **Tuned Random Forest** | Selected Full (150) | 0.9843 | 0.8883 | 0.0960 | $5,430 | $16,822.89 | $29,272.21 | 5.85% | 70.5% |
| **Ridge Regression** | Full Scaled (214) | 0.9088 | 0.8588 | 0.0500 | $14,210 | $20,090.23 | $32,912.81 | 8.34% | 57.5% |
| **Tuned SVR** | Strict (68) | 0.8939 | 0.8125 | 0.0814 | $9,720 | $18,829.91 | $37,926.21 | 6.42% | 69.5% |
| **OLS Baseline** | Strict (68) | 0.9042 | 0.8542 | 0.0500 | $14,890 | $20,727.00 | $33,439.10 | 8.81% | 55.1% |

### 2.3 Stage 1 Optimal Voting Power (SLSQP Optimizer)
* **LightGBM (150 features):** 51.9%
* **CatBoost (Clean Text features):** 32.6%
* **SVR (68 features):** 8.7%
* **Ridge Regression (214 features):** 6.7%
* **Random Forest (150 features):** 0.0%

---

## 3. The Diagnostic Turning Point (Residual Deep-Dive)

### 3.1 Top 10 Largest Residual Errors Audit
An audit of the top 10 worst dollar prediction errors on the holdout test set revealed an overwhelming systematic pattern:

| Rank | Actual Price | Predicted Price | Absolute Error | Pct Error | SaleCondition | Root Cause Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | $755,000 | $539,554 | $215,446 | 28.5% | `Normal` | Most expensive outlier mansion in dataset (Id 692). |
| **2** | $556,581 | $433,280 | $123,301 | 22.2% | `Partial` | Unfinished construction phase contract. |
| **3** | $611,657 | $492,125 | $119,532 | 19.5% | `Partial` | Unfinished construction phase contract. |
| **4** | $395,000 | $295,875 | $99,125 | 25.1% | `Partial` | Unfinished construction phase contract. |
| **5** | $253,293 | $352,415 | $99,122 | 39.1% | `Partial` | Unfinished construction phase contract. |
| **6** | $311,500 | $222,832 | $88,668 | 28.5% | `Normal` | Unique architectural custom floorplan. |
| **7** | $143,000 | $224,705 | $81,705 | 57.1% | `Partial` | Incomplete build with recorded structural specs. |
| **8** | $276,000 | $355,468 | $79,468 | 28.8% | `Partial` | Unfinished construction phase contract. |
| **9** | $301,000 | $245,224 | $55,776 | 18.5% | `Normal` | Standard home with high spatial land variance. |
| **10** | $135,000 | $188,001 | $53,001 | 39.3% | `Family` | Subsidized inter-family transfer below market value. |

**Key Empirical Finding:** 6 of the top 10 errors were `Partial` construction sales, and 1 was a subsidized `Family` deal. Over 70% of the catastrophic prediction errors stemmed from non-arm's-length contractual mechanisms.

### 3.2 Performance Breakdown by Sale Condition (Stage 1 Models)

| Transaction Segment | Sample Size | Test $R^2$ | Test MAE | Test RMSE | Within 10% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Arm's-Length Market Sales** | **234 homes (80.1%)** | **0.9197** | **$11,889.66** | **$21,393.97** | **79.5%** |
| **Partial / Unfinished Construction Sales** | **27 homes (9.2%)** | **0.8315** | **$31,250.91** | **$50,005.67** | **40.7%** |
| **Abnormal / Distressed Sales** | **24 homes (8.2%)** | **0.8870** | **$18,450.20** | **$28,140.10** | **66.7%** |
| **All Test Transactions Combined** | **292 homes (100%)** | **0.9140** | **$14,419.35** | **$25,679.57** | **74.7%** |

---

## 4. Architectural Decision Log: The Transition to Stage 2

### 4.1 Decision Formulation
* **Decision:** Transition the production valuation engine from an unbounded macro-model to a **Scoped Arm's-Length Residential Valuation Pipeline** (`SaleCondition == 'Normal'`).
* **Why Rule 1 was chosen over Price Truncation (Rule 2):**
  * Filtering by `SalePrice <= $500,000` constitutes target truncation / selection on the dependent variable.
  * In contrast, `SaleCondition == 'Normal'` filters strictly on pre-recorded transaction metadata, preserving all high-value properties ($> $500k) and eliminating selection bias.
* **Domain Alignment:** Conforms directly to Fannie Mae (Form 1004) and Uniform Standards of Professional Appraisal Practice (USPAP) standards, which mandate that Automated Valuation Models (AVMs) must only be trained and evaluated on arm's-length market transactions.

## 5. Stage 2: Official Champion Leaderboard (Normal Sales Only)

Below is the verified, authoritative leaderboard from the retrained Normal Sales pipeline (`SaleCondition == 'Normal'`), sorted by Test MAE:

| Model Name | Train $R^2$ | Test $R^2$ | $R^2$ Gap | Train MAE | Test MAE | MAE Gap | Train RMSE | Test RMSE | RMSE Gap | Train MdAPE | Test MdAPE | Train MAPE | Test MAPE | Test $\le 10\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** | **0.9778** | **0.9263** | **0.0515** | **$6,177** | **$11,442** | **$5,265** | **$10,580** | **$17,468** | **$6,888** | **2.36%** | **4.77%** | **3.61%** | **6.81%** | **80.83%** |
| **CatBoost (Tuned)** | 0.9719 | 0.9260 | 0.0459 | $6,764 | $11,817 | $5,053 | $11,887 | $17,501 | $5,614 | 2.09% | 4.87% | 3.92% | 6.97% | 79.17% |
| **LightGBM (Tuned)** | 0.9883 | 0.9040 | 0.0843 | $5,406 | $12,728 | $7,322 | $7,675 | $19,933 | $12,258 | 2.51% | 4.94% | 3.46% | 7.61% | 76.25% |
| **Random Forest (Tuned)** | 0.9847 | 0.8896 | 0.0951 | $5,425 | $13,572 | $8,147 | $8,769 | $21,376 | $12,607 | 2.19% | 5.12% | 3.29% | 8.29% | 72.92% |
| **Ridge (Tuned)** | 0.9194 | 0.8741 | 0.0454 | $13,346 | $15,904 | $2,558 | $20,139 | $22,830 | $2,691 | 6.10% | 7.29% | 7.96% | 9.82% | 67.08% |
| **SVR (Tuned)** | 0.8933 | 0.7410 | 0.1522 | $8,225 | $16,386 | $8,162 | $23,181 | $32,737 | $9,556 | 1.11% | 5.89% | 4.18% | 9.47% | 76.25% |
| **OLS (Baseline)** | 0.9085 | 0.8695 | 0.0391 | $14,354 | $16,710 | $2,356 | $21,460 | $23,244 | $1,784 | 6.84% | 7.82% | 8.89% | 10.45% | 62.92% |

---

## 6. Drafted Final Report Section 1 (Preserved Text)

### Section 1: Business Problem Framing & Task Formulation
*(Full text archived below for inclusion in the final thesis/capstone submission)*

#### 1.1 Executive Summary & Stakeholder Identification
The residential real estate market is fundamentally characterized by information asymmetry, pricing illiquidity, and reliance on manual appraisals. Traditional human appraisals require 2 to 3 weeks, cost between $500 and $800 per transaction, and suffer from cognitive heuristics and subjective bias. Inaccurate property valuations introduce severe financial exposure across three core stakeholders:
1. **Mortgage Lenders & Underwriters (Primary Stakeholder):** Overvaluing a property inflates the Loan-to-Value (LTV) ratio, exposing the financial institution to under-collateralized default risk during foreclosure. Conversely, undervaluing a property causes creditworthy loans to be rejected, resulting in capital allocation inefficiency.
2. **Residential Sellers & Homeowners:** Overpricing leads to extended "Days on Market" (DOM), requiring painful price cuts and stigmatizing listings. Underpricing results in irreversible capital loss for the homeowner.
3. **Institutional Real Estate Investors & Renovators:** Capital allocators require instantaneous, reproducible valuations and marginal ROI estimates for capital expenditure (e.g., assessing the exact value added by finishing a basement or upgrading a kitchen).

#### 1.2 Decision Need
Stakeholders require an automated, transparent, and mathematically grounded **Automated Valuation Model (AVM)** capable of:
* Replacing subjective appraisal opinions with an objective, empirical fair-market valuation.
* Generating instant point estimates alongside financial confidence intervals for risk underwriting.
* Disentangling standard single-family residential homes from structural and contractual anomalies (e.g., unfinished construction or agricultural parcels).

#### 1.3 The Dual-Lens Solution Architecture
* **Primary Lens: Supervised Predictive Regression:** Predict the continuous fair-market transaction price ($\text{SalePrice} \in \mathbb{R}^+$). Evaluation criteria: Out-of-sample $R^2$, MAE, RMSE, and the industry-standard Business Tolerance Metric ($\le 10\%$ error).
* **Secondary Lens: Unsupervised Market Segmentation & Anomaly Detection:** Cluster housing stock into distinct structural market tiers and isolate spatial outliers using DBSCAN (anomaly detection) and Hierarchical Clustering (actionable tiers: Starter, Suburban, Move-Up, Luxury).

#### 1.4 Unit of Analysis
The fundamental unit of analysis is an **individual, arms-length residential property transaction** in Ames, Iowa, represented by structural, architectural, quality, and geographical attributes at the moment of sale.

#### 1.5 Task & Output
* Point estimate: Predicted fair-market sale price $\hat{y} \in \mathbb{R}^+$.
* Interval estimate: 90% Underwriting Confidence Interval $[\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}]$.
* Structural classification: Market Tier assignment (Tiers 1 through 4).
* Marginal Attribution: Renovation ROI impact simulation.
