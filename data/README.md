# Master Dataset Dictionary
As this project has evolved, we have generated multiple specific feature sets tailored to the mathematical requirements of different algorithms. This document serves as the master reference guide to track every dataset, what preprocessing was applied to it, and which models consume it.

---

## 1. The Raw Data (`data/raw/`)
* **`train.csv` & `test.csv`**
  * **Description:** The original Ames Housing dataset containing 80 features.
  * **State:** Completely raw. Contains missing values, unscaled numbers, and raw text categories.

---

## 2. The Processed Data (`data/processed/`)
This directory contains data that has passed through `02_Data_Preprocessing.ipynb`. Missing values have been imputed, and Yeo-Johnson transformations were applied to fix numerical skewness.

* **`X_train_clean_text.csv` / `X_test_clean_text.csv` (80 Features)**
  * **Preprocessing:** Imputed + Yeo-Johnson (No One-Hot Encoding, No Scaling).
  * **Description:** The dataset where categorical features were left as raw text (e.g., "Northridge").
  * **Used By:** `CatBoost` (Native target-encoding handles the text).

* **`X_train_scaled.csv` / `X_test_scaled.csv` (214 Features)**
  * **Preprocessing:** Imputed + Yeo-Johnson + One-Hot Encoded + Standard Scaled.
  * **Description:** The fully expanded dataset. One-Hot Encoding shattered the categorical columns into 130+ sparse 0/1 columns. 
  * **Used By:** `Ridge Regression` (L2 penalty safely regularizes the 214 features).

* **`y_train.csv` / `y_test.csv` (The Target Variable)**
  * **Preprocessing:** Raw Dollars.
  * **Description:** Contains the true `SalePrice`. *(Note: Due to a discovered bug in Notebook 02, these were saved as raw exponential dollars, not Log-Transformed percentages).*

---

## 3. The Feature-Selected Data (`data/final_for_modeling/`)
This directory contains subsets of the `scaled` (214) dataset. These subsets were generated in `03_Feature_Selection.ipynb` to remove noise for algorithms that cannot handle high dimensionality.

* **`X_train_strict.csv` / `X_test_strict.csv` (68 Features)**
  * **Selection Method:** Linear RFECV (Recursive Feature Elimination with Cross-Validation).
  * **Description:** The absolute strictest dataset. 146 noisy columns were deleted to prevent multicollinearity and distance-distortion.
  * **Used By:** `OLS` (Requires strict features to prevent crashing) and `SVR` (Requires low dimensions to accurately calculate geometric distance).

* **`X_train_full.csv` / `X_test_full.csv` (150 Features)**
  * **Selection Method:** Random Forest Feature Importance.
  * **Description:** 64 columns of pure noise were deleted, leaving 150 features containing strong linear and non-linear signals.
  * **Used By:** `Random Forest` and `LightGBM` (Requires the removal of pure noise to prevent trees from making bad random splits).

---

## Quick Reference: Model to Dataset Mapping
| Model | Consumes Dataset | Why? |
|-------|------------------|------|
| **OLS** | `strict` (68) | Prevents coefficient explosion (Multicollinearity). |
| **Ridge** | `scaled` (214) | L2 Regularization shrinks noise safely; needs all data. |
| **SVR** | `strict` (68) | Mitigates the "Curse of Dimensionality" in geometric space. |
| **Random Forest** | `full` (150) | Prevents bad splits by dropping pure 0/1 noise. |
| **LightGBM** | `full` (150) | Same as Random Forest. |
| **CatBoost** | `clean_text` (80) | Bypasses One-Hot Encoding to use native Target Encoding. |
| **Stacker** | *All of the above* | Feeds each dataset to its respective model for maximum diversity. |
