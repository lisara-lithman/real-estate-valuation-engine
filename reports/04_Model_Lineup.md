# The Official Model Roster

This is the locked-in list of models we will train, evaluate, and compare.

### 1. Multiple Linear Regression (OLS)
*   **The Category:** The Fundamental Academic Baseline.
*   **Why it’s perfect here:** You must start with the textbook basics. OLS tests pure linear relationships (e.g., "$100 per square foot").
*   **Report Rationale:** We used OLS as our baseline to prove that simple linear models fail when a dataset has "multicollinearity" (overlapping features like `GarageArea` and `GarageCars`).

### 2. Ridge Regression
*   **The Category:** The Regularized Linear Fix.
*   **Why it’s perfect here:** It takes the basic OLS model and adds an $L_2$ mathematical penalty to shrink the coefficients.
*   **Report Rationale:** We implemented Ridge to specifically fix the multicollinearity problem discovered in our OLS baseline, allowing us to keep all structural features without the model breaking.

### 3. Support Vector Regressor (SVR with RBF Kernel)
*   **The Category:** The Geometric / Margin-Based Model.
*   **Why it’s perfect here:** Ames Housing has extreme outliers (massive $700,000+ mansions). SVR creates an "epsilon tube" ($\epsilon$-insensitive loss) that ignores small errors and is highly resistant to extreme luxury outliers.
*   **Report Rationale:** We selected SVR because its margin-based penalty handles the extreme right-skewed pricing outliers better than standard distance-based algorithms.

### 4. Random Forest Regressor
*   **The Category:** The Bagging Tree Ensemble.
*   **Why it’s perfect here:** It handles non-linear patterns (e.g., an extra bathroom is worth more in a luxury neighborhood than a budget one) by building hundreds of randomized decision trees and averaging them out.
*   **Report Rationale:** Random Forest was chosen to capture complex, non-linear architectural interactions while inherently resisting overfitting through variance reduction (bagging).

### 5. LightGBM (or XGBoost) Regressor
*   **The Category:** The Gradient Boosting Powerhouse.
*   **Why it’s perfect here:** This is the industry standard for tabular data like real estate. Instead of building trees independently like Random Forest, it builds them sequentially, with each new tree fixing the errors of the last one. It almost always delivers the lowest RMSE.
*   **Report Rationale:** We deployed LightGBM as our state-of-the-art benchmark to sequentially minimize residual pricing errors and maximize our $R^2$ score.

### 🏆 The "Overachiever" Bonus: Stacking Regressor
*   If time permits, we will stack the best predictions from the models above into an ultimate meta-model to achieve the lowest possible error.


---
## Feature Set Alignment (The Dataset Strategy)

> **The Overarching Philosophy:** *"I didn't force my algorithms to take a standardized test. I fed each algorithm the exact data structure that its underlying mathematics was built to digest."*

### Why do Feature Selection at all?
Noise actively confuses algorithms. If a dataset has columns that are completely random or highly overlapping (multicollinearity), algorithms might accidentally use them to make predictions, leading to massive overfitting. Feature selection deletes the noise so the model is forced to learn the true signal.

### Why use DIFFERENT datasets for DIFFERENT models?
In Applied Machine Learning, we don't deploy an algorithm; we deploy a **pipeline**. Because every algorithm has a completely different mathematical engine, we paired each model with the dataset that satisfied its mathematical assumptions to find its absolute peak performance.

### 1. Ordinary Least Squares (OLS Baseline)
* **Feature Set Used:** `X_strict` (68 Features) — *Selected via RFECV*
* **The 'Why':** OLS has no defense mechanism against multicollinearity. If fed overlapping features, its coefficients explode. We gave OLS the strictest, smallest dataset to ensure mathematical stability and establish a safe baseline.

### 2. Ridge Regression
* **Feature Set Used:** `X_scaled` (214 Features) — *The Full, Unfiltered Dataset*
* **The 'Why':** Ridge possesses an L2 Regularization penalty designed specifically to handle overlapping features by shrinking their impact. Because Ridge filters noise internally, manually deleting features actually robs it of tiny bits of useful signal. We fed Ridge all 214 features to maximize its potential.

### 3. Support Vector Regression (SVR)
* **Feature Set Used:** `X_strict` (68 Features) — *Selected via RFECV*
* **The 'Why':** SVR (RBF kernel) relies on the geometric distance between houses. It suffers severely from the "Curse of Dimensionality"—meaning too many columns distort the physical distance between data points. Even if features hold good signal, sheer volume ruins the math. We aggressively dropped the dataset to 68 features to clean up the geometric space, achieving massive accuracy gains over larger datasets.

### 4. Random Forest Regressor
* **Feature Set Used:** `X_full` (150 Features) — *Selected via RF Importance*
* **The 'Why':** Decision trees do not care about geometric distance or multicollinearity. However, if forced to select from subsets of pure noise (`max_features`), they will make suboptimal splits. We deleted the 64 features that were pure noise, leaving 150 features packed with strong, non-linear signals. This ensured the trees never received a "bad hand" of features when deciding how to split.
