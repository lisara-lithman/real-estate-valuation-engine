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
