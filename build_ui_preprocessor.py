import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
import joblib

# 1. Load Data
train = pd.read_csv('data/raw/train.csv')
outlier_mask = (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
train = train[~outlier_mask]
X_train = train.drop(['Id', 'SalePrice'], axis=1, errors='ignore')

# 2. Skewed Features
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train_num = X_train[num_cols].fillna(0)
skewness = X_train_num.skew()
skewed_features = skewness[abs(skewness) > 0.75].index.tolist()

# 3. Fit PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=False)
pt.fit(X_train[skewed_features].fillna(0))

# 4. We need the exact scaler used for Ridge.
# Since we have `X_test_cb` (before OHE) and `X_test_scaled` (after OHE+Scale), 
# we can just use `X_train` from Notebook 02. Let's replicate Notebook 02 processing quickly.
# ... Actually, the easiest way to get the exact scaler is to look at `X_test_full` vs `X_test_scaled`.
# But wait, X_test_full has 150 cols, X_test_scaled has 214 cols.
# To properly build the scaler, we just load `data/processed/X_test_clean_text.csv` (which has 80 cols), 
# run pd.get_dummies to get 214 cols (wait, test set get_dummies might not equal 214).
# To perfectly match, we can just save the MAX and MIN of every column in `X_test_scaled`.
