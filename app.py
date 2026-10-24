import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
PROC_DIR = 'data/processed/'
FINAL_DIR = 'data/final_for_modeling/'
X_test_strict = pd.read_csv(f'{FINAL_DIR}X_test_strict.csv').fillna(0).astype(float)
X_test_scaled = pd.read_csv(f'{PROC_DIR}X_test_scaled.csv').fillna(0).astype(float)
X_test_full = pd.read_csv(f'{FINAL_DIR}X_test_full.csv').fillna(0).astype(float)

X_test_cb = pd.read_csv(f'{PROC_DIR}X_test_clean_text.csv')
cat_cols = X_test_cb.select_dtypes(include=['object']).columns
num_cols = X_test_cb.select_dtypes(exclude=['object']).columns
X_test_cb[cat_cols] = X_test_cb[cat_cols].fillna('Missing')
X_test_cb[num_cols] = X_test_cb[num_cols].fillna(0)
y_test = pd.read_csv(f'{PROC_DIR}y_test.csv').squeeze()

# 2. Load Models
ridge = joblib.load('models/ridge_tuned.pkl')
svr = joblib.load('models/svr_68_features_tuned.pkl')
rf = joblib.load('models/random_forest_tuned.pkl')
lgbm = joblib.load('models/lightgbm_tuned.pkl')
cb = joblib.load('models/catboost_tuned.pkl')

# Feature alignment safeguard: ensure X_test_strict matches the exact 67 features SVR was trained on
if hasattr(svr, 'feature_names_in_'):
    svr_cols = list(svr.feature_names_in_)
    if list(X_test_strict.columns) != svr_cols:
        X_test_strict = X_test_scaled[svr_cols].copy()

ui_prep = joblib.load('models/ui_preprocessor.pkl')
tier_scaler = joblib.load('models/tier_scaler.pkl')
tier_classifier = joblib.load('models/tier_classifier.pkl')
tier_mapping = joblib.load('models/tier_mapping.pkl')
from renovation_catalog import RENOVATION_CATALOG

# Calculate training statistics for mathematically rigorous feature scaling
df_train_raw = pd.read_csv(f"{PROC_DIR}X_train_clean_text.csv")
train_stats = {
    col: {
        "mean": float(df_train_raw[col].mean()),
        "std": float(df_train_raw[col].std()) if df_train_raw[col].std() != 0 else 1.0
    }
    for col in ["KitchenQual", "OverallQual", "BsmtQual", "BsmtFinType1", "FullBath", "TotalBath", "WoodDeckSF"]
}
pt = ui_prep['pt']
skewed_features = ui_prep['skewed_features']

hoods = sorted(X_test_cb['Neighborhood'].dropna().unique().tolist())

# --- HELPER FORMATTING FUNCTIONS ---
def format_outputs(final_price, p_arr, preds, stats, area, bsmt, actual_price_text="N/A"):
    # Stage 2 Champion SLSQP Ensemble Weights:
    # CatBoost: 52.7%, LightGBM: 21.3%, Random Forest: 13.8%, Ridge: 6.4%, SVR: 5.9%
    w_dict = {'CatBoost': 0.527, 'LightGBM': 0.213, 'Random Forest': 0.138, 'Ridge': 0.064, 'SVR': 0.059}
    w_arr = np.array([w_dict[m] for m in preds.keys()])
    variance = np.average((p_arr - final_price)**2, weights=w_arr)
    cv = np.sqrt(variance) / final_price
    spread = (np.max(p_arr) - np.min(p_arr)) / final_price
    
    if spread > 0.25 or cv > 0.12:
        tier_html = "<div style='background-color: #ffeaea; padding: 20px; border-radius: 10px; border: 2px solid #e74c3c; text-align: center;'><h2 style='color: #e74c3c; margin: 0;'>🔴 RED TIER (Override)</h2><p style='margin: 5px 0 0 0; color: #c0392b;'><strong>Automation Blocked:</strong> Extreme algorithmic disagreement.</p></div>"
    elif spread > 0.15 or cv > 0.07:
        tier_html = "<div style='background-color: #fdf5e6; padding: 20px; border-radius: 10px; border: 2px solid #f39c12; text-align: center;'><h2 style='color: #f39c12; margin: 0;'>🟡 YELLOW TIER (Review)</h2><p style='margin: 5px 0 0 0; color: #d35400;'><strong>Desktop Review:</strong> Moderate variance detected.</p></div>"
    else:
        tier_html = "<div style='background-color: #eafaf1; padding: 20px; border-radius: 10px; border: 2px solid #27ae60; text-align: center;'><h2 style='color: #27ae60; margin: 0;'>🟢 GREEN TIER (Auto)</h2><p style='margin: 5px 0 0 0; color: #1e8449;'><strong>Safe to List:</strong> High algorithmic confidence.</p></div>"
        
    stats_md = f"| Feature | Value |\n| :--- | :--- |\n| 🏘️ **Neighborhood** | {stats.get('Neighborhood', 'N/A')} |\n| ⭐ **Overall Quality** | {stats.get('OverallQual', 'N/A')} / 10 |\n| 📐 **Square Feet** | {area:,.0f} |\n| 🧱 **Basement (SqFt)** | {bsmt:,.0f} |\n| 🚗 **Garage Cars** | {stats.get('GarageCars', 'N/A')} |\n| 💰 **Actual Target** | {actual_price_text} |"
    price_html = f"<div style='text-align: center; padding: 10px;'><p style='font-size: 1.2em; color: #7f8c8d; margin-bottom: 0;'>Stacked Meta-Model Prediction</p><h1 style='font-size: 3.5em; color: #2c3e50; margin-top: 0;'>${final_price:,.0f}</h1></div>"
    metrics_md = f"<div style='display: flex; justify-content: space-around; background-color: #f1f2f6; padding: 15px; border-radius: 8px;'><div style='text-align: center;'><h3>{cv*100:.1f}%</h3><p style='margin:0;'>Weighted Variance</p></div><div style='text-align: center;'><h3>{spread*100:.1f}%</h3><p style='margin:0;'>Max-Min Spread</p></div></div>"
    df_chart = pd.DataFrame({"Model": list(preds.keys()), "Predicted Price": list(preds.values())})
    return stats_md, tier_html, price_html, metrics_md, df_chart


# --- TAB 1 FUNCTION: Existing DB ---
def analyze_database_property(house_idx):
    idx = int(house_idx.split(" ")[1]) if (isinstance(house_idx, str) and " " in house_idx) else int(house_idx)
    stats = X_test_cb.iloc[idx]
    actual_price = y_test.iloc[idx]
    preds = {
        'CatBoost': cb.predict(X_test_cb.iloc[[idx]])[0],
        'LightGBM': lgbm.predict(X_test_full.iloc[[idx]])[0],
        'Random Forest': rf.predict(X_test_full.iloc[[idx]])[0],
        'Ridge': ridge.predict(X_test_scaled.iloc[[idx]])[0],
        'SVR': svr.predict(X_test_strict.iloc[[idx]])[0]
    }
    p_arr = np.array(list(preds.values()))
    w_arr = np.array([0.527, 0.213, 0.138, 0.064, 0.059]) 
    final_price = np.average(p_arr, weights=w_arr)
    
    dummy = pd.DataFrame(0, index=[0], columns=skewed_features)
    dummy['GrLivArea'] = stats.get('GrLivArea', 0)
    try:
        raw_area = pt.inverse_transform(dummy)[0][skewed_features.index('GrLivArea')]
    except:
        raw_area = 0
        
    raw_bsmt = stats.get('TotalBsmtSF', 0)
    
    return format_outputs(final_price, p_arr, preds, stats, raw_area, raw_bsmt, f"${actual_price:,.0f}")

# --- TAB 2 FUNCTION: Custom Calculator ---
def analyze_custom(hood, qual, area, bsmt, garage):
    stats = {
        'Neighborhood': hood,
        'OverallQual': qual,
        'GrLivArea': area,
        'TotalBsmtSF': bsmt,
        'GarageCars': garage,
        'TotalSF': area + bsmt
    }
    
    # CRITICAL FIX 1 & 2: Use Median and Align Columns for CatBoost
    base_cb = X_test_cb.median(numeric_only=True).to_frame().T.copy()
    for col in X_test_cb.select_dtypes(include=['object']).columns:
        base_cb[col] = X_test_cb[col].mode()[0]
    base_cb = base_cb[X_test_cb.columns]
    
    base_full = X_test_full.median().to_frame().T.copy()
    base_scaled = X_test_scaled.median().to_frame().T.copy()
    
    for k, v in stats.items():
        if k in skewed_features:
            dummy = pd.DataFrame(0, index=[0], columns=skewed_features)
            dummy[k] = v
            yeo_val = pt.transform(dummy)[0][skewed_features.index(k)]
            base_cb[k] = yeo_val
        else:
            base_cb[k] = v
            
    for k, v in stats.items():
        if k == 'Neighborhood':
            prefix = 'Neighborhood'
            cols = [c for c in X_test_full.columns if c.startswith(prefix + '_')]
            for c in cols:
                is_active = (c == f"{prefix}_{v}")
                base_full[c] = 1.0 if is_active else 0.0
                if c in base_scaled.columns:
                    base_scaled[c] = X_test_scaled[c].max() if is_active else X_test_scaled[c].min()
        else:
            if k in skewed_features:
                val = base_cb[k].iloc[0]
            else:
                val = v
                
            mean_val = X_test_cb[k].mean()
            std_val = X_test_cb[k].std()
            if std_val == 0: std_val = 1
            scaled_val = (val - mean_val) / std_val
            
            if k in base_full.columns: base_full[k] = scaled_val
            if k in base_scaled.columns: base_scaled[k] = scaled_val

    base_strict = base_scaled[X_test_strict.columns]
    
    preds = {
        'CatBoost': cb.predict(base_cb)[0],
        'LightGBM': lgbm.predict(base_full)[0],
        'Random Forest': rf.predict(base_full)[0],
        'Ridge': ridge.predict(base_scaled)[0],
        'SVR': svr.predict(base_strict)[0]
    }
    p_arr = np.array(list(preds.values()))
    w_arr = np.array([0.527, 0.213, 0.138, 0.064, 0.059]) 
    final_price = np.average(p_arr, weights=w_arr)
    
    # In Tab 2, the user inputs the raw area directly via the slider. We don't need to inverse transform it!
    raw_area = stats.get('GrLivArea', 0)
        
    raw_bsmt = stats.get('TotalBsmtSF', 0)
    
    return format_outputs(final_price, p_arr, preds, stats, raw_area, raw_bsmt, "N/A (Custom House)")


# --- TAB 3 FUNCTION: Renovation ROI ---
def analyze_renovation(property_id, package_name, user_budget=None):
    if property_id not in X_test_cb.index:
        return "<h3 style='color:red;'>Error: Property ID not found.</h3>"
    if package_name not in RENOVATION_CATALOG:
        return "<h3 style='color:red;'>Error: Renovation package not found.</h3>"
        
    pkg = RENOVATION_CATALOG[package_name]
    cost = float(user_budget) if user_budget is not None and user_budget > 0 else float(pkg["default_cost"])
    
    stats = X_test_cb.loc[property_id].copy()
    
    # 1. Determine Market Tier
    cluster_features = pd.DataFrame({
        'TotalSF': [(stats.get('1stFlrSF', 0) + stats.get('2ndFlrSF', 0) + stats.get('TotalBsmtSF', 0))],
        'OverallQual': [stats.get('OverallQual', 5)],
        'YearBuilt': [stats.get('YearBuilt', 1950)],
        'LotArea': [stats.get('LotArea', 5000)],
        'GarageCars': [stats.get('GarageCars', 1)]
    })
    scaled_feats = tier_scaler.transform(cluster_features)
    tier_label = tier_classifier.predict(scaled_feats)[0]
    tier_name = tier_mapping[tier_label]
    
    # 2. Baseline Multi-Model Prediction
    base_cb = X_test_cb.loc[property_id].to_frame().T.copy()
    base_full = X_test_full.loc[property_id].to_frame().T.copy()
    base_scaled = X_test_scaled.loc[property_id].to_frame().T.copy()
    base_strict = X_test_strict.loc[property_id].to_frame().T.copy()
    
    weights = [0.527, 0.213, 0.138, 0.064, 0.059]
    current_preds = {
        'CatBoost': cb.predict(base_cb)[0],
        'LightGBM': lgbm.predict(base_full)[0],
        'Random Forest': rf.predict(base_full)[0],
        'Ridge': ridge.predict(base_scaled)[0],
        'SVR': svr.predict(base_strict)[0]
    }
    current_price = np.average(list(current_preds.values()), weights=weights)
    
    # 3. Mathematically Rigorous Feature Transformation Pipeline
    up_cb = base_cb.copy()
    up_full = base_full.copy()
    up_scaled = base_scaled.copy()
    up_strict = base_strict.copy()
    
    # Apply catalog updates dynamically
    for feat_name, rule in pkg["updates"].items():
        if feat_name == "WoodDeckSF_raw":
            raw_sqft = rule["value"]
            dummy = pd.DataFrame(0, index=[0], columns=skewed_features)
            dummy["WoodDeckSF"] = raw_sqft
            yeo_val = pt.transform(dummy)[0][skewed_features.index("WoodDeckSF")]
            up_cb["WoodDeckSF"] = yeo_val
            z_val = (yeo_val - train_stats["WoodDeckSF"]["mean"]) / train_stats["WoodDeckSF"]["std"]
            if "WoodDeckSF" in up_full.columns: up_full["WoodDeckSF"] = z_val
            if "WoodDeckSF" in up_scaled.columns: up_scaled["WoodDeckSF"] = z_val
            if "WoodDeckSF" in up_strict.columns: up_strict["WoodDeckSF"] = z_val
        else:
            curr_val = stats.get(feat_name, 5)
            if rule["mode"] == "set":
                new_val = rule["value"]
            elif rule["mode"] == "add":
                new_val = curr_val + rule["value"]
            elif rule["mode"] == "boost":
                new_val = min(rule.get("max", 10), curr_val + rule["value"])
            elif rule["mode"] == "boost_if_below":
                new_val = curr_val + rule["value"] if curr_val < rule["threshold"] else curr_val
            else:
                new_val = curr_val
                
            up_cb[feat_name] = new_val
            if feat_name in train_stats:
                z_score = (new_val - train_stats[feat_name]["mean"]) / train_stats[feat_name]["std"]
                if feat_name in up_full.columns: up_full[feat_name] = z_score
                if feat_name in up_scaled.columns: up_scaled[feat_name] = z_score
                if feat_name in up_strict.columns: up_strict[feat_name] = z_score

    up_preds = {
        'CatBoost': cb.predict(up_cb)[0],
        'LightGBM': lgbm.predict(up_full)[0],
        'Random Forest': rf.predict(up_full)[0],
        'Ridge': ridge.predict(up_scaled)[0],
        'SVR': svr.predict(up_strict)[0]
    }
    upgraded_price = np.average(list(up_preds.values()), weights=weights)
    
    # 4. Financial ROI & Break-Even Analytics
    value_added = upgraded_price - current_price
    roi = value_added - cost
    roi_pct = (roi / cost) * 100 if cost > 0 else 0
    is_profitable = roi > 0
    status_color = "#16A34A" if is_profitable else "#DC2626"
    badge_bg = "#DCFCE7" if is_profitable else "#FEE2E2"
    
    if is_profitable:
        decision_title = f"✅ PROCEED — Net Profit of ${roi:,.0f} (+{roi_pct:.1f}% ROI)"
    else:
        decision_title = f"⚠️ DO NOT RENOVATE — Net Loss of ${abs(roi):,.0f} ({roi_pct:.1f}% ROI)"
        
    report = f"""
    <div style='padding:22px; border:2px solid {status_color}; border-radius:12px; background:#FAFAFA; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <div style='background:{badge_bg}; padding:12px 16px; border-radius:8px; margin-bottom:14px; text-align:center;'>
            <h2 style='margin:0; color:{status_color}; font-size:22px;'>{decision_title}</h2>
        </div>
        
        <p style='color:#334155; font-size:14px; margin: 4px 0 16px 0;'>
            <b>Scope:</b> {pkg["description"]}
        </p>
        
        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:18px;'>
            <div style='background:white; padding:16px; border-radius:8px; border:1px solid #E2E8F0;'>
                <h4 style='margin-top:0; color:#1E293B; border-bottom:2px solid #3B82F6; padding-bottom:6px;'>📊 5-Model Ensemble Valuation</h4>
                <ul style='list-style:none; padding-left:0; font-size:15px; line-height:1.9; margin-bottom:0;'>
                    <li><b>Current As-Is Value:</b> <span style='float:right; font-weight:600;'>${current_price:,.0f}</span></li>
                    <li><b>Post-Renovation Value:</b> <span style='float:right; font-weight:600;'>${upgraded_price:,.0f}</span></li>
                    <li style='border-top:1px dashed #CBD5E1; padding-top:6px; color:#16A34A; font-weight:bold;'>
                        Equity Lift (Market Gain): <span style='float:right;'>+${value_added:,.0f}</span>
                    </li>
                </ul>
            </div>
            
            <div style='background:white; padding:16px; border-radius:8px; border:1px solid #E2E8F0;'>
                <h4 style='margin-top:0; color:#1E293B; border-bottom:2px solid #8B5CF6; padding-bottom:6px;'>💼 Financial Investment Return</h4>
                <ul style='list-style:none; padding-left:0; font-size:15px; line-height:1.9; margin-bottom:0;'>
                    <li><b>Contractor Capital Cost:</b> <span style='float:right; color:#DC2626; font-weight:600;'>-${cost:,.0f}</span></li>
                    <li><b>Max Profitable Cost (Break-Even):</b> <span style='float:right; color:#475569;'>${value_added:,.0f}</span></li>
                    <li style='border-top:1px dashed #CBD5E1; padding-top:6px; color:{status_color}; font-weight:bold; font-size:16px;'>
                        Net ROI (Investor Cash Flow): <span style='float:right;'>${roi:,.0f}</span>
                    </li>
                </ul>
            </div>
        </div>
        
        <div style='margin-top:16px; padding:10px 14px; background:#F1F5F9; border-radius:6px; font-size:13px; color:#475569;'>
            🏛️ <b>Market Segment:</b> <span style='color:#2563EB; font-weight:600;'>{tier_name}</span> &nbsp;|&nbsp; 
            ⚡ <b>Architecture:</b> Mathematical transformation using dynamic training z-scores ($z = \frac{{x - mu}}{{sigma}}$) across Ridge, SVR, RF, LightGBM, and CatBoost.
        </div>
    </div>
    """
    return report

# --- GUI LAYOUT ---
with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.HTML("<h1 style='text-align: center; color: #2c3e50; margin-bottom: 20px;'>🏢 Enterprise AI Pricing & Risk Engine</h1>")
    
    with gr.Tabs():
        
        # TAB 1: EXISTING TEST DATA
        with gr.TabItem("🗄️ Database Explorer (Test Set)"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### ⚙️ Property Control Panel")
                    house_dropdown = gr.Dropdown(choices=[f"House {i}" for i in range(len(y_test))], label="Database Record ID", value="House 15")
                    btn_db = gr.Button("🤖 Execute AI Analysis", variant="primary", size="lg")
                    gr.Markdown("---")
                    gr.Markdown("### 📊 Database Profile")
                    stats_out_db = gr.Markdown("*Select a property and execute analysis.*")
                    
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column(scale=1):
                            price_out_db = gr.HTML("<h2 style='text-align: center; color: #bdc3c7;'>Awaiting Execution...</h2>")
                        with gr.Column(scale=1):
                            tier_out_db = gr.HTML()
                    gr.HTML("<hr>")
                    gr.Markdown("### 📈 Uncertainty Metrics")
                    metrics_out_db = gr.HTML()
                    gr.Markdown("### 🧠 Base Model Breakdown")
                    plot_out_db = gr.BarPlot(x="Model", y="Predicted Price", title="Ensemble Predictions", tooltip=["Model", "Predicted Price"], y_title="Dollar Value ($)", x_title="")
                    
            btn_db.click(fn=analyze_database_property, inputs=house_dropdown, outputs=[stats_out_db, tier_out_db, price_out_db, metrics_out_db, plot_out_db])
            
        # TAB 2: CUSTOM CALCULATOR
        with gr.TabItem("🛠️ Custom Pricing Calculator (Blank Slate)"):
            with gr.Row():
                with gr.Column(scale=1, variant="panel"):
                    gr.Markdown("### 🎛️ Input Custom Features")
                    c_hood = gr.Dropdown(choices=hoods, label="Neighborhood", value="NAmes")
                    c_qual = gr.Slider(minimum=1, maximum=10, step=1, label="Overall Quality (1-10)", value=6)
                    c_area = gr.Slider(minimum=500, maximum=5000, step=10, label="Square Footage (GrLivArea)", value=1500)
                    c_bsmt = gr.Slider(minimum=0, maximum=3000, step=10, label="Basement Square Feet", value=1000)
                    c_garage = gr.Slider(minimum=0, maximum=4, step=1, label="Garage Cars", value=2)
                    btn_custom = gr.Button("⚡ Run Custom AI Prediction", variant="primary", size="lg")
                    gr.Markdown("---")
                    gr.Markdown("### 📊 Custom Profile")
                    stats_out_c = gr.Markdown("*Enter your custom data.*")
                    
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column(scale=1):
                            price_out_c = gr.HTML("<h2 style='text-align: center; color: #bdc3c7;'>Awaiting Custom Input...</h2>")
                        with gr.Column(scale=1):
                            tier_out_c = gr.HTML()
                    gr.HTML("<hr>")
                    gr.Markdown("### 📈 Uncertainty Metrics")
                    metrics_out_c = gr.HTML()
                    gr.Markdown("### 🧠 Base Model Breakdown")
                    plot_out_c = gr.BarPlot(x="Model", y="Predicted Price", title="Ensemble Predictions", tooltip=["Model", "Predicted Price"], y_title="Dollar Value ($)", x_title="")
                    
            btn_custom.click(fn=analyze_custom, inputs=[c_hood, c_qual, c_area, c_bsmt, c_garage], outputs=[stats_out_c, tier_out_c, price_out_c, metrics_out_c, plot_out_c])

        # TAB 3: RENOVATION ROI ADVISOR
        with gr.Tab("Tab 3: Renovation ROI Advisor"):
            gr.Markdown("### 🔨 Dynamic Renovation ROI & Capital Allocation Advisor")
            gr.Markdown("Select a property and a standardized improvement package from our **Renovation Catalog**, or adjust the contractor cost to benchmark your specific estimate.")
            with gr.Row():
                with gr.Column(scale=1):
                    roi_dropdown = gr.Dropdown(choices=X_test_cb.index.tolist(), label="Select Database Property ID", value=X_test_cb.index.tolist()[0])
                    renov_pkg_dropdown = gr.Dropdown(
                        choices=list(RENOVATION_CATALOG.keys()),
                        label="Standard Renovation Package",
                        value=list(RENOVATION_CATALOG.keys())[0]
                    )
                    user_cost_slider = gr.Slider(
                        minimum=2000,
                        maximum=50000,
                        step=500,
                        value=15000,
                        label="Contractor Budget / Estimated Cost ($)"
                    )
                    btn_roi = gr.Button("Calculate Automated ROI", variant="primary")
                with gr.Column(scale=2):
                    roi_output = gr.HTML()
            
            # Helper to update default cost slider when package changes
            def update_package_default(pkg_name):
                if pkg_name in RENOVATION_CATALOG:
                    return gr.update(value=RENOVATION_CATALOG[pkg_name]["default_cost"])
                return gr.update()
                
            renov_pkg_dropdown.change(fn=update_package_default, inputs=renov_pkg_dropdown, outputs=user_cost_slider)
            btn_roi.click(fn=analyze_renovation, inputs=[roi_dropdown, renov_pkg_dropdown, user_cost_slider], outputs=roi_output)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)

