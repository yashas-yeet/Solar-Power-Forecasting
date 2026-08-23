import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def evaluate_and_plot(model_name, y_true, y_pred, ghi_array, out_dir, filename):
    # 1. Create a mask to isolate ONLY daylight hours
    daylight_mask = ghi_array >= 0.05
    
    # 2. Filter both actuals and predictions to exclude night completely for the math
    y_true_day = y_true[daylight_mask]
    y_pred_day = y_pred[daylight_mask]
    
    # 3. Apply Nighttime Override to the full array (so the graph still plots the zeros)
    y_pred_cleaned = np.where(ghi_array < 0.05, 0.0, y_pred)
    
    # 4. Calculate metrics on Active Daylight hours ONLY
    rmse = np.sqrt(mean_squared_error(y_true_day, y_pred_day))
    r2 = r2_score(y_true_day, y_pred_day)
    
    print(f"Generating plot for {model_name}... Active Daylight (R2: {r2:.4f}, RMSE: {rmse:.4f})")
    
    # --- Generate Individual Plot ---
    plt.figure(figsize=(6, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred_cleaned, alpha=0.3, color='#1f77b4', edgecolors='none', s=10)
    
    # Calculate global min and max for perfectly square axes
    min_val = 0
    max_val = max(y_true.max(), y_pred_cleaned.max())
    
    # Plot the perfect prediction line (y=x)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.5, label='Ideal Fit (y=x)')
    
    # Formatting
    plt.title(f'{model_name} Prediction Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Actual Power Output (kWh)', fontsize=12)
    plt.ylabel('Predicted Power Output (kWh)', fontsize=12)
    
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Add a text box with the daylight metrics directly inside the plot
    textstr = '\n'.join((
        f'$R^2= {r2:.4f}$',
        f'$RMSE= {rmse:.4f}$'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.legend(loc='lower right', facecolor='white', framealpha=1)
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved successfully to: {save_path}")
    
    # Close the figure to free up memory before the next one
    plt.close()

def train_and_save_scatters(input_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    features = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
    target = 'P_act'
    
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    ghi_test = test_df['GHI'].values 
    
    # Setup Output Directory
    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)
    
    params = {
        'max_depth': 8,
        'learning_rate': 0.015,
        'n_estimators': 1500,
        'subsample': 0.8,
        'random_state': 42
    }
    
    # 1. XGBoost
    print("\nTraining XGBoost...")
    model_xgb = xgb.XGBRegressor(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=0.8,
        random_state=params['random_state'],
        n_jobs=-1
    )
    model_xgb.fit(X_train, y_train)
    preds_xgb = model_xgb.predict(X_test)
    evaluate_and_plot("XGBoost", y_test, preds_xgb, ghi_test, out_dir, "scatter_1_xgboost.png")
    
    # 2. LightGBM
    print("\nTraining LightGBM...")
    model_lgb = lgb.LGBMRegressor(
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=0.8,
        random_state=params['random_state'],
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(X_train, y_train)
    preds_lgb = model_lgb.predict(X_test)
    evaluate_and_plot("LightGBM", y_test, preds_lgb, ghi_test, out_dir, "scatter_2_lightgbm.png")
    
    # 3. CatBoost (Proposed)
    print("\nTraining CatBoost...")
    model_cat = CatBoostRegressor(
        depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        iterations=params['n_estimators'],
        subsample=params['subsample'],
        random_state=params['random_state'],
        verbose=False 
    )
    model_cat.fit(X_train, y_train)
    preds_cat = model_cat.predict(X_test)
    evaluate_and_plot("CatBoost (Proposed Framework)", y_test, preds_cat, ghi_test, out_dir, "scatter_3_catboost.png")

    print("\nAll separate scatter plots generated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the 'model_ready_data.csv' file from the popup window...")
    
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    
    if not in_file_path:
        print("No file selected. Exiting script.")
        sys.exit()
        
    train_and_save_scatters(in_file_path)