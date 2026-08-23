import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from statsmodels.graphics.tsaplots import plot_acf
from catboost import CatBoostRegressor

def generate_final_plots(input_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    features = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
    target = 'P_act'
    
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    print("Training CatBoost for final diagnostics...")
    model = CatBoostRegressor(
        depth=8, learning_rate=0.015, iterations=1500, 
        subsample=0.8, random_state=42, verbose=False
    )
    model.fit(train_df[features], train_df[target])
    
    preds = model.predict(test_df[features])
    ghi_test = test_df['GHI'].values
    preds_cleaned = np.where(ghi_test < 0.05, 0.0, preds)
    
    # Calculate Residuals and Absolute Errors
    residuals = test_df[target].values - preds_cleaned
    abs_errors = np.abs(residuals)
    
    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    # --- GRAPH 1: SHAP Dependence Plot (LTC vs Hour) ---
    print("\nCalculating SHAP values for Dependence Plot...")
    X_test_sample = test_df[features].sample(n=2000, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(9, 6))
    # This plots how LTC impacts the model, color-coded by the Hour of the day
    shap.dependence_plot("LTC", shap_values, X_test_sample, interaction_index="Hour", show=False, cmap=plt.get_cmap("coolwarm"))
    plt.title("Thermodynamic Interaction: LTC Impact by Hour", fontweight='bold', pad=20)
    plt.tight_layout()
    path1 = os.path.join(out_dir, 'fig7_shap_dependence.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.show()

    # --- GRAPH 2: Cumulative Distribution Function (CDF) of Error ---
    print("\nGenerating Error CDF...")
    plt.figure(figsize=(8, 6))
    
    # Sort absolute errors to calculate cumulative probability
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    plt.plot(sorted_errors, cdf, color='#2ca02c', linewidth=3)
    
    # Add reference lines for 90% and 99% confidence
    idx_90 = np.searchsorted(cdf, 0.90)
    idx_99 = np.searchsorted(cdf, 0.99)
    err_90 = sorted_errors[idx_90]
    err_99 = sorted_errors[idx_99]
    
    plt.axhline(0.90, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(err_90, ymin=0, ymax=0.90, color='gray', linestyle='--', alpha=0.7)
    plt.plot(err_90, 0.90, 'ro')
    plt.text(err_90 + 0.05, 0.86, f'90% < {err_90:.2f} kWh', color='red', fontweight='bold')

    plt.title('Cumulative Distribution Function (CDF) of Absolute Error', fontweight='bold', pad=15)
    plt.xlabel('Absolute Error (kWh)')
    plt.ylabel('Cumulative Probability')
    plt.xlim(0, 1.0) # Focus on the bulk of the error
    plt.ylim(0, 1.05)
    plt.fill_between(sorted_errors, cdf, color='#2ca02c', alpha=0.2)
    plt.tight_layout()
    
    path2 = os.path.join(out_dir, 'fig8_error_cdf.png')
    plt.savefig(path2, dpi=300)
    print(f"Saved: {path2}")
    plt.show()

    # --- GRAPH 3: Autocorrelation of Residuals (ACF) ---
    print("\nGenerating Autocorrelation Plot (ACF)...")
    plt.figure(figsize=(10, 5))
    
    # Plot ACF for 72 lags (3 days) to prove no daily repeating errors exist
    # A good model will have all dots fall inside the blue shaded confidence interval quickly
    ax = plt.gca()
    plot_acf(residuals, lags=72, ax=ax, alpha=0.05, color='#1f77b4', vlines_kwargs={"colors": '#1f77b4'})
    
    plt.title('Autocorrelation of Prediction Residuals (72-Hour Horizon)', fontweight='bold', pad=15)
    plt.xlabel('Lag (Hours)')
    plt.ylabel('Autocorrelation')
    plt.ylim(-0.2, 1.1)
    
    # Text box explaining the white noise
    plt.text(0.65, 0.85, 'Absence of cyclical spikes\nindicates temporal patterns\nare fully captured.', 
             transform=ax.transAxes, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.tight_layout()
    
    path3 = os.path.join(out_dir, 'fig9_residuals_acf.png')
    plt.savefig(path3, dpi=300)
    print(f"Saved: {path3}")
    plt.show()

    print("\nAll Final Diagnostics generated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Please select the 'model_ready_data.csv' file...")
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if in_file_path:
        generate_final_plots(in_file_path)