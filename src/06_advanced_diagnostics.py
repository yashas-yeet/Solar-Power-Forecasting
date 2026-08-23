import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from catboost import CatBoostRegressor

def generate_advanced_plots(input_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    # Ensure month column exists for the heatmap
    df['Month'] = df.index.month
    
    features = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
    target = 'P_act'
    
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    print("Training CatBoost for advanced diagnostics...")
    model = CatBoostRegressor(
        depth=8, learning_rate=0.015, iterations=1500, 
        subsample=0.8, random_state=42, verbose=False
    )
    model.fit(train_df[features], train_df[target])
    
    preds = model.predict(test_df[features])
    ghi_test = test_df['GHI'].values
    preds_cleaned = np.where(ghi_test < 0.05, 0.0, preds)
    
    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)
    
    # Adjust seaborn context for slightly thicker lines in academic print
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    # ... [GRAPH 1: SHAP Beeswarm Plot - Unchanged] ...
    print("\nCalculating SHAP values (This takes a few moments)...")
    # We sample 2000 points from the test set for SHAP to keep computation time reasonable
    X_test_sample = test_df[features].sample(n=2000, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Feature Impact on Model Output", fontweight='bold', pad=20)
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    path1 = os.path.join(out_dir, 'fig4_shap_summary.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.show()

    # ... [GRAPH 2: Temporal Error Heatmap - Unchanged] ...
    print("\nGenerating Temporal Error Heatmap...")
    results = pd.DataFrame({
        'Actual': test_df[target].values,
        'Predicted': preds_cleaned,
        'Hour': test_df['Hour'].values,
        'Month': test_df['Month'].values
    })
    results['Absolute_Error'] = np.abs(results['Actual'] - results['Predicted'])
    
    # Pivot table: average error by Month and Hour
    heatmap_data = results.groupby(['Month', 'Hour'])['Absolute_Error'].mean().unstack()
    # Filter to daylight hours (6 to 19)
    heatmap_data = heatmap_data.loc[:, 6:19]
    
    plt.figure(figsize=(12, 5))
    ax = sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Mean Absolute Error (kWh)'})
    plt.title('Spatiotemporal Error Distribution (Month vs Hour)', fontweight='bold', pad=15)
    plt.ylabel('Month of Year')
    plt.xlabel('Hour of Day (24H)')
    # Format Y-axis to show month names instead of numbers
    ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
    plt.tight_layout()
    
    path2 = os.path.join(out_dir, 'fig5_error_heatmap.png')
    plt.savefig(path2, dpi=300)
    print(f"Saved: {path2}")
    plt.show()

    # --- GRAPH 3: Bivariate KDE Density Plot (UPDATED) ---
    print("\nGenerating High-Contrast KDE Density Plot...")
    plt.figure(figsize=(8, 8))
    
    # UPDATED CONFIGURATION FOR IEEE CONTRAST:
    # 1. cmap="rocket": Vibrant sequential map (sand -> fiery orange -> black). Perfect contrast.
    # 2. thresh=0.005: dropped threshold dramatically to capture extreme tails.
    # 3. levels=10: reduced levels to make the density bands broader and more distinct.
    sns.kdeplot(x=results['Actual'], y=results['Predicted'], cmap="rocket", fill=True, thresh=0.005, levels=10)
    
    # Calculate min/max for the y=x line
    min_val = 0
    max_val = max(results['Actual'].max(), results['Predicted'].max())
    
    # Use a highly contrasting color (bright white or green) for theIdeal Fit line, 
    # since 'rocket' ends in deep black/purple. Bright green is standard.
    plt.plot([min_val, max_val], [min_val, max_val], color='#00FF00', linestyle='--', linewidth=2.5, label='Ideal Fit (y=x)')
    
    plt.title('Prediction Density vs Actual Output', fontweight='bold', pad=15)
    plt.xlabel('Actual Power Output (kWh)')
    plt.ylabel('Predicted Power Output (kWh)')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left', facecolor='white', framealpha=1)
    plt.tight_layout()
    
    path3 = os.path.join(out_dir, 'fig6_kde_density.png')
    plt.savefig(path3, dpi=300)
    print(f"Saved: {path3}")
    plt.show()

    print("\nAll advanced academic diagnostics generated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Please select the 'model_ready_data.csv' file...")
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if in_file_path:
        generate_advanced_plots(in_file_path)