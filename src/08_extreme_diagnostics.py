import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor

def generate_edge_case_plots(input_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    features = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
    target = 'P_act'
    
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    print("Training CatBoost for Extreme Edge-Case diagnostics...")
    model = CatBoostRegressor(
        depth=8, learning_rate=0.015, iterations=1500, 
        subsample=0.8, random_state=42, verbose=False
    )
    model.fit(train_df[features], train_df[target])
    
    preds = model.predict(test_df[features])
    ghi_test = test_df['GHI'].values
    preds_cleaned = np.where(ghi_test < 0.05, 0.0, preds)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Actual': test_df[target].values,
        'Predicted': preds_cleaned,
        'Cloud_Cover': test_df['Cloud_Cover'].values
    }, index=test_df.index)
    
    results['Residuals'] = results['Actual'] - results['Predicted']
    
    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    # --- GRAPH 1: Violin Plot by Cloud Cover Category ---
    print("\nGenerating Weather Stratification (Violin Plot)...")
    
    # Categorize Cloud Cover (0-100%) into 4 distinct meteorological buckets
    bins = [-1, 15, 50, 85, 101]
    labels = ['Clear Sky\n(0-15%)', 'Partly Cloudy\n(16-50%)', 'Mostly Cloudy\n(51-85%)', 'Overcast\n(86-100%)']
    results['Weather_Type'] = pd.cut(results['Cloud_Cover'], bins=bins, labels=labels)
    
    # Filter to daylight hours only to avoid skewing the clear sky with nighttime zeros
    daylight_results = results[ghi_test > 0.05]
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Weather_Type', y='Residuals', data=daylight_results, 
                   palette='muted', inner='quartile', linewidth=1.5)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.title('Prediction Error Distribution by Weather State (Daylight Only)', fontweight='bold', pad=15)
    plt.ylabel('Prediction Error ($y_{true} - y_{pred}$)')
    plt.xlabel('Meteorological State (Satellite Cloud Cover)')
    plt.tight_layout()
    
    path1 = os.path.join(out_dir, 'fig10_weather_violin.png')
    plt.savefig(path1, dpi=300)
    print(f"Saved: {path1}")
    plt.show()

    # --- GRAPH 2: Ramp Rate (Volatility) Tracking ---
    print("\nGenerating Ramp Rate Volatility Plot...")
    
    # Calculate hour-to-hour change in power (Ramp Rate)
    results['Actual_Ramp'] = results['Actual'].diff()
    results['Predicted_Ramp'] = results['Predicted'].diff()
    
    # Drop NaNs from the diff calculation
    ramp_df = results.dropna()
    
    plt.figure(figsize=(8, 8))
    
    # We use a hexbin plot because scatter points will overlap too much
    hb = plt.hexbin(ramp_df['Actual_Ramp'], ramp_df['Predicted_Ramp'], 
                    gridsize=40, cmap='inferno', bins='log', mincnt=1)
    
    # Plot the ideal 1:1 volatility line
    min_ramp = max(ramp_df['Actual_Ramp'].min(), ramp_df['Predicted_Ramp'].min())
    max_ramp = min(ramp_df['Actual_Ramp'].max(), ramp_df['Predicted_Ramp'].max())
    plt.plot([min_ramp, max_ramp], [min_ramp, max_ramp], color='#00FF00', linestyle='--', linewidth=2.5, label='Ideal Tracking')
    
    cb = plt.colorbar(hb, label='Log(Count) of Hourly Events')
    plt.title('Grid Volatility: Hourly Ramp Rate Tracking', fontweight='bold', pad=15)
    plt.xlabel('Actual Hourly Power Change $\Delta P$ (kWh)')
    plt.ylabel('Predicted Hourly Power Change $\Delta P$ (kWh)')
    
    # Text box explaining what this proves
    ax = plt.gca()
    plt.text(0.3, 0.9, 'Top-Right: Accurately predicted power surges\nBottom-Left: Accurately predicted drop-offs', 
             transform=ax.transAxes, ha='left', va='center', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.legend(loc='upper left', facecolor='white', framealpha=1)
    plt.tight_layout()
    
    path2 = os.path.join(out_dir, 'fig11_ramp_rate_hexbin.png')
    plt.savefig(path2, dpi=300)
    print(f"Saved: {path2}")
    plt.show()

    print("\nExtreme Edge-Case Diagnostics generated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Please select the 'model_ready_data.csv' file...")
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if in_file_path:
        generate_edge_case_plots(in_file_path)