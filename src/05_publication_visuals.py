import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

def generate_individual_plots(input_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    features = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
    target = 'P_act'
    
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    print("Training the proposed CatBoost Framework...")
    model = CatBoostRegressor(
        depth=8, learning_rate=0.015, iterations=1500, 
        subsample=0.8, random_state=42, verbose=False
    )
    model.fit(train_df[features], train_df[target])
    
    # Generate predictions
    preds = model.predict(test_df[features])
    ghi_test = test_df['GHI'].values
    preds_cleaned = np.where(ghi_test < 0.05, 0.0, preds)
    
    # Create a results dataframe for easy plotting
    results = pd.DataFrame({
        'Actual': test_df[target].values,
        'Predicted': preds_cleaned,
        'Hour': test_df['Hour'].values
    }, index=test_df.index)
    
    results['Residuals'] = results['Actual'] - results['Predicted']
    
    print("Generating individual IEEE-formatted visuals...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    
    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)

    # --- GRAPH 1: 72-Hour Hero Plot ---
    plt.figure(figsize=(10, 5))
    slice_start = 1200 # roughly 50 days into the test set
    slice_end = slice_start + 72
    subset = results.iloc[slice_start:slice_end]
    
    plt.plot(subset.index, subset['Actual'], label='Actual Power', color='#1f77b4', linewidth=2)
    plt.plot(subset.index, subset['Predicted'], label='CatBoost Prediction', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.title('72-Hour Day-Ahead Operational Simulation', fontweight='bold', pad=15)
    plt.ylabel('Power Output (kWh)')
    plt.xlabel('Timeline')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    path1 = os.path.join(out_dir, 'fig1_hero_plot.png')
    plt.savefig(path1, dpi=300)
    print(f"Saved: {path1}")
    plt.show() # Close the window to generate the next plot

    # --- GRAPH 2: Diurnal Error Profile (MAE by Hour) ---
    plt.figure(figsize=(8, 5))
    hourly_mae = results.groupby('Hour').apply(lambda x: mean_absolute_error(x['Actual'], x['Predicted']))
    daylight_hours = hourly_mae.loc[6:19]
    
    ax = sns.barplot(x=daylight_hours.index, y=daylight_hours.values, color='#2ca02c', alpha=0.8)
    plt.title('Diurnal Error Profile (Daylight Hours)', fontweight='bold', pad=15)
    plt.ylabel('Mean Absolute Error (kWh)')
    plt.xlabel('Hour of Day (24H Format)')
    
    # Text box to highlight the heat-soak stability
    plt.text(0.5, 0.85, 'Stable error during\n14:00-17:00 heat-soak', 
             transform=ax.transAxes, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.tight_layout()
    
    path2 = os.path.join(out_dir, 'fig2_diurnal_error.png')
    plt.savefig(path2, dpi=300)
    print(f"Saved: {path2}")
    plt.show()

    # --- GRAPH 3: Residual Distribution ---
    plt.figure(figsize=(8, 5))
    q_low = results['Residuals'].quantile(0.01)
    q_hi  = results['Residuals'].quantile(0.99)
    filtered_residuals = results['Residuals'][(results['Residuals'] > q_low) & (results['Residuals'] < q_hi)]
    
    sns.histplot(filtered_residuals, bins=50, kde=True, color='#d62728', stat='density')
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Residual Error Distribution', fontweight='bold', pad=15)
    plt.ylabel('Density')
    plt.xlabel('Prediction Error ($y_{true} - y_{pred}$)')
    plt.tight_layout()
    
    path3 = os.path.join(out_dir, 'fig3_residuals.png')
    plt.savefig(path3, dpi=300)
    print(f"Saved: {path3}")
    plt.show()

    print("\nAll publication visuals generated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Please select the 'model_ready_data.csv' file...")
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if in_file_path:
        generate_individual_plots(in_file_path)