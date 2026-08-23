import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

def generate_creative_visuals(input_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    features = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
    target = 'P_act'
    
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    print("Training CatBoost framework to calculate errors...")
    model = CatBoostRegressor(
        depth=8, learning_rate=0.015, iterations=1500, 
        subsample=0.8, random_state=42, verbose=False
    )
    model.fit(train_df[features], train_df[target])
    
    preds = model.predict(test_df[features])
    preds_cleaned = np.where(test_df['GHI'].values < 0.05, 0.0, preds)
    
    results = pd.DataFrame({
        'Actual': test_df[target].values,
        'Predicted': preds_cleaned,
        'Hour': test_df['Hour'].values,
        'T_amb': test_df['T_amb'].values,
        'LTC': test_df['LTC'].values
    }, index=test_df.index)
    results['MAE'] = np.abs(results['Actual'] - results['Predicted'])
    
    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    # --- PLOT 1: Polar Clock Error Dial ---
    print("\nGenerating Graph 1: Polar Clock Error Dial...")
    hourly_metrics = results.groupby('Hour')['MAE'].mean()
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Convert 24 hours to radians (360 degrees = 2pi radians)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    
    # Align 0 hours to the exact top (12 o'clock position) and run clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Plot the error area
    bars = ax.fill(angles, hourly_metrics.values, color='#ff7f0e', alpha=0.4, label='Mean Absolute Error')
    ax.plot(angles, hourly_metrics.values, color='#d62728', linewidth=2.5)
    
    # Format the circular grid to look like a literal clock face
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{h}:00" for h in range(24)], fontsize=10)
    
    plt.title('Spatiotemporal Clock Dial:\nPredictive Error Magnitude across a 24-Hour Cycle', 
              fontweight='bold', pad=25, fontsize=14)
    ax.set_rlabel_position(180) # Move radial labels out of the way
    plt.tight_layout()
    
    path1 = os.path.join(out_dir, 'fig12_polar_clock_error.png')
    plt.savefig(path1, dpi=300)
    print(f"Saved: {path1}")
    plt.show()

    # --- PLOT 2: Thermodynamic Hysteresis Loop ---
    print("\nGenerating Graph 2: Thermodynamic Hysteresis Loop...")
    # Calculate the mean profile for each hour to plot the clean structural trajectory
    hysteresis_data = results.groupby('Hour')[['T_amb', 'LTC']].mean()
    
    # Close the loop mathematically by appending the first hour to the end
    hysteresis_data = pd.concat([hysteresis_data, hysteresis_data.iloc[[0]]])
    hours_loop = list(range(24)) + [0]
    
    plt.figure(figsize=(9, 7))
    
    # Plot the continuous path
    x = hysteresis_data['T_amb'].values
    y = hysteresis_data['LTC'].values
    
    # Draw colored segments to show time progression smoothly
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color code by hour using the 'twilight_shifted' cyclic colormap
    norm = plt.Normalize(0, 24)
    lc = cm.ScalarMappable(norm=norm, cmap='twilight_shifted')
    
    # Plot line segment by segment to embed the color gradient
    for i in range(len(segments)):
        plt.plot(segments[i][:, 0], segments[i][:, 1], color=lc.to_rgba(hours_loop[i]), linewidth=4)
        
        # Add subtle arrows every 3 hours to indicate physical time direction
        if i % 3 == 0 and i < 23:
            plt.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                         arrowprops=dict(arrowstyle="->", color=lc.to_rgba(hours_loop[i]), lw=3))

    # Label key hours explicitly on the loop to tell the story
    for h in [8, 12, 16, 20]:
        plt.text(x[h], y[h], f" {h}:00", fontsize=11, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    cbar = plt.colorbar(lc, ax=plt.gca(), pad=0.03)
    cbar.set_label('Hour of Day (Temporal Progression)', rotation=270, labelpad=20)
    
    plt.title('Thermodynamic Hysteresis Loop:\nQuantifying the Latent Lag between Ambient Air and LTC State', 
              fontweight='bold', pad=15)
    plt.xlabel('Mean Ambient Temperature $T_{amb}$ (°C)')
    plt.ylabel('Physics-Informed Thermal State ($LTC$)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    path2 = os.path.join(out_dir, 'fig13_thermal_hysteresis.png')
    plt.savefig(path2, dpi=300)
    print(f"Saved: {path2}")
    plt.show()

    print("\nCreative visual portfolio generated successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Please select the 'model_ready_data.csv' file...")
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if in_file_path:
        generate_creative_visuals(in_file_path)