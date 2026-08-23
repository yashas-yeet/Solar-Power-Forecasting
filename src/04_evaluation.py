import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ── Feature groups for systematic ablation ────────────────────────────────────
FEATURES_ALL_26 = [
    'GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear',
    'Hour_sin', 'Hour_cos', 'DayOfYear_sin', 'DayOfYear_cos',
    'GHI_lag1', 'GHI_lag2', 'DNI_lag1', 'DNI_lag2',
    'DHI_lag1', 'DHI_lag2', 'Cloud_Cover_lag1', 'Cloud_Cover_lag2',
    'GHI_roll3_mean', 'GHI_roll3_std', 'Cloud_roll3_mean',
    'GHI_x_Tamb', 'Solar_Zenith', 'Solar_Azimuth',
]

FEATURE_GROUPS = {
    'Cyclical Encoding':   ['Hour_sin', 'Hour_cos', 'DayOfYear_sin', 'DayOfYear_cos'],
    'Autoregressive Lags': ['GHI_lag1', 'GHI_lag2', 'DNI_lag1', 'DNI_lag2',
                            'DHI_lag1', 'DHI_lag2', 'Cloud_Cover_lag1', 'Cloud_Cover_lag2'],
    'Rolling Statistics':  ['GHI_roll3_mean', 'GHI_roll3_std', 'Cloud_roll3_mean'],
    'Interaction Term':    ['GHI_x_Tamb'],
    'Solar Geometry':      ['Solar_Zenith', 'Solar_Azimuth'],
    'LTC (Thermal Proxy)': ['LTC'],
    'DHI (Diffuse Decoupling)': ['DHI'],
}

TARGET = 'P_act'
MODEL_PARAMS = dict(depth=8, learning_rate=0.015, iterations=1500,
                    subsample=0.8, random_state=42, verbose=False)


def run_ablation_study(input_path):
    print(f"Loading data from {input_path}...\n")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    # Check which features are available
    available = [f for f in FEATURES_ALL_26 if f in df.columns]
    if len(available) < len(FEATURES_ALL_26):
        missing = [f for f in FEATURES_ALL_26 if f not in df.columns]
        print(f"WARNING: Missing {len(missing)} features: {missing}")
        print("Run 02_feature_engineering.py first to unlock all 26 features.\n")

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    ghi_test = test_df['GHI'].values
    daylight_mask = ghi_test >= 0.05

    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)

    def train_eval(features, name):
        """Train CatBoost and evaluate under both benchmarks."""
        # Use only features actually in the dataset
        feats = [f for f in features if f in df.columns]
        X_tr = train_df[feats]
        X_te = test_df[feats]
        y_tr = train_df[TARGET]
        y_te = test_df[TARGET].values

        model = CatBoostRegressor(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        preds_cleaned = np.where(ghi_test < 0.05, 0.0, preds)

        # Daylight-only
        r2_day  = r2_score(y_te[daylight_mask], preds_cleaned[daylight_mask])
        rmse_day = np.sqrt(mean_squared_error(y_te[daylight_mask], preds_cleaned[daylight_mask]))
        mae_day  = mean_absolute_error(y_te[daylight_mask], preds_cleaned[daylight_mask])

        # Full 24-hour
        r2_24   = r2_score(y_te, preds_cleaned)
        rmse_24  = np.sqrt(mean_squared_error(y_te, preds_cleaned))
        mae_24   = mean_absolute_error(y_te, preds_cleaned)

        print(f"  [{name}]")
        print(f"    Daylight → R2: {r2_day:.4f}  RMSE: {rmse_day:.4f} kWh  MAE: {mae_day:.4f} kWh")
        print(f"    24-Hour  → R2: {r2_24:.4f}  RMSE: {rmse_24:.4f} kWh  MAE: {mae_24:.4f} kWh")

        return model, feats, rmse_day, rmse_24, r2_day, r2_24

    # ── 1. Full 26-feature model ───────────────────────────────────────────────
    print("=" * 65)
    print("1. FULL PROPOSED FRAMEWORK (26 Features)")
    print("=" * 65)
    model_full, feats_full, rmse_full_day, rmse_full_24, r2_full_day, r2_full_24 = \
        train_eval(available, "Full Framework (CatBoost + 26 Features)")

    # ── 2. SHAP Feature Importance ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2. SHAP FEATURE IMPORTANCE (26-Feature Model)")
    print("=" * 65)
    X_te_full = test_df[feats_full]
    X_sample  = X_te_full[daylight_mask].sample(n=min(2000, daylight_mask.sum()), random_state=42)

    print("  Computing SHAP values (daylight sample)...")
    explainer   = shap.TreeExplainer(model_full)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP per feature
    shap_df = pd.DataFrame({
        'Feature':    feats_full,
        'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean |SHAP|', ascending=False)

    print("\n  SHAP Feature Importance Ranking:")
    print(f"  {'Rank':<5} {'Feature':<25} {'Mean |SHAP|':>12}")
    print(f"  {'-'*44}")
    for rank, (_, row) in enumerate(shap_df.iterrows(), 1):
        print(f"  {rank:<5} {row['Feature']:<25} {row['Mean |SHAP|']:>12.4f}")

    # SHAP summary plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    plt.figure(figsize=(10, 7))
    plt.title("SHAP Feature Impact — Full 26-Feature Framework\n(Active Daylight Hours)",
              fontweight='bold', pad=20)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    shap_path = os.path.join(out_dir, 'ablation_shap_importance.png')
    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
    print(f"\n  SHAP plot saved: {shap_path}")
    plt.show()

    # ── 3. Systematic Ablation: Remove each feature group ────────────────────
    print("\n" + "=" * 65)
    print("3. SYSTEMATIC FEATURE GROUP ABLATION (Remove One Group at a Time)")
    print("=" * 65)

    ablation_results = []

    # Baseline — original 8-feature set
    baseline_8 = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
    print(f"\n  [Baseline — Original 8 Features]")
    _, _, b_rmse_day, b_rmse_24, b_r2_day, b_r2_24 = train_eval(baseline_8, "Baseline (8 Features)")
    ablation_results.append({
        'Experiment': 'Baseline (8 Features)',
        'Features Removed': '18 engineered features',
        'N Features': 8,
        'RMSE Day': b_rmse_day,
        'R2 Day': b_r2_day,
        'RMSE 24h': b_rmse_24,
        'RMSE Day Δ%': (b_rmse_day - rmse_full_day) / rmse_full_day * 100,
    })

    # Remove each group
    for group_name, group_feats in FEATURE_GROUPS.items():
        feats_reduced = [f for f in available if f not in group_feats]
        print(f"\n  [Remove: {group_name} ({len(group_feats)} feature(s))]")
        _, _, a_rmse_day, a_rmse_24, a_r2_day, a_r2_24 = \
            train_eval(feats_reduced, f"No {group_name}")
        ablation_results.append({
            'Experiment': f'Remove {group_name}',
            'Features Removed': ', '.join(group_feats),
            'N Features': len(feats_reduced),
            'RMSE Day': a_rmse_day,
            'R2 Day': a_r2_day,
            'RMSE 24h': a_rmse_24,
            'RMSE Day Δ%': (a_rmse_day - rmse_full_day) / rmse_full_day * 100,
        })

    # ── 4. Ablation Summary Table ──────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("ABLATION SUMMARY (vs. Full 26-Feature Framework)")
    print("=" * 75)
    print(f"  Full Model → Daylight RMSE: {rmse_full_day:.4f} kWh  | "
          f"24h RMSE: {rmse_full_24:.4f} kWh")
    print(f"  {'-' * 73}")
    print(f"  {'Experiment':<35} {'N Feats':>7} {'RMSE Day':>10} {'R2 Day':>8} {'ΔRMSE Day':>10}")
    print(f"  {'-' * 73}")

    for r in sorted(ablation_results, key=lambda x: -x['RMSE Day Δ%']):
        delta_str = f"+{r['RMSE Day Δ%']:.2f}%" if r['RMSE Day Δ%'] >= 0 else f"{r['RMSE Day Δ%']:.2f}%"
        print(f"  {r['Experiment']:<35} {r['N Features']:>7} "
              f"{r['RMSE Day']:>10.4f} {r['R2 Day']:>8.4f} {delta_str:>10}")

    print("=" * 75)
    print("  ↑ ΔRMSE = % increase vs full model. Higher = that group matters more.")

    # ── 5. Ablation Bar Chart ─────────────────────────────────────────────────
    experiments = [r['Experiment'] for r in ablation_results]
    deltas      = [r['RMSE Day Δ%'] for r in ablation_results]
    colors      = ['#d62728' if d > 1 else '#ff7f0e' if d > 0.3 else '#2ca02c' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(experiments, deltas, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel('RMSE Increase (%) when Feature Group is Removed', fontsize=12)
    ax.set_title('Feature Group Ablation Study\n(Daylight-Only Benchmark, CatBoost)',
                 fontweight='bold', pad=15)
    for bar, val in zip(bars, deltas):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'+{val:.2f}%' if val >= 0 else f'{val:.2f}%',
                va='center', fontsize=10)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, 'ablation_feature_group_impact.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"\n  Ablation bar chart saved: {bar_path}")
    plt.show()

    print("\nAblation study complete.")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Select the 'model_ready_data.csv' file...")
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if in_file_path:
        run_ablation_study(in_file_path)
    else:
        print("No file selected. Exiting.")
        sys.exit()