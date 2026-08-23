import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# ── Feature set matching 02_feature_engineering.py output ─────────────────────
FEATURES = [
    # Raw irradiance channels
    'GHI', 'DNI', 'DHI',
    # Atmospheric state
    'T_amb', 'Cloud_Cover',
    # Physics-informed thermal proxy
    'LTC',
    # Raw temporal
    'Hour', 'DayOfYear',
    # Cyclical temporal encodings
    'Hour_sin', 'Hour_cos', 'DayOfYear_sin', 'DayOfYear_cos',
    # Autoregressive lags
    'GHI_lag1', 'GHI_lag2', 'DNI_lag1', 'DNI_lag2',
    'DHI_lag1', 'DHI_lag2', 'Cloud_Cover_lag1', 'Cloud_Cover_lag2',
    # Rolling statistics
    'GHI_roll3_mean', 'GHI_roll3_std', 'Cloud_roll3_mean',
    # Non-linear interaction
    'GHI_x_Tamb',
    # Solar geometry
    'Solar_Zenith', 'Solar_Azimuth',
]
TARGET = 'P_act'


def apply_nighttime_override(y_pred, ghi_array, threshold=0.05):
    """Hard zero-out predictions during nighttime. Guarantees zero error at night."""
    return np.where(ghi_array < threshold, 0.0, y_pred)


def evaluate_model(name, y_true, y_pred, ghi_array):
    """
    Evaluates a model under two benchmarks:
      1. Active Daylight Only — strict test on sun-up hours (GHI >= 0.05)
      2. Full 24-Hour         — includes nighttime zeros (matches original visual style)
    Returns nighttime-zeroed predictions and both sets of metrics.
    """
    y_pred_cleaned = apply_nighttime_override(y_pred, ghi_array)

    # ── Benchmark 1: Active Daylight Only ─────────────────────────────────────
    daylight_mask = ghi_array >= 0.05
    y_true_day    = y_true[daylight_mask]
    y_pred_day    = y_pred_cleaned[daylight_mask]

    rmse_day = np.sqrt(mean_squared_error(y_true_day, y_pred_day))
    mae_day  = mean_absolute_error(y_true_day, y_pred_day)
    r2_day   = r2_score(y_true_day, y_pred_day)

    # ── Benchmark 2: Full 24-Hour (incl. nights) ───────────────────────────────
    rmse_24h = np.sqrt(mean_squared_error(y_true, y_pred_cleaned))
    mae_24h  = mean_absolute_error(y_true, y_pred_cleaned)
    r2_24h   = r2_score(y_true, y_pred_cleaned)

    print(f"  {'─' * 50}")
    print(f"  {name}")
    print(f"  [Daylight Only]  R2: {r2_day:.4f}  RMSE: {rmse_day:.4f} kWh  MAE: {mae_day:.4f} kWh")
    print(f"  [Full 24-Hour]   R2: {r2_24h:.4f}  RMSE: {rmse_24h:.4f} kWh  MAE: {mae_24h:.4f} kWh")

    return y_pred_cleaned, r2_day, rmse_day, mae_day, r2_24h, rmse_24h, mae_24h


def build_stacking_ensemble(random_state=42):
    """
    Builds a Hybrid Deep Learning + Gradient Boosting Stacking Regressor.

    Base Learners:
        - XGBoost  : fast, robust, handles sparse features well
        - LightGBM : leaf-wise growth, excellent for large datasets
        - CatBoost : oblivious symmetric trees, strong regularisation
        - MLP      : neural network in a StandardScaler pipeline; captures
                     non-linear manifolds the tree ensembles may miss

    Meta-Learner:
        - Ridge Regression: linear blender that avoids overfitting at the
          meta-level. passthrough=True also feeds raw features directly.
    """
    base_learners = [
        ('xgb', xgb.XGBRegressor(
            max_depth=7, learning_rate=0.015, n_estimators=1500,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, verbosity=0
        )),
        ('lgb', lgb.LGBMRegressor(
            max_depth=7, learning_rate=0.015, n_estimators=1500,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, verbose=-1
        )),
        ('cat', CatBoostRegressor(
            depth=7, learning_rate=0.015, iterations=1500,
            subsample=0.8, random_state=random_state, verbose=False
        )),
        # Neural network must be preceded by StandardScaler (mean=0, var=1)
        ('mlp', Pipeline([
            ('scaler', StandardScaler()),
            ('nn', MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            ))
        ]))
    ]

    meta_model = Ridge(alpha=1.0)

    return StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_model,
        cv=5,            # 5-fold OOF to generate meta-features without leakage
        passthrough=True, # also feed raw features to meta-model
        n_jobs=-1
    )


def plot_results(y_test, ghi_test, results_dict, output_dir):
    """
    Generates TWO sets of side-by-side scatter plots for all models:
      1. hybrid_ensemble_comparison_daylight.png — Active daylight hours only
      2. hybrid_ensemble_comparison_24hour.png   — Full 24-hour (incl. nights)
    """
    print("\nGenerating comparison scatter plots (both benchmarks)...")
    os.makedirs(output_dir, exist_ok=True)

    y_test_np     = y_test.values if hasattr(y_test, 'values') else y_test
    daylight_mask = ghi_test >= 0.05
    n_models      = len(results_dict)
    max_val       = y_test_np.max() * 1.05

    benchmarks = [
        {
            'label':       'Active Daylight Hours (GHI ≥ 0.05 kW/m²)',
            'filename':    'hybrid_ensemble_comparison_daylight.png',
            'y_true':      y_test_np[daylight_mask],
            'pred_filter': daylight_mask,           # apply mask to preds too
            'r2_idx': 1, 'rmse_idx': 2, 'mae_idx': 3,
        },
        {
            'label':       'Full 24-Hour (Including Nighttime Zeros)',
            'filename':    'hybrid_ensemble_comparison_24hour.png',
            'y_true':      y_test_np,
            'pred_filter': None,                    # use full preds
            'r2_idx': 4, 'rmse_idx': 5, 'mae_idx': 6,
        },
    ]

    for bm in benchmarks:
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)
        if n_models == 1:
            axes = [axes]

        fig.suptitle(f'Predicted vs Actual Power Output — {bm["label"]}',
                     fontsize=14, fontweight='bold')

        for ax, (model_name, data) in zip(axes, results_dict.items()):
            preds_cleaned = data[0]
            r2   = data[bm['r2_idx']]
            rmse = data[bm['rmse_idx']]
            mae  = data[bm['mae_idx']]

            y_plot = bm['y_true']
            p_plot = preds_cleaned[bm['pred_filter']] if bm['pred_filter'] is not None else preds_cleaned

            ax.scatter(y_plot, p_plot, alpha=0.25, color='#4C72B0', edgecolors='none', s=8)
            ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='Ideal (y=x)')
            ax.set_title(model_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Actual Power (kWh)', fontsize=10)
            if model_name == list(results_dict.keys())[0]:
                ax.set_ylabel('Predicted Power (kWh)', fontsize=10)
            ax.set_xlim([0, max_val])
            ax.set_ylim([0, max_val])
            ax.grid(True, linestyle=':', alpha=0.6)

            textstr = f'$R^2 = {r2:.4f}$\n$RMSE = {rmse:.4f}$\n$MAE = {mae:.4f}$'
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
            ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, bm['filename'])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.show()


def train_and_evaluate(input_path):
    print(f"\nLoading engineered dataset from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    # Verify all required features exist
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features in dataset: {missing}")
        print("Make sure you have run the updated 02_feature_engineering.py first.")
        sys.exit(1)

    # Chronological 80/20 split — preserves time-series integrity
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
    ghi_test = test_df['GHI'].values

    print(f"Training: {len(X_train):,} hours | Testing: {len(X_test):,} hours")
    print(f"Feature count: {len(FEATURES)}\n")

    # ── Robustness Test ────────────────────────────────────────────────────────
    print("=" * 52)
    print("Running Robustness Test across 5 random seeds...")
    print("(Individual base models only — for speed)")
    print("=" * 52)

    seeds = [42, 123, 456, 789, 999]
    xgb_rmses, lgb_rmses, cat_rmses = [], [], []
    daylight_mask = ghi_test >= 0.05

    for seed in seeds:
        print(f"  Seed {seed}...", end=' ', flush=True)

        m_xgb = xgb.XGBRegressor(max_depth=7, learning_rate=0.015, n_estimators=1500,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=seed, n_jobs=-1, verbosity=0).fit(X_train, y_train)
        xgb_rmses.append(np.sqrt(mean_squared_error(
            y_test[daylight_mask], m_xgb.predict(X_test)[daylight_mask])))

        m_lgb = lgb.LGBMRegressor(max_depth=7, learning_rate=0.015, n_estimators=1500,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, n_jobs=-1, verbose=-1).fit(X_train, y_train)
        lgb_rmses.append(np.sqrt(mean_squared_error(
            y_test[daylight_mask], m_lgb.predict(X_test)[daylight_mask])))

        m_cat = CatBoostRegressor(depth=7, learning_rate=0.015, iterations=1500,
                                   subsample=0.8, random_state=seed, verbose=False).fit(X_train, y_train)
        cat_rmses.append(np.sqrt(mean_squared_error(
            y_test[daylight_mask], m_cat.predict(X_test)[daylight_mask])))
        print("done")

    print("\n=== ROBUSTNESS RESULTS (ACTIVE DAYLIGHT) ===")
    print(f"  XGBoost:  {np.mean(xgb_rmses):.4f} +/- {np.std(xgb_rmses):.4f} kWh")
    print(f"  LightGBM: {np.mean(lgb_rmses):.4f} +/- {np.std(lgb_rmses):.4f} kWh")
    print(f"  CatBoost: {np.mean(cat_rmses):.4f} +/- {np.std(cat_rmses):.4f} kWh")
    print("=" * 52)

    # ── Final Models + Ensemble ────────────────────────────────────────────────
    print("\nTraining final models (seed=42) for evaluation and visuals...")
    results_dict = {}

    for name, model in [
        ("XGBoost", xgb.XGBRegressor(
            max_depth=7, learning_rate=0.015, n_estimators=1500,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0)),
        ("LightGBM", lgb.LGBMRegressor(
            max_depth=7, learning_rate=0.015, n_estimators=1500,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbose=-1)),
        ("CatBoost", CatBoostRegressor(
            depth=7, learning_rate=0.015, iterations=1500,
            subsample=0.8, random_state=42, verbose=False)),
    ]:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results_dict[name] = evaluate_model(name, y_test, preds, ghi_test)

    # Stacking Ensemble — slowest step due to 5-fold cross-validation
    print("\nTraining Hybrid Stacking Ensemble (XGB + LGB + CAT + MLP -> Ridge)...")
    print("  Using 5-fold OOF cross-validation to build meta-features...")
    stack = build_stacking_ensemble(random_state=42)
    stack.fit(X_train, y_train)
    stack_preds = stack.predict(X_test)
    results_dict["Hybrid Ensemble"] = evaluate_model(
        "Hybrid Ensemble", y_test, stack_preds, ghi_test)

    # ── Final Summary ──────────────────────────────────────────────────────────
    col = 28
    print("\n" + "=" * 85)
    print("FINAL SUMMARY")
    print("=" * 85)
    print(f"  {'Model':<{col}}  {'── Daylight Only ──':^33}  {'── Full 24-Hour ──':^25}")
    print(f"  {'':<{col}}  {'R2':>8}  {'RMSE':>9}  {'MAE':>9}  {'R2':>8}  {'RMSE':>9}  {'MAE':>9}")
    print(f"  {'-' * 83}")
    for model_name, (_, r2d, rmse_d, mae_d, r2_24, rmse_24, mae_24) in results_dict.items():
        print(f"  {model_name:<{col}}  {r2d:>8.4f}  {rmse_d:>8.4f}  {mae_d:>8.4f}  {r2_24:>8.4f}  {rmse_24:>8.4f}  {mae_24:>8.4f}")
    print("=" * 85)
    print("  * Daylight Only: GHI >= 0.05 kW/m² (strict solar benchmark)")
    print("  * Full 24-Hour : includes night hours where predicted = actual = 0")

    output_directory = os.path.join(os.path.dirname(input_path), '../results')
    plot_results(y_test, ghi_test, results_dict, output_directory)


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

    train_and_evaluate(in_file_path)
