import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ── Full 26-feature set matching 02_feature_engineering.py ──────────────────────
FEATURES_26 = [
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
FEATURES_8 = ['GHI', 'DNI', 'DHI', 'T_amb', 'Cloud_Cover', 'LTC', 'Hour', 'DayOfYear']
TARGET = 'P_act'


def generate_merged_visuals(input_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    if 'Month' not in df.columns:
        df['Month'] = df.index.month

    # Auto-detect feature set
    missing_26 = [f for f in FEATURES_26 if f not in df.columns]
    if missing_26:
        print(f"WARNING: Missing 26-feature columns: {missing_26}")
        print("Falling back to 8-feature set. Run 02_feature_engineering.py to unlock 26 features.")
        features = FEATURES_8
    else:
        features = FEATURES_26
        print(f"Using full {len(features)}-feature set.")

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[TARGET]
    X_test,  y_test  = test_df[features],  test_df[TARGET]
    ghi_test        = test_df['GHI'].values
    daylight_mask   = ghi_test >= 0.05

    out_dir = os.path.join(os.path.dirname(input_path), '../results')
    os.makedirs(out_dir, exist_ok=True)

    # ── Train models ─────────────────────────────────────────────────────────────
    model_params = dict(depth=8, learning_rate=0.015, iterations=1500, subsample=0.8,
                        random_state=42, verbose=False)

    print("\nTraining CatBoost (Primary Model)...")
    model_cat = CatBoostRegressor(**model_params)
    model_cat.fit(X_train, y_train)
    preds_cat_raw     = model_cat.predict(X_test)
    preds_cat_cleaned = np.where(ghi_test < 0.05, 0.0, preds_cat_raw)

    print("Training XGBoost...")
    model_xgb = xgb.XGBRegressor(
        max_depth=model_params['depth'], learning_rate=model_params['learning_rate'],
        n_estimators=model_params['iterations'], subsample=model_params['subsample'],
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
    model_xgb.fit(X_train, y_train)
    preds_xgb_cleaned = np.where(ghi_test < 0.05, 0.0, model_xgb.predict(X_test))

    print("Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(
        max_depth=model_params['depth'], learning_rate=model_params['learning_rate'],
        n_estimators=model_params['iterations'], subsample=model_params['subsample'],
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    model_lgb.fit(X_train, y_train)
    preds_lgb_cleaned = np.where(ghi_test < 0.05, 0.0, model_lgb.predict(X_test))

    # ── Print dual-benchmark metrics ──────────────────────────────────────────────
    y_np = y_test.values
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"  {'Model':<30}  {'── Daylight Only ──':^28}  {'── Full 24-Hour ──':^24}")
    print(f"  {'':30}  {'R2':>8}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}  {'RMSE':>8}")
    print(f"  {'-' * 73}")
    for name, preds in [("CatBoost", preds_cat_cleaned),
                         ("XGBoost",  preds_xgb_cleaned),
                         ("LightGBM", preds_lgb_cleaned)]:
        r2d   = r2_score(y_np[daylight_mask], preds[daylight_mask])
        rmse_d = np.sqrt(mean_squared_error(y_np[daylight_mask], preds[daylight_mask]))
        mae_d  = mean_absolute_error(y_np[daylight_mask], preds[daylight_mask])
        r2_24  = r2_score(y_np, preds)
        rmse_24 = np.sqrt(mean_squared_error(y_np, preds))
        print(f"  {name:<30}  {r2d:>8.4f}  {rmse_d:>8.4f}  {mae_d:>8.4f}  {r2_24:>8.4f}  {rmse_24:>8.4f}")
    print("=" * 75)

    # ── SHAP values: two samples (daylight and full) ───────────────────────────
    print("\nCalculating SHAP values (this may take a moment)...")
    X_test_day    = X_test[daylight_mask]
    shap_sample_day  = X_test_day.sample(n=min(2000, len(X_test_day)), random_state=42)
    shap_sample_full = X_test.sample(n=min(2000, len(X_test)),         random_state=42)
    explainer        = shap.TreeExplainer(model_cat)
    shap_vals_day    = explainer.shap_values(shap_sample_day)
    shap_vals_full   = explainer.shap_values(shap_sample_full)

    # ── Build results DataFrames ───────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    results_full = pd.DataFrame({
        'Actual':      y_np,
        'Predicted':   preds_cat_cleaned,
        'Hour':        test_df['Hour'].values,
        'Month':       test_df['Month'].values,
        'Cloud_Cover': test_df['Cloud_Cover'].values,
        'T_amb':       test_df['T_amb'].values,
        'LTC':         test_df['LTC'].values,
    }, index=test_df.index)
    results_full['Residuals']     = results_full['Actual'] - results_full['Predicted']
    results_full['Absolute_Error']= np.abs(results_full['Residuals'])

    results_day = results_full[daylight_mask].copy()

    # ── Benchmark descriptor registry ─────────────────────────────────────────
    benchmarks = [
        {
            'suffix':      'daylight',
            'label':       'Active Daylight Hours (GHI ≥ 0.05 kW/m²)',
            'results':     results_day,
            'shap_vals':   shap_vals_day,
            'shap_X':      shap_sample_day,
            'preds_xgb':   preds_xgb_cleaned[daylight_mask],
            'preds_lgb':   preds_lgb_cleaned[daylight_mask],
            'preds_cat':   preds_cat_cleaned[daylight_mask],
            'y_true':      y_np[daylight_mask],
        },
        {
            'suffix':      '24hour',
            'label':       'Full 24-Hour (Including Nighttime Zeros)',
            'results':     results_full,
            'shap_vals':   shap_vals_full,
            'shap_X':      shap_sample_full,
            'preds_xgb':   preds_xgb_cleaned,
            'preds_lgb':   preds_lgb_cleaned,
            'preds_cat':   preds_cat_cleaned,
            'y_true':      y_np,
        },
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    #  MAIN PLOTTING LOOP — generates 2 versions of every plot
    # ═══════════════════════════════════════════════════════════════════════════
    for bm in benchmarks:
        sfx     = bm['suffix']
        lbl     = bm['label']
        results = bm['results']

        print(f"\n{'='*65}")
        print(f"  Generating plots → {lbl}")
        print(f"{'='*65}")

        # ── Plot 1: 72-Hour Hero Plot ─────────────────────────────────────────
        print("  [1] 72-Hour Hero Plot...")
        slice_start = 800 if sfx == 'daylight' else 1200
        slice_end   = slice_start + min(72, len(results) - slice_start)
        subset      = results.iloc[slice_start:slice_end]
        plt.figure(figsize=(10, 5))
        plt.plot(subset.index, subset['Actual'],    label='Actual Power',        color='#1f77b4', linewidth=2)
        plt.plot(subset.index, subset['Predicted'], label='CatBoost Prediction', color='#ff7f0e', linestyle='--', linewidth=2)
        plt.title(f'72-Hour Day-Ahead Operational Simulation\n({lbl})', fontweight='bold', pad=15)
        plt.ylabel('Power Output (kWh)')
        plt.xlabel('Timeline')
        plt.legend(loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig01_hero_plot_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 2: Diurnal Error Profile ─────────────────────────────────────
        print("  [2] Diurnal Error Profile...")
        hourly_mae    = results.groupby('Hour').apply(lambda x: mean_absolute_error(x['Actual'], x['Predicted']))
        daylight_hrs  = hourly_mae[hourly_mae.index.isin(range(6, 20))]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=daylight_hrs.index, y=daylight_hrs.values, color='#2ca02c', alpha=0.8)
        plt.title(f'Diurnal Error Profile (Daylight Hours)\n({lbl})', fontweight='bold', pad=15)
        plt.ylabel('Mean Absolute Error (kWh)')
        plt.xlabel('Hour of Day (24H Format)')
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig02_diurnal_error_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 3: Residual Distribution ─────────────────────────────────────
        print("  [3] Residual Distribution...")
        q_low = results['Residuals'].quantile(0.01)
        q_hi  = results['Residuals'].quantile(0.99)
        filt  = results['Residuals'][(results['Residuals'] > q_low) & (results['Residuals'] < q_hi)]
        plt.figure(figsize=(8, 5))
        sns.histplot(filt, bins=50, kde=True, color='#d62728', stat='density')
        plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
        plt.title(f'Residual Error Distribution\n({lbl})', fontweight='bold', pad=15)
        plt.ylabel('Density')
        plt.xlabel('Prediction Error ($y_{true} - y_{pred}$)')
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig03_residuals_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 4: SHAP Summary ──────────────────────────────────────────────
        print("  [4] SHAP Summary...")
        plt.figure(figsize=(10, 6))
        plt.title(f"SHAP Feature Impact on Model Output\n({lbl})", fontweight='bold', pad=20)
        shap.summary_plot(bm['shap_vals'], bm['shap_X'], show=False)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig04_shap_summary_{sfx}.png')
        plt.savefig(p, dpi=300, bbox_inches='tight'); print(f"    Saved: {p}"); plt.show()

        # ── Plot 5: Temporal Error Heatmap ───────────────────────────────────
        print("  [5] Temporal Error Heatmap...")
        heatmap_data = results.groupby(['Month', 'Hour'])['Absolute_Error'].mean().unstack()
        day_cols     = [c for c in range(6, 20) if c in heatmap_data.columns]
        heatmap_data = heatmap_data[day_cols]
        month_names  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        tick_labels  = [month_names[i-1] for i in heatmap_data.index if 1 <= i <= 12]
        plt.figure(figsize=(12, 5))
        ax = sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False,
                         cbar_kws={'label': 'Mean Absolute Error (kWh)'})
        ax.set_yticklabels(tick_labels, rotation=0)
        plt.title(f'Spatiotemporal Error Distribution (Month vs Hour)\n({lbl})', fontweight='bold', pad=15)
        plt.ylabel('Month of Year')
        plt.xlabel('Hour of Day (24H)')
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig05_error_heatmap_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 6: Bivariate KDE Density ────────────────────────────────────
        print("  [6] Bivariate KDE Density...")
        plt.figure(figsize=(8, 8))
        sns.kdeplot(x=results['Actual'], y=results['Predicted'],
                    cmap="rocket", fill=True, thresh=0.005, levels=10)
        mv = max(results['Actual'].max(), results['Predicted'].max())
        plt.plot([0, mv], [0, mv], color='#00FF00', linestyle='--', linewidth=2.5, label='Ideal Fit (y=x)')
        plt.title(f'Prediction Density vs Actual Output\n({lbl})', fontweight='bold', pad=15)
        plt.xlabel('Actual Power Output (kWh)')
        plt.ylabel('Predicted Power Output (kWh)')
        plt.xlim(0, mv); plt.ylim(0, mv)
        plt.legend(loc='upper left', facecolor='white', framealpha=1)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig06_kde_density_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 7: SHAP Dependence Plot ──────────────────────────────────────
        print("  [7] SHAP Dependence Plot...")
        plt.figure(figsize=(9, 6))
        shap.dependence_plot("LTC", bm['shap_vals'], bm['shap_X'],
                             interaction_index="Hour", show=False, cmap=plt.get_cmap("coolwarm"))
        plt.title(f"Thermodynamic Interaction: LTC Impact by Hour\n({lbl})", fontweight='bold', pad=20)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig07_shap_dependence_{sfx}.png')
        plt.savefig(p, dpi=300, bbox_inches='tight'); print(f"    Saved: {p}"); plt.show()

        # ── Plot 8: Error CDF ─────────────────────────────────────────────────
        print("  [8] Error CDF...")
        sorted_errors = np.sort(results['Absolute_Error'].values)
        cdf           = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        idx_90        = min(np.searchsorted(cdf, 0.90), len(sorted_errors) - 1)
        err_90        = sorted_errors[idx_90]
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_errors, cdf, color='#2ca02c', linewidth=3)
        plt.axhline(0.90, color='gray', linestyle='--', alpha=0.7)
        plt.axvline(err_90, ymin=0, ymax=0.90, color='gray', linestyle='--', alpha=0.7)
        plt.plot(err_90, 0.90, 'ro')
        plt.text(err_90 + 0.02, 0.86, f'90% < {err_90:.2f} kWh', color='red', fontweight='bold')
        plt.title(f'CDF of Absolute Error\n({lbl})', fontweight='bold', pad=15)
        plt.xlabel('Absolute Error (kWh)')
        plt.ylabel('Cumulative Probability')
        plt.xlim(0, np.percentile(sorted_errors, 99) * 1.1)
        plt.ylim(0, 1.05)
        plt.fill_between(sorted_errors, cdf, color='#2ca02c', alpha=0.2)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig08_error_cdf_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 9: Residual ACF ──────────────────────────────────────────────
        print("  [9] Residual ACF...")
        plt.figure(figsize=(10, 5))
        plot_acf(results['Residuals'], lags=72, ax=plt.gca(), alpha=0.05,
                 color='#1f77b4', vlines_kwargs={"colors": '#1f77b4'})
        plt.title(f'Autocorrelation of Prediction Residuals (72-Hour Horizon)\n({lbl})',
                  fontweight='bold', pad=15)
        plt.xlabel('Lag (Hours)')
        plt.ylabel('Autocorrelation')
        plt.ylim(-0.2, 1.1)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig09_residuals_acf_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 10: Weather Stratification Violin ────────────────────────────
        print("  [10] Weather Stratification Violin...")
        bins_w   = [-1, 15, 50, 85, 101]
        labs_w   = ['Clear Sky\n(0-15%)', 'Partly Cloudy\n(16-50%)',
                    'Mostly Cloudy\n(51-85%)', 'Overcast\n(86-100%)']
        res_copy = results.copy()
        res_copy['Weather_Type'] = pd.cut(res_copy['Cloud_Cover'], bins=bins_w, labels=labs_w)
        # violin always uses daylight-only data (physically meaningful)
        if sfx == '24hour':
            violin_data = res_copy[res_copy.index.isin(test_df.index[daylight_mask])]
        else:
            violin_data = res_copy
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Weather_Type', y='Residuals', data=violin_data,
                       palette='muted', inner='quartile', linewidth=1.5)
        plt.axhline(0, color='black', linestyle='--', linewidth=2)
        plt.title(f'Prediction Error Distribution by Weather State\n({lbl})', fontweight='bold', pad=15)
        plt.ylabel('Prediction Error ($y_{true} - y_{pred}$)')
        plt.xlabel('Meteorological State (Satellite Cloud Cover)')
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig10_weather_violin_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 11: Ramp Rate Hexbin ─────────────────────────────────────────
        print("  [11] Ramp Rate Volatility...")
        ramp = results.copy()
        ramp['Actual_Ramp']    = ramp['Actual'].diff()
        ramp['Predicted_Ramp'] = ramp['Predicted'].diff()
        ramp = ramp.dropna(subset=['Actual_Ramp', 'Predicted_Ramp'])
        plt.figure(figsize=(8, 8))
        hb = plt.hexbin(ramp['Actual_Ramp'], ramp['Predicted_Ramp'],
                        gridsize=40, cmap='inferno', bins='log', mincnt=1)
        rmin = max(ramp['Actual_Ramp'].min(), ramp['Predicted_Ramp'].min())
        rmax = min(ramp['Actual_Ramp'].max(), ramp['Predicted_Ramp'].max())
        plt.plot([rmin, rmax], [rmin, rmax], color='#00FF00', linestyle='--', linewidth=2.5, label='Ideal Tracking')
        plt.colorbar(hb, label='Log(Count) of Hourly Events')
        plt.title(f'Grid Volatility: Hourly Ramp Rate Tracking\n({lbl})', fontweight='bold', pad=15)
        plt.xlabel('Actual Hourly Power Change $\\Delta P$ (kWh)')
        plt.ylabel('Predicted Hourly Power Change $\\Delta P$ (kWh)')
        plt.legend(loc='upper left', facecolor='white', framealpha=1)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig11_ramp_rate_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plots 12–14: Individual Model Scatter Plots ───────────────────────
        print("  [12–14] Individual Scatter Plots (XGBoost, LightGBM, CatBoost)...")

        def plot_scatter(model_name, y_true_arr, y_pred_arr, filename):
            rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
            r2   = r2_score(y_true_arr, y_pred_arr)
            mae  = mean_absolute_error(y_true_arr, y_pred_arr)
            maxv = max(y_true_arr.max(), y_pred_arr.max()) * 1.02
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true_arr, y_pred_arr, alpha=0.3, color='#1f77b4', edgecolors='none', s=10)
            plt.plot([0, maxv], [0, maxv], 'k--', lw=2.5, label='Ideal Fit (y=x)')
            plt.title(f'{model_name}\n({lbl})', fontsize=13, fontweight='bold', pad=15)
            plt.xlabel('Actual Power Output (kWh)', fontsize=12)
            plt.ylabel('Predicted Power Output (kWh)', fontsize=12)
            plt.xlim(0, maxv); plt.ylim(0, maxv)
            plt.grid(True, linestyle=':', alpha=0.7)
            textstr = f'$R^2= {r2:.4f}$\n$RMSE= {rmse:.4f}$\n$MAE= {mae:.4f}$'
            plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
            plt.legend(loc='lower right', facecolor='white', framealpha=1)
            plt.tight_layout()
            sp = os.path.join(out_dir, filename)
            plt.savefig(sp, dpi=300, bbox_inches='tight')
            print(f"    Saved: {sp}")
            plt.show()

        plot_scatter("XGBoost Prediction Accuracy",              bm['y_true'], bm['preds_xgb'], f'fig12_scatter_xgboost_{sfx}.png')
        plot_scatter("LightGBM Prediction Accuracy",             bm['y_true'], bm['preds_lgb'], f'fig13_scatter_lightgbm_{sfx}.png')
        plot_scatter("CatBoost (Proposed Framework) Prediction", bm['y_true'], bm['preds_cat'], f'fig14_scatter_catboost_{sfx}.png')

        # ── Plot 15: Polar Clock Error Dial ───────────────────────────────────
        print("  [15] Polar Clock Error Dial...")
        hourly_err = results.groupby('Hour')['Absolute_Error'].mean()
        full_hours = hourly_err.reindex(range(24), fill_value=0).values
        angles     = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        plt.figure(figsize=(8, 8))
        ax_pol = plt.subplot(111, projection='polar')
        ax_pol.set_theta_offset(np.pi / 2)
        ax_pol.set_theta_direction(-1)
        ax_pol.fill(angles, full_hours, color='#ff7f0e', alpha=0.4)
        ax_pol.plot(angles, full_hours, color='#d62728', linewidth=2.5)
        ax_pol.set_xticks(angles)
        ax_pol.set_xticklabels([f"{h}:00" for h in range(24)], fontsize=9)
        ax_pol.set_rlabel_position(180)
        plt.title(f'Spatiotemporal Clock Dial: Error Across 24-Hour Cycle\n({lbl})',
                  fontweight='bold', pad=25, fontsize=12)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig15_polar_clock_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

        # ── Plot 16: Thermodynamic Hysteresis Loop ────────────────────────────
        print("  [16] Thermodynamic Hysteresis Loop...")
        hyst = results.groupby('Hour')[['T_amb', 'LTC']].mean()
        hyst = pd.concat([hyst, hyst.iloc[[0]]])
        hrs_loop = list(range(24)) + [0]
        x_h = hyst['T_amb'].values
        y_h = hyst['LTC'].values
        pts  = np.array([x_h, y_h]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm_h = plt.Normalize(0, 24)
        lc_map = cm.ScalarMappable(norm=norm_h, cmap='twilight_shifted')
        plt.figure(figsize=(9, 7))
        for i in range(len(segs)):
            plt.plot(segs[i][:, 0], segs[i][:, 1], color=lc_map.to_rgba(hrs_loop[i]), linewidth=4)
            if i % 3 == 0 and i < 23:
                plt.annotate('', xy=(x_h[i+1], y_h[i+1]), xytext=(x_h[i], y_h[i]),
                             arrowprops=dict(arrowstyle="->", color=lc_map.to_rgba(hrs_loop[i]), lw=3))
        for h in [8, 12, 16, 20]:
            if h < len(x_h):
                plt.text(x_h[h], y_h[h], f" {h}:00", fontsize=11, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        cbar = plt.colorbar(lc_map, ax=plt.gca(), pad=0.03)
        cbar.set_label('Hour of Day (Temporal Progression)', rotation=270, labelpad=20)
        plt.title(f'Thermodynamic Hysteresis Loop\n({lbl})', fontweight='bold', pad=15)
        plt.xlabel('Mean Ambient Temperature $T_{amb}$ (°C)')
        plt.ylabel('Physics-Informed Thermal State ($LTC$)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        p = os.path.join(out_dir, f'fig16_thermal_hysteresis_{sfx}.png')
        plt.savefig(p, dpi=300); print(f"    Saved: {p}"); plt.show()

    print(f"\n{'='*65}")
    print(f"  Done! 32 plots saved to: {os.path.abspath(out_dir)}")
    print(f"    • 16 plots with suffix _daylight  (GHI ≥ 0.05 kW/m²)")
    print(f"    • 16 plots with suffix _24hour    (all hours)")
    print(f"{'='*65}")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    print("Please select the 'model_ready_data.csv' file...")
    in_file_path = filedialog.askopenfilename(
        title="Select Model Ready CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    if in_file_path:
        generate_merged_visuals(in_file_path)
    else:
        print("No file selected. Exiting.")
        sys.exit()
