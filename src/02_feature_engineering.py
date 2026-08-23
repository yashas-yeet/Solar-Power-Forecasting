import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys

# ── Site Coordinates: DKASC Alice Springs ──────────────────────────────────────
LAT_DEG = -23.76
LON_DEG = 133.87

def compute_solar_geometry(df):
    """
    Computes deterministic Solar Zenith and Azimuth angles based on site
    coordinates and UTC timestamp. Uses standard astronomical declination
    and hour-angle equations — no external libraries required.
    """
    lat_rad = np.radians(LAT_DEG)
    day_of_year = df.index.dayofyear
    hour        = df.index.hour

    # Solar declination angle (degrees)
    declination_deg = -23.45 * np.cos(np.radians(360 / 365 * (day_of_year + 10)))
    declination_rad = np.radians(declination_deg)

    # Hour angle: 0 degrees at solar noon, +-15 degrees per hour offset
    hour_angle_rad = np.radians(15 * (hour - 12))

    # Zenith angle
    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination_rad) +
        np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    cos_zenith  = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad  = np.arccos(cos_zenith)
    zenith_deg  = np.degrees(zenith_rad)

    # Azimuth angle
    sin_azimuth = (-np.cos(declination_rad) * np.sin(hour_angle_rad)) / (np.sin(zenith_rad) + 1e-9)
    sin_azimuth = np.clip(sin_azimuth, -1.0, 1.0)
    azimuth_deg = np.degrees(np.arcsin(sin_azimuth))

    df = df.copy()
    df['Solar_Zenith']  = zenith_deg
    df['Solar_Azimuth'] = azimuth_deg
    return df


def engineer_features(input_path, output_path):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    # ── 1. Physics-Informed Feature: Lagged Temperature Coefficient (LTC) ──────
    # CRITICAL: adjust=False ensures strict backward calculation.
    # This prevents temporal leakage, satisfying rigorous peer-review standards.
    alpha_val = 0.15
    print(f"Calculating LTC with alpha={alpha_val}...")
    df['LTC'] = df['T_amb'].ewm(alpha=alpha_val, adjust=False).mean()

    # ── 2. Cyclical Temporal Encoding ──────────────────────────────────────────
    # Trees split poorly on raw integers because they don't know Hour 23 is
    # adjacent to Hour 0. Sine/cosine transforms fix this continuous cycle.
    print("Encoding temporal features cyclically (sin/cos)...")
    df['Hour_sin']        = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_cos']        = np.cos(2 * np.pi * df.index.hour / 24)
    df['DayOfYear_sin']   = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['DayOfYear_cos']   = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # Keep raw integer values too — gradient boosters can still use ordinal splits
    df['Hour']      = df.index.hour
    df['DayOfYear'] = df.index.dayofyear

    # ── 3. Autoregressive Lag Features ─────────────────────────────────────────
    # Short-term weather trends: if clouds were clearing an hour ago, the model
    # should know. Lags provide a temporal memory of incoming weather fronts.
    print("Generating 1h and 2h autoregressive lag features...")
    for col in ['GHI', 'DNI', 'DHI', 'Cloud_Cover']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)

    # ── 4. Rolling Window Statistics ────────────────────────────────────────────
    # Rolling mean smooths sensor noise; rolling std captures cloud variability
    # (high std = intermittent cloud cover, difficult to forecast accurately).
    print("Computing 3-hour rolling window statistics...")
    df['GHI_roll3_mean']   = df['GHI'].rolling(window=3, min_periods=1).mean()
    df['GHI_roll3_std']    = df['GHI'].rolling(window=3, min_periods=1).std().fillna(0)
    df['Cloud_roll3_mean'] = df['Cloud_Cover'].rolling(window=3, min_periods=1).mean()

    # ── 5. Interaction Term: GHI x T_amb ────────────────────────────────────────
    # PV panels lose efficiency as temperature rises (temperature coefficient).
    # This explicit cross-term saves the model from approximating it over many splits.
    print("Computing GHI x T_amb interaction term...")
    df['GHI_x_Tamb'] = df['GHI'] * df['T_amb']

    # ── 6. Deterministic Solar Geometry ─────────────────────────────────────────
    # Provides hard astronomical bounds on possible irradiance, decoupled from
    # cloud-cover-driven uncertainty.
    print("Computing Solar Zenith and Azimuth angles...")
    df = compute_solar_geometry(df)

    # ── 7. Final Feature Selection ──────────────────────────────────────────────
    required_cols = [
        # Raw irradiance channels
        'GHI', 'DNI', 'DHI',
        # Atmospheric state
        'T_amb', 'Cloud_Cover',
        # Physics-informed thermal proxy
        'LTC',
        # Raw temporal (ordinal splits in trees)
        'Hour', 'DayOfYear',
        # Cyclical temporal encodings
        'Hour_sin', 'Hour_cos', 'DayOfYear_sin', 'DayOfYear_cos',
        # Autoregressive lags
        'GHI_lag1', 'GHI_lag2',
        'DNI_lag1', 'DNI_lag2',
        'DHI_lag1', 'DHI_lag2',
        'Cloud_Cover_lag1', 'Cloud_Cover_lag2',
        # Rolling statistics
        'GHI_roll3_mean', 'GHI_roll3_std', 'Cloud_roll3_mean',
        # Non-linear interaction
        'GHI_x_Tamb',
        # Solar geometry
        'Solar_Zenith', 'Solar_Azimuth',
        # Target
        'P_act'
    ]

    try:
        df_final = df[required_cols].copy()
    except KeyError as e:
        print(f"ERROR: Missing expected column: {e}")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    # Lag operations introduce NaNs at the start of the series — drop those rows
    initial_len = len(df_final)
    df_final.dropna(inplace=True)
    rows_dropped = initial_len - len(df_final)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows from lag/rolling warm-up period.")

    # ── 8. Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path)

    n_features = len(required_cols) - 1  # Exclude target
    print(f"\nSuccess! Engineered dataset saved to {output_path}")
    print(f"  -> Features: {n_features} | Rows: {len(df_final):,}")
    print(f"  Ready for hybrid ensemble training.")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    print("Please select the 'cleaned_aligned_data.csv' file from the popup window...")

    in_file_path = filedialog.askopenfilename(
        title="Select Cleaned Data CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not in_file_path:
        print("No file selected. Exiting script.")
        sys.exit()

    out_file_path = 'data/processed/model_ready_data.csv'
    engineer_features(in_file_path, out_file_path)