import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import tkinter as tk
from tkinter import filedialog
import sys

def fetch_open_meteo_data(start_date, end_date, lat=-23.76, lon=133.87):
    print(f"Fetching Open-Meteo data from {start_date} to {end_date}...")
    
    # Setup robust session with retry logic
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
        "timezone": "Australia/Darwin" # Explicit timezone alignment
    }
    
    response = session.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
    data = response.json()
    
    # Reconstruct the satellite data into a DataFrame
    df_sat = pd.DataFrame(data['hourly'])
    df_sat['time'] = pd.to_datetime(df_sat['time'])
    
    # Ensure the dataframe is explicitly timezone-naive to match standard CSV parsing
    df_sat['time'] = df_sat['time'].dt.tz_localize(None)
    df_sat.set_index('time', inplace=True)
    
    df_sat.rename(columns={
        'temperature_2m': 'T_amb',
        'relative_humidity_2m': 'RH',
        'cloud_cover': 'Cloud_Cover',
        'shortwave_radiation': 'GHI',
        'direct_normal_irradiance': 'DNI',
        'diffuse_radiation': 'DHI'
    }, inplace=True)
    
    # Convert W/m^2 to kW/m^2
    df_sat['GHI'] = df_sat['GHI'] / 1000.0
    df_sat['DNI'] = df_sat['DNI'] / 1000.0
    df_sat['DHI'] = df_sat['DHI'] / 1000.0
    
    return df_sat

def process_and_align_data(raw_csv_path, output_path):
    print("Loading raw DKASC telemetry...")
    
    # 1. Load the raw data, explicitly parsing the timestamp format
    # The format in your snippet is DD-MM-YYYY HH:MM
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], format='%d-%m-%Y %H:%M')
    df_raw.set_index('timestamp', inplace=True)
    
    # 2. Isolate target variable and handle strings/missing values
    # We only keep Active_Power and force any weird characters to NaN, then fill with 0
    df_power = df_raw[['Active_Power']].copy()
    df_power['Active_Power'] = pd.to_numeric(df_power['Active_Power'], errors='coerce').fillna(0)
    
    # 3. Downsample 5-minute data to 1-hour averages
    print("Downsampling 5-minute electrical telemetry to hourly blocks...")
    df_power_hourly = df_power.resample('1H').mean()
    df_power_hourly.rename(columns={'Active_Power': 'P_act'}, inplace=True)
    
    # 4. Determine date ranges for the API call
    # Extracting standard YYYY-MM-DD strings for Open-Meteo
    start_date_str = df_power_hourly.index.min().strftime('%Y-%m-%d')
    end_date_str = df_power_hourly.index.max().strftime('%Y-%m-%d')
    
    # 5. Fetch corresponding satellite data
    df_satellite = fetch_open_meteo_data(start_date=start_date_str, end_date=end_date_str)
    
    # 6. Merge datasets
    print("Aligning physical and atmospheric data streams...")
    # Using an inner join ensures we only keep hours where both sources have data
    df_merged = pd.merge(df_satellite, df_power_hourly, left_index=True, right_index=True, how='inner')
    
    # 7. Hardware Anomaly Filtering
    # Drop instances where GHI > 0.4 kW/m^2 but Actual Power < 0.2 kW (Inverter Faults)
    initial_len = len(df_merged)
    anomaly_mask = (df_merged['GHI'] > 0.4) & (df_merged['P_act'] < 0.2)
    df_clean = df_merged[~anomaly_mask].copy()
    
    dropped_rows = initial_len - len(df_clean)
    print(f"Dropped {dropped_rows} unphysical anomaly records.")
    
    # 8. Save the final aligned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path)
    print(f"Success. Cleaned and aligned dataset saved to: {output_path}")

if __name__ == "__main__":
    # Hide the main, empty tkinter window
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the raw DKASC CSV file from the popup window...")
    
    # Open the file explorer dialog
    raw_file_path = filedialog.askopenfilename(
        title="Select Raw DKASC Telemetry CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    
    # Check if the user clicked 'Cancel'
    if not raw_file_path:
        print("No file selected. Exiting script.")
        sys.exit()
        
    print(f"Selected File: {raw_file_path}")
    
    # Define where the output goes
    out_file_path = 'data/interim/cleaned_aligned_data.csv'
    
    # Run the pipeline
    process_and_align_data(raw_file_path, out_file_path)