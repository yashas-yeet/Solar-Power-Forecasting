# Source Code (`src/`)

This directory contains the core Python scripts that make up the pipeline of the solar forecasting project. The scripts are numbered sequentially to indicate the order of execution from raw data ingestion to advanced visualizations.

## Pipeline Overview

### 1. Data Processing & Feature Engineering
* **`01_data_cleaning.py`**: Handles the ingestion of raw DKASC telemetry and fetches historical atmospheric data via the Open-Meteo API. It aligns the data streams, downsamples 5-minute electrical telemetry to hourly blocks, and removes physical hardware anomalies (e.g., inverter faults).
* **`02_feature_engineering.py`**: Constructs the final predictive features used by the models, such as temporal variables (Hour, Day of Year) and localized corrections (Local Time Correction).

### 2. Modeling & Evaluation
* **`03_model_training.py`**: Trains the Day-Ahead Market (DAM) predictive models using advanced gradient boosting architectures (XGBoost, LightGBM, and CatBoost). It evaluates the models strictly during active daylight hours and outputs initial accuracy metrics (R², RMSE, MAE) and scatter comparisons.
* **`04_evaluation.py`**: Contains deeper evaluation routines and quantitative assessments to rigorously validate the models' predictive capabilities.

### 3. Diagnostics & Visualizations
* **`05_publication_visuals.py`**: Generates high-resolution, IEEE-formatted charts and figures suitable for academic and professional publication.
* **`06_advanced_diagnostics.py`**, **`07_final_diagnostics.py`**, **`08_extreme_diagnostics.py`**: A comprehensive suite of diagnostic scripts designed to probe model reliability. These scripts analyze error distributions, final performance benchmarks, and prediction stability under extreme or anomalous weather conditions.
* **`09_individual_scatters.py`**: Produces highly detailed, individual scatter plots of Actual vs. Predicted power output for each machine learning model.
* **`10_creative_plots.py`**: Generates supplementary exploratory graphics and novel visualizations to better communicate forecasting accuracy and data relationships.
* **`11_merged_visuals.py`**: A unified, high-performance program that consolidates all plotting logic from scripts 05 through 10. It trains the required models and calculates SHAP values exactly once to vastly improve execution speed while generating all 16 figures sequentially.

## Usage

To reproduce the study or run the pipeline from scratch, execute the scripts sequentially from `01` to `04`, and then you can either run scripts `05` to `10` individually or use the consolidated `11_merged_visuals.py` to generate all plots at once. Ensure that you have the necessary dependencies installed (e.g., `pandas`, `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `matplotlib`, `shap`) and an active internet connection for the initial data fetching script.
