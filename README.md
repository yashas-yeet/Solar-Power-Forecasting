# Physics-Informed Virtual Sensor Framework for Solar Power Forecasting

This repository contains the complete machine learning pipeline for the Physics-Informed Virtual Sensor Framework. The codebase is designed to ingest open-access satellite meteorological data (Open-Meteo), engineer localized solar geometry and thermal mass features, train boundary-constrained Gradient Boosting models, and generate publication-ready diagnostics.

## 📂 Repository Structure & Pipeline

The workflow is strictly sequential, divided into 10 modular Python scripts. Execution should follow the numerical order provided below.

| Step | Script Name | Primary Function |
| :--- | :--- | :--- |
| **01** | `01_data_cleaning.py` | Ingests raw satellite and ground sensor telemetry; handles missing values, outlier removal, and timestamp alignment. |
| **02** | `02_feature_engineering.py` | Computes analytical solar geometry (Zenith, Azimuth) and empirical thermal mass proxies (EWMA) to append physical constraints to the feature space. |
| **03** | `03_model_training.py` | Executes the core training loop for the ensemble gradient boosting models (XGBoost, LightGBM, CatBoost) with physics-informed bounding penalties. |
| **04** | `04_evaluation.py` | Conducts blind hold-out validation and computes standard operational metrics (RMSE, MAE, R-squared) against deterministic physical baselines. |
| **05** | `05_publication_visuals.py` | Generates high-resolution (300 DPI), IEEE-formatted static graphs, including the primary forecasting time-series comparisons. |
| **06** | `06_advanced_diagnostics.py` | Computes and plots feature importance matrices and SHAP (SHapley Additive exPlanations) values to interpret model decisions. |
| **07** | `07_final_diagnostics.py` | Analyzes overall residual distributions to check for heteroscedasticity and systemic biases. |
| **08** | `08_extreme_diagnostics.py` | Isolates model performance during severe weather anomalies, telemetry dropouts, and non-clear-sky events. |
| **09** | `09_individual_scatters.py` | Produces scatter mapping of Predicted vs. Actual power output for granular accuracy validation. |
| **10** | `10_creative_plots.py` | Generates specialized visualizations, such as the literature mapping radar chart and citation networks, for academic presentation. |



## 🧮 Mathematical Framework & Physics Constraints

The framework supplements standard meteorological features with deterministic mathematical models of solar geometry and heat retention. This embeds physical constraints into the tree-based models, significantly reducing non-physical predictions.

### 1. Deterministic Solar Geometry

To give the model a hard astronomical bound decoupled from cloud cover uncertainty, we compute the Solar Zenith ($\\theta_z$) and Solar Azimuth ($\\gamma_s$) angles.

**Solar Declination ($\\delta$)**
$$ \\delta = -23.45^\\circ \\cdot \\cos\\left( \\frac{360^\\circ}{365} (d + 10) \\right) $$
*(Where $d$ is the day of the year)*

**Hour Angle ($h$)**
$$ h = 15^\\circ \\cdot (t - 12) $$
*(Where $t$ is the hour of the day)*

**Solar Zenith Angle ($\\theta_z$)**
$$ \\cos(\\theta_z) = \\sin(\\phi)\\sin(\\delta) + \\cos(\\phi)\\cos(\\delta)\\cos(h) $$
*(Where $\\phi$ is the site latitude)*

**Solar Azimuth Angle ($\\gamma_s$)**
$$ \\sin(\\gamma_s) = \\frac{-\\cos(\\delta)\\sin(h)}{\\sin(\\theta_z)} $$

### 2. Lagged Temperature Coefficient (LTC)

Photovoltaic panel efficiency drops as the temperature rises. Because panels retain heat, current output is a function of *past* temperatures as well as current ones. We model this thermal mass using an Exponentially Weighted Moving Average (EWMA) of the ambient temperature ($T_{amb}$):

$$ LTC_t = \\alpha \\cdot T_{amb, t} + (1 - \\alpha) \\cdot LTC_{t-1} $$

We employ a strict backward calculation ($\\alpha = 0.15$) to prevent temporal leakage, satisfying rigorous peer-review standards.

### 3. Physics-Informed Boundary Constraints

Tree-based models can occasionally predict small positive power outputs during the night due to latent heat or residual feature correlations. We enforce a hard physical boundary constraint:

$$ P_{pred\\_final} = \\begin{cases} 0 & \\text{if } GHI < 0.05 \\text{ kW/m}^2 \\\\ P_{pred} & \\text{otherwise} \\end{cases} $$

This deterministic override guarantees absolute zero error during nocturnal periods, strictly aligning the model's output with the physical reality of solar energy generation.


## 👨‍💻 Credits & Acknowledgments

**Author & Lead Researcher:** 
* **Yashas Vishwakarma** (Yashas Anand Kumar Vishwakarma)

**Associated Research:** 
This codebase serves as the empirical foundation for the IEEE research paper: 
*"A Physics-Informed Virtual Sensor Framework for Satellite-Based Solar Power Forecasting Using Gradient Boosting."*

**Data Source Acknowledgments:**
* **Open-Meteo API:** For providing the open-access satellite meteorological and historical weather grids utilized in the spatial feature engineering.
* **DKASC (Desert Knowledge Australia Solar Centre):** For providing the raw ground-sensor electrical telemetry used to calibrate the virtual pyranometer baselines.

---

## ⚖️ Dual Licensing Notice

This source code is protected by copyright law and is available under two distinct licensing models. You may choose to use it under:

**1. OPEN SOURCE (GPLv3):**
Free for academic, personal, and open-source projects.
*Condition:* If you distribute software using this code, your ENTIRE project must also be open-source under the GPLv3 license.

**2. COMMERCIAL LICENSE:**
Required for proprietary (closed-source) commercial products. This model allows you to keep your source code private and provides legal support.

*Full terms are available in the `LICENSE` file.*

---

## 📫 Contact & Support

For academic inquiries, collaboration requests, or commercial licensing, please reach out via:
* **Email:** [yashasakvish@gmail.com](mailto:yashasakvish@gmail.com)

**Copyright (c) 2025 Yashas Vishwakarma. All Rights Reserved.**


---

## ⚙️ Installation & Requirements

Ensure you have Python 3.9+ installed. The pipeline requires standard scientific computing and machine learning libraries.

Install the required dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn
```
