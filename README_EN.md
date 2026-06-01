# Displacement Prediction and Early Warning of Reservoir Landslides Using Machine Learning — A Case Study of the Outang Landslide

Undergraduate thesis project on landslide displacement prediction and early warning for the Outang landslide in the Three Gorges Reservoir area.

[中文版本](README.md) | **Download Thesis PDF**: [Wei_2026_ML_Reservoir_Landslide_Prediction.pdf](https://github.com/wcqqq1214/Undergraduate-Thesis/releases/latest/download/Wei_2026_ML_Reservoir_Landslide_Prediction.pdf)

> **Note**: The released paper in the Releases page was written in Microsoft Word and is nearly identical to the LaTeX source (`docs/latex/`) in content, with only minor typesetting differences.

## Research Objective

This study investigates the Outang landslide, a typical step-like reservoir landslide in the Three Gorges Reservoir area, aiming to:

1. Quantify the contribution of triggering factors (reservoir water level, rainfall, groundwater level) to landslide displacement
2. Build machine learning and deep learning models for displacement prediction
3. Develop a probabilistic early warning framework based on predicted displacement rates

## Dataset

The dataset consists of long-term field monitoring records from the Outang landslide, including:

| Data Type | Description |
|-----------|-------------|
| **Surface displacement** | Cumulative displacement (mm) measured at multiple GNSS surface displacement monitoring points (MJ9, MJ1, MJ3), covering both slow-creep and step-like deformation phases |
| **Reservoir water level** | Daily water level (m) of the Three Gorges Reservoir, varying seasonally between 145–175 m |
| **Rainfall** | Daily precipitation (mm) recorded at nearby meteorological stations |
| **Groundwater level** | Groundwater table elevation (m) at boreholes within the landslide area |

The monitoring period spans from July 2016 to June 2020, covering 4 complete hydrological years with 1,457 daily records, capturing multiple reservoir operation cycles and several distinct step-like acceleration events.

## Methodology

### Feature Engineering and Interpretability (LightGBM + SHAP)

- **LightGBM** gradient boosting models are trained to predict displacement increments from hydrological triggers
- Both regression (displacement increment) and classification (warning state) targets are modelled
- **SHAP** (SHapley Additive exPlanations) values are computed to quantify the marginal contribution of each triggering factor
- SHAP dependence plots reveal non-linear, threshold-like responses — for example, displacement accelerates when the reservoir water level drops below a critical elevation and when cumulative rainfall exceeds certain thresholds

### LSTM Time-Series Displacement Prediction

- A multi-input **LSTM** (Long Short-Term Memory) network predicts daily displacement at three monitoring points simultaneously
- Input features are the historical cumulative displacement time series from four monitoring points (MJ9, MJ1, MJ3, ATU4), leveraging spatial kinematic correlations across the landslide body
- **50 independent training runs** with random weight initialization are performed to capture model (epistemic) uncertainty
- The ensemble of predictions forms a predictive distribution, from which **50% and 90% confidence intervals** are constructed

### V₀ Threshold-Based Four-Level Probabilistic Early Warning

- A baseline deformation rate $V_0 = 1.5\bar{V} + 2\sigma$ is computed from the constant-creep phase for each monitoring point, where $\bar{V}$ and $\sigma$ are the mean and standard deviation of displacement rates during that phase
- Three warning thresholds are set at **$V_0$**, **$5V_0$**, and **$10V_0$**, defining four levels:
  - **Green**: normal (rate < $V_0$)
  - **Yellow**: attention (rate ∈ [$V_0$, $5V_0$))
  - **Orange**: alert (rate ∈ [$5V_0$, $10V_0$))
  - **Red**: alarm (rate ≥ $10V_0$)
- For each day, the 50 LSTM predictions of monthly displacement rate are mapped to these four levels; the level with the highest probability is taken as the day's warning status

## Key Results

| Metric | Value |
|--------|-------|
| **LSTM test R²** (MJ1) | 0.7916 |
| **LSTM test RMSE** (MJ1) | 5.55 mm |
| **Relative error** (MJ1) | 0.56% |
| **Specificity** (all 3 monitoring points) | 100% |
| **False alarm rate** (all 3 monitoring points) | 0% |

- The SHAP analysis identified reservoir water level drawdown and cumulative rainfall as the dominant triggers of step-like displacement
- The LSTM ensemble effectively captured both slow-creep and sudden acceleration phases, with narrow confidence intervals during stable periods and wider intervals during step-like events
- During the 2017 step-like deformation event, the early warning system triggered **70-day yellow alerts** at MJ1 and **129-day yellow alerts** at MJ3, demonstrating timely hazard detection without false alarms

## Project Structure

```
├── code/                        # Experiment code
│   ├── chapter2/                # Ch.2: LightGBM + SHAP feature analysis
│   ├── chapter3/                # Ch.3: LSTM displacement prediction
│   ├── chapter4/                # Ch.4: V₀ probabilistic early warning
│   ├── scripts/                 # Shared utilities
│   └── utils/                   # Helper tools
├── data/                        # Raw monitoring data
├── docs/
│   └── latex/                   # LaTeX source files
│       ├── figures/             # Figures (by chapter)
│       ├── main.tex             # Thesis manuscript
│       ├── references.bib       # Bibliography
│       └── slides.tex           # Defense slides (beamer)
├── scripts/                     # Project-level utilities
├── .gitignore
├── LICENSE
├── README.md
└── README_EN.md
```

## Tech Stack

- **Language**: Python 3.x
- **Deep Learning**: PyTorch
- **ML Libraries**: scikit-learn, LightGBM
- **Interpretability**: SHAP
- **Time-Series**: LSTM
- **Typesetting**: LaTeX (XeTeX)

## Build

```bash
cd docs/latex
latexmk -xelatex main.tex    # Thesis
latexmk -xelatex slides.tex  # Defense slides
```

## License

- **Code** (`/code`, `/scripts`): [MIT License](LICENSE)
- **Thesis Text & Figures** (`/docs`): [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
