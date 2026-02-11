# Efficient Hybrid Learning-and-Reasoning for Multi-Horizon Multivariate Time-Series Forecasting under Resource Constraints: A Comparative Evaluation of dCeNN–ELM–ASP approach with LSTM and CNN on Weather and Energy Data

A comprehensive thesis research project implementing advanced time series forecasting for **weather features** and **energy features** using a hybrid deep learning and symbolic AI pipeline. The approach combines **dCeNN (Discrete Cellular Neural Networks)**, **ELM (Extreme Learning Machines)**, and **ASP (Answer Set Programming)** for constraint-based optimization, benchmarked against baseline models including **LSTM** and **CNN**.

## Overview

This repository presents an integrated forecasting framework designed for:
- **Multi-target multivariate forecasting** (wind generation, solar generation, load, weather variables)
- **Flexible temporal horizons** (12h, 24h, 72h ahead predictions)
- **Variable context windows** (lookback periods of 24h, 72h, 168h)
- **ASP-based constraint repair** for physics-aware and business-rule-compliant predictions
- **Rigorous benchmarking** across accuracy, computational efficiency, and inference latency metrics

## Architecture

### Core Components

1. **Data Pipeline** (`src/dataio/`)
   - Preprocessing and temporal feature engineering
   - Train/validation/test splitting with strict UTC boundaries
   - Windowing and scaling without data leakage

2. **Neural Models** 
   - **dCeNN (Discrete Cellular Neural Networks)**: Advanced cellular automata-based neural architecture for spatiotemporal pattern recognition
   - **ELM (Extreme Learning Machines)**: Fast, single-pass learning with random projections for efficient predictions
   - **Baseline implementations**: LSTM (Seq2Seq), CNN for comparison

3. **Optimization Layer** (`scripts/run_asp.py`)
   - Answer Set Programming (Clingo solver) for post-processing
   - Constraint-based repair of raw predictions
   - Physical constraints (e.g., renewable generation capacity bounds)
   - Business rules (e.g., night-time solar generation = 0)

4. **Benchmarking & Visualization**
   - Comprehensive metrics: MAE, RMSE, sMAPE with baseline ratios
   - Resource tracking: CPU time, memory, model parameters, inference latency
   - Pareto frontier analysis and model ranking

## Datasets

- **Weather Forecasting**: Historical weather features (radiation, temperature, humidity, wind speed, pressure, precipitation)
- **Energy Forecasting**: Wind/solar capacity factors and load demand

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

#### Step 1: Train & Predict (ELM / dCeNN)
```bash
# Example: Weather task, 24-hour lookback, 12-hour horizon
python scripts/run_lstm_baseline.py --lookback 24
# or run full pipeline with various configurations
```

#### Step 2: Apply ASP Post-Processing
```bash
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 24 --horizon 12
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 24 --horizon 24
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 24 --horizon 72
```

#### Step 3: Visualization
```bash
# Plot Pareto frontier (accuracy vs. latency trade-off)
python scripts/plot_pareto.py

# Generate benchmark tables
python scripts/make_benchmark_table.py

# Rank and visualize models
python scripts/rank_and_plot_benchmarks.py

# Plot weekly predictions with ground truth
python scripts/viz_results.py --config configs/weather_full.yaml

# Weather meteogram visualization
python scripts/viz_weather_meteogram.py --config configs/weather_full.yaml --date 2022-08-01
```

## Directory Structure

```
thesis_forecasting/
├── src/                           # Core library code
│   ├── config.py                  # Configuration loader
│   ├── dataio/                    # Data loading & preprocessing
│   │   ├── preprocess.py          # Train/val/test split, feature engineering
│   │   └── window.py              # Temporal windowing
│   └── inference/                 # Prediction pipelines
│       └── predict.py             # Inference runner
│
├── scripts/                       # Standalone execution scripts
│   ├── run_lstm_baseline.py       # LSTM baseline training & benchmarking
│   ├── run_asp.py                 # ASP post-processing for weather
│   ├── run_energy_asp.py          # ASP post-processing for energy
│   ├── predict.py                 # Standalone inference
│   ├── plot_pareto.py             # Pareto frontier visualization
│   ├── make_benchmark_table.py    # Generate summary tables
│   ├─��� rank_and_plot_benchmarks.py # Model ranking & scoring
│   ├── viz_results.py             # Weekly forecast visualization
│   └── viz_weather_meteogram.py   # Weather variable time series plots
│
├── configs/                       # YAML configuration files
│   ├── weather_full.yaml          # Weather forecasting config
│   └── energy_full.yaml           # Energy forecasting config
│
├── data/                          # Input data (not included; prepare locally)
├── checkpoints/                   # Trained model weights
├── outputs/                       # Raw predictions & intermediate files
├── outputs_weather_full/          # Weather task outputs
├── outputs_energy_full/           # Energy task outputs
├── artifacts_lstm_baseline/       # LSTM baseline results
├── artifacts_cnn_baseline/        # CNN baseline results
├── plots/                         # Generated visualizations
│
├── benchmarks_master.csv          # Full benchmark results (all configs)
├── benchmarks_scored.csv          # Scored & ranked results
├── benchmark_table_rmse_ratio.csv # Accuracy comparison table
├── benchmark_table_latency_ms.csv # Inference latency comparison
├── benchmark_winners_rmse_ratio.csv # Best model per setting
│
├── model_ranking_overall.csv      # Overall model ranking
├── model_ranking_with_compute_latency.csv
├── model_means_scatter.png        # Model performance scatter plot
├── model_ranking_bar.png          # Ranking bar chart
├── rank_overall.png               # Overall ranking visualization
├── rank_compute_only.png          # Computation-only ranking
├── rank_latency.png               # Latency-only ranking
├── rank_compute_latency.png       # Combined metrics ranking
├── benchmark_pareto_all.png       # Pareto frontier plot
├── pareto_points.png              # Detailed Pareto visualization
│
└── requirements.txt               # Python dependencies
```

## Key Results

### Benchmarking Framework
- **Accuracy Metrics**: RMSE ratio (relative to persistence baseline), MAE, sMAPE
- **Efficiency Metrics**: 
  - Training: CPU time, peak RAM, parameter count
  - Inference: latency per sample (ms), model size (MB)
  - Compute-aware scoring across multiple dimensions

### Model Comparison
Results stored in `benchmarks_master.csv` and visualized across:
- **Pareto frontier**: Accuracy vs. inference latency trade-off
- **Model rankings**: Overall, computation-only, latency-only scores
- **Per-task winners**: Best model for each (task, lookback, horizon) configuration

### Supported Grid
- **Tasks**: Weather, Energy
- **Lookback Windows**: 24h, 72h, 168h (1 week)
- **Forecast Horizons**: 12h, 24h, 72h (3 days)
- **Models**: dCeNN, ELM, LSTM baseline, CNN baseline

## Configuration

Edit `configs/weather_full.yaml` or `configs/energy_full.yaml` to customize:
- Feature selection & engineering
- Train/val/test date ranges
- Model hyperparameters
- ASP constraint rules (night hours for solar, capacity bounds, etc.)
- Output directories

Example:
```yaml
features:
  context_hours: 24         # Lookback window
  horizon_hours: 12         # Forecast horizon

asp:
  pv_night_hours: [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]  # Solar = 0 at night
```

## Metrics & Evaluation

### Accuracy
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)
- **RMSE Ratio**: Normalized by persistence baseline

### Efficiency
- **Latency**: Milliseconds per sample (inference only)
- **Model Size**: Megabytes on disk
- **Training Time**: Wall-clock and CPU seconds
- **Peak Memory**: RAM usage during training
- **Parameter Count**: Total & deployment parameters

### Pareto Analysis
- Identifies non-dominated models (best on multiple objectives)
- Visualization in scatter plots and bar charts

## Use Cases

1. **Renewable Energy Integration**: Wind/solar forecasting for grid operations
2. **Load Forecasting**: Electricity demand prediction for grid planning
3. **Weather-Driven Analytics**: Fine-grained weather forecasting for operational decisions
4. **Constraint-Aware ML**: Learning with physics and business rule constraints via ASP

## ASP Post-Processing

The ASP pipeline applies logical constraints to refine raw neural predictions:

**Example facts** (generated from raw predictions):
```prolog
pred(solar, sample_123, horizon_1, 50).      % Raw prediction: 50 units
night(sample_123, horizon_1).                 % Night time detected
```

**Constraint repair logic**:
```prolog
% If solar forecast during night, repair to 0
repair(solar, sample, S, H) :- 
    pred(solar, S, H, X), X > 0, night(S, H).
```

**Repair statistics** tracked:
- Total repairs applied
- Cells changed
- Mean/max adjustment magnitude
- Repairs by kind & target

## Discrete Cellular Neural Networks (dCeNN)

dCeNN is an advanced neural architecture that leverages cellular automata principles for spatiotemporal forecasting:
- **Cellular Structure**: Operates on discrete grid-based representations of temporal sequences
- **Local Interactions**: Neurons update based on neighbors, enabling efficient parallel computation
- **Spatiotemporal Patterns**: Captures complex temporal dependencies through local update rules
- **Computational Efficiency**: Lower memory footprint and inference latency compared to traditional RNNs/CNNs

## Dependencies

- **Data Processing**: pandas, numpy
- **ML Frameworks**: TensorFlow/Keras (for LSTM, CNN, dCeNN)
- **ASP Solver**: Clingo (Answer Set Programming)
- **Utilities**: scikit-learn, matplotlib, seaborn, PyYAML, psutil

See `requirements.txt` for exact versions.

## Citation

If you use this work in research, please cite the associated thesis:

```bibtex
@thesis{binura_thesis_2024,
  author={Binura Widanagama},
  title={Advanced Time Series Forecasting with Hybrid Neural and Symbolic AI},
  year={2024}
}
```

## Author

**Binura Widanagama**  
University Of Klagenfurt

---

## Contact & Support

For questions, issues, or contributions:
- Email: biwidanagama@edu.aau.at
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

**Last Updated**: February 11, 2026
