Thesis Plot Pack
================

This folder contains publication-style figures generated from evaluate_all_from_parquets.py outputs.

Folders:
- dashboards : one compact RMSE summary figure per task/lookback/horizon
- overall    : overall model comparison bar charts
- by_horizon : how error changes as the forecast goes further into the future
- by_target  : model comparison for each target variable
- heatmaps   : target × forecast-step error maps for each model
- ratios     : persistence-baseline comparison plots (when summary CSVs exist)

Interpretation rules:
- Lower RMSE, MAE, and sMAPE are better.
- RMSE_ratio below 1.0 means the model beats the persistence baseline.
- RMSE_gain above 0.0 means the model improves over the persistence baseline.

Files created:
- dashboards: 18
- overall: 54
- by_horizon: 54
- by_target: 54
- heatmaps: 54
- ratios: 36