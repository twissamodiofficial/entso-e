# NL Day-Ahead Load Forecasting

Day-ahead electricity load forecasting for the Netherlands, using public ENTSO-E data, with both a point forecast and a calibrated probabilistic (quantile) forecast.

## Problem

- **Target**: hourly NL electricity load (MW).
- **Task**: day-ahead   given data available at forecast origin (midnight), predict the next 24 hours.
- Public load-forecasting examples are common but usually stop at a point-forecast LSTM/Prophet with no probabilistic angle and little attention to whether the features used would actually be available at real forecast time. This project focuses on doing the leak-free, probabilistic version properly.

## Data

- Source: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) (load) + [Open-Meteo](https://open-meteo.com/) (weather).
- Hourly resolution (resampled from 15-min).
- Chronological splits: train 2019–2025, val Jan–Mar 2026, test Apr–Jun 2026.

## Pipeline

- `ingest_data.py` / `ingest_weather.py`   pull raw data into train/val/test CSVs.
- `feature_engineering.py`   lag/rolling features, calendar, holiday, and weather features; transformers fit on train only, applied (not refit) to val/test.
- `train_lightgbm.py`   point-forecast model.
- `train_quantile.py`   q0.1/q0.5/q0.9 probabilistic models.
- `cqr.py`   conformal calibration to fix prediction-interval coverage.
- `evaluate_test.py`   final held-out evaluation.

## Results

| | Val | Test |
|---|---|---|
| Point forecast MAPE | 3.00% | **2.65%** |
| Point forecast MAE | 461 MW | 346 MW |
| [q10, q90] coverage, raw | ~62% | ~70% |
| [q10, q90] coverage, calibrated | 85.8% | 89.9% |

Target coverage is 80%. Comparable operational systems (PJM, ISO-NE, ERCOT) run ~1.9–2.7% MAPE with real weather forecasts and ensembling; this project's 2.65% test MAPE, using a single model and idealized (actual-historical, not forecast) weather, is competitive with that range and ahead of comparable single-model academic benchmarks. Full reasoning and experiment log in `NOTES.md`.

## Notable design points

- **Leakage-checked feature engineering**: short lags (1–3h) are invalid for a full day-ahead horizon and are excluded; only lags that stay valid across the entire next-day horizon are used.
- **Quantile miscalibration, diagnosed and fixed**: raw prediction intervals undercovered (~62–70% vs. target 80%) even after ruling out training and feature causes. Fixed with Conformalized Quantile Regression (CQR), calibrated on validation and verified on a fully held-out test set.
- **Weather is currently idealized** (actual historical values, not a real forecast)   a documented simplification, not a hidden one. See `NOTES.md` for the reasoning and the path to a more operationally realistic version.

## Future work

- Real NWP-forecast weather (vs. current idealized actual-historical).
- Periodic model retraining (currently frozen after initial training).
- Deployment: daily ingestion + forecast refresh via a scheduled pipeline.