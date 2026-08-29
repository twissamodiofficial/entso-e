# NL Day-Ahead Load Forecasting

Day-ahead electricity load forecasting for the Netherlands, using public ENTSO-E data, with both a point forecast and a calibrated probabilistic (quantile) forecast.

## Problem

- **Target**: hourly NL electricity load (MW).
- **Task**: day-ahead, given data available at forecast origin (midnight), predict the next 24 hours.
- Public load-forecasting examples are common but usually stop at a point-forecast LSTM/Prophet with no probabilistic angle and little attention to whether the features used would actually be available at real forecast time. This project focuses on doing the leak-free, probabilistic version properly.

## Data

- Source: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) (load) + [Open-Meteo](https://open-meteo.com/) (weather).
- Hourly resolution (resampled from 15-min).
- Chronological splits: train 2019–2025, val Jan–Mar 2026, test Apr–Jun 2026.

## Pipeline

**Batch (historical, run once / rerun to extend history):**
- `scripts/run_batch_backfill.py`, pulls raw load (`ingestion/load.py`) + weather (`ingestion/weather.py`) into `data/raw/*.csv` for the fixed train/val/test ranges in `config.SPLITS`. Defines the frozen dataset the model is trained/evaluated against.
- `scripts/run_training.py`, runs backfill (unless `--skip-ingest`), builds features (`features/engineering.py`, lag/rolling/calendar/holiday/weather features, transformers fit on train only, applied not refit to val/test), trains the point model (`modeling/point.py`) and quantile models (`modeling/quantile.py`), calibrates (`modeling/calibration.py`, CQR), and logs everything to MLflow via `registry.py`.
- `scripts/run_test_eval.py`, final held-out evaluation on the untouched test split.

**Live (daily, scheduled via GitHub Actions):**
- `scripts/bootstrap_live_store.py`, one-time (watermark-driven, safe to rerun) catch-up of Supabase from the end of the test split through yesterday. Run this once before enabling the daily workflows, so `run_daily_ingest.py`'s first run already has enough history for lag features.
- `scripts/run_daily_ingest.py`, pulls *yesterday's* actual load + weather into Supabase, and scores yesterday's forecast against it. Scheduled via `.github/workflows/daily_ingest.yml` (note: GitHub's scheduled-workflow runner queue can delay the actual run time by hours vs. the cron target).
- `scripts/run_predict.py`, pulls the latest model from the registry, builds next-24h features (`engineering.transform_future`), and writes the new forecast to Supabase. Triggered via `.github/workflows/daily_predict.yml` on successful ingest.
- `scripts/sanity_check.py`, cross-checks the storage layer (server-side vs. client-side filtering) to catch silent wrong-data bugs, plus timestamp-alignment and recent-day-completeness checks.
- `scripts/check_live_accuracy.py`, reviews recent live scoring (`daily_metrics` table, written by `run_daily_ingest.py`'s scoring step).

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
- **Weather is currently idealized** (actual historical values, not a real forecast), a documented simplification, not a hidden one. See `NOTES.md` for the reasoning and the path to a more operationally realistic version.
- **Train/serve feature parity**: live prediction (`engineering.transform_future`) shares the same feature-construction code path as train/val/test, so categorical dtypes and columns are guaranteed to match. No separate/duplicated feature logic for live inference.

## Future work

- Real NWP-forecast weather (vs. current idealized actual-historical).
- Periodic model retraining (currently frozen after initial training).
- Live accuracy dashboard on top of `daily_metrics`.