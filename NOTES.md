# Working Notes

## Pipeline
- Data: ENTSO-E NL load (hourly, resampled from 15-min) + Open-Meteo weather.
- Splits: train 2019–2025, val Jan–Mar 2026, test Apr–Jun 2026 (test untouched until final eval).
- Features: `lag_24`, `lag_168`, `mean_24_24` (rolling mean, lag24-window24), calendar (hour/day_of_week/month/weekend, LightGBM native categoricals), holiday (`is_holiday` + `holiday_name`), weather (`apparent_temperature`, `dew_point_2m`).
- `feature_engineering.py`: fit transformers on train only, `.transform()` (never `.fit_transform()`) on val/test. Train's tail stitched onto val (and val's onto test) before transforming, so boundary rows keep real lookback instead of losing ~168h to NaN. `WARMUP_HOURS = 168` (max lookback across all lag features   confirmed empirically, not 191 as originally miscalculated).

## Key decisions
- Day-ahead framing, midnight forecast origin. `lag_1/2/3` dropped   not valid across a full next-day horizon (leak future data at that origin). `lag_24`/`lag_168` kept   valid across the whole horizon.
- Weather joined at target timestamp (not lagged)   it's exogenous, not autoregressive on the target, so "will there be weather info for the forecast hour" is the right question, not "would this leak." Currently uses **actual historical weather** (idealized-input experiment), not real NWP forecasts   documented limitation, not hidden.
- Categorical features: LightGBM native categorical handling (`.astype('category')` + `categorical_feature=[...]`), not cyclical/one-hot appropriate for tree models, would differ for linear/neural.
- `mean_24_24` tested for removal (looked low-value in feature importance)   actually cost ~0.2 MAPE points when removed. Reverted; low importance ≠ safe to cut.

## Results

| Stage | Val MAPE | Val MAE | Test MAPE | Test MAE |
|---|---|---|---|---|
| Naive (lag_24) | 5.45% | 824.2 |   |   |
| LightGBM (lag_24, calendar) | 3.51% | 537.9 |   |   |
| + lag_168 | 3.48% | 532.0 |   |   |
| + weather | 3.00% | 461.4 | **2.65%** | **345.8** |

- Test MAPE (2.65%) better than val: plausibly milder/lower-variance spring weather (Apr–Jun) vs. val's winter period (Jan–Mar).
- Industry comparison (point forecast): PJM ~1.9%, ISO-NE ~2.1%, ERCOT ~2.7% (real operational systems, real NWP forecasts, ensembling). Academic single-model benchmarks: Spain national ~3.33% (autoregressive), ~2.50% (NN). Test result (2.65%) sits credibly between these   close to operational range, ahead of comparable single-model academic work.
- Error concentrates in hours 9–17 (fastest-changing load); weather feature improved these hours most.

## Quantile regression + coverage

- Trained separate q0.1/q0.5/q0.9 LightGBM models (pinball loss). q0.5 tracks point model closely (sanity check passed).
- **Raw [q10,q90] coverage stuck at ~62–70%** (target 80%) across every tuning attempt: more boosting rounds, lower LR, more `num_leaves`, added weather. Ruled out via elimination   not underfitting, not model capacity, not (primarily) missing weather signal.
- Diagnosed as an information/structural ceiling on the upper tail, not a training bug.
- **Fix: Conformalized Quantile Regression (CQR)**: calibrated `Q` on val, applied (never refit) to test.
  - Val: raw 62–66% → calibrated 85.8% coverage (target 80%), interval +54.7% wider.
  - Test: raw 69.6% → calibrated 89.9% coverage, interval width 1333 MW (vs raw 856 MW).
  - CQR fixes the coverage *symptom*; doesn't fix the underlying model's tail miscalibration real cost is a much wider interval than the model's raw (wrong) confidence implied.
- Quantile crossing (q0.1 > q0.5): rare (~1–5% of rows), mostly small (tens of MW) traced to low-variance overnight hours where independent models' minor noise can flip order. One large exception: **King's Day** (largest-effect holiday, very few training examples of that hour/holiday combo) q0.1 under-weighted the holiday signal relative to q0.5. Legitimate argument for monotonic constraints or joint quantile training as future work, not urgent now.

## Open / future
- Real NWP forecast archive (vs. current idealized actual-historical weather) Open-Meteo's "Previous Runs API" is a viable source, coverage from ~2022 only.
- Model is frozen (trained once through 2025), not retrained during val/test walk-forward only feature inputs refresh daily. Retraining cadence = future work.
- Monotonic constraints / joint quantile training to reduce crossing on rare-event rows (King's Day case).
- Deployment: daily ingestion + prediction via GitHub Actions, results surfaced on a small site/dashboard not yet built.
- Portfolio site: separate 4-project showcase planned (Rossmann, Serengeti, DialogBench, this project) + a standalone writing/blog section.