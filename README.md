# Netherlands load forecasting

The pipeline has separate stages: ingestion → preprocessing → feature
engineering → final validation → training/prediction. Load/weather adapters,
backfill, preprocessing, point/quantile models, conformal interval calibration,
historical training/reporting, and the live forecast/reconciliation commands
are implemented.
No trained model is included.

See [Pipeline flow and timezone boundaries](PIPELINE_FLOW.md) for the data flow,
UTC/Amsterdam conversions, and DST examples.

The read-only dashboard lives in [dashboard/](dashboard/). Deploy it as a
separate Vercel project with `dashboard` as the project root.

## Local setup

Use Python 3.11 or newer. On macOS, LightGBM also needs the OpenMP runtime:

```sh
brew install libomp
python3 -m pip install -e .
python3 -m unittest discover -v
```

## 1. Ingest source readings

`fetch_load` returns source-resolution observations from the ENTSO-E client,
which already supplies Amsterdam-aware timestamps for NL. The adapter trims the
requested window, but does not fill gaps, average readings, or remove duplicates.

Weather is requested in UTC, decoded to Amsterdam-aware timestamps, and trimmed
to the requested window. HTTP failures, malformed responses, missing required
fields, and wrong units are errors. Missing observations reach preprocessing.

Backfill writes source-resolution observations directly to Supabase after
applying [supabase/schema.sql](supabase/schema.sql):

```sh
python3 -m entso_e_pipeline.offline.backfill
```

The importer uses `SUPABASE_URL` and a server-side
`SUPABASE_SECRET_KEY` from `.env` (`SUPABASE_SERVICE_ROLE_KEY` and
`SUPABASE_KEY` remain fallbacks). Get these from Supabase Project Settings →
API. Keep the secret/service-role key on trusted backend or local machines. It
upserts by UTC observation timestamp, so a later source revision replaces the
current stored value. No local data file or resume checkpoint is created.

The same schema includes curated `hourly_load_observations`,
`hourly_weather_observations`, and versioned `model_features`, plus
`model_versions`, `forecast_runs`, `forecast_values`, and
`forecast_daily_metrics`. Raw readings remain unchanged; resampling and feature
engineering write separate derived layers. A future run records which model and
issuance time produced each Amsterdam horizon, then attaches actual load and
daily MAE/MAPE/quantile coverage metrics when observations arrive.
Populate the derived tables from the stored raw data with:

```sh
python3 -m entso_e_pipeline.offline.materialize
```

For live ingestion, `ingest_previous_day_load(reference_date)` uses the stored
Supabase history as its availability watermark. It checks the seven Amsterdam
calendar days before the forecast date, fetches from the first missing day
through the cutoff, and raises an error if any required hourly day remains
incomplete. An API outage therefore delays a forecast instead of silently
breaking its 168-hour lookback.

## Live forecasting

Publish a trained model bundle to DagsHub MLflow and register its artifact URI
in Supabase:

```sh
python3 -m entso_e_pipeline.offline.train --publish-dagshub
```

This uses the existing master-branch convention:
`DAGSHUB_USER_TOKEN` and the repository `twissamodiofficial/entso-e`. You may
override the repository with `DAGSHUB_REPO`, or use an equivalent
`MLFLOW_TRACKING_URI` and MLflow credentials.
The live command loads the newest registered bundle from DagsHub:

```sh
python3 -m entso_e_pipeline.online.forecast
```

It stores the forecast run, calibrated point/quantile values, and materialized
live features in Supabase. After the forecast day completes, reconcile it with:

```sh
python3 -m entso_e_pipeline.online.reconcile
```

The GitHub Actions workflow at
`.github/workflows/live-forecast.yml` runs reconciliation and forecasting in
Amsterdam time. Configure its Supabase, ENTSO-E, and `DAGSHUB_USER_TOKEN` secrets
before enabling the schedule; use manual dispatch for the first run.


## 2. Preprocess

[preprocessing.py](entso_e_pipeline/preprocessing.py) handles cleaning separately:

- Sort timestamps and collapse identical duplicate observations. Conflicting
  duplicates and missing timestamps require a decision and are rejected.
- Convert invalid numeric readings into missing values.
- Average available load readings within each real hour. Three valid quarter-hour
  readings are averaged directly; no fourth value is invented.
- Leave completely missing hours as missing. There is no ordinary load or weather
  interpolation, forward fill, or backward fill.
- Keep weather hourly and preserve its selected forecast source.

An average based on fewer readings may be less representative of the full hour.
This is the explicit policy for both load history and hourly labels. Entirely
missing labels are not fabricated or silently dropped.

```python
from entso_e_pipeline import preprocessing
from entso_e_pipeline.offline.backfill import BackfillSpec
from entso_e_pipeline.storage.supabase import SupabaseRawStore

spec = BackfillSpec()
store = SupabaseRawStore()
raw_load = store.load(spec.load_start, spec.test_end)
raw_weather = store.weather(spec.train_start, spec.test_end)
load = preprocessing.prepare_load(raw_load, start=spec.load_start, end=spec.test_end)
weather = preprocessing.prepare_weather(raw_weather)
```

Pass both load bounds to preserve entirely missing first/last hours in the
output. Otherwise, preprocessing covers the hours between the first and last
available readings. Source-cadence inference is no longer needed.

Preprocessing the whole archive is safe for daily forecasting because each
hourly load value uses only observations within that hour. No next-day reading
can fill a pre-midnight gap.

## 3. Build features

`engineering.transform` takes preprocessed hourly load, a complete local-day
horizon, two stateless transformers, and optional preprocessed weather. It uses
only the seven Amsterdam calendar days before the horizon. Forecast-day actuals
and later observations are excluded from load features.

```python
from entso_e_pipeline.features import engineering
from entso_e_pipeline.time_utils import local_day_hours

load_transformer, calendar_transformer = engineering.make_transformers()
horizon = local_day_hours("2026-01-10", config.TIMEZONE)
X = engineering.transform(load, horizon, load_transformer, calendar_transformer, weather)
y = load.reindex(horizon)[config.TARGET]  # Historical labels only.
```

For January 10 at 09:00, the daily lag uses January 9 at 09:00, the weekly lag
uses January 3 at 09:00, and the rolling mean ends at January 9 at 09:00. Calendar
and holiday features use Amsterdam dates and clock hours.

The load features are:

- `load_lag_1d_same_clock_hour`
- `load_lag_7d_same_clock_hour`
- `load_mean_24_clock_hours_ending_1d_ago`

DST policies apply only to the internal reference clock grid: repeated reference
hours are averaged, and a nonexistent spring reference hour uses the mean of
01:00 and 03:00. Both repeated forecast hours use the same reference features.
Actual rows and predictions retain all 23, 24, or 25 hourly instants. This DST
lookup policy does not fill missing source observations.

`fit_transform_train(load, weather)` constructs each eligible day through that
same feature path and attaches hourly labels. The first seven full local days
are history; an initial partial day is excluded from the lookback. Missing
features and labels remain visible for final validation.

## 4. Validate, then train or predict

Final validation is in [modeling/schema.py](entso_e_pipeline/modeling/schema.py).
`split_target(frame)` checks a labeled frame; `validate_features(X)` checks an
unlabeled feature frame. They require complete hourly indexes, finite values,
valid categorical features, and the appropriate feature/target separation.
Model training and prediction wrappers run their final checks automatically.

An unresolved gap can pass through ingestion, preprocessing, and feature
construction, but cannot silently enter a model. Basic structural checks remain
where needed to interpret source data and define a valid forecast horizon.

`config.SPLITS` reserves train (2019–2025), validation (January–March 2026),
calibration (April–June), and test (July 1–September 15). All bounds are Amsterdam
local and half-open. Build features with their prior history before selecting
these splits; an isolated validation frame would lose its first seven days.

The historical training entry point performs those preprocessing, feature,
split, and final validation steps in one reproducible command:

```sh
python3 -m entso_e_pipeline.offline.train
```

It selects rounds using train/validation, refits on both with fixed rounds, and
saves the point and quantile models under `models/`. It also prints MAE and MAPE
for the validation selection models and for the final models on calibration and
test. Calibration and test stay out of fitting and early stopping. The q10–q90
interval is widened using calibration only, then evaluated on untouched test
data. `PRODUCTION_SPLITS` describes a later fit through June with
July–September calibration.

To compare the saved point model with ENTSO-E's point forecast on the same test
window, run the local-only benchmark:

```sh
python3 -m entso_e_pipeline.offline.compare_entsoe
```

It fetches ENTSO-E actuals and forecasts, aligns both sources to Amsterdam
hourly instants, and prints MAE/MAPE without writing files or Supabase rows.

## Weather policy

Historical and live weather both use Open-Meteo Previous Runs with JMA GSM
(`jma_gsm`), the configured Amsterdam coordinates, Celsius, and the fields
`apparent_temperature_previous_day2` and `dew_point_2m_previous_day2`.
They become the `apparent_temperature` and `dew_point_2m` feature columns.

The fixed two-day lead is separate from the calendar-day load lag. It selects
older forecasts without shifting their valid timestamps. This is not a
reconstruction of one particular midnight-issued weather run. There is no
fallback to observed weather or another model.

The weather adapter owns request chunking: at most 14 UTC dates per call.
`BackfillSpec.chunk_days` controls load requests only. UTC weather requests are
retained to preserve real instants across DST; requesting local clock strings
does not guarantee an unambiguous 25-hour autumn day.

References: [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api),
[JMA models](https://open-meteo.com/en/docs/jma-api), and
[access plans](https://open-meteo.com/en/pricing).

## Storage and migration

Supabase is the only runtime data store. The old local file snapshot is not
read or regenerated; rerun the backfill to obtain source-resolution data in the
canonical raw tables. The old `to_hourly_local`, `load_and_clean`, and
`load_weather` shortcuts are replaced by explicit loading followed by
`prepare_load` and `prepare_weather`.
