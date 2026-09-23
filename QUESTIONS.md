# Questions and explanations

This is a running learning log for confusing functions, design decisions, and
pipeline concepts.

## Training and evaluation

### Why select boosting rounds and then refit the models?

`select_training_rounds(train, validation)` uses a large maximum number of
boosting rounds and early stopping to choose the best round count using the
validation split. The validation rows are evaluated but are not used to fit the
model.

`fit_models(train + validation, rounds)` then trains fresh models on all
available labeled data using those fixed round counts. This lets the final model
learn from the validation rows without using them again for early stopping.

The standard flow is:

```text
train → validation-based round selection
train + validation → final fixed-round model
calibration → interval adjustment
test → final evaluation
```

### Which data is used for calibration and test accuracy?

The final point and quantile models are fitted on `train + validation` using
rounds selected from validation. The calibration split is used only to estimate
the interval adjustment. The test split is kept untouched until final scoring.

### Is validation-selection accuracy a final metric?

No. A validation score from the same data used for early stopping is
selection-biased. It can be useful as a diagnostic, but final generalization
metrics come from the untouched test split. The `validation_selection` report
entry was removed for this reason.

### Why are training and production separate?

`offline.train` performs historical round selection, final historical fitting,
calibration, and test evaluation. `offline.production` loads the saved rounds,
refits through the latest production training window, recalibrates, and writes a
separate production model bundle.

Production model building remains under `offline`; `online` is for serving
forecasts and reconciling completed forecasts.

## Preprocessing and feature construction

### Does processing the full archive in `materialize` leak validation or test data?

No, not with the current implementation. `materialize` preprocesses and builds
features for the full historical range so they can be stored canonically. The
training code later slices that table into train, validation, calibration, and
test windows.

The current preprocessing and feature transformers are deterministic and
stateless. For each forecast day, load features use only the seven calendar
days strictly before that day. The model itself is trained only on the selected
training rows.

If learned preprocessing is added later—such as scaling, mean imputation,
target encoding, or normalization—it must be fitted on training data only and
then applied unchanged to later splits.

### Do categorical features leak validation or test information?

They would if their category vocabulary or encoding statistics were inferred
from the full dataset. This project does not infer them from the materialized
rows. `hour`, `day_of_week`, `month`, `weekend`, and `is_holiday` use fixed
configured categories. `holiday_name` uses the known holiday calendar rather
than discovering names from validation or test values.

Those categories are external schema information, not information learned from
the target. LightGBM still learns model splits only from training rows. If a
future categorical feature is data-derived, its vocabulary, encoder, and any
statistics should be fitted on training data only, with an explicit policy for
unknown categories.

### What happens if a holiday category appears in validation or test but not training?

An allowed category does not automatically give the model a learned effect. If
the category never occurs in training rows, LightGBM cannot learn a
category-specific split for it; the prediction follows the model's other
learned splits and default behavior.

In this project, `"none"` means that the date is not present in the configured
holiday calendar. It is not the fallback for an unknown holiday name. The
configured `holiday_name` categories and the calendar values come from the same
known calendar, currently covering 2019–2040. We intentionally do not add an
unknown bucket: if the calendar library gains a new holiday name, the fixed
calendar schema should be updated and the feature version rematerialized. A
date outside the configured calendar range should likewise be covered by an
updated calendar rather than silently treated as `"none"`.

### What does `build_labeled_features` do?

It builds causal historical features for each eligible local day and attaches
that day’s actual load afterward. It does not train LightGBM.

The function:

1. Converts load timestamps to Amsterdam time.
2. Discards an initial partial day if necessary.
3. Skips the first seven full days because they are needed as lookback history.
4. Builds each day’s lag, rolling, calendar, holiday, and weather features.
5. Joins actual load values as labels after feature construction.

The old name `fit_transform_train` was misleading and was renamed to
`build_labeled_features`.

### Does preprocessing impute missing values?

No. `prepare_load` averages available sub-hourly readings within each hour,
but missing hours remain `NaN`. `prepare_weather` cleans and aligns values but
does not fill gaps. Final validation rejects missing model inputs.

The only special fill is the deterministic handling of a nonexistent spring
DST clock hour using neighboring reference values.

### Why must preprocessing bounds be hourly?

`prepare_load` produces complete hourly observations using `resample("h")`.
A bound such as `00:30` would cut through an hour and could produce a partial
hour labeled as a complete hour. Therefore bounds must represent exact hourly
instants.

### Can `reindex` introduce missing values?

Yes:

```python
hourly = hourly.reindex(hourly_index(start, end))
```

ensures every expected hourly timestamp exists. If a source hour is absent,
`reindex` inserts that timestamp with `NaN`; it does not impute a value.

## Duplicate and invalid source values

### Are identical duplicate timestamps allowed?

Yes. Duplicate rows with identical values are harmless; the first copy is kept
and the rest are removed. Duplicate timestamps with conflicting values raise an
error because the pipeline cannot know which value is correct.

### What does `.gt(1).any(axis=1)` mean?

`.gt(1)` means “greater than 1” element by element. `.any(axis=1)` asks whether
any column in each row is `True`.

After counting distinct values per timestamp, the expression means:

> Does any column contain more than one distinct value for this timestamp?

That identifies conflicting duplicate rows.

## Timezones and DST

### What does `Timestamp.normalize()` do?

It sets the clock time to local midnight while preserving the timestamp’s
timezone. For example:

```text
2026-07-01 15:30+02:00 → 2026-07-01 00:00+02:00
```

It does not convert the timestamp to UTC. In this project it finds the start of
an Amsterdam calendar day.

### How are timestamps handled by backfill, APIs, and Supabase?

Backfill windows are defined in Amsterdam time. Weather requests are converted
to UTC, and the returned weather index is converted back to Amsterdam. Load
requests use Amsterdam-aware timestamps and the load adapter returns an
Amsterdam-aware index.

Supabase storage normalizes aware timestamps to UTC before writing. The database
uses PostgreSQL `timestamptz`, which stores an absolute instant. Reads convert
those instants back to Amsterdam for application use.

`Z` means UTC. For example:

```text
2026-01-01T00:00:00+01:00 = 2025-12-31T23:00:00Z
```

### How are the four DST target-day cases handled?

Normal days have one `02:00` and 24 hourly output rows. DST days work as follows:

1. **Target is the spring DST day:** the horizon has 23 rows and no `02:00`.
   `target_clock` also has no `02:00`, so the feature reindexing cannot create
   one.
2. **Target is the day after spring DST:** the horizon has one `02:00`. The
   previous spring day has a nonexistent local `02:00`, so the explicit DST
   rule creates a reference value from its neighboring hours. That value can
   be used as the target day's one-day lag at `02:00`.
3. **Target is the autumn DST day:** the horizon has 25 rows and two real
   `02:00` timestamps. Both receive the same local-clock load-history,
   calendar, and holiday features. Weather can differ because the two rows are
   different timezone-aware instants, so their predictions can differ.
4. **Target is the day after autumn DST:** the horizon has one `02:00`. The two
   previous-day autumn `02:00` observations are grouped and averaged into one
   local-clock reference value, which is used for the one-day lag.

The output index always remains the original timezone-aware horizon, so it has
23, 24, or 25 real hourly rows as appropriate.

### What does `CalendarLoadFeatures.transform` calculate?

It creates three load-history features for the target horizon:

- same local clock hour one calendar day earlier;
- same local clock hour seven calendar days earlier;
- a 24-clock-hour rolling mean ending one day earlier.

It temporarily removes timezone metadata to match local clock labels. Repeated
autumn `02:00` source observations are averaged into one reference value, and
the nonexistent spring `02:00` reference label is filled only by the explicit
DST rule. Ordinary missing observations remain missing.

## ENTSO-E comparison

`offline.compare_entsoe` compares our saved point forecast with ENTSO-E’s point
forecast over the same test window. Both forecasts are scored against the same
ENTSO-E actual load series. It reports MAE and MAPE for both; it is not treating
ENTSO-E’s forecast as the ground truth.
