-- Canonical source-resolution observations.
-- Store instants in UTC; local Amsterdam conversion belongs to preprocessing.

create table if not exists public.raw_load_observations (
    observed_at timestamptz primary key,
    area_code text not null default '10YNL----------L',
    actual_load_mw double precision,
    source text not null default 'entsoe',
    source_fetched_at timestamptz not null default now()
);

create table if not exists public.raw_weather_observations (
    observed_at timestamptz primary key,
    apparent_temperature double precision,
    dew_point_2m double precision,
    source text not null default 'open_meteo_previous_runs',
    source_fetched_at timestamptz not null default now()
);

create index if not exists raw_load_observed_at_idx
    on public.raw_load_observations (observed_at);

create index if not exists raw_weather_observed_at_idx
    on public.raw_weather_observations (observed_at);

-- Curated hourly observations produced by preprocessing. These rows are
-- replaceable derived data; the raw tables remain the source snapshot.
create table if not exists public.hourly_load_observations (
    observed_at timestamptz primary key,
    actual_load_mw double precision,
    valid_readings smallint,
    processing_version text not null,
    processed_at timestamptz not null default now()
);

create table if not exists public.hourly_weather_observations (
    observed_at timestamptz primary key,
    apparent_temperature double precision,
    dew_point_2m double precision,
    processing_version text not null,
    processed_at timestamptz not null default now()
);

create index if not exists hourly_load_observed_at_idx
    on public.hourly_load_observations (observed_at);

create index if not exists hourly_weather_observed_at_idx
    on public.hourly_weather_observations (observed_at);

-- Materialized model inputs. The feature version is part of the key so a
-- future feature change can coexist with the current training data.
create table if not exists public.model_features (
    feature_version text not null,
    valid_at timestamptz not null,
    load_lag_1d_same_clock_hour double precision,
    load_lag_7d_same_clock_hour double precision,
    load_mean_24_clock_hours_ending_1d_ago double precision,
    hour smallint,
    day_of_week smallint,
    month smallint,
    weekend smallint,
    is_holiday smallint,
    holiday_name text,
    apparent_temperature double precision,
    dew_point_2m double precision,
    actual_load_mw double precision,
    primary key (feature_version, valid_at)
);

create index if not exists model_features_valid_at_idx
    on public.model_features (valid_at);

-- Model and forecast history. Forecast rows stay tied to the model version
-- and issuance time that produced them; actuals and metrics arrive later.
create table if not exists public.model_versions (
    model_version text primary key,
    trained_at timestamptz not null default now(),
    artifact_uri text,
    training_start date,
    training_end date,
    training_rounds jsonb,
    training_metrics jsonb
);

create table if not exists public.forecast_runs (
    run_id uuid primary key default gen_random_uuid(),
    model_version text not null references public.model_versions(model_version),
    forecast_date date not null,
    issued_at timestamptz not null default now(),
    horizon_start timestamptz not null,
    horizon_end timestamptz not null,
    unique (model_version, forecast_date, issued_at)
);

create table if not exists public.forecast_values (
    run_id uuid not null references public.forecast_runs(run_id) on delete cascade,
    valid_at timestamptz not null,
    point_forecast_mw double precision not null,
    q10_forecast_mw double precision,
    q50_forecast_mw double precision,
    q90_forecast_mw double precision,
    actual_load_mw double precision,
    actual_observed_at timestamptz,
    primary key (run_id, valid_at)
);

create index if not exists forecast_values_valid_at_idx
    on public.forecast_values (valid_at);

create table if not exists public.forecast_daily_metrics (
    run_id uuid not null references public.forecast_runs(run_id) on delete cascade,
    valid_date date not null,
    evaluated_rows integer not null,
    point_mae_mw double precision,
    point_mape_percent double precision,
    q10_pinball_loss double precision,
    q50_pinball_loss double precision,
    q90_pinball_loss double precision,
    interval_coverage double precision,
    computed_at timestamptz not null default now(),
    primary key (run_id, valid_date)
);

create index if not exists forecast_daily_metrics_date_idx
    on public.forecast_daily_metrics (valid_date);

-- Local scripts should use the Supabase service-role key. RLS keeps these raw
-- tables inaccessible through the public client key.
alter table public.raw_load_observations enable row level security;
alter table public.raw_weather_observations enable row level security;
alter table public.hourly_load_observations enable row level security;
alter table public.hourly_weather_observations enable row level security;
alter table public.model_features enable row level security;
alter table public.model_versions enable row level security;
alter table public.forecast_runs enable row level security;
alter table public.forecast_values enable row level security;
alter table public.forecast_daily_metrics enable row level security;
