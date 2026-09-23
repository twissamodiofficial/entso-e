"""Supabase storage for source-resolution load and weather observations."""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

from .. import config
from ..features.engineering import cast_categorical_features


load_dotenv()


def _client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = (
        os.environ.get("SUPABASE_SECRET_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
    )
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY, SUPABASE_SERVICE_ROLE_KEY, "
            "or SUPABASE_KEY are required."
        )
    return create_client(url, key)


def _utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Supabase observations require timezone-aware timestamps.")
    return timestamp.tz_convert("UTC")


def _batches(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


class SupabaseRawStore:
    """Read and upsert the canonical raw observations."""

    def __init__(self, client: Client | None = None, batch_size: int = 1000):
        if batch_size <= 0:
            raise ValueError("Supabase batch_size must be positive.")
        self.client = client or _client()
        self.batch_size = batch_size

    def _upsert(
        self,
        table: str,
        rows: list[dict],
        on_conflict: str,
    ) -> int:
        for batch in _batches(rows, self.batch_size):
            self.client.table(table).upsert(
                batch,
                on_conflict=on_conflict,
            ).execute()
        return len(rows)

    def upsert_load(self, frame: pd.DataFrame) -> int:
        if config.TARGET not in frame.columns:
            raise ValueError(f"Load data must contain {config.TARGET!r}.")
        rows = []
        for timestamp, value in frame[config.TARGET].items():
            timestamp = _utc_timestamp(timestamp)
            rows.append({
                "observed_at": timestamp.isoformat(),
                "area_code": "10YNL----------L",
                "actual_load_mw": None if pd.isna(value) else float(value),
            })
        return self._upsert("raw_load_observations", rows, "observed_at")

    def upsert_weather(self, frame: pd.DataFrame) -> int:
        missing = [column for column in config.WEATHER_FEATURES if column not in frame]
        if missing:
            raise ValueError(f"Weather data is missing columns: {missing}")
        rows = []
        for timestamp, values in frame[config.WEATHER_FEATURES].iterrows():
            timestamp = _utc_timestamp(timestamp)
            rows.append({
                "observed_at": timestamp.isoformat(),
                "apparent_temperature": (
                    None if pd.isna(values["apparent_temperature"])
                    else float(values["apparent_temperature"])
                ),
                "dew_point_2m": (
                    None if pd.isna(values["dew_point_2m"])
                    else float(values["dew_point_2m"])
                ),
            })
        return self._upsert("raw_weather_observations", rows, "observed_at")

    def upsert_hourly_load(
        self,
        frame: pd.DataFrame,
        processing_version: str = config.PROCESSING_VERSION,
    ) -> int:
        if config.TARGET not in frame.columns:
            raise ValueError(f"Load data must contain {config.TARGET!r}.")
        rows = []
        for timestamp, value in frame[config.TARGET].items():
            rows.append({
                "observed_at": _utc_timestamp(timestamp).isoformat(),
                "actual_load_mw": None if pd.isna(value) else float(value),
                "valid_readings": None,
                "processing_version": processing_version,
            })
        return self._upsert("hourly_load_observations", rows, "observed_at")

    def upsert_hourly_weather(
        self,
        frame: pd.DataFrame,
        processing_version: str = config.PROCESSING_VERSION,
    ) -> int:
        missing = [column for column in config.WEATHER_FEATURES if column not in frame]
        if missing:
            raise ValueError(f"Weather data is missing columns: {missing}")
        rows = []
        for timestamp, values in frame[config.WEATHER_FEATURES].iterrows():
            rows.append({
                "observed_at": _utc_timestamp(timestamp).isoformat(),
                "apparent_temperature": (
                    None if pd.isna(values["apparent_temperature"])
                    else float(values["apparent_temperature"])
                ),
                "dew_point_2m": (
                    None if pd.isna(values["dew_point_2m"])
                    else float(values["dew_point_2m"])
                ),
                "processing_version": processing_version,
            })
        return self._upsert("hourly_weather_observations", rows, "observed_at")

    def upsert_features(
        self,
        frame: pd.DataFrame,
        feature_version: str = config.FEATURE_VERSION,
    ) -> int:
        feature_columns = [
            *config.LOAD_FEATURE_NAMES.values(),
            *config.CATEGORICAL_FEATURES,
            *config.WEATHER_FEATURES,
            config.TARGET,
        ]
        missing = [column for column in feature_columns if column not in frame]
        if missing:
            raise ValueError(f"Feature data is missing columns: {missing}")

        def value_for(column, value):
            if pd.isna(value):
                return None
            if column == "holiday_name":
                return str(value)
            if column in config.CATEGORICAL_FEATURES:
                return int(value)
            return float(value)

        rows = []
        for timestamp, values in frame[feature_columns].iterrows():
            row = {
                "feature_version": feature_version,
                "valid_at": _utc_timestamp(timestamp).isoformat(),
            }
            for column in feature_columns:
                destination = "actual_load_mw" if column == config.TARGET else column
                row[destination] = value_for(column, values[column])
            rows.append(row)
        return self._upsert(
            "model_features",
            rows,
            "feature_version,valid_at",
        )

    def upsert_model_version(
        self,
        model_version: str,
        *,
        artifact_uri: str | None = None,
        training_start=None,
        training_end=None,
        training_rounds: dict | None = None,
        training_metrics: dict | None = None,
    ) -> None:
        row = {
            "model_version": model_version,
            "artifact_uri": artifact_uri,
            "training_start": None if training_start is None else str(training_start),
            "training_end": None if training_end is None else str(training_end),
            "training_rounds": training_rounds,
            "training_metrics": training_metrics,
        }
        self.client.table("model_versions").upsert(
            row,
            on_conflict="model_version",
        ).execute()

    def latest_model_version(self) -> dict | None:
        """Return the newest published model artifact registration."""

        response = (
            self.client.table("model_versions")
            .select(
                "model_version,artifact_uri,training_start,training_end,"
                "training_rounds,training_metrics,trained_at"
            )
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    def create_forecast_run(
        self,
        model_version: str,
        forecast_date,
        horizon_start,
        horizon_end,
        issued_at=None,
    ) -> str:
        row = {
            "model_version": model_version,
            "forecast_date": str(forecast_date),
            "issued_at": (
                pd.Timestamp.now(tz="UTC").isoformat()
                if issued_at is None else _utc_timestamp(issued_at).isoformat()
            ),
            "horizon_start": _utc_timestamp(horizon_start).isoformat(),
            "horizon_end": _utc_timestamp(horizon_end).isoformat(),
        }
        response = self.client.table("forecast_runs").insert(row).execute()
        data = response.data or []
        if not data or "run_id" not in data[0]:
            raise RuntimeError("Supabase did not return the created forecast run ID.")
        return data[0]["run_id"]

    def latest_forecast_run(self, model_version: str, forecast_date) -> dict | None:
        """Return the newest run for a model and Amsterdam forecast date."""

        response = (
            self.client.table("forecast_runs")
            .select(
                "run_id,model_version,forecast_date,issued_at,horizon_start,horizon_end"
            )
            .eq("model_version", model_version)
            .eq("forecast_date", str(forecast_date))
            .order("issued_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    def forecast_run_for_date(self, forecast_date, model_version=None) -> dict | None:
        """Return the newest forecast run for a local forecast date."""

        query = (
            self.client.table("forecast_runs")
            .select(
                "run_id,model_version,forecast_date,issued_at,horizon_start,horizon_end"
            )
            .eq("forecast_date", str(forecast_date))
            .order("issued_at", desc=True)
            .limit(1)
        )
        if model_version is not None:
            query = query.eq("model_version", model_version)
        response = query.execute()
        rows = response.data or []
        return rows[0] if rows else None

    def oldest_unreconciled_forecast_run(
        self,
        before_date,
        model_version=None,
    ) -> dict | None:
        """Return the oldest latest run before a date without daily metrics.

        A rerun can create multiple forecast runs for one local date. Only the
        newest run for each date is eligible, matching ``forecast_run_for_date``
        and preventing an obsolete rerun from blocking reconciliation.
        """

        query = (
            self.client.table("forecast_runs")
            .select(
                "run_id,model_version,forecast_date,issued_at,horizon_start,horizon_end"
            )
            .lt("forecast_date", str(before_date))
            .order("forecast_date")
            .order("issued_at", desc=True)
            .limit(100)
        )
        if model_version is not None:
            query = query.eq("model_version", model_version)
        rows = query.execute().data or []

        latest_by_date = {}
        for row in rows:
            latest_by_date.setdefault(row["forecast_date"], row)

        for row in latest_by_date.values():
            metrics = (
                self.client.table("forecast_daily_metrics")
                .select("run_id")
                .eq("run_id", row["run_id"])
                .limit(1)
                .execute()
                .data
                or []
            )
            if not metrics:
                return row
        return None

    def upsert_forecasts(self, run_id: str, forecasts: pd.DataFrame) -> int:
        required = {"point_forecast_mw", "q10_forecast_mw", "q50_forecast_mw", "q90_forecast_mw"}
        missing = sorted(required - set(forecasts.columns))
        if missing:
            raise ValueError(f"Forecast data is missing columns: {missing}")
        rows = []
        for timestamp, values in forecasts.iterrows():
            timestamp = _utc_timestamp(timestamp)
            row = {
                "run_id": run_id,
                "valid_at": timestamp.isoformat(),
                **{
                    column: None if pd.isna(values[column]) else float(values[column])
                    for column in required
                },
            }
            if "actual_load_mw" in forecasts:
                value = values["actual_load_mw"]
                row["actual_load_mw"] = None if pd.isna(value) else float(value)
            if "actual_observed_at" in forecasts:
                value = values["actual_observed_at"]
                row["actual_observed_at"] = (
                    None if pd.isna(value) else _utc_timestamp(value).isoformat()
                )
            rows.append(row)
        return self._upsert("forecast_values", rows, "run_id,valid_at")

    def forecast_values(self, run_id: str) -> pd.DataFrame:
        """Read a forecast run with UTC storage converted to Amsterdam time."""

        rows = (
            self.client.table("forecast_values")
            .select(
                "valid_at,point_forecast_mw,q10_forecast_mw,q50_forecast_mw,"
                "q90_forecast_mw,actual_load_mw,actual_observed_at"
            )
            .eq("run_id", run_id)
            .order("valid_at")
            .execute()
            .data
            or []
        )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "point_forecast_mw", "q10_forecast_mw", "q50_forecast_mw",
                    "q90_forecast_mw", "actual_load_mw", "actual_observed_at",
                ],
                index=pd.DatetimeIndex([], tz=config.TIMEZONE, name="time"),
            )
        frame = pd.DataFrame(rows)
        frame["valid_at"] = pd.to_datetime(frame["valid_at"], utc=True)
        frame = frame.set_index("valid_at").rename_axis("time")
        frame.index = frame.index.tz_convert(config.TIMEZONE)
        if "actual_observed_at" in frame:
            frame["actual_observed_at"] = pd.to_datetime(
                frame["actual_observed_at"], utc=True, errors="coerce"
            ).dt.tz_convert(config.TIMEZONE)
        return frame

    def upsert_daily_metrics(self, run_id: str, metrics: pd.DataFrame) -> int:
        rows = []
        for valid_date, values in metrics.iterrows():
            row = {
                "run_id": run_id,
                "valid_date": str(valid_date),
            }
            for column in (
                "evaluated_rows",
                "point_mae_mw",
                "point_mape_percent",
                "q10_pinball_loss",
                "q50_pinball_loss",
                "q90_pinball_loss",
                "interval_coverage",
            ):
                if column in metrics.columns:
                    value = values[column]
                    if pd.isna(value):
                        row[column] = None
                    elif column == "evaluated_rows":
                        row[column] = int(value)
                    else:
                        row[column] = float(value)
            rows.append(row)
        return self._upsert(
            "forecast_daily_metrics",
            rows,
            "run_id,valid_date",
        )

    def _read(
        self,
        table: str,
        columns: str,
        start=None,
        end=None,
    ) -> list[dict]:
        rows = []
        offset = 0
        while True:
            query = (
                self.client.table(table)
                .select(columns)
                .order("observed_at")
                .range(offset, offset + self.batch_size - 1)
            )
            if start is not None:
                query = query.gte("observed_at", _utc_timestamp(start).isoformat())
            if end is not None:
                query = query.lt("observed_at", _utc_timestamp(end).isoformat())
            page = query.execute().data or []
            rows.extend(page)
            if len(page) < self.batch_size:
                return rows
            offset += self.batch_size

    @staticmethod
    def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=columns,
                index=pd.DatetimeIndex([], tz=config.TIMEZONE),
            )
            frame.index.name = "time"
            return frame
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        frame = frame.set_index("observed_at").rename_axis("time")
        frame.index = frame.index.tz_convert(config.TIMEZONE)
        return frame

    def load(self, start=None, end=None) -> pd.DataFrame:
        rows = self._read(
            "raw_load_observations",
            "observed_at,actual_load_mw",
            start,
            end,
        )
        frame = self._frame(rows, ["actual_load_mw"])
        return frame.rename(columns={"actual_load_mw": config.TARGET})

    def weather(self, start=None, end=None) -> pd.DataFrame:
        rows = self._read(
            "raw_weather_observations",
            "observed_at,apparent_temperature,dew_point_2m",
            start,
            end,
        )
        return self._frame(rows, config.WEATHER_FEATURES)

    def features(self, feature_version: str = config.FEATURE_VERSION) -> pd.DataFrame:
        columns = [
            "feature_version",
            "valid_at",
            *config.LOAD_FEATURE_NAMES.values(),
            *config.CATEGORICAL_FEATURES,
            *config.WEATHER_FEATURES,
            "actual_load_mw",
        ]
        rows = []
        offset = 0
        selection = ",".join(columns)
        while True:
            page = (
                self.client.table("model_features")
                .select(selection)
                .eq("feature_version", feature_version)
                .order("valid_at")
                .range(offset, offset + self.batch_size - 1)
                .execute()
                .data
                or []
            )
            rows.extend(page)
            if len(page) < self.batch_size:
                break
            offset += self.batch_size

        if not rows:
            return pd.DataFrame(
                columns=[
                    *config.LOAD_FEATURE_NAMES.values(),
                    *config.CATEGORICAL_FEATURES,
                    *config.WEATHER_FEATURES,
                    config.TARGET,
                ],
                index=pd.DatetimeIndex([], tz=config.TIMEZONE, name="time"),
            )
        frame = pd.DataFrame(rows)
        frame["valid_at"] = pd.to_datetime(frame["valid_at"], utc=True)
        frame = frame.set_index("valid_at").rename_axis("time")
        frame.index = frame.index.tz_convert(config.TIMEZONE)
        frame = frame.drop(columns=["feature_version"])
        frame = frame.rename(columns={"actual_load_mw": config.TARGET})
        return cast_categorical_features(frame)
