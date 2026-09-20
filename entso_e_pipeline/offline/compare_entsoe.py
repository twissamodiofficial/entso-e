"""Compare our saved point model with ENTSO-E's point load forecast."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from .. import config, preprocessing
from ..modeling import metrics, point
from ..modeling.schema import split_target
from ..storage.supabase import SupabaseRawStore
from ..time_utils import as_local, as_local_index


load_dotenv()


def _test_window(start=None, end=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    configured_start, configured_end = config.SPLITS["test"][0]
    start = as_local(start or configured_start, config.TIMEZONE)
    end = as_local(end or configured_end, config.TIMEZONE)
    if start >= end:
        raise ValueError("Comparison start must precede end.")
    return start, end


def _as_load_frame(data, column: str) -> pd.DataFrame:
    """Normalize an ENTSO-E Series/DataFrame to the source-load schema."""

    if isinstance(data, pd.Series):
        frame = data.to_frame(name=column)
    elif isinstance(data, pd.DataFrame):
        if column in data.columns:
            frame = data[[column]].copy()
        elif len(data.columns) == 1:
            frame = data.rename(columns={data.columns[0]: column})
        else:
            raise ValueError(f"ENTSO-E response does not contain {column!r}.")
    else:
        raise TypeError("ENTSO-E response must be a pandas Series or DataFrame.")
    frame.index = as_local_index(frame.index, config.TIMEZONE)
    return frame


def _hourly(data, column: str, start, end) -> pd.DataFrame:
    frame = _as_load_frame(data, column).rename(columns={column: config.TARGET})
    return preprocessing.prepare_load(
        frame,
        start=start,
        end=end,
    )


def compare(
    *,
    start=None,
    end=None,
    models_dir: str = config.MODELS_DIR,
    store: SupabaseRawStore | None = None,
    client: EntsoePandasClient | None = None,
) -> dict:
    """Score both point forecasts against the same ENTSO-E actual series."""

    start, end = _test_window(start, end)
    store = store or SupabaseRawStore()
    client = client or EntsoePandasClient(api_key=os.environ.get("ENTSOE_API_KEY"))

    features = store.features(config.FEATURE_VERSION)
    test_frame = features.loc[
        (features.index >= start) & (features.index < end)
    ]
    model_features, _ = split_target(test_frame, "Model test data")
    our_prediction = point.predict(point.load(f"{models_dir}/lightgbm_v1.txt"), model_features)

    entsoe_actual = _hourly(
        client.query_load(config.COUNTRY_CODE, start=start, end=end),
        config.TARGET,
        start,
        end,
    )
    entsoe_forecast = _hourly(
        client.query_load_forecast(config.COUNTRY_CODE, start=start, end=end),
        "Forecasted Load",
        start,
        end,
    )

    aligned = pd.concat(
        [
            entsoe_actual[config.TARGET].rename("actual"),
            entsoe_forecast[config.TARGET].rename("entsoe_forecast"),
            our_prediction.rename("our_forecast"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError("No overlapping complete hourly rows were available.")

    actual = aligned["actual"]
    our = aligned["our_forecast"]
    entsoe = aligned["entsoe_forecast"]
    our_metrics = metrics.point_metrics(actual, our)
    entsoe_metrics = metrics.point_metrics(actual, entsoe)
    return {
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "rows_compared": len(aligned),
        "our_model": our_metrics,
        "entsoe_forecast": entsoe_metrics,
        "mape_difference_percentage_points": (
            entsoe_metrics["mape_percent"] - our_metrics["mape_percent"]
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--models-dir", default=config.MODELS_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(compare(
        start=args.start,
        end=args.end,
        models_dir=args.models_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
