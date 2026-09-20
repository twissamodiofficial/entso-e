"""Materialize hourly preprocessing and model features in Supabase."""

from __future__ import annotations

import argparse
import json

from .. import config, preprocessing
from ..features import engineering
from .backfill import BackfillSpec
from ..storage.supabase import SupabaseRawStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args(argv)

    spec = BackfillSpec()
    store = SupabaseRawStore(batch_size=args.batch_size)
    raw_load = store.load(spec.load_start, spec.test_end)
    raw_weather = store.weather(spec.train_start, spec.test_end)
    load = preprocessing.prepare_load(
        raw_load,
        start=spec.load_start,
        end=spec.test_end,
    )
    weather = preprocessing.prepare_weather(raw_weather)
    features, _, _ = engineering.fit_transform_train(load, weather)

    print(json.dumps({
        "hourly_load_rows": store.upsert_hourly_load(load),
        "hourly_weather_rows": store.upsert_hourly_weather(weather),
        "feature_rows": store.upsert_features(features),
        "processing_version": config.PROCESSING_VERSION,
        "feature_version": config.FEATURE_VERSION,
    }, indent=2))


if __name__ == "__main__":
    main()
