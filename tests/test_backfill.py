import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.offline.backfill import BackfillDownloader, BackfillSpec
from entso_e_pipeline.time_utils import local_day_hours


class RecordingStore:
    """Small in-memory stand-in for the Supabase raw store."""

    def __init__(self):
        self.load_frames = []
        self.weather_frames = []

    def upsert_load(self, frame):
        self.load_frames.append(frame.copy())
        return len(frame)

    def upsert_weather(self, frame):
        self.weather_frames.append(frame.copy())
        return len(frame)

    @property
    def load(self):
        return pd.concat(self.load_frames) if self.load_frames else pd.DataFrame()

    @property
    def weather(self):
        return pd.concat(self.weather_frames) if self.weather_frames else pd.DataFrame()


class BackfillTests(unittest.TestCase):
    def make_fetcher(self, kind, calls):
        def fetch(start, end):
            calls.append((kind, start, end))
            index = pd.date_range(start, end, freq="h", inclusive="left")
            if kind == "load":
                return pd.DataFrame(
                    {config.TARGET: np.arange(len(index), dtype=float)},
                    index=index,
                )
            result = pd.DataFrame(
                {name: 10.0 for name in config.WEATHER_FEATURES}, index=index
            )
            result.attrs.update(
                weather_source="open_meteo_previous_runs",
                weather_model="jma_gsm",
                weather_forecast_previous_days=2,
            )
            return result

        return fetch

    def test_downloads_chunks_and_upserts_without_files(self):
        spec = BackfillSpec(
            train_start=pd.Timestamp("2026-01-01", tz=config.TIMEZONE),
            test_end=pd.Timestamp("2026-01-15", tz=config.TIMEZONE),
            chunk_days=4,
        )
        calls = []
        store = RecordingStore()
        downloader = BackfillDownloader(
            spec=spec,
            load_fetcher=self.make_fetcher("load", calls),
            weather_fetcher=self.make_fetcher("weather", calls),
            store=store,
        )

        first = downloader.download()
        self.assertGreater(len(store.load_frames), 1)
        self.assertEqual(first["load_rows"], len(store.load))
        self.assertEqual(first["weather_rows"], len(store.weather))
        self.assertTrue(store.load.index.tz)
        self.assertTrue(store.weather.index.tz)

        first_call_count = len(calls)
        second = downloader.download()
        self.assertEqual(len(calls), first_call_count * 2)
        self.assertEqual(second, first)

    def test_dst_windows_keep_real_local_row_counts(self):
        spec = BackfillSpec(
            train_start=pd.Timestamp("2026-03-28", tz=config.TIMEZONE),
            test_end=pd.Timestamp("2026-04-02", tz=config.TIMEZONE),
            chunk_days=2,
        )
        store = RecordingStore()
        BackfillDownloader(
            spec=spec,
            load_fetcher=self.make_fetcher("load", []),
            weather_fetcher=self.make_fetcher("weather", []),
            store=store,
        ).download()
        expected_weather = sum(
            len(local_day_hours(day, config.TIMEZONE))
            for day in pd.date_range(
                spec.train_start, spec.test_end, freq="D", inclusive="left"
            )
        )
        self.assertEqual(len(store.weather), expected_weather)

    def test_incomplete_fetched_chunk_is_saved_for_preprocessing(self):
        spec = BackfillSpec(
            train_start="2026-01-01", test_end="2026-01-05", chunk_days=4
        )

        def missing_load(start, end):
            index = pd.date_range(start, end, freq="h", inclusive="left").delete(0)
            return pd.DataFrame({config.TARGET: 1.0}, index=index)

        store = RecordingStore()
        BackfillDownloader(
            spec=spec,
            load_fetcher=missing_load,
            weather_fetcher=self.make_fetcher("weather", []),
            store=store,
        ).download()
        self.assertNotIn(spec.load_start, store.load.index)

    def test_internal_subhour_gap_is_preserved_without_filling(self):
        spec = BackfillSpec(train_start="2026-01-01", test_end="2026-01-02")

        def missing_quarter(start, end):
            index = pd.date_range(start, end, freq="15min", inclusive="left").delete(1)
            return pd.DataFrame({config.TARGET: 1.0}, index=index)

        store = RecordingStore()
        BackfillDownloader(
            spec=spec,
            load_fetcher=missing_quarter,
            weather_fetcher=self.make_fetcher("weather", []),
            store=store,
        ).download()
        self.assertTrue(store.load[config.TARGET].eq(1.0).all())
        self.assertNotIn(spec.load_start + pd.Timedelta(minutes=15), store.load.index)
        self.assertIn(spec.load_start + pd.Timedelta(minutes=30), store.load.index)

    def test_missing_boundary_quarter_hours_are_saved_without_filling(self):
        spec = BackfillSpec(train_start="2026-01-01", test_end="2026-01-02")
        for position in (0, -1):
            with self.subTest(position=position):
                def incomplete_load(start, end):
                    index = pd.date_range(start, end, freq="15min", inclusive="left")
                    return pd.DataFrame(
                        {config.TARGET: 1.0}, index=index.delete(position)
                    )

                store = RecordingStore()
                BackfillDownloader(
                    spec=spec,
                    load_fetcher=incomplete_load,
                    weather_fetcher=self.make_fetcher("weather", []),
                    store=store,
                ).download()
                missing = (
                    spec.load_start
                    if position == 0
                    else spec.test_end - pd.Timedelta(minutes=15)
                )
                self.assertNotIn(missing, store.load.index)

    @patch("entso_e_pipeline.ingestion.weather.requests.get")
    def test_weather_requests_are_utc_and_retrieved_as_amsterdam(self, get):
        spec = BackfillSpec(train_start="2026-03-15", test_end="2026-04-20")

        def response(url, *, params, timeout):
            index = pd.date_range(
                pd.Timestamp(params["start_date"], tz="UTC"),
                pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(days=1),
                freq="h",
                inclusive="left",
            )
            fields = params["hourly"].split(",")
            result = Mock()
            result.json.return_value = {
                "hourly": {
                    "time": index.strftime("%Y-%m-%dT%H:%M").tolist(),
                    **{field: [10.0] * len(index) for field in fields},
                },
                "hourly_units": {field: "°C" for field in fields},
            }
            return result

        get.side_effect = response
        store = RecordingStore()
        BackfillDownloader(
            spec=spec,
            load_fetcher=self.make_fetcher("load", []),
            store=store,
        ).download()

        expected = pd.date_range(
            spec.train_start, spec.test_end, freq="h", inclusive="left"
        )
        pd.testing.assert_index_equal(store.weather.index, expected.rename("time"))
        self.assertEqual(store.weather.index.tz.zone, config.TIMEZONE)
        self.assertTrue(store.weather.eq(10.0).all().all())
        self.assertEqual(get.call_count, 3)
        previous_end = None
        for call in get.call_args_list:
            params = call.kwargs["params"]
            self.assertEqual(params["timezone"], "UTC")
            start = pd.Timestamp(params["start_date"])
            end = pd.Timestamp(params["end_date"])
            self.assertLessEqual((end - start).days + 1, 14)
            if previous_end is not None:
                self.assertEqual(start, previous_end + pd.Timedelta(days=1))
            previous_end = end


if __name__ == "__main__":
    unittest.main()
