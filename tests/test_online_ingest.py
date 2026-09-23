import unittest
from unittest.mock import patch

import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.online.ingest import ensure_actual_day, ensure_forecast_history


class MemoryStore:
    def __init__(self, frame):
        self.frame = frame.copy()

    def load(self, start, end):
        return self.frame.loc[
            (self.frame.index >= start) & (self.frame.index < end)
        ].copy()

    def upsert_load(self, frame):
        if frame.empty:
            return 0
        self.frame = pd.concat([self.frame, frame])
        self.frame = self.frame[~self.frame.index.duplicated(keep="last")].sort_index()
        return len(frame)


def hourly_load(start, end, value=100.0):
    index = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame({config.TARGET: value}, index=index)


class OnlineIngestTests(unittest.TestCase):
    def test_catches_up_from_first_missing_day_through_cutoff(self):
        reference = pd.Timestamp("2026-01-10", tz=config.TIMEZONE)
        history_start = reference - pd.DateOffset(days=config.LOOKBACK_DAYS)
        missing_day = reference - pd.DateOffset(days=2)
        existing = pd.concat([
            hourly_load(history_start, missing_day),
            hourly_load(reference - pd.DateOffset(days=1), reference),
        ])
        store = MemoryStore(existing)
        calls = []

        def fetch(start, end):
            calls.append((start, end))
            return hourly_load(start, end, value=110.0)

        with patch("entso_e_pipeline.online.ingest.fetch_load", fetch):
            rows = ensure_forecast_history(reference, store)

        self.assertEqual(rows, 48)
        self.assertEqual(calls, [(missing_day, reference)])

    def test_complete_history_does_not_fetch_again(self):
        reference = pd.Timestamp("2026-01-10", tz=config.TIMEZONE)
        history_start = reference - pd.DateOffset(days=config.LOOKBACK_DAYS)
        store = MemoryStore(hourly_load(history_start, reference))

        with patch("entso_e_pipeline.online.ingest.fetch_load") as fetch:
            self.assertEqual(ensure_forecast_history(reference, store), 0)

        fetch.assert_not_called()

    def test_unresolved_history_blocks_forecast(self):
        reference = pd.Timestamp("2026-01-10", tz=config.TIMEZONE)
        history_start = reference - pd.DateOffset(days=config.LOOKBACK_DAYS)
        store = MemoryStore(hourly_load(history_start, reference).iloc[:24])

        with patch(
            "entso_e_pipeline.online.ingest.fetch_load",
            return_value=pd.DataFrame({config.TARGET: []}),
        ):
            with self.assertRaisesRegex(RuntimeError, "still incomplete"):
                ensure_forecast_history(reference, store)

    def test_actual_day_fetches_only_the_completed_day(self):
        day = pd.Timestamp("2026-01-10", tz=config.TIMEZONE)
        store = MemoryStore(hourly_load(day, day + pd.DateOffset(days=1)).iloc[:12])
        calls = []

        def fetch(start, end):
            calls.append((start, end))
            return hourly_load(start, end, value=110.0)

        with patch("entso_e_pipeline.online.ingest.fetch_load", fetch):
            rows = ensure_actual_day(day, store)

        self.assertEqual(rows, 24)
        self.assertEqual(
            calls,
            [(day, day + pd.DateOffset(days=1))],
        )


if __name__ == "__main__":
    unittest.main()
