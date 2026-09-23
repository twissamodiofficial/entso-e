import unittest
from unittest.mock import Mock

import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.storage.supabase import SupabaseRawStore


class SupabaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.table = Mock()
        self.client.table.return_value = self.table
        self.table.upsert.return_value = self.table
        self.table.select.return_value = self.table
        self.table.eq.return_value = self.table
        self.table.lt.return_value = self.table
        self.table.order.return_value = self.table
        self.table.limit.return_value = self.table
        self.table.execute.return_value = Mock(data=[])
        self.store = SupabaseRawStore(self.client, batch_size=2)

    def test_load_upsert_stores_utc_observations_in_batches(self):
        index = pd.date_range(
            "2026-01-01", periods=3, freq="15min", tz=config.TIMEZONE
        )
        frame = pd.DataFrame({config.TARGET: [1.0, None, 3.0]}, index=index)

        self.assertEqual(self.store.upsert_load(frame), 3)

        calls = self.table.upsert.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["on_conflict"], "observed_at")
        self.assertEqual(calls[0].args[0][0]["observed_at"], "2025-12-31T23:00:00+00:00")
        self.assertIsNone(calls[0].args[0][1]["actual_load_mw"])

    def test_weather_upsert_uses_selected_columns(self):
        index = pd.DatetimeIndex([pd.Timestamp("2026-07-01", tz=config.TIMEZONE)])
        frame = pd.DataFrame(
            {"apparent_temperature": [20.0], "dew_point_2m": [12.0], "extra": [99]},
            index=index,
        )

        self.assertEqual(self.store.upsert_weather(frame), 1)

        row = self.table.upsert.call_args.args[0][0]
        self.assertEqual(row["observed_at"], "2026-06-30T22:00:00+00:00")
        self.assertEqual(row["apparent_temperature"], 20.0)
        self.assertNotIn("extra", row)

    def test_reads_utc_storage_as_amsterdam_aware_index(self):
        self.store._read = Mock(return_value=[
            {
                "observed_at": "2026-01-01T23:00:00+00:00",
                "actual_load_mw": 100.0,
            },
            {
                "observed_at": "2026-07-01T22:00:00+00:00",
                "actual_load_mw": 200.0,
            },
        ])

        frame = self.store.load()

        self.assertEqual(str(frame.index.tz), config.TIMEZONE)
        self.assertEqual(frame.index[0], pd.Timestamp("2026-01-02 00:00", tz=config.TIMEZONE))
        self.assertEqual(frame.index[1], pd.Timestamp("2026-07-02 00:00", tz=config.TIMEZONE))
        self.assertEqual(frame[config.TARGET].tolist(), [100.0, 200.0])

    def test_empty_reads_still_have_amsterdam_aware_index(self):
        self.store._read = Mock(return_value=[])

        frame = self.store.weather()

        self.assertEqual(str(frame.index.tz), config.TIMEZONE)
        self.assertEqual(frame.index.name, "time")

    def test_daily_metrics_store_row_count_as_integer(self):
        metrics = pd.DataFrame(
            {
                "evaluated_rows": [24],
                "point_mae_mw": [100.5],
                "point_mape_percent": [2.5],
            },
            index=pd.Index(["2026-09-20"], name="valid_date"),
        )

        self.assertEqual(self.store.upsert_daily_metrics("run-1", metrics), 1)

        row = self.table.upsert.call_args.args[0][0]
        self.assertEqual(row["evaluated_rows"], 24)
        self.assertIsInstance(row["evaluated_rows"], int)
        self.assertEqual(row["point_mae_mw"], 100.5)

    def test_oldest_unreconciled_uses_newest_run_per_date(self):
        self.table.execute.side_effect = [
            Mock(data=[
                {
                    "run_id": "run-20-new",
                    "forecast_date": "2026-09-20",
                    "issued_at": "2026-09-20T01:00:00+00:00",
                },
                {
                    "run_id": "run-20-old",
                    "forecast_date": "2026-09-20",
                    "issued_at": "2026-09-20T00:00:00+00:00",
                },
                {
                    "run_id": "run-21",
                    "forecast_date": "2026-09-21",
                    "issued_at": "2026-09-21T00:00:00+00:00",
                },
            ]),
            Mock(data=[{"run_id": "run-20-new"}]),
            Mock(data=[]),
        ]

        run = self.store.oldest_unreconciled_forecast_run("2026-09-22")

        self.assertEqual(run["run_id"], "run-21")


if __name__ == "__main__":
    unittest.main()
