import unittest

import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.ingestion.load import fetch_load


class FakeLoadClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def query_load(self, country, start, end):
        self.calls.append((country, start, end))
        return self.response


class LoadAdapterTests(unittest.TestCase):
    def test_missing_readings_and_duplicates_are_left_for_preprocessing(self):
        index = pd.date_range("2026-07-01", periods=8, freq="15min", tz=config.TIMEZONE)
        raw = pd.DataFrame({config.TARGET: range(8)}, index=index).astype(float)
        raw.iloc[1, 0] = float("nan")
        raw = pd.concat([raw.drop(index[3]), raw.iloc[-1:]])
        result = fetch_load(index[0], index[-1] + pd.Timedelta(minutes=15), FakeLoadClient(raw))
        pd.testing.assert_frame_equal(result, raw)

    def test_fetch_load_preserves_source_readings_without_resampling(self):
        raw_index = pd.date_range(
            "2026-07-01 14:00", periods=8, freq="15min", tz=config.TIMEZONE
        )
        client = FakeLoadClient(
            pd.DataFrame(
                {config.TARGET: range(100, 108)},
                index=raw_index,
            )
        )

        result = fetch_load(
            "2026-07-01 14:00",
            "2026-07-01 15:00",
            client=client,
        )

        self.assertEqual(len(result), 4)
        self.assertEqual(result.index[0].tz.zone, config.TIMEZONE)
        self.assertEqual(result[config.TARGET].tolist(), [100, 101, 102, 103])
        self.assertEqual(client.calls[0][0], config.COUNTRY_CODE)


if __name__ == "__main__":
    unittest.main()
