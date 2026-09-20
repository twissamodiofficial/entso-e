import unittest

import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.common.normalization import to_local_weather
from entso_e_pipeline.time_utils import as_local, as_local_index, local_day_hours


class TimeNormalizationTests(unittest.TestCase):
    def test_utc_timestamp_is_converted_to_amsterdam(self):
        self.assertEqual(
            as_local("2026-09-18T08:00:00Z", config.TIMEZONE),
            pd.Timestamp("2026-09-18 10:00:00", tz=config.TIMEZONE),
        )

    def test_local_days_preserve_dst_row_counts(self):
        self.assertEqual(len(local_day_hours("2026-03-29", config.TIMEZONE)), 23)
        autumn = local_day_hours("2026-10-25", config.TIMEZONE)
        self.assertEqual(len(autumn), 25)
        self.assertEqual(list(autumn.hour).count(2), 2)

    def test_mixed_offsets_are_preserved_as_distinct_instants(self):
        result = as_local_index([
            "2026-10-25 02:00:00+02:00", "2026-10-25 02:00:00+01:00",
        ], config.TIMEZONE)
        self.assertNotEqual(result[0].value, result[1].value)

    def test_weather_decoding_preserves_both_autumn_hours(self):
        raw = pd.DataFrame({
            "time": ["2026-10-25T00:00Z", "2026-10-25T01:00Z"],
            "apparent_temperature": [10.0, 11.0], "dew_point_2m": [5.0, 6.0],
        })
        result = to_local_weather(raw)
        self.assertEqual(result.index.hour.tolist(), [2, 2])
        self.assertNotEqual(result.index[0], result.index[1])

    def test_weather_decoding_does_not_clean_source_duplicates(self):
        raw = pd.DataFrame({
            "time": ["2026-01-01T00:00Z"] * 2,
            "apparent_temperature": [10.0, 99.0], "dew_point_2m": [5.0, 5.0],
        })
        self.assertEqual(len(to_local_weather(raw)), 2)


if __name__ == "__main__":
    unittest.main()
