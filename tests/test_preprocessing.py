import unittest

import numpy as np
import pandas as pd

from entso_e_pipeline import config, preprocessing
from entso_e_pipeline.features import engineering
from entso_e_pipeline.modeling.schema import split_target, validate_features
from entso_e_pipeline.time_utils import local_day_hours


class PreprocessingTests(unittest.TestCase):
    def test_partial_hour_uses_available_mean_without_interpolation(self):
        index = pd.date_range("2026-01-01", periods=8, freq="15min", tz=config.TIMEZONE)
        raw = pd.DataFrame({config.TARGET: [100., 110., 120., np.nan, 1000., 1000., 1000., 1000.]}, index=index)
        original = raw.copy(deep=True)
        result = preprocessing.prepare_load(raw)
        self.assertEqual(result.iloc[0, 0], 110.)
        self.assertEqual(result.iloc[1, 0], 1000.)
        pd.testing.assert_frame_equal(raw, original)

    def test_missing_whole_hour_stays_missing(self):
        index = pd.date_range("2026-01-01", periods=12, freq="15min", tz=config.TIMEZONE)
        raw = pd.DataFrame({config.TARGET: 100.}, index=index).drop(index[4:8])
        result = preprocessing.prepare_load(raw)
        self.assertEqual(len(result), 3)
        self.assertTrue(pd.isna(result.iloc[1, 0]))

    def test_explicit_bounds_keep_missing_boundary_hours(self):
        index = local_day_hours("2026-01-01", config.TIMEZONE)
        raw = pd.DataFrame({config.TARGET: 100.}, index=index[1:-1])
        result = preprocessing.prepare_load(raw, start=index[0], end=index[-1] + pd.Timedelta(hours=1))
        pd.testing.assert_index_equal(result.index, index)
        self.assertTrue(result.iloc[[0, -1], 0].isna().all())

    def test_invalid_readings_are_cleaned_and_not_rejected_before_aggregation(self):
        index = pd.date_range("2026-01-01", periods=8, freq="15min", tz=config.TIMEZONE)
        raw = pd.DataFrame({config.TARGET: [100., "bad", np.inf, 120., None, None, None, None]}, index=index)
        result = preprocessing.prepare_load(raw)
        self.assertEqual(result.iloc[0, 0], 110.)
        self.assertTrue(pd.isna(result.iloc[1, 0]))

    def test_duplicates_are_cleaned_before_averaging(self):
        index = pd.date_range("2026-01-01", periods=4, freq="15min", tz=config.TIMEZONE)
        raw = pd.DataFrame({config.TARGET: [100., 110., 120., 130.]}, index=index)
        duplicated = pd.concat([raw.iloc[:1], raw]).iloc[::-1]
        self.assertEqual(preprocessing.prepare_load(duplicated).iloc[0, 0], 115.)
        duplicated.iloc[-1, 0] = 999.
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            preprocessing.prepare_load(duplicated)

    def test_hourly_aggregation_preserves_real_dst_instants(self):
        for day in ("2026-03-29", "2026-10-25"):
            horizon = local_day_hours(day, config.TIMEZONE)
            for frequency in ("15min", "30min", "h"):
                with self.subTest(day=day, frequency=frequency):
                    index = pd.date_range(horizon[0], horizon[-1] + pd.Timedelta(hours=1), freq=frequency, inclusive="left")
                    raw = pd.DataFrame({config.TARGET: np.arange(len(index), dtype=float)}, index=index)
                    result = preprocessing.prepare_load(raw)
                    pd.testing.assert_index_equal(result.index, horizon)
                    self.assertTrue(result[config.TARGET].is_monotonic_increasing)

    def test_preprocessing_and_features_ignore_future_values_even_with_midnight_gaps(self):
        for day in ("2026-01-11", "2026-03-30", "2026-10-26"):
            horizon = local_day_hours(day, config.TIMEZONE)
            cutoff = horizon[0]
            index = pd.date_range(cutoff - pd.DateOffset(days=10), cutoff + pd.DateOffset(days=1), freq="15min", inclusive="left")
            raw = pd.DataFrame({config.TARGET: 100.}, index=index)
            raw = raw.drop(cutoff - pd.Timedelta(minutes=15))
            original = preprocessing.prepare_load(raw)
            raw.loc[raw.index >= cutoff, config.TARGET] = 1000000.
            changed = preprocessing.prepare_load(raw)
            live = preprocessing.prepare_load(raw.loc[raw.index < cutoff])
            with self.subTest(day=day):
                pd.testing.assert_frame_equal(original.loc[original.index < cutoff], live)
                load_transformer, calendar = engineering.make_transformers()
                expected = engineering.transform(original, horizon, load_transformer, calendar)
                actual = engineering.transform(changed, horizon, load_transformer, calendar)
                pd.testing.assert_frame_equal(expected, actual)
                validate_features(actual)
                self.assertEqual(actual.iloc[-1][config.LOAD_FEATURE_NAMES["lag_day"]], 100.)

    def test_unresolved_feature_gaps_fail_only_at_final_validation(self):
        horizon = local_day_hours("2026-01-11", config.TIMEZONE)
        index = pd.date_range(pd.Timestamp("2026-01-01", tz=config.TIMEZONE), horizon[0], freq="h", inclusive="left")
        raw = pd.DataFrame({config.TARGET: 100.}, index=index).drop(index[-1])
        load = preprocessing.prepare_load(raw, start=index[0], end=horizon[0])
        features = engineering.transform(load, horizon, *engineering.make_transformers())
        self.assertTrue(features.isna().any().any())
        with self.assertRaisesRegex(ValueError, "missing values"):
            validate_features(features)

    def test_missing_labels_are_never_filled_or_silently_dropped(self):
        index = pd.date_range("2026-01-01", periods=240, freq="h", tz=config.TIMEZONE)
        raw = pd.DataFrame({config.TARGET: 100.}, index=index).drop(index[200])
        load = preprocessing.prepare_load(raw)
        frame, _, _ = engineering.build_labeled_features(load)
        self.assertEqual(len(frame), 72)
        self.assertTrue(pd.isna(frame.loc[index[200], config.TARGET]))
        with self.assertRaisesRegex(ValueError, "missing values"):
            split_target(frame)

    def test_weather_gaps_are_cleaned_and_checked_after_feature_engineering(self):
        index = pd.date_range("2026-01-01", periods=240, freq="h", tz=config.TIMEZONE)
        raw_load = pd.DataFrame({config.TARGET: 100.}, index=index)
        raw_weather = pd.DataFrame({name: 10. for name in config.WEATHER_FEATURES}, index=index).drop(index[200])
        raw_weather.iloc[201, 0] = np.inf
        weather = preprocessing.prepare_weather(raw_weather)
        frame, _, _ = engineering.build_labeled_features(
            preprocessing.prepare_load(raw_load), weather
        )
        self.assertTrue(frame.loc[index[200], config.WEATHER_FEATURES].isna().all())
        with self.assertRaisesRegex(ValueError, "missing values"):
            split_target(frame)


if __name__ == "__main__":
    unittest.main()
