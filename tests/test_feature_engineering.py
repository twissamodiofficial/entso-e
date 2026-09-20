import unittest

import numpy as np
import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.features import engineering
from entso_e_pipeline.time_utils import local_day_hours
from tests.helpers import validated_training, validated_transform


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.history_index = pd.date_range(
            "2026-01-01 00:00",
            periods=240,
            freq="h",
            tz=config.TIMEZONE,
        )
        self.load = pd.DataFrame(
            {config.TARGET: range(len(self.history_index))},
            index=self.history_index,
        )
        self.weather = pd.DataFrame(
            {
                "apparent_temperature": 10.0,
                "dew_point_2m": 5.0,
            },
            index=self.history_index,
        )
        self.features, self.load_transformer, self.dtf = validated_training(
            self.load, self.weather
        )

    def test_training_discards_only_feature_warmup_rows(self):
        self.assertEqual(len(self.features), len(self.load) - (config.LOOKBACK_DAYS * 24))
        self.assertIn(config.LOAD_FEATURE_NAMES["lag_day"], self.features.columns)
        self.assertIn(config.LOAD_FEATURE_NAMES["lag_week"], self.features.columns)
        self.assertIn(config.LOAD_FEATURE_NAMES["mean_day"], self.features.columns)
        self.assertNotIn("Actual Load_lag_24", self.features.columns)
        self.assertIn("apparent_temperature", self.features.columns)

    def test_future_features_cover_the_requested_horizon(self):
        horizon = pd.date_range(
            self.history_index[-1] + pd.Timedelta(hours=1),
            periods=24,
            freq="h",
        )
        forecast_weather = pd.DataFrame(
            {
                "apparent_temperature": 11.0,
                "dew_point_2m": 6.0,
            },
            index=horizon,
        )

        result = validated_transform(
            self.load,
            horizon,
            self.load_transformer,
            self.dtf,
            forecast_weather,
        )

        self.assertTrue(result.index.equals(horizon))
        self.assertNotIn(config.TARGET, result.columns)

    def test_future_prediction_rejects_missing_weather_hours(self):
        horizon = pd.date_range(
            self.history_index[-1] + pd.Timedelta(hours=1),
            periods=24,
            freq="h",
        )
        forecast_weather = pd.DataFrame(
            {
                "apparent_temperature": 11.0,
                "dew_point_2m": 6.0,
            },
            index=horizon,
        ).drop(index=horizon[3])

        with self.assertRaisesRegex(ValueError, "missing values"):
            validated_transform(
                self.load,
                horizon,
                self.load_transformer,
                self.dtf,
                forecast_weather,
            )


    def test_training_without_weather_removes_only_warmup(self):
        result, _, _ = validated_training(self.load)
        self.assertTrue(result.index.equals(self.load.index[(config.LOOKBACK_DAYS * 24):]))
        self.assertFalse(result.isna().any().any())

    def test_ordinary_lag_values_are_unchanged(self):
        row = self.features.loc[self.load.index[200]]
        self.assertEqual(row[config.LOAD_FEATURE_NAMES["lag_day"]], 176)
        self.assertEqual(row[config.LOAD_FEATURE_NAMES["lag_week"]], 32)
        self.assertEqual(row[config.LOAD_FEATURE_NAMES["mean_day"]], 164.5)

    def test_training_rejects_missing_and_infinite_load_without_dropping_rows(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                load = self.load.astype(float)
                load.iloc[200, 0] = value
                with self.assertRaisesRegex(ValueError, "missing|non-finite"):
                    validated_training(load, self.weather)

    def test_training_rejects_gapped_history(self):
        load = self.load.drop(self.load.index[200])
        with self.assertRaisesRegex(ValueError, "missing values"):
            validated_training(load, self.weather)

    def test_training_rejects_nonfinite_weather(self):
        weather = self.weather.copy()
        weather.iloc[200, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validated_training(self.load, weather)

    def test_historical_days_use_actuals_only_after_the_cutoff_advances(self):
        index = pd.date_range(self.load.index[-1] + pd.Timedelta(hours=1),
                              periods=72, freq="h")
        actuals = pd.DataFrame({config.TARGET: np.arange(240., 312.)}, index=index)
        full_load = pd.concat([self.load, actuals])
        result = pd.concat([
            validated_transform(
                full_load, local_day_hours(day, config.TIMEZONE),
                self.load_transformer, self.dtf,
            )
            for day in index.normalize().unique()
        ])
        self.assertTrue(result.index.equals(index))
        self.assertNotIn(config.TARGET, result.columns)
        self.assertFalse(result.isna().any().any())
        daily = config.LOAD_FEATURE_NAMES["lag_day"]
        self.assertEqual(result.iloc[0][daily], 216.)
        self.assertEqual(result.iloc[24][daily], 240.)
        self.assertEqual(result.iloc[48][daily], 264.)

        changed = full_load.copy()
        first_day = local_day_hours(index[0], config.TIMEZONE)
        second_day = local_day_hours(index[24], config.TIMEZONE)
        changed.loc[first_day, config.TARGET] += 1000.
        first = validated_transform(
            changed, first_day, self.load_transformer, self.dtf
        )
        pd.testing.assert_frame_equal(first, result.loc[first_day])
        second = validated_transform(
            changed, second_day, self.load_transformer, self.dtf
        )
        np.testing.assert_allclose(second[daily], result.loc[second_day, daily] + 1000.)

    def test_only_selected_history_is_validated(self):
        horizon = local_day_hours("2026-01-11", config.TIMEZONE)
        expected = validated_transform(
            self.load, horizon, self.load_transformer, self.dtf
        )
        future_index = horizon.append(local_day_hours("2026-01-12", config.TIMEZONE))
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                unused = pd.DataFrame({config.TARGET: value}, index=future_index)
                full_load = pd.concat([self.load, unused]).astype(float)
                # Invalid older observations and missing target-day rows are
                # irrelevant to the selected seven-day history as well.
                full_load.iloc[0, 0] = value
                full_load = full_load.drop(horizon[4])
                actual = validated_transform(
                    full_load, horizon, self.load_transformer, self.dtf
                )
                pd.testing.assert_frame_equal(actual, expected)

                bad_history = self.load.astype(float)
                bad_history.iloc[-1, 0] = value
                with self.assertRaisesRegex(ValueError, "missing|non-finite"):
                    validated_transform(
                        bad_history, horizon, self.load_transformer, self.dtf
                    )

    def test_missing_target_day_hour_is_rejected_when_it_becomes_history(self):
        first_day = local_day_hours("2026-01-11", config.TIMEZONE)
        second_day = local_day_hours("2026-01-12", config.TIMEZONE)
        actuals = pd.DataFrame({config.TARGET: 10.}, index=first_day).drop(first_day[4])
        full_load = pd.concat([self.load, actuals])
        expected = validated_transform(
            self.load, first_day, self.load_transformer, self.dtf
        )
        actual = validated_transform(
            full_load, first_day, self.load_transformer, self.dtf
        )
        pd.testing.assert_frame_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "missing values"):
            validated_transform(full_load, second_day, self.load_transformer, self.dtf)

    def test_history_must_reach_the_forecast_day_boundary(self):
        horizon = local_day_hours("2026-01-11", config.TIMEZONE)
        actuals = pd.DataFrame({config.TARGET: 10.}, index=horizon)
        with self.assertRaisesRegex(ValueError, "missing values"):
            validated_transform(
                pd.concat([self.load.iloc[:-1], actuals]), horizon,
                self.load_transformer, self.dtf,
            )

    def test_training_uses_full_days_after_a_partial_initial_day(self):
        result, _, _ = validated_training(self.load.iloc[1:])
        expected = self.features.loc[self.features.index >= "2026-01-09"].drop(
            columns=config.WEATHER_FEATURES
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_training_rejects_a_partial_final_day(self):
        with self.assertRaisesRegex(ValueError, "missing values"):
            validated_training(self.load.iloc[:-1])

    def test_future_rejects_stale_history_and_invalid_horizons(self):
        tomorrow = local_day_hours("2026-01-11", config.TIMEZONE)
        for horizon in (tomorrow[1:], tomorrow[:12], tomorrow.delete(4),
                        tomorrow.append(tomorrow[-1:]), tomorrow[::-1],
                        tomorrow.tz_localize(None)):
            with self.subTest(horizon=str(horizon)):
                weather = pd.DataFrame({name: 10. for name in config.WEATHER_FEATURES},
                                       index=horizon)
                with self.assertRaises(ValueError):
                    validated_transform(self.load, horizon, self.load_transformer,
                                                 self.dtf, weather)
        horizon = local_day_hours("2026-01-12", config.TIMEZONE)
        weather = pd.DataFrame({name: 10. for name in config.WEATHER_FEATURES},
                               index=horizon)
        with self.assertRaisesRegex(ValueError, "missing values"):
            validated_transform(self.load, horizon, self.load_transformer,
                                         self.dtf, weather)

    def test_historical_and_future_features_match_across_dst_without_target_leakage(self):
        for day, hours in (
            ("2026-01-11", 24), ("2026-03-29", 23), ("2026-03-30", 24),
            ("2026-04-05", 24), ("2026-10-25", 25), ("2026-10-26", 24),
            ("2026-11-01", 24),
        ):
            with self.subTest(day=day):
                horizon = local_day_hours(day, config.TIMEZONE)
                index = pd.date_range(end=horizon[0] - pd.Timedelta(hours=1),
                                      periods=240, freq="h")
                history = pd.DataFrame({config.TARGET: np.arange(240.)}, index=index)
                weather_index = index.append(horizon)
                weather = pd.DataFrame(
                    {name: 10. for name in config.WEATHER_FEATURES}, index=weather_index
                )
                _, load_transformer, dtf = validated_training(history, weather.loc[index])
                future = validated_transform(history, horizon, load_transformer, dtf, weather)
                self.assertEqual(len(future), hours)
                for scale in (1000., -1000.):
                    actual = pd.DataFrame({config.TARGET: scale + np.arange(hours)},
                                          index=horizon)
                    historical = validated_transform(
                        pd.concat([history, actual]), horizon, load_transformer, dtf, weather
                    )
                    pd.testing.assert_frame_equal(future, historical)
                    full, _, _ = validated_training(
                        pd.concat([history, actual]), weather
                    )
                    pd.testing.assert_frame_equal(
                        future, full.loc[horizon].drop(columns=config.TARGET)
                    )
                    pd.testing.assert_series_equal(full.loc[horizon, config.TARGET], actual[config.TARGET])
                for value in (np.nan, np.inf, -np.inf):
                    unavailable = pd.DataFrame({config.TARGET: value}, index=horizon)
                    historical = validated_transform(
                        pd.concat([history, unavailable]), horizon,
                        load_transformer, dtf, weather,
                    )
                    pd.testing.assert_frame_equal(future, historical)
                if hours == 25:
                    self.assertEqual(future.iloc[-1][config.LOAD_FEATURE_NAMES["lag_day"]],
                                     239.)
                    self.assertEqual(
                        future.iloc[-1][config.LOAD_FEATURE_NAMES["mean_day"]], 227.5
                    )

    def test_utc_inputs_produce_local_calendar_features(self):
        load = self.load.tz_convert("UTC")
        features, _, _ = validated_training(load, self.weather)
        pd.testing.assert_frame_equal(features, self.features)

    def test_categorical_cast_leaves_unknown_values_missing_for_final_validation(self):
        for value in (None, 99):
            with self.subTest(value=value):
                result = engineering.cast_categorical_features(pd.DataFrame({"hour": [value]}))
                self.assertTrue(result["hour"].isna().all())


if __name__ == "__main__":
    unittest.main()
