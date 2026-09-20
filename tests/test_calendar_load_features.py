import pickle
import unittest

import numpy as np
import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.features import engineering
from entso_e_pipeline.time_utils import local_day_hours
from tests.helpers import validated_training, validated_transform


class CalendarLoadFeatureTests(unittest.TestCase):
    def history(self, day, days=10):
        origin = pd.Timestamp(day, tz=config.TIMEZONE)
        index = pd.date_range(
            origin - pd.DateOffset(days=days), origin, freq="h", inclusive="left"
        )
        return pd.DataFrame(
            {config.TARGET: (index.dayofyear * 100 + index.hour).astype(float)},
            index=index,
        )

    def forecast(self, day, history=None):
        if history is None:
            history = self.history(day)
        _, load_transformer, calendar = validated_training(history)
        horizon = local_day_hours(day, config.TIMEZONE)
        weather = pd.DataFrame(
            {name: 10. for name in config.WEATHER_FEATURES}, index=horizon
        )
        return validated_transform(
            history, horizon, load_transformer, calendar, weather
        )

    def test_daily_and_weekly_lags_match_clock_hours_across_dst_and_new_year(self):
        cases = (
            ("2026-03-29", "2026-03-28", "2026-03-22"),
            ("2026-10-25", "2026-10-24", "2026-10-18"),
            ("2026-01-01", "2025-12-31", "2025-12-25"),
        )
        for day, yesterday, last_week in cases:
            history = self.history(day)
            result = self.forecast(day, history)
            for hour in (3, 9, 23):
                with self.subTest(day=day, hour=hour):
                    row = result.loc[pd.Timestamp(f"{day} {hour}:00", tz=config.TIMEZONE)]
                    daily_source = pd.Timestamp(f"{yesterday} {hour}:00", tz=config.TIMEZONE)
                    weekly_source = pd.Timestamp(f"{last_week} {hour}:00", tz=config.TIMEZONE)
                    self.assertEqual(row[config.LOAD_FEATURE_NAMES["lag_day"]],
                                     history.loc[daily_source, config.TARGET])
                    self.assertEqual(row[config.LOAD_FEATURE_NAMES["lag_week"]],
                                     history.loc[weekly_source, config.TARGET])

    def test_both_repeated_forecast_hours_use_the_same_reference_values(self):
        history = self.history("2026-10-25")
        history.loc[pd.Timestamp("2026-10-24 02:00", tz=config.TIMEZONE), config.TARGET] = 100.
        history.loc[pd.Timestamp("2026-10-18 02:00", tz=config.TIMEZONE), config.TARGET] = 200.
        result = self.forecast("2026-10-25", history)
        repeated = result.loc[result.index.hour == 2]
        self.assertEqual(len(repeated), 2)
        self.assertTrue(repeated.index.is_unique)
        self.assertEqual(repeated[config.LOAD_FEATURE_NAMES["lag_day"]].tolist(), [100., 100.])
        self.assertEqual(repeated[config.LOAD_FEATURE_NAMES["lag_week"]].tolist(), [200., 200.])
        self.assertEqual(repeated.iloc[0][config.LOAD_FEATURE_NAMES["mean_day"]],
                         repeated.iloc[1][config.LOAD_FEATURE_NAMES["mean_day"]])
        self.assertEqual(len(result), 25)

    def test_repeated_reference_hour_is_averaged_for_daily_and_weekly_lags(self):
        for day, feature in (("2026-10-26", "lag_day"), ("2026-11-01", "lag_week")):
            with self.subTest(day=day):
                history = self.history(day)
                reference = (history.index.date == pd.Timestamp("2026-10-25").date()) & (
                    history.index.hour == 2
                )
                history.loc[reference, config.TARGET] = [100., 500.]
                result = self.forecast(day, history)
                row = result.loc[pd.Timestamp(f"{day} 02:00", tz=config.TIMEZONE)]
                self.assertEqual(row[config.LOAD_FEATURE_NAMES[feature]], 300.)

    def test_nonexistent_reference_hour_is_interpolated_for_daily_and_weekly_lags(self):
        for day, feature in (("2026-03-30", "lag_day"), ("2026-04-05", "lag_week")):
            with self.subTest(day=day):
                history = self.history(day)
                history.loc[pd.Timestamp("2026-03-29 01:00", tz=config.TIMEZONE), config.TARGET] = 100.
                history.loc[pd.Timestamp("2026-03-29 03:00", tz=config.TIMEZONE), config.TARGET] = 300.
                result = self.forecast(day, history)
                row = result.loc[pd.Timestamp(f"{day} 02:00", tz=config.TIMEZONE)]
                self.assertEqual(row[config.LOAD_FEATURE_NAMES[feature]], 200.)

    def test_clock_hour_mean_uses_the_same_dst_policies(self):
        for day, reference_day, expected in (
            ("2026-03-30", "2026-03-29", (276 - 1 - 2 - 3 + 100 + 200 + 300) / 24),
            ("2026-10-26", "2026-10-25", (276 - 2 + 200) / 24),
        ):
            with self.subTest(day=day):
                history = self.history(day)
                reference = history.index.date == pd.Timestamp(reference_day).date()
                history.loc[reference, config.TARGET] = history.index[reference].hour.astype(float)
                if reference_day == "2026-03-29":
                    history.loc[reference & (history.index.hour == 1), config.TARGET] = 100.
                    history.loc[reference & (history.index.hour == 3), config.TARGET] = 300.
                else:
                    history.loc[reference & (history.index.hour == 2), config.TARGET] = [100., 300.]
                result = self.forecast(day, history)
                self.assertAlmostEqual(
                    result.iloc[-1][config.LOAD_FEATURE_NAMES["mean_day"]], expected
                )

    def test_training_warmup_is_seven_calendar_days(self):
        for start, expected_rows in (("2026-03-23", 167), ("2026-10-19", 169)):
            with self.subTest(start=start):
                first = pd.Timestamp(start, tz=config.TIMEZONE)
                cutoff = first + pd.DateOffset(days=7)
                index = pd.date_range(first, first + pd.DateOffset(days=9),
                                      freq="h", inclusive="left")
                load = pd.DataFrame({config.TARGET: np.arange(len(index), dtype=float)}, index=index)
                result, _, _ = validated_training(load)
                self.assertEqual(result.index[0], cutoff)
                self.assertEqual(len(load) - len(result), expected_rows)
                self.assertTrue(result.index.equals(index[index >= cutoff]))
                self.assertFalse(result.isna().any().any())

    def test_full_and_past_only_inputs_accept_exact_calendar_history_and_reject_short_history(self):
        _, load_transformer, calendar = validated_training(self.history("2026-01-11"))
        for day, expected_rows in (("2026-03-30", 167), ("2026-10-26", 169)):
            with self.subTest(day=day):
                history = self.history(day, days=7)
                self.assertEqual(len(history), expected_rows)
                horizon = local_day_hours(day, config.TIMEZONE)
                weather = pd.DataFrame({name: 10. for name in config.WEATHER_FEATURES}, index=horizon)
                result = validated_transform(history, horizon, load_transformer, calendar, weather)
                self.assertEqual(result.iloc[0][config.LOAD_FEATURE_NAMES["lag_week"]], history.iloc[0, 0])
                actual = pd.DataFrame({config.TARGET: -1.}, index=horizon)
                historical = validated_transform(
                    pd.concat([history, actual]), horizon, load_transformer, calendar, weather
                )
                pd.testing.assert_frame_equal(result, historical)
                with self.assertRaisesRegex(ValueError, "missing values"):
                    validated_transform(history.iloc[1:], horizon, load_transformer, calendar, weather)
                with self.assertRaisesRegex(ValueError, "missing values"):
                    validated_transform(
                        pd.concat([history.iloc[1:], actual]), horizon,
                        load_transformer, calendar, weather,
                    )

    def test_real_missing_reference_observations_are_never_interpolated(self):
        history = self.history("2026-03-30")
        missing = pd.Timestamp("2026-03-29 01:00", tz=config.TIMEZONE)
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.forecast("2026-03-30", history.drop(missing))
        history.loc[missing, config.TARGET] = np.nan
        with self.assertRaisesRegex(ValueError, "missing values"):
            self.forecast("2026-03-30", history)

    def test_serialized_transformers_preserve_dst_features(self):
        day = "2026-11-01"
        history = self.history(day)
        _, load_transformer, calendar = validated_training(history)
        restored_load, restored_calendar = pickle.loads(pickle.dumps((load_transformer, calendar)))
        horizon = local_day_hours(day, config.TIMEZONE)
        weather = pd.DataFrame({name: 10. for name in config.WEATHER_FEATURES}, index=horizon)
        expected = validated_transform(history, horizon, load_transformer, calendar, weather)
        restored = validated_transform(history, horizon, restored_load, restored_calendar, weather)
        pd.testing.assert_frame_equal(expected, restored)

    def test_calendar_values_on_holidays_and_dst_days(self):
        for day, hours, weekday, month, weekend, holiday in (
            ("2026-01-01", list(range(24)), 3, 1, 0, 1),
            ("2026-03-29", [0, 1, *range(3, 24)], 6, 3, 1, 0),
            ("2026-10-25", [0, 1, 2, *range(2, 24)], 6, 10, 1, 0),
        ):
            with self.subTest(day=day):
                result = self.forecast(day)
                self.assertEqual(result["hour"].tolist(), hours)
                for column, value in (
                    ("day_of_week", weekday), ("month", month),
                    ("weekend", weekend), ("is_holiday", holiday),
                ):
                    self.assertTrue(result[column].eq(value).all())
                for column, categories in config.FIXED_CATEGORIES.items():
                    self.assertEqual(result[column].cat.categories.tolist(), categories)


if __name__ == "__main__":
    unittest.main()
