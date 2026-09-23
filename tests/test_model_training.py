import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from entso_e_pipeline import config, preprocessing
from entso_e_pipeline.features import engineering
from entso_e_pipeline.time_utils import local_day_hours

from entso_e_pipeline.modeling import point, quantile


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range(
            "2025-01-01", periods=360, freq="h", tz=config.TIMEZONE
        )
        rng = np.random.default_rng(7)
        temperature = rng.normal(10, 5, len(index))
        frame = pd.DataFrame(
            {
                "hour": index.hour,
                "day_of_week": index.dayofweek,
                "month": index.month,
                "weekend": (index.dayofweek >= 5).astype(int),
                "is_holiday": 0,
                "holiday_name": "none",
                "temperature": temperature,
                config.TARGET: 10000 - 100 * temperature
                + 20 * index.hour + rng.normal(0, 20, len(index)),
            },
            index=index,
        )
        for column, categories in config.FIXED_CATEGORIES.items():
            frame[column] = pd.Categorical(frame[column], categories=categories)
        frame["holiday_name"] = pd.Categorical(
            frame["holiday_name"], categories=["none"]
        )
        cls.training = frame.iloc[:240]
        cls.validation = frame.iloc[240:]
        cls.combined = frame

    def test_point_selection_final_fit_and_loaded_predictions(self):
        selected = point.train(self.training, self.validation, num_boost_round=8)
        self.assertGreater(selected.best_iteration, 0)

        final = point.train(self.combined, num_boost_round=selected.best_iteration)
        self.assertEqual(final.current_iteration(), selected.best_iteration)
        self.assertEqual(final.best_iteration, 0)  # No early stopping in final fit.

        features = self.validation.drop(columns=config.TARGET)
        expected = final.predict(features)
        # Column order changes must not change values or timestamp alignment.
        reordered = features.loc[:, list(reversed(features.columns))]
        predictions = point.predict(final, reordered)
        pd.testing.assert_index_equal(predictions.index, features.index)
        np.testing.assert_allclose(predictions, expected)

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "point.txt")
            point.save(final, path)
            restored = point.predict(point.load(path), features)
        np.testing.assert_allclose(restored, predictions)

    def test_quantiles_refit_with_their_own_rounds_and_round_trip(self):
        selected = quantile.train_all(
            self.training,
            self.validation,
            num_boost_round={0.1: 3, 0.5: 5, 0.9: 7},
        )
        rounds = {q: model.best_iteration for q, model in selected.items()}
        final = quantile.train_all(self.combined, num_boost_round=rounds)
        for q, model in final.items():
            self.assertGreater(rounds[q], 0)
            self.assertEqual(model.current_iteration(), rounds[q])
            self.assertEqual(model.best_iteration, 0)
            self.assertEqual(model.params["alpha"], q)

        features = self.validation.drop(columns=config.TARGET)
        predictions = quantile.predict_all(final, features)
        with tempfile.TemporaryDirectory() as directory:
            quantile.save(final, directory)
            restored = quantile.predict_all(quantile.load_all(directory), features)
        for q in config.QUANTILES:
            pd.testing.assert_index_equal(predictions[q].index, features.index)
            np.testing.assert_allclose(predictions[q], final[q].predict(features))
            np.testing.assert_allclose(predictions[q], restored[q])

    def test_final_fit_requires_selected_positive_rounds(self):
        with self.assertRaisesRegex(ValueError, "rounds selected on validation"):
            point.train(self.combined)
        with self.assertRaisesRegex(ValueError, "rounds selected on validation"):
            quantile.train_all(self.combined)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            point.train(self.combined, num_boost_round=0)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            quantile.train_one(self.combined, None, 0.5, num_boost_round=0)


    def test_source_readings_preprocess_then_build_features_and_train(self):
        index = pd.date_range("2026-10-15", "2026-10-26", freq="15min",
                              inclusive="left", tz=config.TIMEZONE)
        raw_load = pd.DataFrame(
            {config.TARGET: 10000 + 500 * np.sin(np.arange(len(index)) / 96)},
            index=index,
        ).drop(index[3])
        raw_weather = pd.DataFrame(
            {name: 10. for name in config.WEATHER_FEATURES},
            index=pd.date_range("2026-10-15", "2026-10-26", freq="h",
                                inclusive="left", tz=config.TIMEZONE),
        )
        load = preprocessing.prepare_load(raw_load)
        weather = preprocessing.prepare_weather(raw_weather)
        frame, _, _ = engineering.build_labeled_features(load, weather)
        horizon = local_day_hours("2026-10-25", config.TIMEZONE)
        train = frame.loc[frame.index < horizon[0]]
        val = frame.loc[horizon]
        features = val.drop(columns=config.TARGET)
        model = point.train(train, val, num_boost_round=4)
        models = quantile.train_all(train, val, num_boost_round={q: 4 for q in config.QUANTILES})
        results = {"point": point.predict(model, features), **quantile.predict_all(models, features)}
        for result in results.values():
            pd.testing.assert_index_equal(result.index, horizon)
            self.assertEqual(len(result), 25)
            self.assertTrue(np.isfinite(result).all())


    def test_engineered_features_train_and_predict_a_full_autumn_dst_day(self):
        horizon = local_day_hours("2026-10-25", config.TIMEZONE)
        index = pd.date_range(end=horizon[0] - pd.Timedelta(hours=1),
                              periods=600, freq="h")
        load = pd.DataFrame(
            {config.TARGET: 10000 + 500 * np.sin(np.arange(600) / 24)}, index=index
        )
        weather = pd.DataFrame(
            {"apparent_temperature": 10., "dew_point_2m": 5.},
            index=index.append(horizon),
        )
        train, load_transformer, dtf = engineering.build_labeled_features(load.iloc[:480], weather)
        val_features = pd.concat([
            engineering.transform(
                load, local_day_hours(day, config.TIMEZONE), load_transformer, dtf, weather
            )
            for day in load.index[480:].normalize().unique()
        ])
        val = val_features.join(load[[config.TARGET]])
        selected = point.train(train, val, num_boost_round=4)
        selected_quantiles = quantile.train_all(
            train, val, num_boost_round={q: 4 for q in config.QUANTILES}
        )
        combined = pd.concat([train, val])
        model = point.train(combined, num_boost_round=selected.best_iteration)
        models = quantile.train_all(
            combined,
            num_boost_round={q: m.best_iteration for q, m in selected_quantiles.items()},
        )
        future = engineering.transform(load, horizon, load_transformer, dtf, weather)
        predictions = {"point": point.predict(model, future),
                       **quantile.predict_all(models, future)}
        for values in predictions.values():
            pd.testing.assert_index_equal(values.index, horizon)
            self.assertEqual(len(values), 25)
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
