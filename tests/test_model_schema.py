import unittest

import pandas as pd

from entso_e_pipeline import config
from entso_e_pipeline.modeling.schema import (
    select_prediction_features,
    split_target,
    validate_train_val_schema,
    validate_temporal_splits,
)


class FakeModel:
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def feature_name(self):
        return self._feature_names


class ModelSchemaTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range(
            "2026-01-01",
            periods=2,
            freq="h",
            tz=config.TIMEZONE,
        )
        self.train = pd.DataFrame(
            {
                "hour": pd.Categorical([1, 2], categories=list(range(24))),
                "day_of_week": pd.Categorical([3, 3], categories=list(range(7))),
                "month": pd.Categorical([1, 1], categories=list(range(1, 13))),
                "weekend": pd.Categorical([0, 0], categories=[0, 1]),
                "is_holiday": pd.Categorical([0, 0], categories=[0, 1]),
                "holiday_name": pd.Categorical(["none", "none"]),
                "temperature": [10.0, 11.0],
                config.TARGET: [100.0, 110.0],
            },
            index=index,
        )
        self.validation = self.train.copy()
        self.validation.index += pd.Timedelta(hours=2)

    def test_validation_uses_training_feature_order(self):
        validation = self.validation[
            [
                config.TARGET,
                "temperature",
                "holiday_name",
                "is_holiday",
                "weekend",
                "month",
                "day_of_week",
                "hour",
            ]
        ]

        self.assertEqual(
            validate_train_val_schema(self.train, validation),
            [
                "hour",
                "day_of_week",
                "month",
                "weekend",
                "is_holiday",
                "holiday_name",
                "temperature",
            ],
        )

    def test_validation_rejects_missing_feature(self):
        validation = self.validation.drop(columns="temperature")

        with self.assertRaisesRegex(ValueError, "missing features"):
            validate_train_val_schema(self.train, validation)

    def test_split_target_returns_feature_only_frame_and_target(self):
        features, target = split_target(self.train)

        self.assertEqual(
            list(features.columns),
            [
                "hour",
                "day_of_week",
                "month",
                "weekend",
                "is_holiday",
                "holiday_name",
                "temperature",
            ],
        )
        self.assertNotIn(config.TARGET, features.columns)
        self.assertEqual(target.tolist(), [100.0, 110.0])

    def test_prediction_reorders_columns_using_model_schema(self):
        frame = self.train[["temperature", "hour"]].copy()
        selected = select_prediction_features(
            FakeModel(["hour", "temperature"]),
            frame,
        )

        self.assertEqual(list(selected.columns), ["hour", "temperature"])

    def test_prediction_rejects_target_column(self):
        with self.assertRaisesRegex(ValueError, "must not contain the target"):
            select_prediction_features(
                FakeModel(["hour", "temperature"]),
                self.train,
            )

    def test_prediction_rejects_unexpected_feature(self):
        frame = self.train.drop(columns=config.TARGET).assign(unexpected=1.0)

        with self.assertRaisesRegex(ValueError, "unexpected features"):
            select_prediction_features(FakeModel(["hour", "temperature"]), frame)

    def test_training_rejects_missing_target_value(self):
        training = self.train.copy()
        training.loc[training.index[0], config.TARGET] = float("nan")

        with self.assertRaisesRegex(ValueError, "missing values"):
            validate_train_val_schema(training, self.validation)

    def test_prediction_rejects_missing_feature_value(self):
        frame = self.train[["hour", "temperature"]].copy()
        frame.loc[frame.index[0], "temperature"] = float("nan")

        with self.assertRaisesRegex(ValueError, "missing values"):
            select_prediction_features(
                FakeModel(["hour", "temperature"]),
                frame,
            )


    def test_training_rejects_infinite_target_and_features(self):
        for column in (config.TARGET, "temperature"):
            for value in (float("inf"), float("-inf")):
                with self.subTest(column=column, value=value):
                    frame = self.train.copy()
                    frame.loc[frame.index[0], column] = value
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        split_target(frame)

    def test_prediction_rejects_infinite_features(self):
        frame = self.train[["hour", "temperature"]].copy()
        frame.loc[frame.index[0], "temperature"] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            select_prediction_features(FakeModel(list(frame.columns)), frame)

    def test_train_validation_overlap_or_reversed_order_is_rejected(self):
        for validation in (self.train.copy(), self.train.set_axis(
                self.train.index - pd.Timedelta(days=1))):
            with self.assertRaisesRegex(ValueError, "chronological and non-overlapping"):
                validate_train_val_schema(self.train, validation)

    def test_all_four_splits_require_chronological_separation(self):
        frames = {
            name: self.train.set_axis(self.train.index + pd.Timedelta(days=i))
            for i, name in enumerate(("train", "val", "calibration", "test"))
        }
        validate_temporal_splits(frames)
        for name in ("calibration", "test"):
            invalid = {**frames, name: self.train}
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "chronological and non-overlapping"):
                    validate_temporal_splits(invalid)

    def test_model_frames_reject_invalid_time_indexes(self):
        for index in (self.train.index[::-1], self.train.index.tz_localize(None),
                      pd.DatetimeIndex([self.train.index[0]] * 2),
                      self.train.index + pd.Timedelta(minutes=15),
                      pd.date_range(self.train.index[0], periods=2, freq="2h")):
            with self.subTest(index=str(index)):
                with self.assertRaises(ValueError):
                    split_target(self.train.set_axis(index))


if __name__ == "__main__":
    unittest.main()
