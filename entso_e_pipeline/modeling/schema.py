"""Shared feature and value checks for model training and prediction."""

import pandas as pd

from .. import config
from ..common.normalization import validate_numeric_values
from ..time_utils import validate_hourly_index


def _validate_frame(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError(f"{label} is empty.")
    if not frame.columns.is_unique:
        raise ValueError(f"{label} contains duplicate column names.")
    validate_hourly_index(frame.index, label)


def _require_target(frame: pd.DataFrame, label: str) -> None:
    if config.TARGET not in frame.columns:
        raise ValueError(f"{label} is missing target {config.TARGET!r}.")


def _validate_values(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    for column in columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            if series.isna().any():
                raise ValueError(f"{label} contains missing values in {column!r}.")
        else:
            validate_numeric_values(frame[[column]], label)


def _validate_categorical_dtypes(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    require_all: bool,
) -> None:
    configured = list(config.CATEGORICAL_FEATURES)
    missing = [column for column in configured if column not in feature_cols]
    if require_all and missing:
        raise ValueError(
            f"{label} is missing configured categorical features: {missing}"
        )

    invalid = [
        column
        for column in configured
        if column in feature_cols
        and not isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]
    if invalid:
        raise TypeError(
            f"{label} configured categorical features must use pandas "
            f"categorical dtype: {invalid}"
        )


def _validate_labeled_frame(frame: pd.DataFrame, label: str) -> list[str]:
    _validate_frame(frame, label)
    _require_target(frame, label)
    feature_cols = [column for column in frame.columns if column != config.TARGET]
    if not feature_cols:
        raise ValueError(f"{label} contains no feature columns.")

    if not pd.api.types.is_numeric_dtype(frame[config.TARGET].dtype):
        raise TypeError(f"{label} target must be numeric.")
    _validate_values(frame, [config.TARGET, *feature_cols], label)
    _validate_categorical_dtypes(frame, feature_cols, label, require_all=True)
    return feature_cols


def _validate_feature_dtypes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    for column in feature_cols:
        train_dtype = train_df[column].dtype
        val_dtype = val_df[column].dtype
        if train_dtype != val_dtype:
            raise TypeError(
                f"Feature {column!r} has different dtypes: "
                f"training={train_dtype}, validation={val_dtype}."
            )


def _raise_schema_mismatch(
    label: str,
    missing: list[str],
    unexpected: list[str],
) -> None:
    problems = []
    if missing:
        problems.append(f"missing features: {missing}")
    if unexpected:
        problems.append(f"unexpected features: {unexpected}")
    if problems:
        raise ValueError(f"{label} feature schema mismatch; " + "; ".join(problems))


def split_target(
    frame: pd.DataFrame,
    label: str = "Labeled data",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled frame into feature-only ``X`` and target ``y``."""

    feature_cols = _validate_labeled_frame(frame, label)
    return frame.loc[:, feature_cols].copy(), frame.loc[:, config.TARGET].copy()


def validate_features(frame: pd.DataFrame) -> None:
    """Final check after preprocessing and feature engineering, before prediction."""

    _validate_frame(frame, "Features")
    if config.TARGET in frame.columns:
        raise ValueError("Features must not contain the target column.")
    columns = list(frame.columns)
    _validate_values(frame, columns, "Features")
    _validate_categorical_dtypes(frame, columns, "Features", require_all=True)


def validate_temporal_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Require complete splits supplied in train/val/calibration/test order.

    Splits may be separated by time gaps, but cannot overlap or run backwards.
    Call with all four frames before an offline run; model selection also
    checks its training and validation frames here.
    """

    if not splits:
        raise ValueError("At least one temporal split is required.")
    previous_name = None
    previous = None
    for name, frame in splits.items():
        _validate_frame(frame, name)
        if previous is not None and previous.index[-1] >= frame.index[0]:
            raise ValueError(
                f"Temporal splits must be chronological and non-overlapping: "
                f"{previous_name!r} must end before {name!r} starts."
            )
        previous_name, previous = name, frame


def validate_train_val_schema(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[str]:
    """Validate train/validation columns and return train feature order."""

    feature_cols = _validate_labeled_frame(train_df, "Training data")
    validation_features = _validate_labeled_frame(val_df, "Validation data")

    missing = [column for column in feature_cols if column not in validation_features]
    unexpected = [
        column for column in validation_features if column not in feature_cols
    ]
    _raise_schema_mismatch("Validation data", missing, unexpected)
    _validate_feature_dtypes(train_df, val_df, feature_cols)
    validate_temporal_splits({"train": train_df, "val": val_df})
    return feature_cols


def select_prediction_features(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Return target-free prediction columns in the trained model's order."""

    _validate_frame(frame, "Prediction data")
    if config.TARGET in frame.columns:
        raise ValueError(
            "Prediction data must not contain the target column "
            f"{config.TARGET!r}."
        )
    feature_name = getattr(model, "feature_name", None)
    if not callable(feature_name):
        raise TypeError("Model must expose a callable feature_name() method.")

    feature_cols = list(feature_name())
    if not feature_cols:
        raise ValueError("Model does not contain a feature schema.")

    available_features = list(frame.columns)
    missing = [column for column in feature_cols if column not in available_features]
    unexpected = [
        column for column in available_features if column not in feature_cols
    ]
    _raise_schema_mismatch("Prediction data", missing, unexpected)
    selected = frame.loc[:, feature_cols]
    _validate_values(selected, feature_cols, "Prediction data")
    _validate_categorical_dtypes(
        selected,
        feature_cols,
        "Prediction data",
        require_all=False,
    )
    return selected
