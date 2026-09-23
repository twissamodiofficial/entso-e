import pandas as pd

from .. import config
from ..time_utils import hourly_index, local_day_hours, validate_hourly_index
from .load import CalendarLoadFeatures


class CalendarFeatures:
    """Stateless calendar extraction with the existing transformer interface."""

    def fit(self, features: pd.DataFrame):
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        index = features.index
        return features.assign(
            hour=index.hour,
            day_of_week=index.dayofweek,
            month=index.month,
            weekend=(index.dayofweek >= 5).astype(int),
        )


def make_transformers():
    return CalendarLoadFeatures(), CalendarFeatures()


def _category_dtypes() -> dict:
    dtypes = {
        col: pd.CategoricalDtype(categories=cats)
        for col, cats in config.FIXED_CATEGORIES.items()
    }
    holiday_names = sorted(set(config.COUNTRY_HOLIDAYS.values()) | {"none"})
    dtypes["holiday_name"] = pd.CategoricalDtype(categories=holiday_names)
    return dtypes


def cast_categorical_features(features: pd.DataFrame) -> pd.DataFrame:
    """Apply the same fixed category definitions to every feature frame."""

    features = features.copy()
    for col, dtype in _category_dtypes().items():
        if col not in features.columns:
            continue
        features[col] = features[col].astype(dtype)
    return features


def _add_holiday_and_cast(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["is_holiday"] = [
        int(d in config.COUNTRY_HOLIDAYS) for d in features.index.date
    ]
    features["holiday_name"] = [
        config.COUNTRY_HOLIDAYS.get(d, "none") for d in features.index.date
    ]
    return cast_categorical_features(features)


def _append_weather(
    features: pd.DataFrame, weather: pd.DataFrame = None
) -> pd.DataFrame:
    if weather is None:
        return features
    aligned_weather = weather.reindex(features.index)[config.WEATHER_FEATURES]
    return pd.concat([features, aligned_weather], axis=1)


def build_labeled_features(
    load_train: pd.DataFrame, weather_train: pd.DataFrame = None
):
    """Build causal features and attach labels for historical observations.

    Build each day using only prior actuals, then attach that day's labels.
    The transformers are stateless. A partial initial history day is
    excluded from the seven-day lookback. Missing features or labels remain
    missing for final validation at the model boundary.
    """

    if load_train.empty:
        raise ValueError("No preprocessed load to build training features from.")
    load_train = load_train[[config.TARGET]].tz_convert(config.TIMEZONE)
    load_transformer, dtf = make_transformers()
    first_full_day = load_train.index[0].normalize()
    if load_train.index[0] != first_full_day:
        first_full_day += pd.DateOffset(days=1)
    first_day = first_full_day + pd.DateOffset(days=load_transformer.lookback_days)
    last_day = load_train.index[-1].normalize()
    if first_day > last_day:
        raise ValueError("Training requires seven calendar days of history and subsequent rows.")
    daily_features = []
    day = first_day
    while day <= last_day:
        daily_features.append(
            transform(
                load_train, local_day_hours(day, config.TIMEZONE),
                load_transformer, dtf, weather_train,
            )
        )
        day += pd.DateOffset(days=1)

    features = pd.concat(daily_features)
    # Labels stay outside feature construction and are joined only afterward.
    labeled = features.join(load_train[[config.TARGET]])
    return labeled, load_transformer, dtf


def _select_history(
    load_data: pd.DataFrame, start: pd.Timestamp, load_transformer: CalendarLoadFeatures
) -> pd.DataFrame:
    # Apply the forecast cutoff before calculating features.
    # Keep the entire reference date, including both occurrences of 02:00.
    required_start = start - pd.DateOffset(days=load_transformer.lookback_days)
    history = load_data.loc[
        (load_data.index >= required_start) & (load_data.index < start)
    ]
    return history.reindex(hourly_index(required_start, start))


def transform(
    load_data: pd.DataFrame,
    horizon: pd.DatetimeIndex,
    load_transformer: CalendarLoadFeatures,
    dtf: CalendarFeatures,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build feature-only inputs for one historical or future forecast day.

    Load data is already preprocessed and may include the forecast day's actuals.
    Only the seven calendar days strictly before the horizon's local midnight
    are selected, before any load features are calculated. The day's actuals
    are never returned as features; callers keep labels separately. Supply
    weather consistently with the model's feature schema.
    """

    validate_hourly_index(horizon, "Forecast horizon")
    horizon = horizon.tz_convert(config.TIMEZONE)
    if not horizon.equals(local_day_hours(horizon[0], config.TIMEZONE)):
        raise ValueError("Forecast horizon must cover one complete local calendar day.")
    history = _select_history(load_data, horizon[0], load_transformer)
    load_features = load_transformer.transform(history, horizon)
    features = _add_holiday_and_cast(dtf.transform(load_features))
    return _append_weather(features, weather)
