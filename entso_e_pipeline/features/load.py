"""Load features aligned by local calendar date and clock hour."""

import pandas as pd

from .. import config


class CalendarLoadFeatures:
    """Stateless daily/weekly lookups with explicit Amsterdam DST policies.

    Reference days have 24 clock-hour slots: average repeated hours and
    interpolate nonexistent spring hours from their adjacent observations.
    This grid is only for feature lookups; original target timestamps and
    actual load observations retain every real hourly instant.
    """

    def __init__(self):
        self.timezone = config.TIMEZONE
        self.lookback_days = config.LOOKBACK_DAYS
        self.feature_names = dict(config.LOAD_FEATURE_NAMES)

    def fit_transform(self, load: pd.DataFrame) -> pd.DataFrame:
        # No training observations are cached or needed by later transforms.
        return self.transform(load, load.index)

    def transform(
        self, load: pd.DataFrame, horizon: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Align historical reference values with the supplied target timestamps."""

        source_clock = load.index.tz_convert(self.timezone).tz_localize(None)
        target_clock = horizon.tz_convert(self.timezone).tz_localize(None)
        clock_load = load[config.TARGET].groupby(source_clock).mean()
        clock_grid = pd.date_range(clock_load.index[0], clock_load.index[-1], freq="h")
        clock_load = clock_load.reindex(clock_grid)

        # Only fill clock labels that never existed due to spring DST.
        # Ordinary missing observations remain missing and fail validation.
        nonexistent = clock_grid.tz_localize(
            self.timezone, ambiguous=True, nonexistent="NaT"
        ).isna()
        neighbors = (clock_load.shift(1) + clock_load.shift(-1)) / 2
        clock_load.loc[nonexistent] = neighbors.loc[nonexistent]

        daily = clock_load.shift(freq=pd.DateOffset(days=1))
        weekly = clock_load.shift(freq=pd.DateOffset(days=self.lookback_days))
        mean = clock_load.rolling(24, min_periods=24).mean().shift(
            freq=pd.DateOffset(days=1)
        )
        return pd.DataFrame(
            {
                self.feature_names["lag_day"]: daily.reindex(target_clock).to_numpy(),
                self.feature_names["lag_week"]: weekly.reindex(target_clock).to_numpy(),
                self.feature_names["mean_day"]: mean.reindex(target_clock).to_numpy(),
            },
            index=horizon,
        )
