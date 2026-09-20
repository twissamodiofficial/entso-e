import holidays

MODELS_DIR = "models"

COUNTRY_CODE = "NL"
TIMEZONE = "Europe/Amsterdam"
LATITUDE = 52.37
LONGITUDE = 4.90

TARGET = "Actual Load"
PROCESSING_VERSION = "hourly_mean_v1"
FEATURE_VERSION = "calendar_load_v1"
# Calendar-day and calendar-week lookups match Amsterdam clock hours.
LOAD_FEATURE_NAMES = {
    "lag_day": "load_lag_1d_same_clock_hour",
    "lag_week": "load_lag_7d_same_clock_hour",
    "mean_day": "load_mean_24_clock_hours_ending_1d_ago",
}
CATEGORICAL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "weekend",
    "is_holiday",
    "holiday_name",
]
WEATHER_FEATURES = ["apparent_temperature", "dew_point_2m"]
# Use the same weather model and forecast-age policy for history and live use.
WEATHER_MODEL = "jma_gsm"
# Fixed elapsed-day lead offset, independent of the calendar-day load lags.
# Two days leave a buffer before midnight even on a 25-hour target day.
WEATHER_FORECAST_PREVIOUS_DAYS = 2
LOOKBACK_DAYS = 7
QUANTILES = [0.1, 0.5, 0.9]
TARGET_COVERAGE = 0.80

# Cover the frozen historical data and a reasonable live-serving horizon.
# Extend this range when the model's supported serving horizon changes.
COUNTRY_HOLIDAYS = holidays.Netherlands(years=range(2019, 2041))

# Historical evaluation: select settings on train/val, then fit on both.
# Calibration and test stay out of that model's fitting and early stopping.
# All boundaries are Amsterdam-local and half-open: [start, end).
SPLITS = {
    "train": [
        (f"{y}0101", f"{y + 1}0101") for y in range(2019, 2026)
    ],
    "val": [("20260101", "20260401")],
    "calibration": [("20260401", "20260701")],
    "test": [("20260701", "20260916")],
}

# After historical evaluation, refit with the selected settings through June.
# Recompute Q from this new model's July–September predictions. These rows
# are now calibration data; evaluate this model on subsequent live outcomes.
PRODUCTION_SPLITS = {
    "train": [
        (f"{y}0101", f"{y + 1}0101") for y in range(2019, 2026)
    ] + [("20260101", "20260701")],
    "calibration": [("20260701", "20260916")],
}

FIXED_CATEGORIES = {
    "hour": list(range(24)),
    "day_of_week": list(range(7)),
    "month": list(range(1, 13)),
    "weekend": [0, 1],
    "is_holiday": [0, 1],
}
