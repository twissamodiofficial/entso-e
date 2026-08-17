import holidays

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

COUNTRY_CODE = "NL"
TIMEZONE = "Europe/Amsterdam"
LATITUDE = 52.37
LONGITUDE = 4.90

TARGET = "Actual Load"
CATEGORICAL_FEATURES = ["hour", "day_of_week", "month", "weekend", "is_holiday", "holiday_name"]
WEATHER_FEATURES = ["apparent_temperature", "dew_point_2m"]
WARMUP_HOURS = 168
QUANTILES = [0.1, 0.5, 0.9]
TARGET_COVERAGE = 0.80

COUNTRY_HOLIDAYS = holidays.Netherlands(years=range(2019, 2028))

SPLITS = {
    "train": [
        (f"{y}0101", f"{y+1}0101") for y in range(2019, 2026)
    ],
    "val": [("20260101", "20260401")],
    "test": [("20260401", "20260701")],
}

FIXED_CATEGORIES = {
    "hour": list(range(24)),
    "day_of_week": list(range(7)),
    "month": list(range(1, 13)),
    "weekend": [0, 1],
    "is_holiday": [0, 1],
}