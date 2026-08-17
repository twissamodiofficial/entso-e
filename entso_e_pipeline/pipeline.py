import pandas as pd
import joblib

from . import config
from .features import engineering
from .modeling import point, quantile, calibration
from .ingestion import load as load_ingestion
from .ingestion import weather as weather_ingestion


class ForecastPipeline:
    def __init__(self):
        self.ws = None
        self.dtf = None
        self.point_model = None
        self.quantile_models = None
        self.cqr_q = None
        self.features = {}

    def ingest(self):
        load_ingestion.ingest_all()
        weather_ingestion.ingest_all()

    def build_features(self):
        load_train = engineering.load_and_clean(f"{config.RAW_DIR}/load_train.csv")
        load_val = engineering.load_and_clean(f"{config.RAW_DIR}/load_val.csv")
        load_test = engineering.load_and_clean(f"{config.RAW_DIR}/load_test.csv")

        weather_train = engineering.load_weather(f"{config.RAW_DIR}/weather_train.csv")
        weather_val = engineering.load_weather(f"{config.RAW_DIR}/weather_val.csv")
        weather_test = engineering.load_weather(f"{config.RAW_DIR}/weather_test.csv")

        features_train, self.ws, self.dtf = engineering.fit_transform_train(load_train, weather_train)
        features_val = engineering.transform(load_val, self.ws, self.dtf, load_prior=load_train, weather_split=weather_val)
        features_test = engineering.transform(load_test, self.ws, self.dtf, load_prior=load_val, weather_split=weather_test)

        self.features = {"train": features_train, "val": features_val, "test": features_test}

        for name, df in self.features.items():
            df.to_parquet(f"{config.PROCESSED_DIR}/features_{name}.parquet")

        return self.features

    def load_features(self):
        self.features = {
            name: pd.read_parquet(f"{config.PROCESSED_DIR}/features_{name}.parquet")
            for name in ("train", "val", "test")
        }
        for df in self.features.values():
            for col in config.CATEGORICAL_FEATURES:
                df[col] = df[col].astype("category")
        return self.features

    def train(self):
        train_df, val_df = self.features["train"], self.features["val"]
        self.point_model = point.train(train_df, val_df)
        self.quantile_models = quantile.train_all(train_df, val_df)
        return self.point_model, self.quantile_models

    def calibrate(self):
        val_df = self.features["val"]
        preds = quantile.predict_all(self.quantile_models, val_df)
        self.cqr_q = calibration.calibrate(val_df[config.TARGET], preds[0.1], preds[0.9])
        return self.cqr_q

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        point_pred = point.predict(self.point_model, df)
        preds = quantile.predict_all(self.quantile_models, df)
        q10_cal, q90_cal = calibration.apply(preds[0.1], preds[0.9], self.cqr_q)
        return pd.DataFrame({
            "point_pred": point_pred,
            "q10": preds[0.1],
            "q50": preds[0.5],
            "q90": preds[0.9],
            "q10_calibrated": q10_cal,
            "q90_calibrated": q90_cal,
        })

    def load_models(self):
        self.point_model = point.load()
        self.quantile_models = quantile.load_all()
        self.load_transformers()
        self.cqr_q = joblib.load(f"{config.MODELS_DIR}/cqr_q.joblib")

    def save_models(self):
        point.save(self.point_model)
        quantile.save(self.quantile_models)
        self.save_transformers()
        joblib.dump(self.cqr_q, f"{config.MODELS_DIR}/cqr_q.joblib")

    def save_transformers(self):
        joblib.dump(self.ws, f"{config.MODELS_DIR}/ws.joblib")
        joblib.dump(self.dtf, f"{config.MODELS_DIR}/dtf.joblib")

    def load_transformers(self):
        self.ws = joblib.load(f"{config.MODELS_DIR}/ws.joblib")
        self.dtf = joblib.load(f"{config.MODELS_DIR}/dtf.joblib")