import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import config


def pinball_loss(y_true, y_pred, quantile):
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def train_one(train_df: pd.DataFrame, val_df: pd.DataFrame, quantile: float,
              num_boost_round=2000, early_stopping_rounds=50,
              learning_rate=0.05, num_leaves=31) -> lgb.Booster:
    feature_cols = [c for c in train_df.columns if c != config.TARGET]
    X_train, y_train = train_df[feature_cols], train_df[config.TARGET]
    X_val, y_val = val_df[feature_cols], val_df[config.TARGET]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=config.CATEGORICAL_FEATURES)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=config.CATEGORICAL_FEATURES, reference=train_set)

    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "verbose": -1,
    }

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds), lgb.log_evaluation(period=0)],
    )


def train_all(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    return {q: train_one(train_df, val_df, q) for q in config.QUANTILES}

def predict(model: lgb.Booster, df: pd.DataFrame) -> pd.Series:
    feature_cols = [c for c in df.columns if c != config.TARGET]
    preds = model.predict(df[feature_cols], num_iteration=model.best_iteration)
    return pd.Series(preds, index=df.index)

def predict_all(models: dict, df: pd.DataFrame) -> dict:
    return {q: predict(m, df) for q, m in models.items()}

def save(models: dict, models_dir: str = config.MODELS_DIR):
    for q, model in models.items():
        model.save_model(f"{models_dir}/lightgbm_q{q}.txt")


def load_all(models_dir: str = config.MODELS_DIR) -> dict:
    return {q: lgb.Booster(model_file=f"{models_dir}/lightgbm_q{q}.txt") for q in config.QUANTILES}
