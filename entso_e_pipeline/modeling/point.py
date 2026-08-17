import lightgbm as lgb
import pandas as pd

from .. import config

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
}


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> lgb.Booster:
    feature_cols = [c for c in train_df.columns if c != config.TARGET]
    X_train, y_train = train_df[feature_cols], train_df[config.TARGET]
    X_val, y_val = val_df[feature_cols], val_df[config.TARGET]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=config.CATEGORICAL_FEATURES)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=config.CATEGORICAL_FEATURES, reference=train_set)

    return lgb.train(
        PARAMS,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )


def predict(model: lgb.Booster, df: pd.DataFrame) -> pd.Series:
    feature_cols = [c for c in df.columns if c != config.TARGET]
    preds = model.predict(df[feature_cols], num_iteration=model.best_iteration)
    return pd.Series(preds, index=df.index)


def save(model: lgb.Booster, path: str = f"{config.MODELS_DIR}/lightgbm_v1.txt"):
    model.save_model(path)


def load(path: str = f"{config.MODELS_DIR}/lightgbm_v1.txt") -> lgb.Booster:
    return lgb.Booster(model_file=path)
