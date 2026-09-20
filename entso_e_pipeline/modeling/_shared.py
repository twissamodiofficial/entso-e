"""Common training and prediction for point and quantile models."""

import lightgbm as lgb
import pandas as pd

from .. import config
from .schema import select_prediction_features, split_target, validate_train_val_schema


def train_booster(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    *,
    params: dict,
    num_boost_round: int | None,
    default_rounds: int,
    early_stopping_rounds: int = 50,
    log_period: int = 0,
    include_training_metrics: bool = False,
) -> lgb.Booster:
    if val_df is None and num_boost_round is None:
        raise ValueError("Final fitting requires the rounds selected on validation.")
    if num_boost_round is None:
        num_boost_round = default_rounds
    if num_boost_round <= 0:
        raise ValueError("num_boost_round must be positive.")

    if val_df is None:
        X_train, y_train = split_target(train_df, "Training data")
        feature_cols = list(X_train.columns)
    else:
        # Validate both frames once, before splitting or reordering columns.
        feature_cols = validate_train_val_schema(train_df, val_df)
        X_train, y_train = train_df.loc[:, feature_cols], train_df[config.TARGET]

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_cols,
        categorical_feature=config.CATEGORICAL_FEATURES,
    )
    if val_df is None:
        return lgb.train(params, train_set, num_boost_round=num_boost_round)

    val_set = lgb.Dataset(
        val_df.loc[:, feature_cols],
        label=val_df[config.TARGET],
        feature_name=feature_cols,
        categorical_feature=config.CATEGORICAL_FEATURES,
        reference=train_set,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set] if include_training_metrics else [val_set],
        valid_names=["train", "val"] if include_training_metrics else ["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=log_period),
        ],
    )


def predict(model: lgb.Booster, features: pd.DataFrame) -> pd.Series:
    """Predict from validated feature columns in the trained model's order."""

    features = select_prediction_features(model, features)
    preds = model.predict(features, num_iteration=model.best_iteration or None)
    return pd.Series(preds, index=features.index)
