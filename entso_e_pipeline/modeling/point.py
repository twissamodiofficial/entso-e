"""Point-forecast LightGBM model."""

import lightgbm as lgb
import pandas as pd

from .. import config
from ._shared import predict, train_booster

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
}


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    num_boost_round: int | None = None,
) -> lgb.Booster:
    """Select rounds with validation, or fit fresh with fixed selected rounds.

    For final fitting, pass combined train/validation features, omit ``val_df``,
    and pass the selection model's ``best_iteration`` as ``num_boost_round``.
    """

    return train_booster(
        train_df,
        val_df,
        params=PARAMS,
        num_boost_round=num_boost_round,
        default_rounds=1000,
        log_period=50,
        include_training_metrics=True,
    )


def save(
    model: lgb.Booster,
    path: str = f"{config.MODELS_DIR}/lightgbm_v1.txt",
):
    model.save_model(path)


def load(
    path: str = f"{config.MODELS_DIR}/lightgbm_v1.txt",
) -> lgb.Booster:
    return lgb.Booster(model_file=path)
