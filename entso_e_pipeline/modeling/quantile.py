"""Quantile-forecast LightGBM models."""

import lightgbm as lgb
import pandas as pd

from .. import config
from ._shared import predict, train_booster
from .metrics import pinball_loss


def train_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    quantile: float,
    num_boost_round: int | None = None,
    early_stopping_rounds=50,
    learning_rate=0.05,
    num_leaves=31,
) -> lgb.Booster:
    """Select rounds on validation, or fit fresh using fixed selected rounds.

    Pass ``val_df=None`` and the selection model's ``best_iteration`` for
    final fitting on combined train/validation features.
    """

    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "verbose": -1,
    }
    return train_booster(
        train_df,
        val_df,
        params=params,
        num_boost_round=num_boost_round,
        default_rounds=2000,
        early_stopping_rounds=early_stopping_rounds,
    )


def train_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    num_boost_round: dict[float, int] | None = None,
) -> dict:
    """Train each quantile, using its own selected round count for final fitting."""

    return {
        q: train_one(
            train_df,
            val_df,
            q,
            num_boost_round=None if num_boost_round is None else num_boost_round[q],
        )
        for q in config.QUANTILES
    }


def predict_all(models: dict, features: pd.DataFrame) -> dict:
    return {q: predict(model, features) for q, model in models.items()}


def save(models: dict, models_dir: str = config.MODELS_DIR):
    for quantile, model in models.items():
        model.save_model(f"{models_dir}/lightgbm_q{quantile}.txt")


def load_all(models_dir: str = config.MODELS_DIR) -> dict:
    return {
        quantile: lgb.Booster(
            model_file=f"{models_dir}/lightgbm_q{quantile}.txt"
        )
        for quantile in config.QUANTILES
    }
