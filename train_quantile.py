import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_engineering import CATEGORICAL_FEATURES

TARGET = "Actual Load"
DROP_COLS = [TARGET]
QUANTILES = [0.1, 0.5, 0.9]


def pinball_loss(y_true, y_pred, quantile):
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def coverage(y_true, lower, upper):
    inside = (y_true >= lower) & (y_true <= upper)
    return inside.mean()


def train_quantile_model(X_train, y_train, X_val, y_val, quantile, num_boost_round=2000, early_stopping_rounds=50, learning_rate=0.05, num_leaves=31):
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=train_set)

    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds), lgb.log_evaluation(period=0)],
    )
    return model


if __name__ == "__main__":
    train = pd.read_parquet("data/processed/features_weather_train.parquet")
    val = pd.read_parquet("data/processed/features_weather_val.parquet")

    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    models = {}
    preds = {}

    for q in QUANTILES:
        print(f"\nTraining quantile {q}...")
        model = train_quantile_model(X_train, y_train, X_val, y_val, q)
        models[q] = model
        preds[q] = pd.Series(model.predict(X_val, num_iteration=model.best_iteration), index=X_val.index)
        print(f"  best iteration: {model.best_iteration}")

    crossing_low_mid = (preds[0.1] > preds[0.5]).sum()
    crossing_mid_high = (preds[0.5] > preds[0.9]).sum()
    print(f"\nQuantile crossing: q0.1 > q0.5 in {crossing_low_mid} rows, "
          f"q0.5 > q0.9 in {crossing_mid_high} rows (out of {len(y_val)})")

    print("\nPinball loss (val):")
    for q in QUANTILES:
        print(f"  q{q}: {pinball_loss(y_val, preds[q], q):.3f}")

    cov = coverage(y_val, preds[0.1], preds[0.9])
    print(f"\n80% interval [q0.1, q0.9] coverage: {cov*100:.1f}% (target: 80%)")

    median_mae = np.mean(np.abs(y_val - preds[0.5]))
    print(f"\nMedian (q0.5) model MAE: {median_mae:.3f} (compare to point-forecast model's MAE)")

    for q, model in models.items():
        model.save_model(f"models/lightgbm_q{q}.txt")

    interval_summary = pd.DataFrame({
        "actual": y_val,
        "q10": preds[0.1],
        "q50": preds[0.5],
        "q90": preds[0.9],
    })
    interval_summary.to_parquet("data/processed/val_quantile_predictions.parquet")
    print("\nSaved predictions to data/processed/val_quantile_predictions.parquet")