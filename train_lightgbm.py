import lightgbm as lgb
import pandas as pd

from baseline_eval import evaluate, evaluate_by_group
from feature_engineering import CATEGORICAL_FEATURES

TARGET = "Actual Load"
DROP_COLS = [TARGET]

if __name__ == "__main__":
    train = pd.read_parquet("data/processed/features_weather_train.parquet")
    val = pd.read_parquet("data/processed/features_weather_val.parquet")

    feature_cols = [c for c in train.columns if c not in DROP_COLS]

    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )

    y_pred = pd.Series(model.predict(X_val, num_iteration=model.best_iteration), index=X_val.index)

    print("\n--- LightGBM (val) ---")
    overall = evaluate(y_val, y_pred, label="lightgbm_val")
    for k, v in overall.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nBy hour of day:")
    print(evaluate_by_group(y_val, y_pred, val["hour"], "hour").to_string(index=False))

    print("\nBy weekday/weekend:")
    print(evaluate_by_group(y_val, y_pred, val["weekend"], "weekend").to_string(index=False))

    print("\nFeature importance (gain):")
    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=feature_cols
    ).sort_values(ascending=False)
    print(importance.to_string())

    model.save_model("models/lightgbm_v1.txt")