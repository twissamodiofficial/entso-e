import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(y_true: pd.Series, y_pred: pd.Series, label: str = "") -> dict:
    metrics = {
        "label": label,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "n": len(y_true),
    }
    return metrics


def evaluate_by_group(y_true: pd.Series, y_pred: pd.Series, group: pd.Series, group_name: str) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group})
    rows = []
    for g, sub in df.groupby("group"):
        rows.append({
            group_name: g,
            "mae": mae(sub["y_true"], sub["y_pred"]),
            "rmse": rmse(sub["y_true"], sub["y_pred"]),
            "mape": mape(sub["y_true"], sub["y_pred"]),
            "n": len(sub),
        })
    return pd.DataFrame(rows).sort_values(group_name).reset_index(drop=True)


if __name__ == "__main__":
    val = pd.read_parquet("data/processed/features_weather_val.parquet")

    y_true = val["Actual Load"]
    y_pred_naive = val["Actual Load_lag_24"]

    overall = evaluate(y_true, y_pred_naive, label="naive_lag24_val")
    print("Overall (val):")
    for k, v in overall.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nBy hour of day:")
    print(evaluate_by_group(y_true, y_pred_naive, val["hour"], "hour").to_string(index=False))

    print("\nBy weekday/weekend:")
    print(evaluate_by_group(y_true, y_pred_naive, val["weekend"], "weekend").to_string(index=False))