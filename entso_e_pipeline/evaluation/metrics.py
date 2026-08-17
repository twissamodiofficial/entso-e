import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(y_true: pd.Series, y_pred: pd.Series, label: str = "") -> dict:
    return {
        "label": label,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "n": len(y_true),
    }


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


def print_metrics(metrics: dict):
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
