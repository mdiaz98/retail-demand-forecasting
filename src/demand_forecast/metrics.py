"""Forecast accuracy metrics.

- WAPE: sum |error| / sum |actual|. Scale-free and dominated by high-volume
  series, which is what matters for stock and revenue. Main metric here.
- MAE: in sales units, easy to read.
- WMAE: the Kaggle competition metric, MAE with holiday weeks weighted x5.
- Bias: sum(forecast - actual) / sum(actual). Positive = over-forecasting.
"""

import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def wmae(y_true: np.ndarray, y_pred: np.ndarray, holiday: np.ndarray) -> float:
    w = np.where(holiday > 0, 5.0, 1.0)
    w = np.broadcast_to(w, y_true.shape)
    return float((w * np.abs(y_true - y_pred)).sum() / w.sum())


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred - y_true).sum() / np.abs(y_true).sum())


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, holiday: np.ndarray) -> dict[str, float]:
    return {
        "WAPE": wape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "WMAE": wmae(y_true, y_pred, holiday),
        "Bias": bias(y_true, y_pred),
    }
