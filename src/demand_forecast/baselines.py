"""Baselines the deep model has to beat.

1. Seasonal naive: sales in the same week last year. Hard to beat in retail.
2. XGBoost: one global gradient-boosted model, direct multi-horizon
   (the horizon step is a feature), trained on the same scaled windows.
"""

import numpy as np
import xgboost as xgb

from .windows import Windows


def seasonal_naive(w: Windows) -> np.ndarray:
    """Unscaled forecast equal to last year's sales for each target week."""
    return w.future[:, :, 0] * w.scale[:, None]


def _tabular(w: Windows) -> np.ndarray:
    """One row per (window, horizon step)."""
    n, h = w.target.shape
    lags = w.hist[:, :, 0]  # scaled history
    recent = np.stack([lags[:, -4:].mean(1), lags[:, -13:].mean(1), lags[:, -52:].std(1)], axis=1)
    base = np.concatenate([lags, recent, w.store[:, None], w.dept[:, None]], axis=1)
    rows = np.repeat(base, h, axis=0)
    step = np.tile(np.arange(h), n)[:, None]
    fut = w.future.reshape(n * h, -1)  # last-year value, holiday, sin, cos of the target week
    return np.concatenate([rows, step, fut], axis=1).astype(np.float32)


class XGBForecaster:
    def __init__(self, params: dict, seed: int = 42):
        self.model = xgb.XGBRegressor(**params, random_state=seed, n_jobs=-1)

    def fit(self, train: Windows, val: Windows) -> "XGBForecaster":
        self.model.fit(
            _tabular(train),
            train.target.reshape(-1),
            eval_set=[(_tabular(val), val.target.reshape(-1))],
            verbose=False,
        )
        return self

    def predict(self, w: Windows) -> np.ndarray:
        pred = self.model.predict(_tabular(w)).reshape(w.target.shape)
        return np.clip(pred, 0, None) * w.scale[:, None]
