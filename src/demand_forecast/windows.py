"""Cut the panel into (history, future) windows at given forecast origins.

A window at origin ``o`` uses weeks ``[o - lookback, o)`` as history and
weeks ``[o, o + horizon)`` as targets. Everything a model receives about the
future (holiday flags, week of year, sales on the same week last year) is
known at time ``o``, so there is no leakage from the target period.

Sales are divided by a per-window scale (mean of the history + 1) so that
one global model can learn from departments that sell tens and tens of
thousands per week. Forecasts are multiplied back by the same scale.
"""

from dataclasses import dataclass

import numpy as np

from .data import Panel


@dataclass
class Windows:
    hist: np.ndarray  # (n, lookback, 4): scaled sales, holiday, sin(week), cos(week)
    future: np.ndarray  # (n, horizon, 4): last-year scaled sales, holiday, sin, cos
    target: np.ndarray  # (n, horizon): scaled sales to predict
    scale: np.ndarray  # (n,)
    store: np.ndarray  # (n,)
    dept: np.ndarray  # (n,)
    series: np.ndarray  # (n,) row index in the panel
    origin: np.ndarray  # (n,) forecast origin (week index)

    def __len__(self) -> int:
        return len(self.scale)

    def unscaled_target(self) -> np.ndarray:
        return self.target * self.scale[:, None]


def make_windows(panel: Panel, origins: list[int], lookback: int, horizon: int) -> Windows:
    if lookback < 52:
        raise ValueError("lookback must cover a full year: the models use last year's sales")
    cal = np.concatenate([panel.holiday[:, None], panel.week_of_year_features()], axis=1)

    parts = []
    for o in origins:
        if o - lookback < 0 or o + horizon > panel.n_weeks:
            raise ValueError(f"origin {o} does not fit lookback={lookback}, horizon={horizon}")
        hist_sales = panel.sales[:, o - lookback : o]
        scale = hist_sales.mean(axis=1) + 1.0
        last_year = panel.sales[:, o - 52 : o - 52 + horizon] / scale[:, None]

        n = panel.n_series
        hist = np.concatenate(
            [
                (hist_sales / scale[:, None])[..., None],
                np.broadcast_to(cal[o - lookback : o], (n, lookback, 3)),
            ],
            axis=2,
        )
        future = np.concatenate(
            [last_year[..., None], np.broadcast_to(cal[o : o + horizon], (n, horizon, 3))],
            axis=2,
        )
        parts.append(
            dict(
                hist=hist,
                future=future,
                target=panel.sales[:, o : o + horizon] / scale[:, None],
                scale=scale,
                store=panel.store,
                dept=panel.dept,
                series=np.arange(n),
                origin=np.full(n, o),
            )
        )

    return Windows(
        **{
            k: np.concatenate([p[k] for p in parts]).astype(
                np.float32 if k in {"hist", "future", "target", "scale"} else np.int64
            )
            for k in parts[0]
        }
    )


def train_origins(
    panel_weeks: int, lookback: int, horizon: int, val_origin: int, step: int = 1
) -> list[int]:
    """All origins whose target window ends at or before the validation origin."""
    return list(range(lookback, val_origin - horizon + 1, step))
