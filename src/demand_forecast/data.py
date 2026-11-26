"""Load the Walmart weekly sales data and turn it into a dense panel.

The raw file has one row per (Store, Dept, week). Models work on a panel:
a 2-D array with one row per store-department series and one column per week,
plus the calendar covariates that are known in advance for every week.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Panel:
    sales: np.ndarray  # (n_series, n_weeks), float32, weekly sales >= 0
    store: np.ndarray  # (n_series,), int, 0-based store code
    dept: np.ndarray  # (n_series,), int, 0-based department code
    dates: pd.DatetimeIndex  # (n_weeks,)
    holiday: np.ndarray  # (n_weeks,), float32, 1 if the week contains a major holiday
    series_ids: list[tuple[int, int]]  # original (Store, Dept) for each row

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]

    @property
    def n_weeks(self) -> int:
        return self.sales.shape[1]

    def week_of_year_features(self) -> np.ndarray:
        """sin/cos encoding of ISO week of year, shape (n_weeks, 2)."""
        week = self.dates.isocalendar().week.to_numpy().astype(np.float32)
        angle = 2 * np.pi * week / 52.0
        return np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    expected = {"Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    return df


def build_panel(df: pd.DataFrame, min_mean_sales: float = 100.0) -> Panel:
    """Keep series observed in every week, pivot to a panel.

    - Series with gaps are dropped rather than imputed: about 80% of
      store-department pairs have the full history, and dropping avoids
      inventing sales for departments that did not exist yet.
    - Negative weekly sales (returns exceeding sales, ~0.3% of rows) are clipped to 0.
    - Near-empty series (mean below ``min_mean_sales``) are dropped because
      percentage-type errors on them are noise.
    """
    dates = pd.DatetimeIndex(sorted(df["Date"].unique()))
    wide = df.pivot_table(index=["Store", "Dept"], columns="Date", values="Weekly_Sales")
    wide = wide.reindex(columns=dates)
    wide = wide.dropna(axis=0, how="any")
    wide = wide.clip(lower=0)
    wide = wide[wide.mean(axis=1) >= min_mean_sales]

    holiday = df.groupby("Date")["IsHoliday"].max().reindex(dates).fillna(False).astype(np.float32)

    stores = wide.index.get_level_values("Store").to_numpy()
    depts = wide.index.get_level_values("Dept").to_numpy()
    store_codes = pd.Index(sorted(set(stores))).get_indexer(stores)
    dept_codes = pd.Index(sorted(set(depts))).get_indexer(depts)

    return Panel(
        sales=wide.to_numpy(dtype=np.float32),
        store=store_codes.astype(np.int64),
        dept=dept_codes.astype(np.int64),
        dates=dates,
        holiday=holiday.to_numpy(),
        series_ids=list(wide.index),
    )


def load_panel(path: str | Path) -> Panel:
    return build_panel(load_raw(path))
