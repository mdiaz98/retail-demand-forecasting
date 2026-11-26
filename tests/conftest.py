"""Small synthetic dataset in the same format as Walmart's train.csv.

Tests never need the real data, so they run in CI on a clean checkout.
"""

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data import build_panel


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-02-05", periods=80, freq="7D")
    holidays = {pd.Timestamp("2010-11-26"), pd.Timestamp("2010-12-31")}
    rows = []
    for store in (1, 2):
        for dept in (1, 2, 3):
            level = 1000 * store * dept
            week = np.arange(len(dates))
            season = 1 + 0.3 * np.sin(2 * np.pi * week / 52)
            sales = level * season + rng.normal(0, 50, len(dates))
            for d, s in zip(dates, sales, strict=True):
                rows.append((store, dept, d, float(s), d in holidays))
    # A series with a gap (should be dropped) and one negative value (should be clipped).
    rows.append((3, 1, dates[0], 500.0, False))
    df = pd.DataFrame(rows, columns=["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"])
    df.loc[5, "Weekly_Sales"] = -20.0
    return df


@pytest.fixture
def panel(raw_df):
    return build_panel(raw_df)
