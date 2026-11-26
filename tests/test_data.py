import numpy as np

from demand_forecast.windows import make_windows, train_origins


def test_panel_keeps_only_complete_series(panel):
    assert panel.n_series == 6  # the gapped (Store 3, Dept 1) series is dropped
    assert (3, 1) not in panel.series_ids
    assert panel.sales.shape == (6, 80)


def test_negative_sales_are_clipped(panel):
    assert panel.sales.min() >= 0


def test_holiday_flags(panel):
    assert panel.holiday.sum() == 2


def test_windows_shapes_and_scaling(panel):
    w = make_windows(panel, origins=[60, 70], lookback=52, horizon=8)
    assert w.hist.shape == (12, 52, 4)
    assert w.future.shape == (12, 8, 4)
    assert w.target.shape == (12, 8)
    # unscaling gives back the raw sales
    s, o = w.series[0], w.origin[0]
    np.testing.assert_allclose(w.unscaled_target()[0], panel.sales[s, o : o + 8], rtol=1e-5)


def test_no_future_leakage(panel):
    """History must stop right before the origin; last-year values come from the past."""
    o = 60
    w = make_windows(panel, origins=[o], lookback=52, horizon=8)
    hist_raw = w.hist[0, :, 0] * w.scale[0]
    np.testing.assert_allclose(hist_raw, panel.sales[0, o - 52 : o], rtol=1e-5)
    last_year_raw = w.future[0, :, 0] * w.scale[0]
    np.testing.assert_allclose(last_year_raw, panel.sales[0, o - 52 : o - 44], rtol=1e-5)


def test_train_origins_end_before_validation():
    origins = train_origins(143, lookback=52, horizon=8, val_origin=111)
    assert origins[0] == 52
    assert max(origins) + 8 <= 111
