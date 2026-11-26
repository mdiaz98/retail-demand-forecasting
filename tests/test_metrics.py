import numpy as np
import pytest

from demand_forecast.metrics import bias, mae, wape, wmae


def test_perfect_forecast():
    y = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert wape(y, y) == 0
    assert mae(y, y) == 0
    assert bias(y, y) == 0


def test_known_values():
    y = np.array([[100.0, 100.0]])
    p = np.array([[110.0, 80.0]])
    assert wape(y, p) == pytest.approx(30 / 200)
    assert mae(y, p) == pytest.approx(15)
    assert bias(y, p) == pytest.approx(-10 / 200)


def test_wmae_weights_holidays():
    y = np.array([[100.0, 100.0]])
    p = np.array([[110.0, 100.0]])
    # error only in the holiday week -> weight 5 out of total weight 6
    assert wmae(y, p, np.array([[1.0, 0.0]])) == pytest.approx(5 * 10 / 6)
