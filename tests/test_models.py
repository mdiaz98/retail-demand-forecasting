import numpy as np
import torch

from demand_forecast.baselines import XGBForecaster, seasonal_naive
from demand_forecast.config import Config
from demand_forecast.model import GRUForecaster, TorchForecaster
from demand_forecast.windows import make_windows


def test_seasonal_naive_is_last_year(panel):
    w = make_windows(panel, origins=[60], lookback=52, horizon=8)
    np.testing.assert_allclose(seasonal_naive(w)[0], panel.sales[0, 8:16], rtol=1e-5)


def test_gru_output_shape():
    net = GRUForecaster(n_stores=3, n_depts=5, horizon=8)
    out = net(
        torch.randn(4, 52, 4),
        torch.randn(4, 8, 4),
        torch.tensor([0, 1, 2, 0]),
        torch.tensor([0, 1, 4, 3]),
    )
    assert out.shape == (4, 8)


def test_gru_can_overfit_tiny_batch():
    """Sanity check that gradients flow: loss on one batch should drop a lot."""
    torch.manual_seed(0)
    net = GRUForecaster(n_stores=2, n_depts=2, horizon=8)
    hist, fut = torch.randn(16, 52, 4), torch.randn(16, 8, 4)
    s, d = torch.randint(0, 2, (16,)), torch.randint(0, 2, (16,))
    y = torch.randn(16, 8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        loss = torch.abs(net(hist, fut, s, d) - y).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first


def test_end_to_end_training_is_reproducible(panel):
    cfg = Config(max_epochs=2, batch_size=16)
    train = make_windows(panel, list(range(52, 64)), 52, 8)
    val = make_windows(panel, [64], 52, 8)
    preds = []
    for _ in range(2):
        m = TorchForecaster(n_stores=2, n_depts=3, horizon=8, cfg=cfg, device="cpu")
        preds.append(m.fit(train, val, verbose=False).predict(val))
    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-5)
    assert (preds[0] >= 0).all()


def test_xgboost_baseline_runs(panel):
    params = dict(Config().xgb_params, n_estimators=10)
    train = make_windows(panel, list(range(52, 64)), 52, 8)
    val = make_windows(panel, [64], 52, 8)
    pred = XGBForecaster(params).fit(train, val).predict(val)
    assert pred.shape == val.target.shape
    assert (pred >= 0).all()
