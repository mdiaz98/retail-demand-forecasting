"""Shared settings for the experiment.

Week indices refer to columns of the (series x week) panel built in ``data.py``:
week 0 is the first week in the data and the panel has ``n_weeks`` columns.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    lookback: int = 52  # weeks of history the models see
    horizon: int = 8  # weeks forecast at each origin
    # Forecast origins for the rolling backtest, counted from the end of the panel.
    # With 143 weeks and horizon 8 these are weeks 119, 127 and 135,
    # so the three test windows cover the last 24 weeks without overlap.
    n_test_origins: int = 3
    # Training uses windows whose targets end before the validation window,
    # which in turn ends before the first test origin (no leakage).
    seed: int = 42

    # PyTorch model
    hidden_size: int = 64
    store_emb_dim: int = 8
    dept_emb_dim: int = 16
    dropout: float = 0.1
    batch_size: int = 512
    lr: float = 2e-3
    max_epochs: int = 30
    patience: int = 5

    # XGBoost baseline
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
        }
    )

    def test_origins(self, n_weeks: int) -> list[int]:
        last = n_weeks - self.horizon
        return [last - self.horizon * i for i in reversed(range(self.n_test_origins))]

    def val_origin(self, n_weeks: int) -> int:
        return self.test_origins(n_weeks)[0] - self.horizon
