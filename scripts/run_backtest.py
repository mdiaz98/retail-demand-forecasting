"""Rolling-origin backtest: seasonal naive vs XGBoost vs GRU (PyTorch).

Usage:
    uv run python scripts/run_backtest.py --data data/raw/train.csv

Models are trained once on windows that end before the validation week,
then frozen and evaluated at three later forecast origins (8 weeks each).
Writes tables and plots to results/.
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from demand_forecast.baselines import XGBForecaster, seasonal_naive
from demand_forecast.config import Config
from demand_forecast.data import load_panel
from demand_forecast.metrics import all_metrics, wape
from demand_forecast.model import TorchForecaster
from demand_forecast.windows import make_windows, train_origins


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="few epochs, for a smoke test")
    args = ap.parse_args()

    cfg = Config(max_epochs=2) if args.quick else Config()
    set_seeds(cfg.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.data)
    n_w = panel.n_weeks
    test_o, val_o = cfg.test_origins(n_w), cfg.val_origin(n_w)
    tr_o = train_origins(n_w, cfg.lookback, cfg.horizon, val_o)
    print(
        f"{panel.n_series} series x {n_w} weeks "
        f"({panel.dates[0].date()} to {panel.dates[-1].date()})"
    )
    print(
        f"train origins {tr_o[0]}-{tr_o[-1]} | validation origin {val_o} "
        f"({panel.dates[val_o].date()}) | test origins {test_o} "
        f"({', '.join(str(panel.dates[o].date()) for o in test_o)})"
    )

    train = make_windows(panel, tr_o, cfg.lookback, cfg.horizon)
    val = make_windows(panel, [val_o], cfg.lookback, cfg.horizon)
    test = make_windows(panel, test_o, cfg.lookback, cfg.horizon)
    print(f"windows: train {len(train):,} | val {len(val):,} | test {len(test):,}")

    preds, timings = {"Seasonal naive": seasonal_naive(test)}, {}

    t = time.time()
    print("\nXGBoost")
    xgbm = XGBForecaster(cfg.xgb_params, seed=cfg.seed).fit(train, val)
    preds["XGBoost"] = xgbm.predict(test)
    timings["XGBoost"] = time.time() - t

    t = time.time()
    print("\nGRU (PyTorch)")
    gru = TorchForecaster(
        n_stores=int(panel.store.max()) + 1,
        n_depts=int(panel.dept.max()) + 1,
        horizon=cfg.horizon,
        cfg=cfg,
    ).fit(train, val)
    preds["GRU (PyTorch)"] = gru.predict(test)
    timings["GRU (PyTorch)"] = time.time() - t

    # ---- evaluation -------------------------------------------------------
    y = test.unscaled_target()
    hol = np.stack([panel.holiday[o : o + cfg.horizon] for o in test.origin])

    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "origin": "all", **all_metrics(y, p, hol)})
        for o in test_o:
            m = test.origin == o
            date = str(panel.dates[o].date())
            rows.append({"model": name, "origin": date, **all_metrics(y[m], p[m], hol[m])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)

    overall = table[table.origin == "all"].set_index("model")[["WAPE", "MAE", "WMAE", "Bias"]]
    print("\nTest metrics (3 origins x 8 weeks):")
    print(overall.round(4).to_string())

    by_h = pd.DataFrame(
        {name: [wape(y[:, h], p[:, h]) for h in range(cfg.horizon)] for name, p in preds.items()},
        index=pd.Index(range(1, cfg.horizon + 1), name="weeks ahead"),
    )
    by_h.to_csv(out / "wape_by_horizon.csv")

    # Simple significance check: share of series where GRU beats XGBoost on absolute error.
    err = {n: np.abs(y - p).sum(1) for n, p in preds.items()}
    win_rate = float((err["GRU (PyTorch)"] < err["XGBoost"]).mean())

    summary = {
        "n_series": panel.n_series,
        "n_weeks": n_w,
        "test_origins": [str(panel.dates[o].date()) for o in test_o],
        "metrics": overall.round(5).to_dict(orient="index"),
        "gru_beats_xgb_share_of_windows": round(win_rate, 4),
        "train_seconds": {k: round(v, 1) for k, v in timings.items()},
        "gru_epochs": len(gru.history),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(gru.history).to_csv(out / "gru_training_history.csv", index=False)

    # ---- plots --------------------------------------------------------------
    colors = {"Seasonal naive": "#9a9a9a", "XGBoost": "#2b6cb0", "GRU (PyTorch)": "#d9534f"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in by_h.columns:
        ax.plot(by_h.index, by_h[name], marker="o", label=name, color=colors[name])
    ax.set_xlabel("Weeks ahead")
    ax.set_ylabel("WAPE")
    ax.set_title("Error by forecast horizon (test set)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "wape_by_horizon.png", dpi=150)
    plt.close(fig)

    top = np.argsort(-panel.sales.sum(1))[:3]
    fig, axes = plt.subplots(len(top), 1, figsize=(10, 8), sharex=True)
    start = test_o[0] - 30
    for ax, s in zip(axes, top, strict=True):
        ax.plot(panel.dates[start:], panel.sales[s, start:], color="black", lw=1.5, label="Actual")
        for name in ["XGBoost", "GRU (PyTorch)"]:
            for k, o in enumerate(test_o):
                idx = np.where((test.series == s) & (test.origin == o))[0][0]
                ax.plot(
                    panel.dates[o : o + cfg.horizon],
                    preds[name][idx],
                    color=colors[name],
                    lw=1.5,
                    label=name if k == 0 else None,
                )
        for o in test_o:
            ax.axvline(panel.dates[o], color="#cccccc", ls="--", lw=0.8)
        st, dp = panel.series_ids[s]
        ax.set_title(f"Store {st}, Dept {dp}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Forecasts at the three test origins (largest series)")
    fig.tight_layout()
    fig.savefig(out / "forecast_examples.png", dpi=150)
    plt.close(fig)

    print(f"\nGRU beats XGBoost on {win_rate:.1%} of test windows")
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    main()
