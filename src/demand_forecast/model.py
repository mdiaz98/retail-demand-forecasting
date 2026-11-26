"""Global deep forecaster in PyTorch.

One model for all store-department series:

    history (52 weeks x 4 features) --GRU--> h
    store, dept --embeddings--> e
    known future covariates (8 weeks x 4) --flatten--> f
    [h, e, f] --MLP--> 8 forecasts (scaled)

The output is added to last year's (scaled) sales for each target week, so the
network learns a correction on top of the seasonal-naive forecast instead of
the whole level. Trained with L1 loss on scaled sales, which matches WAPE.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .windows import Windows


class GRUForecaster(nn.Module):
    def __init__(
        self,
        n_stores: int,
        n_depts: int,
        horizon: int,
        hidden_size: int = 64,
        store_emb_dim: int = 8,
        dept_emb_dim: int = 16,
        dropout: float = 0.1,
        n_hist_features: int = 4,
        n_future_features: int = 4,
    ):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.GRU(n_hist_features, hidden_size, batch_first=True)
        self.store_emb = nn.Embedding(n_stores, store_emb_dim)
        self.dept_emb = nn.Embedding(n_depts, dept_emb_dim)
        head_in = hidden_size + store_emb_dim + dept_emb_dim + horizon * n_future_features
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon),
        )

    def forward(self, hist, future, store, dept):
        _, h = self.encoder(hist)  # h: (1, batch, hidden)
        x = torch.cat([h[-1], self.store_emb(store), self.dept_emb(dept), future.flatten(1)], dim=1)
        last_year = future[:, :, 0]
        return last_year + self.head(x)


def _tensors(w: Windows):
    return (
        torch.from_numpy(w.hist),
        torch.from_numpy(w.future),
        torch.from_numpy(w.store),
        torch.from_numpy(w.dept),
        torch.from_numpy(w.target),
    )


class TorchForecaster:
    def __init__(self, n_stores: int, n_depts: int, horizon: int, cfg, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.net = GRUForecaster(
            n_stores,
            n_depts,
            horizon,
            hidden_size=cfg.hidden_size,
            store_emb_dim=cfg.store_emb_dim,
            dept_emb_dim=cfg.dept_emb_dim,
            dropout=cfg.dropout,
        ).to(self.device)
        self.history: list[dict] = []

    def _loss(self, w: Windows) -> float:
        self.net.eval()
        total, n = 0.0, 0
        loader = DataLoader(TensorDataset(*_tensors(w)), batch_size=4096)
        with torch.no_grad():
            for hist, fut, s, d, y in loader:
                pred = self.net(*(t.to(self.device) for t in (hist, fut, s, d)))
                total += torch.abs(pred - y.to(self.device)).sum().item()
                n += y.numel()
        return total / n

    def fit(self, train: Windows, val: Windows, verbose: bool = True) -> "TorchForecaster":
        gen = torch.Generator().manual_seed(self.cfg.seed)
        loader = DataLoader(
            TensorDataset(*_tensors(train)),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=gen,
        )
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0

        for epoch in range(1, self.cfg.max_epochs + 1):
            self.net.train()
            for hist, fut, s, d, y in loader:
                hist, fut, s, d, y = (t.to(self.device) for t in (hist, fut, s, d, y))
                loss = torch.abs(self.net(hist, fut, s, d) - y).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            train_l1, val_l1 = self._loss(train), self._loss(val)
            self.history.append({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1})
            if verbose:
                print(f"  epoch {epoch:2d}  train L1 {train_l1:.4f}  val L1 {val_l1:.4f}")
            if val_l1 < best - 1e-4:
                best, best_state, bad = val_l1, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    if verbose:
                        print(f"  early stopping, best val L1 {best:.4f}")
                    break

        self.net.load_state_dict(best_state)
        return self

    def predict(self, w: Windows) -> np.ndarray:
        self.net.eval()
        out = []
        loader = DataLoader(TensorDataset(*_tensors(w)[:4]), batch_size=4096)
        with torch.no_grad():
            for hist, fut, s, d in loader:
                out.append(self.net(*(t.to(self.device) for t in (hist, fut, s, d))).cpu().numpy())
        pred = np.concatenate(out)
        return np.clip(pred, 0, None) * w.scale[:, None]
