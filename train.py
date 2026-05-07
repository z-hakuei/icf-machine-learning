"""Train the AnyOrderARTransformer on combined_data.xlsx.

Outputs:
    artifacts/model.pt          - trained weights + config + normalisation stats
    artifacts/history.json      - per-epoch loss / RMSE curves
    artifacts/test_indices.npy  - row indices used for the held-out test set
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import AnyOrderARTransformer, ModelConfig


def main():
    ART = Path("artifacts")
    ART.mkdir(exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    # --- load data --------------------------------------------------------
    df = pd.read_excel("combined_data.xlsx")
    input_cols = [c for c in df.columns if c.startswith("input")]
    output_cols = [c for c in df.columns if c.startswith("output")]
    cols = input_cols + output_cols
    print(f"Loaded {len(df)} rows | {len(input_cols)} inputs | {len(output_cols)} outputs")

    X = df[cols].values.astype(np.float32)              # (N, 20)
    n_dim = X.shape[1]

    # --- normalisation: standardise to zero mean / unit std ---------------
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma

    # --- train / test split (90 / 10) ------------------------------------
    n_total = Xn.shape[0]
    n_test = n_total // 10
    perm = np.random.permutation(n_total)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    np.save(ART / "test_indices.npy", test_idx)

    Xtr = torch.from_numpy(Xn[train_idx])
    Xte = torch.from_numpy(Xn[test_idx])

    train_loader = DataLoader(
        TensorDataset(Xtr), batch_size=256, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(TensorDataset(Xte), batch_size=512, shuffle=False)

    # --- model & optimiser ------------------------------------------------
    cfg = ModelConfig(
        n_dim=n_dim, d_model=64, n_layers=4, n_heads=4,
        d_ff=256, n_gmm=10, dropout=0.0,
    )
    model = AnyOrderARTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.3f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)

    n_epochs = 120
    history = {"train_nll": [], "test_nll": [], "fwd_rmse": [], "inv_rmse": [],
               "fwd_r2": [], "inv_r2": [], "epoch_time": []}

    n_in = len(input_cols)
    n_out = len(output_cols)
    fwd_mask_eval = torch.zeros(1, n_dim)
    fwd_mask_eval[0, n_in:] = 1.0          # hide outputs
    inv_mask_eval = torch.zeros(1, n_dim)
    inv_mask_eval[0, :n_in] = 1.0          # hide inputs

    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        train_nll_sum, train_count = 0.0, 0
        for (x,) in train_loader:
            B, N = x.shape
            # any-order masking: pick r uniform on [1/N, 1-1/N], Bernoulli per dim,
            # ensure at least one masked position per sample
            r = torch.empty(B, 1).uniform_(1.0 / N, 1.0 - 1.0 / N)
            mask = (torch.rand(B, N) < r).float()
            empty = mask.sum(dim=1) == 0
            if empty.any():
                idx = torch.randint(0, N, (int(empty.sum()),))
                mask[empty, idx] = 1.0
            loss = model.loss(x, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_nll_sum += loss.item() * B
            train_count += B
        sched.step()

        # ---- eval ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            test_nll_sum, test_count = 0.0, 0
            fwd_pred, fwd_true = [], []
            inv_pred, inv_true = [], []
            for (x,) in test_loader:
                B = x.shape[0]
                # use the same any-order objective for the test NLL,
                # with a fixed mask ratio of 0.5 for stability
                mask = (torch.rand(B, n_dim) < 0.5).float()
                empty = mask.sum(dim=1) == 0
                if empty.any():
                    idx = torch.randint(0, n_dim, (int(empty.sum()),))
                    mask[empty, idx] = 1.0
                test_nll_sum += model.loss(x, mask).item() * B
                test_count += B
                # forward surrogate
                fmask = fwd_mask_eval.expand(B, n_dim)
                fwd_pred.append(model.predict_mean(x, fmask)[:, n_in:])
                fwd_true.append(x[:, n_in:])
                # inverse surrogate
                imask = inv_mask_eval.expand(B, n_dim)
                inv_pred.append(model.predict_mean(x, imask)[:, :n_in])
                inv_true.append(x[:, :n_in])

            fwd_pred = torch.cat(fwd_pred).numpy()
            fwd_true = torch.cat(fwd_true).numpy()
            inv_pred = torch.cat(inv_pred).numpy()
            inv_true = torch.cat(inv_true).numpy()
            fwd_rmse = float(np.sqrt(((fwd_pred - fwd_true) ** 2).mean()))
            inv_rmse = float(np.sqrt(((inv_pred - inv_true) ** 2).mean()))
            fwd_r2 = float(1.0 - ((fwd_pred - fwd_true) ** 2).sum() /
                                  ((fwd_true - fwd_true.mean(0)) ** 2).sum())
            inv_r2 = float(1.0 - ((inv_pred - inv_true) ** 2).sum() /
                                  ((inv_true - inv_true.mean(0)) ** 2).sum())

        history["train_nll"].append(train_nll_sum / train_count)
        history["test_nll"].append(test_nll_sum / test_count)
        history["fwd_rmse"].append(fwd_rmse)
        history["inv_rmse"].append(inv_rmse)
        history["fwd_r2"].append(fwd_r2)
        history["inv_r2"].append(inv_r2)
        history["epoch_time"].append(time.time() - t0)

        if ep == 1 or ep % 5 == 0 or ep == n_epochs:
            print(f"ep {ep:3d} | trainNLL {history['train_nll'][-1]:+.4f}"
                  f" | testNLL {history['test_nll'][-1]:+.4f}"
                  f" | fwdR2 {fwd_r2:.4f} (RMSE {fwd_rmse:.4f})"
                  f" | invR2 {inv_r2:.4f} (RMSE {inv_rmse:.4f})"
                  f" | {history['epoch_time'][-1]:.1f}s/ep")

    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "norm_mu": mu, "norm_sigma": sigma,
        "input_cols": input_cols, "output_cols": output_cols,
    }, ART / "model.pt")
    with open(ART / "history.json", "w") as f:
        json.dump(history, f)
    print("Saved artifacts/model.pt and artifacts/history.json")


if __name__ == "__main__":
    main()
