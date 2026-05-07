"""Generate all evaluation figures and a metrics.json from the trained model.

Outputs (figures saved to artifacts/figs/):
    01_training_curves.png     - train / test NLL + surrogate RMSE per epoch
    02_marginals.png           - learned marginal p(x_i) vs data histograms (all 20 dims)
    03_forward_scatter.png     - predicted-vs-true outputs given all inputs
    04_inverse_scatter.png     - predicted-vs-true inputs given all outputs
    05_posterior_corner.png    - corner plot of the posterior over all 5 inputs given a measured output vector
    06_conditional_slice.png   - 1D conditional p(output-1 | input-1 = c) for several c
    metrics.json               - final R²/RMSE per dimension
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import AnyOrderARTransformer, GMMHead, ModelConfig


def load():
    ART = Path("artifacts")
    ckpt = torch.load(ART / "model.pt", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = AnyOrderARTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    df = pd.read_excel("combined_data.xlsx")
    cols = ckpt["input_cols"] + ckpt["output_cols"]
    X = df[cols].values.astype(np.float32)
    Xn = (X - ckpt["norm_mu"]) / ckpt["norm_sigma"]
    test_idx = np.load(ART / "test_indices.npy")
    train_idx = np.setdiff1d(np.arange(len(X)), test_idx)
    return model, ckpt, cols, X, Xn, train_idx, test_idx


def fig_training_curves(out: Path):
    with open("artifacts/history.json") as f:
        h = json.load(f)
    ep = np.arange(1, len(h["train_nll"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(ep, h["train_nll"], label="train NLL", color="C0")
    axs[0].plot(ep, h["test_nll"],  label="test  NLL", color="C0", ls="--")
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("− log likelihood")
    axs[0].set_title("Training / test negative log-likelihood")
    axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].plot(ep, h["fwd_rmse"], label=f"forward RMSE (final {h['fwd_rmse'][-1]:.3f})", color="C3")
    axs[1].plot(ep, h["inv_rmse"], label=f"inverse RMSE (final {h['inv_rmse'][-1]:.3f})", color="C2")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("RMSE (standardised)")
    axs[1].set_title("Surrogate-mode RMSE on the 1k test set")
    axs[1].legend(); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_marginals(model, ckpt, X, Xn, out: Path):
    """Compare data histograms (real units) with the model's predicted
    marginal p(x_i) obtained by feeding an all-MASK token to the network."""
    n_dim = Xn.shape[1]
    cols = ckpt["input_cols"] + ckpt["output_cols"]
    mu_n, sig_n = ckpt["norm_mu"], ckpt["norm_sigma"]

    # all-mask conditioning
    values = torch.zeros(1, n_dim)
    mask = torch.ones(1, n_dim)
    with torch.no_grad():
        log_w, mu_g, sigma_g = model.predict_marginals(values, mask)
        # build a fine grid in normalised space for each dim
        fig, axs = plt.subplots(4, 5, figsize=(15, 10))
        axs = axs.flatten()
        for i in range(n_dim):
            xi = X[:, i]
            lo, hi = xi.min(), xi.max()
            grid_real = np.linspace(lo, hi, 400)
            grid_n = torch.from_numpy(((grid_real - mu_n[i]) / sig_n[i]).astype(np.float32))
            pdf_n = GMMHead.pdf_grid(log_w[0, i], mu_g[0, i], sigma_g[0, i], grid_n).numpy()
            # change of variables back to real units: divide by sigma
            pdf_real = pdf_n / sig_n[i]
            ax = axs[i]
            ax.hist(xi, bins=40, density=True, color="0.85", edgecolor="0.4", linewidth=0.4)
            ax.plot(grid_real, pdf_real, color="C3", lw=1.6)
            ax.set_title(cols[i], fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(n_dim, len(axs)):
            axs[j].axis("off")
        fig.suptitle("Learned marginal density (red) vs. data histogram (grey)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out, dpi=130); plt.close(fig)


def fig_io_scatter(model, ckpt, Xn, idx, out: Path, mode: str):
    n_dim = Xn.shape[1]
    n_in = len(ckpt["input_cols"])
    Xt = torch.from_numpy(Xn[idx])
    mask = torch.zeros(1, n_dim).expand(Xt.shape[0], n_dim).clone()
    if mode == "fwd":
        mask[:, n_in:] = 1.0
        target_idx = list(range(n_in, n_dim))
        names = ckpt["output_cols"]
        title = "Forward surrogate: predicted vs. true (outputs given inputs)"
    else:
        mask[:, :n_in] = 1.0
        target_idx = list(range(n_in))
        names = ckpt["input_cols"]
        title = "Inverse surrogate: predicted vs. true (inputs given outputs)"
    with torch.no_grad():
        pred = model.predict_mean(Xt, mask).numpy()
    pred = pred[:, target_idx]
    true = Xn[idx][:, target_idx]
    n = len(target_idx)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.6))
    axs = np.atleast_2d(axs).flatten()
    for i in range(n):
        ax = axs[i]
        ax.plot(true[:, i], pred[:, i], ".", ms=2, color="0.2", alpha=.5)
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r-", lw=0.8)
        rmse = np.sqrt(((pred[:, i] - true[:, i]) ** 2).mean())
        var = true[:, i].var() + 1e-12
        r2 = 1.0 - ((pred[:, i] - true[:, i]) ** 2).mean() / var
        ax.set_title(f"{names[i]}\nR²={r2:.3f}  RMSE={rmse:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, len(axs)):
        axs[j].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130); plt.close(fig)
    # return per-dim metrics
    rmse = np.sqrt(((pred - true) ** 2).mean(axis=0))
    var = true.var(axis=0) + 1e-12
    r2 = 1.0 - ((pred - true) ** 2).mean(axis=0) / var
    return {n: {"r2": float(r), "rmse": float(s)} for n, r, s in zip(names, r2, rmse)}


def fig_posterior_corner(model, ckpt, X, Xn, idx, out: Path):
    """Pick one held-out sample, treat its outputs as 'measurements' and sample
    the posterior over the 5 inputs by ancestral sampling (any-order AR)."""
    n_dim = Xn.shape[1]
    n_in = len(ckpt["input_cols"])
    n_samples = 4000
    rng = np.random.default_rng(7)
    pick = int(idx[rng.integers(0, len(idx))])
    obs = Xn[pick:pick + 1].copy()
    # build conditioning: outputs visible, inputs masked
    cond_values = torch.from_numpy(obs).repeat(n_samples, 1)
    cond_mask = torch.zeros(n_samples, n_dim)
    cond_mask[:, :n_in] = 1.0   # mask inputs
    samples = []
    batch = 400
    with torch.no_grad():
        for s in range(0, n_samples, batch):
            v = cond_values[s:s + batch].clone()
            m = cond_mask[s:s + batch].clone()
            order = list(range(n_in))
            rng.shuffle(order)
            full = model.sample_joint(v, m, order=order)
            samples.append(full[:, :n_in].numpy())
    S_n = np.concatenate(samples, axis=0)
    # de-normalise to real units
    mu_n = ckpt["norm_mu"][:n_in]; sig_n = ckpt["norm_sigma"][:n_in]
    S = S_n * sig_n + mu_n
    truth = X[pick, :n_in]
    cols = ckpt["input_cols"]

    fig, axs = plt.subplots(n_in, n_in, figsize=(2.0 * n_in, 2.0 * n_in))
    for i in range(n_in):
        for j in range(n_in):
            ax = axs[i, j]
            if i == j:
                ax.hist(S[:, i], bins=40, color="C0", alpha=.7, density=True)
                ax.axvline(truth[i], color="r", ls="--", lw=1)
                ax.set_yticks([])
            elif i > j:
                ax.hexbin(S[:, j], S[:, i], gridsize=28, cmap="Blues", mincnt=1)
                ax.plot(truth[j], truth[i], "r*", ms=10)
            else:
                ax.axis("off")
            if i == n_in - 1:
                ax.set_xlabel(cols[j], fontsize=8)
            else:
                ax.set_xticks([])
            if j == 0 and i > 0:
                ax.set_ylabel(cols[i], fontsize=8)
            else:
                if j > 0:
                    ax.set_yticks([])
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Bayesian posterior p(inputs | outputs) for held-out sample #{pick}\n"
                 f"(red star / dashed line = ground-truth inputs, n_samples={n_samples})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130); plt.close(fig)
    return pick


def fig_conditional_slice(model, ckpt, X, Xn, out: Path):
    """Show how a single output's conditional density p(output-1 | input-1 = c)
    deforms as the conditioning value c is swept."""
    n_dim = Xn.shape[1]
    n_in = len(ckpt["input_cols"])
    mu_n = ckpt["norm_mu"]; sig_n = ckpt["norm_sigma"]
    cs_real = np.linspace(X[:, 0].min(), X[:, 0].max(), 5)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    out_idx = n_in     # output-1
    grid_real = np.linspace(X[:, out_idx].min(), X[:, out_idx].max(), 400)
    grid_n = torch.from_numpy(((grid_real - mu_n[out_idx]) / sig_n[out_idx]).astype(np.float32))
    cmap = plt.get_cmap("viridis")
    for k, c in enumerate(cs_real):
        v = torch.zeros(1, n_dim)
        m = torch.ones(1, n_dim)
        v[0, 0] = float((c - mu_n[0]) / sig_n[0])
        m[0, 0] = 0.0
        with torch.no_grad():
            log_w, mu_g, sigma_g = model.predict_marginals(v, m)
            pdf = GMMHead.pdf_grid(log_w[0, out_idx], mu_g[0, out_idx],
                                    sigma_g[0, out_idx], grid_n).numpy() / sig_n[out_idx]
        ax.plot(grid_real, pdf, color=cmap(k / 4), lw=1.6,
                label=f"input-1 = {c:+.2f}")
    ax.set_xlabel("output-1")
    ax.set_ylabel("predicted p(output-1 | input-1 = c)")
    ax.set_title("Conditional density of output-1 sweeping the conditioning input-1")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ART = Path("artifacts"); FIG = ART / "figs"; FIG.mkdir(parents=True, exist_ok=True)
    model, ckpt, cols, X, Xn, train_idx, test_idx = load()

    fig_training_curves(FIG / "01_training_curves.png")
    fig_marginals(model, ckpt, X, Xn, FIG / "02_marginals.png")
    fwd_metrics = fig_io_scatter(model, ckpt, Xn, test_idx,
                                  FIG / "03_forward_scatter.png", mode="fwd")
    inv_metrics = fig_io_scatter(model, ckpt, Xn, test_idx,
                                  FIG / "04_inverse_scatter.png", mode="inv")
    pick = fig_posterior_corner(model, ckpt, X, Xn, test_idx,
                                 FIG / "05_posterior_corner.png")
    fig_conditional_slice(model, ckpt, X, Xn,
                           FIG / "06_conditional_slice.png")

    # aggregate metrics
    fwd_r2 = float(np.mean([m["r2"] for m in fwd_metrics.values()]))
    fwd_rmse = float(np.mean([m["rmse"] for m in fwd_metrics.values()]))
    inv_r2 = float(np.mean([m["r2"] for m in inv_metrics.values()]))
    inv_rmse = float(np.mean([m["rmse"] for m in inv_metrics.values()]))
    metrics = {
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "forward":  {"mean_r2": fwd_r2, "mean_rmse": fwd_rmse, "per_dim": fwd_metrics},
        "inverse":  {"mean_r2": inv_r2, "mean_rmse": inv_rmse, "per_dim": inv_metrics},
        "posterior_pick_idx": int(pick),
    }
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"forward_R2": fwd_r2, "forward_RMSE": fwd_rmse,
                       "inverse_R2": inv_r2, "inverse_RMSE": inv_rmse}, indent=2))
    print("Figures and artifacts/metrics.json written.")


if __name__ == "__main__":
    main()
