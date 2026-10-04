"""Experiment 1 — sanity check: without costs, the network must rediscover BS delta.

A recurrent deep hedger is trained from scratch on Black-Scholes paths with no
transaction costs. It never sees the Black-Scholes formula; if the
implementation is right, the learned hedge should coincide with N(d1).

Run:  python experiments/gbm_frictionless.py [--quick]
"""
from __future__ import annotations

import time

import numpy as np

from common import evaluate, markdown_table, output_dirs, parse_args, save_json, settings_dict

from deephedging import GBM, DeepHedger, EntropicRisk
from deephedging import plotting as P
import matplotlib.pyplot as plt  # noqa: E402  (backend set by deephedging.plotting)
from deephedging.pricing import bs_call_delta, bs_call_price
from deephedging.problems import gbm_call_env
from deephedging.strategies import bs_delta, no_hedge, stack_instruments


def main(settings=None):
    s = settings or parse_args(__doc__.splitlines()[0])
    out = output_dirs(s)
    P.setup()
    print("\n=== Experiment 1: GBM, no transaction costs ===")

    model = GBM(sigma=s.sigma, S0=s.S0)
    train_paths = model.simulate(s.n_train, s.maturity, s.n_steps, seed=s.seed_train)
    test_paths = model.simulate(s.n_test, s.maturity, s.n_steps, seed=s.seed_test)
    train = gbm_call_env(train_paths, s.K, s.sigma, cost=0.0)
    test = gbm_call_env(test_paths, s.K, s.sigma, cost=0.0)
    premium = float(bs_call_price(s.S0, s.K, s.maturity, s.sigma))

    t0 = time.time()
    hedger = DeepHedger(train.n_features, 1, policy="mlp", hidden=(32, 32),
                        risk=EntropicRisk(s.risk_aversion), seed=0)
    hedger.fit(train, n_iters=s.iters_mlp, batch_size=s.batch_size, lr=5e-3, lr_end=1e-4,
               log_every=max(s.iters_mlp // 10, 1), valid_env=test.subset(slice(0, 32768)))
    train_time = time.time() - t0
    hedger.save(out["models"] / "gbm_frictionless_mlp.npz")

    positions = {
        "no_hedge": stack_instruments(no_hedge(test_paths)),
        "bs_delta": stack_instruments(bs_delta(test_paths, s.K, s.sigma)),
        "deep_mlp": hedger.positions(test),
    }
    rows = {k: evaluate(v, test, premium, s) for k, v in positions.items()}
    labels = {k: P.label(k) for k in rows}

    # how close is the learned hedge to the BS delta along the test paths?
    gap = np.abs(positions["deep_mlp"] - positions["bs_delta"])[..., 0]
    agreement = {"mean_abs_gap_to_bs_delta": float(gap.mean()),
                 "p99_abs_gap_to_bs_delta": float(np.quantile(gap, 0.99))}

    table = markdown_table(rows, labels)
    print("\n" + table)
    print(f"\nMean |deep - BS delta| along test paths: {agreement['mean_abs_gap_to_bs_delta']:.4f}")
    (out["tables"] / "gbm_frictionless.md").write_text(table + "\n")
    save_json(out["results"] / "gbm_frictionless.json", {
        "settings": settings_dict(s), "premium_bs": premium, "metrics": rows,
        "agreement": agreement, "train_seconds": train_time, "history": hedger.history,
    })

    # ---- Figure 1: learned hedge vs BS delta --------------------------------
    S = np.linspace(85, 115, 241)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, days in zip(axes, (20, 10, 2)):
        tau = days / 365
        delta = bs_call_delta(S, s.K, tau, s.sigma)
        feats = np.stack([np.log(S / s.K) / (s.sigma * np.sqrt(s.maturity)),
                          np.full_like(S, tau / s.maturity)], axis=1)
        learned = hedger.act(feats, delta[:, None])[:, 0]
        P.line(ax, S, delta, "bs_delta")
        P.line(ax, S, learned, "deep_mlp")
        ax.set_title(f"{days} days to maturity")
        ax.set_xlabel("Spot")
    axes[0].set_ylabel("Shares held per option")
    axes[0].legend(loc="upper left")
    fig.suptitle("Without costs, the network learns the Black-Scholes delta from scratch",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=P.TEXT)
    fig.tight_layout()
    fig.savefig(out["figures"] / "gbm_frictionless_hedge.png")
    plt.close(fig)

    # ---- Figure 2: P&L distribution ----------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = P.bins(rows["bs_delta"]["_pnl"], rows["deep_mlp"]["_pnl"])
    P.hist(ax, rows["bs_delta"]["_pnl"], bins, "bs_delta")
    P.hist(ax, rows["deep_mlp"]["_pnl"], bins, "deep_mlp")
    ax.set_xlabel("Terminal P&L per option (premium = BS price, S0 = 100)")
    ax.set_ylabel("Density")
    ax.set_title("Hedged P&L, no transaction costs", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(out["figures"] / "gbm_frictionless_pnl.png")
    plt.close(fig)
    print(f"\nFigures and tables written to {out['results']}")
    return rows


if __name__ == "__main__":
    main()
