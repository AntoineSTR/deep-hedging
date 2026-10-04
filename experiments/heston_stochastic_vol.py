"""Experiment 3 — stochastic volatility (Heston): incomplete vs complete markets.

With the spot as the only hedging instrument the Heston market is incomplete:
no strategy removes the volatility risk. The deep hedger is compared with the
model delta and the minimum-variance delta, which corrects for the
spot/variance correlation. Adding a variance swap completes the market; the
deep hedger is then compared with the model's delta-vega hedge.

Run:  python experiments/heston_stochastic_vol.py [--quick]
"""
from __future__ import annotations

import time

import numpy as np

from common import evaluate, markdown_table, output_dirs, parse_args, save_json, settings_dict

from deephedging import DeepHedger, EntropicRisk, Heston, HestonParams
from deephedging import plotting as P
import matplotlib.pyplot as plt  # noqa: E402  (backend set by deephedging.plotting)
from deephedging.pricing import heston_call
from deephedging.problems import heston_call_env
from deephedging.strategies import bs_delta, heston_strategies, stack_instruments

PARAMS = HestonParams(kappa=1.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
VS_NOTIONAL = 100.0


def main(settings=None):
    s = settings or parse_args(__doc__.splitlines()[0])
    out = output_dirs(s)
    P.setup()
    print("\n=== Experiment 3: Heston stochastic volatility ===")

    model = Heston(params=PARAMS, S0=s.S0, substeps=8)
    t0 = time.time()
    train_paths = model.simulate(s.n_train, s.maturity, s.n_steps, seed=s.seed_train)
    test_paths = model.simulate(s.n_test_heston, s.maturity, s.n_steps, seed=s.seed_test)
    print(f"simulated paths in {time.time() - t0:.0f}s")
    premium = float(heston_call(np.array([s.S0]), np.array([PARAMS.v0]), s.maturity, s.K, PARAMS)[0][0])
    print(f"Heston price of the call: {premium:.4f}")

    envs = {
        "spot": (heston_call_env(train_paths, model, s.K),
                 heston_call_env(test_paths, model, s.K)),
        "spot_vs": (heston_call_env(train_paths, model, s.K, variance_swap=True,
                                    variance_swap_notional=VS_NOTIONAL),
                    heston_call_env(test_paths, model, s.K, variance_swap=True,
                                    variance_swap_notional=VS_NOTIONAL)),
    }

    hedgers, timings = {}, {}
    for name, (train, test) in envs.items():
        print(f"\n--- deep hedger, instruments: {', '.join(train.instrument_names)} ---")
        t0 = time.time()
        h = DeepHedger(train.n_features, train.n_instruments, policy="mlp", hidden=(32, 32),
                       risk=EntropicRisk(s.risk_aversion), seed=0)
        h.fit(train, n_iters=s.iters_mlp, batch_size=s.batch_size, lr=5e-3, lr_end=1e-4,
              log_every=max(s.iters_mlp // 5, 1), valid_env=test)
        timings[name] = time.time() - t0
        h.save(out["models"] / f"heston_{name}_mlp.npz")
        hedgers[name] = h

    print("\ncomputing Heston benchmark hedges (Fourier pricing on every node)...")
    t0 = time.time()
    model_hedges = heston_strategies(test_paths, s.K, PARAMS, VS_NOTIONAL)
    print(f"done in {time.time() - t0:.0f}s")

    test_s, test_sv = envs["spot"][1], envs["spot_vs"][1]
    rows = {
        "bs_delta": evaluate(stack_instruments(bs_delta(test_paths, s.K, np.sqrt(PARAMS.v0))),
                             test_s, premium, s),
        "heston_delta": evaluate(stack_instruments(model_hedges["heston_delta"]["spot"]), test_s, premium, s),
        "min_variance_delta": evaluate(stack_instruments(model_hedges["min_variance_delta"]["spot"]),
                                       test_s, premium, s),
        "deep_spot": evaluate(hedgers["spot"].positions(test_s), test_s, premium, s),
        "delta_vega": evaluate(stack_instruments(model_hedges["delta_vega"]["spot"],
                                                 model_hedges["delta_vega"]["variance_swap"]),
                               test_sv, premium, s),
        "deep_spot_vs": evaluate(hedgers["spot_vs"].positions(test_sv), test_sv, premium, s),
    }
    labels = {k: P.label(k) for k in rows}
    labels["bs_delta"] = "Black-Scholes delta (σ = √v0)"
    table = markdown_table(rows, labels)
    print("\n" + table)
    (out["tables"] / "heston.md").write_text(table + "\n")
    save_json(out["results"] / "heston.json", {
        "settings": settings_dict(s), "heston_params": PARAMS.__dict__, "premium_heston": premium,
        "variance_swap_notional": VS_NOTIONAL, "metrics": rows, "train_seconds": timings,
        "history": {k: h.history for k, h in hedgers.items()},
    })

    # ---- Figure 7: P&L distributions ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    bins = P.bins(*(rows[k]["_pnl"] for k in ("heston_delta", "min_variance_delta", "deep_spot")))
    for key in ("heston_delta", "min_variance_delta", "deep_spot"):
        P.hist(axes[0], rows[key]["_pnl"], bins, key)
    axes[0].set_title("Spot only: the market is incomplete", loc="left")
    bins = P.bins(*(rows[k]["_pnl"] for k in ("delta_vega", "deep_spot_vs")))
    for key in ("delta_vega", "deep_spot_vs"):
        P.hist(axes[1], rows[key]["_pnl"], bins, key)
    axes[1].set_title("Spot + variance swap: the market is complete", loc="left")
    for ax in axes:
        ax.set_xlabel("Terminal P&L per option (premium = Heston price)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(out["figures"] / "heston_pnl.png")
    plt.close(fig)

    # ---- Figure 8: learned spot hedge vs model deltas ----------------------
    S = np.linspace(88, 112, 121)
    days = 15
    tau = days / 365
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, v in zip(axes, (0.02, 0.08)):
        _, delta, dcdv = heston_call(S, np.full_like(S, v), tau, s.K, PARAMS)
        mv = delta + PARAMS.rho * PARAMS.xi / S * dcdv
        feats = np.stack([np.log(S / s.K) / np.sqrt(PARAMS.theta * s.maturity),
                          np.full_like(S, tau / s.maturity),
                          np.full_like(S, (np.sqrt(v) - np.sqrt(PARAMS.theta)) / np.sqrt(PARAMS.theta))], axis=1)
        learned = hedgers["spot"].act(feats, mv[:, None])[:, 0]
        P.line(ax, S, delta, "heston_delta")
        P.line(ax, S, mv, "min_variance_delta")
        P.line(ax, S, learned, "deep_spot")
        ax.set_title(f"Volatility {np.sqrt(v):.0%}, {days} days to maturity")
        ax.set_xlabel("Spot")
    axes[0].set_ylabel("Shares held per option")
    axes[0].legend(loc="upper left")
    fig.suptitle("The network learns the correlation-adjusted (minimum-variance) delta",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=P.TEXT)
    fig.tight_layout()
    fig.savefig(out["figures"] / "heston_hedge.png")
    plt.close(fig)
    print(f"\nFigures and tables written to {out['results']}")
    return rows


if __name__ == "__main__":
    main()
