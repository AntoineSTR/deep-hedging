"""Experiment 2 — proportional transaction costs: learning the no-trade band.

For several cost levels, a No-Transaction Band Network is trained to minimise
the entropic risk of the hedged P&L and compared with three classical answers:
the plain BS delta, Leland's cost-adjusted delta and the Whalley-Wilmott band
(asymptotically optimal for the same exponential-utility objective). At the
reference cost level the recurrent MLP deep hedger is trained as well, to
compare architectures.

Run:  python experiments/gbm_transaction_costs.py [--quick]
"""
from __future__ import annotations

import time

import numpy as np

from common import evaluate, markdown_table, output_dirs, parse_args, save_json, settings_dict

from deephedging import GBM, DeepHedger, EntropicRisk
from deephedging import plotting as P
import matplotlib.pyplot as plt  # noqa: E402  (backend set by deephedging.plotting)
from deephedging.pricing import bs_call_delta, bs_call_price, whalley_wilmott_half_width
from deephedging.problems import gbm_call_env
from deephedging.strategies import bs_delta, leland_delta, stack_instruments, whalley_wilmott

COSTS_BP = (5, 10, 20)
REFERENCE_BP = 10


def main(settings=None):
    s = settings or parse_args(__doc__.splitlines()[0])
    out = output_dirs(s)
    P.setup()
    print("\n=== Experiment 2: GBM with proportional transaction costs ===")

    model = GBM(sigma=s.sigma, S0=s.S0)
    train_paths = model.simulate(s.n_train, s.maturity, s.n_steps, seed=s.seed_train)
    test_paths = model.simulate(s.n_test, s.maturity, s.n_steps, seed=s.seed_test)
    premium = float(bs_call_price(s.S0, s.K, s.maturity, s.sigma))

    all_rows, hedgers, tables, timings = {}, {}, [], {}
    for bp in COSTS_BP:
        cost = bp * 1e-4
        print(f"\n--- cost {bp} bp ---")
        train = gbm_call_env(train_paths, s.K, s.sigma, cost)
        test = gbm_call_env(test_paths, s.K, s.sigma, cost)
        valid = test.subset(slice(0, 32768))

        t0 = time.time()
        ntb = DeepHedger(train.n_features, 1, policy="ntb", hidden=(32, 32),
                         risk=EntropicRisk(s.risk_aversion), seed=0)
        ntb.fit(train, n_iters=s.iters_ntb, batch_size=s.batch_size, lr=1e-2, lr_end=3e-4,
                log_every=max(s.iters_ntb // 5, 1), valid_env=valid)
        timings[f"ntb_{bp}bp"] = time.time() - t0
        ntb.save(out["models"] / f"gbm_costs_{bp}bp_ntb.npz")
        hedgers[bp] = ntb

        positions = {
            "bs_delta": stack_instruments(bs_delta(test_paths, s.K, s.sigma)),
            "leland": stack_instruments(leland_delta(test_paths, s.K, s.sigma, cost)),
            "whalley_wilmott": stack_instruments(
                whalley_wilmott(test_paths, s.K, s.sigma, cost, s.risk_aversion)),
            "deep_ntb": ntb.positions(test),
        }
        if bp == REFERENCE_BP:
            t0 = time.time()
            mlp = DeepHedger(train.n_features, 1, policy="mlp", hidden=(32, 32),
                             risk=EntropicRisk(s.risk_aversion), seed=0)
            mlp.fit(train, n_iters=s.iters_mlp, batch_size=s.batch_size, lr=5e-3, lr_end=1e-4,
                    log_every=max(s.iters_mlp // 5, 1), valid_env=valid)
            timings[f"mlp_{bp}bp"] = time.time() - t0
            mlp.save(out["models"] / f"gbm_costs_{bp}bp_mlp.npz")
            positions["deep_mlp"] = mlp.positions(test)

        rows = {k: evaluate(v, test, premium, s) for k, v in positions.items()}
        all_rows[bp] = rows
        table = markdown_table(rows, {k: P.label(k) for k in rows})
        print("\n" + table)
        tables.append(f"### Proportional cost {bp} bp\n\n{table}\n")

    (out["tables"] / "gbm_transaction_costs.md").write_text("\n".join(tables))
    save_json(out["results"] / "gbm_transaction_costs.json", {
        "settings": settings_dict(s), "premium_bs": premium,
        "metrics": {f"{bp}bp": r for bp, r in all_rows.items()},
        "train_seconds": timings,
        "history": {f"{bp}bp": h.history for bp, h in hedgers.items()},
    })

    ref_cost = REFERENCE_BP * 1e-4
    ref = hedgers[REFERENCE_BP]

    # ---- Figure 3: the learned no-transaction band vs Whalley-Wilmott -------
    S = np.linspace(88, 112, 241)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, days in zip(axes, (20, 5)):
        tau = days / 365
        delta = bs_call_delta(S, s.K, tau, s.sigma)
        feats = np.stack([np.log(S / s.K) / (s.sigma * np.sqrt(s.maturity)),
                          np.full_like(S, tau / s.maturity)], axis=1)
        lower, upper = ref.band(feats, delta[:, None])
        half = whalley_wilmott_half_width(S, s.K, tau, s.sigma, ref_cost, s.risk_aversion)
        ax.fill_between(S, lower[:, 0], upper[:, 0], color=P.BAND_FILL, linewidth=0,
                        label="Learned no-trade band")
        P.line(ax, S, lower[:, 0], "deep_ntb", label="_nolegend_", linewidth=1.2)
        P.line(ax, S, upper[:, 0], "deep_ntb", label="_nolegend_", linewidth=1.2)
        P.line(ax, S, delta - half, "whalley_wilmott")
        P.line(ax, S, delta + half, "whalley_wilmott", label="_nolegend_")
        P.line(ax, S, delta, "bs_delta")
        ax.set_title(f"{days} days to maturity")
        ax.set_xlabel("Spot")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Shares held per option")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Do not trade inside the band — learned vs Whalley-Wilmott ({REFERENCE_BP} bp costs)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=P.TEXT)
    fig.tight_layout()
    fig.savefig(out["figures"] / "tc_no_trade_band.png")
    plt.close(fig)

    # ---- Figure 4: one test path, positions through time -------------------
    test = gbm_call_env(test_paths, s.K, s.sigma, ref_cost)
    path = _pick_path(test_paths, s.K)
    t_days = test_paths.times[:-1] * 365
    sub = test.subset([path])
    deep_pos = ref.positions(sub)[0, :, 0]
    lower, upper = ref.band(sub.features[0], sub.base[0])
    bs_pos = bs_delta(test_paths, s.K, s.sigma)[path]
    ww_pos = whalley_wilmott(test_paths, s.K, s.sigma, ref_cost, s.risk_aversion)[path]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.6]})
    ax1.plot(test_paths.times * 365, test_paths.spot[path], color=P.TEXT_SECONDARY, linewidth=1.6)
    ax1.axhline(s.K, color=P.TEXT_SECONDARY, linewidth=0.9, linestyle=":", zorder=0)
    ax1.text(0.3, s.K, "strike", va="bottom", fontsize=9, color=P.TEXT_SECONDARY)
    ax1.set_ylabel("Spot")
    ax1.set_title("One test path", loc="left")
    # holdings over [t_i, t_{i+1}): repeat the last value at maturity to draw the final interval
    t_ext = np.append(t_days, test_paths.times[-1] * 365)
    ext = lambda a: np.append(a, a[-1])
    ax2.fill_between(t_ext, ext(lower[:, 0]), ext(upper[:, 0]), step="post", color=P.BAND_FILL,
                     linewidth=0, label="Learned no-trade band")
    for key, pos in (("bs_delta", bs_pos), ("whalley_wilmott", ww_pos), ("deep_ntb", deep_pos)):
        lab, colour, ls = P.STYLE[key]
        ax2.step(t_ext, ext(pos), where="post", color=colour, linestyle=ls, label=lab, linewidth=1.8)
    ax2.set_xlabel("Days since inception")
    ax2.set_ylabel("Shares held per option")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out["figures"] / "tc_sample_path.png")
    plt.close(fig)

    # ---- Figure 5: cost of hedging vs transaction costs --------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.array(COSTS_BP)
    for key in ("bs_delta", "leland", "whalley_wilmott", "deep_ntb"):
        y = [all_rows[bp][key]["indifference_price"] - premium for bp in COSTS_BP]
        P.line(ax, x, y, key, marker="o", markersize=6)
    ax.set_xticks(x)
    ax.set_xlabel("Proportional transaction cost (bp)")
    ax.set_ylabel("Indifference price − BS price")
    ax.set_title("Extra premium needed to hedge (lower is better)", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(out["figures"] / "tc_price_vs_cost.png")
    plt.close(fig)

    # ---- Figure 6: P&L distribution at the reference cost ------------------
    rows = all_rows[REFERENCE_BP]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = P.bins(*(rows[k]["_pnl"] for k in ("bs_delta", "whalley_wilmott", "deep_ntb")))
    for key in ("bs_delta", "whalley_wilmott", "deep_ntb"):
        P.hist(ax, rows[key]["_pnl"], bins, key)
    ax.set_xlabel("Terminal P&L per option (premium = BS price, S0 = 100)")
    ax.set_ylabel("Density")
    ax.set_title(f"Hedged P&L with {REFERENCE_BP} bp transaction costs", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(out["figures"] / "tc_pnl.png")
    plt.close(fig)
    print(f"\nFigures and tables written to {out['results']}")
    return all_rows


def _pick_path(paths, K):
    """A representative path that crosses the strike several times."""
    S = paths.spot
    crossings = np.sum(np.diff(np.sign(S - K), axis=1) != 0, axis=1)
    candidates = np.where(crossings >= 3)[0]
    return int(candidates[0]) if candidates.size else 0


if __name__ == "__main__":
    main()
