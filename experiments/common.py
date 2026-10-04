"""Shared setup for the experiments: paths, settings, evaluation and tables."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from deephedging.hedging import terminal_pnl, turnover  # noqa: E402
from deephedging.risk import summarize  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
TABLES = RESULTS / "tables"


@dataclass
class Settings:
    """Contract and training settings shared by all experiments."""

    S0: float = 100.0
    K: float = 100.0                 # at-the-money call
    maturity: float = 30 / 365       # 30 calendar days
    n_steps: int = 30                # daily rebalancing
    sigma: float = 0.20              # Black-Scholes volatility
    risk_aversion: float = 1.0       # entropic risk / exponential utility
    n_train: int = 2**17             # 131,072 training paths
    n_test: int = 2**17              # 131,072 independent test paths
    n_test_heston: int = 2**15       # Heston benchmarks need Fourier pricing on every node
    iters_mlp: int = 3000
    iters_ntb: int = 1500
    batch_size: int = 8192
    seed_train: int = 1
    seed_test: int = 2
    quick: bool = False

    @classmethod
    def from_args(cls, quick: bool) -> "Settings":
        if not quick:
            return cls()
        return cls(n_train=2**13, n_test=2**12, n_test_heston=2**10,
                   iters_mlp=150, iters_ntb=100, batch_size=2048, quick=True)


def parse_args(description: str) -> Settings:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true",
                        help="small run (a minute or two) to check that everything works; "
                             "results are written to results/quick/")
    args = parser.parse_args()
    return Settings.from_args(args.quick)


def output_dirs(settings: Settings):
    base = RESULTS / "quick" if settings.quick else RESULTS
    dirs = {"results": base, "figures": base / "figures", "models": base / "models", "tables": base / "tables"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def evaluate(positions: np.ndarray, env, premium: float, settings: Settings) -> dict:
    """Metrics of a strategy on the test environment."""
    pnl, costs = terminal_pnl(positions, env)
    metrics = summarize(pnl, costs, turnover(positions)[:, 0], premium, settings.risk_aversion)
    metrics["_pnl"] = pnl + premium          # kept in memory for plots, not saved
    return metrics


COLUMNS = [
    ("mean_pnl", "Mean P&L"),
    ("std_pnl", "Std P&L"),
    ("cvar95", "CVaR 95%"),
    ("mean_costs", "Avg. costs"),
    ("turnover", "Turnover"),
    ("indifference_price", "Indifference price"),
]


def markdown_table(rows: dict, labels: dict, best: str = "indifference_price") -> str:
    """Markdown table of metrics; the best indifference price is in bold."""
    head = "| Strategy | " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    candidates = {k: v[best] for k, v in rows.items() if k != "no_hedge"}
    winner = min(candidates, key=candidates.get) if candidates else None
    lines = [head, sep]
    for key, m in rows.items():
        cells = []
        for col, _ in COLUMNS:
            txt = f"{m[col]:.3f}" if col != "turnover" else f"{m[col]:.2f}"
            if col == best and key == winner:
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append(f"| {labels[key]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_json(path: pathlib.Path, obj) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    path.write_text(json.dumps(clean(obj), indent=2))


def settings_dict(settings: Settings) -> dict:
    return asdict(settings)
