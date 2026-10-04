"""Shared figure style.

Colours follow the strategy, not its rank: the same strategy has the same
colour (and line style) in every figure of the project. Benchmarks are drawn
dashed or dotted so that identity never relies on colour alone.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
BAND_FILL = "#cde2fb"

STYLE = {
    # strategy key: (label, colour, linestyle)
    "deep_ntb": ("Deep hedger (band network)", "#2a78d6", "-"),
    "deep_mlp": ("Deep hedger (recurrent MLP)", "#2a78d6", "-"),
    "bs_delta": ("Black-Scholes delta", "#eb6834", "--"),
    "whalley_wilmott": ("Whalley-Wilmott band", "#1baf7a", "-."),
    "leland": ("Leland delta", "#eda100", ":"),
    "no_hedge": ("No hedge", "#8a8984", ":"),
    # Heston
    "deep_spot": ("Deep hedger (spot)", "#2a78d6", "-"),
    "heston_delta": ("Heston delta", "#eb6834", "--"),
    "min_variance_delta": ("Minimum-variance delta", "#1baf7a", "-."),
    "deep_spot_vs": ("Deep hedger (spot + var. swap)", "#4a3aa7", "-"),
    "delta_vega": ("Delta-vega (spot + var. swap)", "#eda100", ":"),
}


def label(key: str) -> str:
    return STYLE[key][0]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "font.size": 10.5,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def line(ax, x, y, key, **kw):
    lab, colour, ls = STYLE[key]
    kw.setdefault("label", lab)
    return ax.plot(x, y, color=colour, linestyle=ls, **kw)


def hist(ax, values, bins, key, **kw):
    lab, colour, ls = STYLE[key]
    kw.setdefault("label", lab)
    return ax.hist(values, bins=bins, density=True, histtype="step", color=colour,
                   linestyle=ls, linewidth=1.8, **kw)


def bins(*samples, n: int = 120, coverage: float = 0.998):
    """Common histogram bins covering the central ``coverage`` of all samples."""
    import numpy as np

    pooled = np.concatenate([np.asarray(x).ravel() for x in samples])
    lo, hi = np.quantile(pooled, [(1 - coverage) / 2, (1 + coverage) / 2])
    return np.linspace(lo, hi, n + 1)
