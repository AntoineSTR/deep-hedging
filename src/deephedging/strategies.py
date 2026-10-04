"""Classical hedging strategies used as benchmarks.

Every function returns holdings of shape (B, N): the quantity of the
underlying held over [t_i, t_{i+1}). Use :func:`stack_instruments` to build the
(B, N, n) array expected by :func:`deephedging.hedging.terminal_pnl`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .market import MarketPaths
from .pricing import (
    HestonParams,
    bs_call_delta,
    d_expected_integrated_variance_dv,
    heston_call,
    leland_sigma,
    whalley_wilmott_half_width,
)


def _grid(paths: MarketPaths):
    S = paths.spot[:, :-1]
    tau = (paths.maturity - paths.times[:-1])[None, :]
    return S, tau


def stack_instruments(*holdings: np.ndarray) -> np.ndarray:
    """Stack (B, N) holdings of each instrument into a (B, N, n) array."""
    return np.stack(holdings, axis=-1)


def no_hedge(paths: MarketPaths) -> np.ndarray:
    return np.zeros_like(paths.spot[:, :-1])


def bs_delta(paths: MarketPaths, K: float, sigma: float) -> np.ndarray:
    """Black-Scholes delta, rebalanced at every date."""
    S, tau = _grid(paths)
    return bs_call_delta(S, K, tau, sigma)


def leland_delta(paths: MarketPaths, K: float, sigma: float, cost: float) -> np.ndarray:
    """Black-Scholes delta computed with Leland's cost-adjusted volatility."""
    dt = paths.times[1] - paths.times[0]
    return bs_delta(paths, K, leland_sigma(sigma, cost, dt))


def whalley_wilmott(paths: MarketPaths, K: float, sigma: float, cost: float,
                    risk_aversion: float) -> np.ndarray:
    """Whalley-Wilmott no-transaction band around the BS delta.

    The position is left unchanged while it stays inside
    [delta - H, delta + H] and moved to the nearest edge otherwise.
    """
    S, tau = _grid(paths)
    delta = bs_call_delta(S, K, tau, sigma)
    half = whalley_wilmott_half_width(S, K, tau, sigma, cost, risk_aversion)
    lower, upper = delta - half, delta + half
    out = np.empty_like(delta)
    prev = np.zeros(delta.shape[0])
    for i in range(delta.shape[1]):
        prev = np.clip(prev, lower[:, i], upper[:, i])
        out[:, i] = prev
    return out


def heston_greeks(paths: MarketPaths, K: float, params: HestonParams):
    """Heston delta and dC/dv on every path and rebalancing date."""
    S, tau = _grid(paths)
    delta = np.empty_like(S)
    dcdv = np.empty_like(S)
    for i in range(S.shape[1]):
        _, delta[:, i], dcdv[:, i] = heston_call(S[:, i], paths.var[:, i], float(tau[0, i]), K, params)
    return delta, dcdv


def heston_strategies(paths: MarketPaths, K: float, params: HestonParams,
                      variance_swap_notional: float) -> Dict[str, Dict[str, np.ndarray]]:
    """Model-based Heston hedges.

    Returns a dict  name -> {"spot": holdings, "variance_swap": holdings}:

    * ``heston_delta``: dC/dS, spot only.
    * ``min_variance_delta``: dC/dS + rho xi / S dC/dv, the spot hedge that
      minimises the local variance of the hedged book (spot only).
    * ``delta_vega``: dC/dS in spot plus dC/dv / (dV/dv) variance swaps,
      the perfect hedge in continuous time (Heston is complete with both).
    """
    S, tau = _grid(paths)
    delta, dcdv = heston_greeks(paths, K, params)
    zeros = np.zeros_like(delta)
    mv = delta + params.rho * params.xi / S * dcdv
    dvs_dv = variance_swap_notional * d_expected_integrated_variance_dv(tau, params.kappa)
    return {
        "heston_delta": {"spot": delta, "variance_swap": zeros},
        "min_variance_delta": {"spot": mv, "variance_swap": zeros},
        "delta_vega": {"spot": delta, "variance_swap": dcdv / dvs_dv},
    }
