"""Builders that turn simulated paths into hedging environments.

Features are normalised so that every input of the network is O(1).
"""
from __future__ import annotations

import numpy as np

from .hedging import HedgingEnv
from .market import Heston, MarketPaths
from .pricing import bs_call_delta


def call_payoff(paths: MarketPaths, K: float) -> np.ndarray:
    return np.maximum(paths.spot[:, -1] - K, 0.0)


def gbm_call_env(paths: MarketPaths, K: float, sigma: float, cost: float) -> HedgingEnv:
    """Short European call hedged with the underlying only.

    Features: log-moneyness in units of total volatility, and the fraction of
    time to maturity remaining. The BS delta is attached as ``base`` for the
    no-transaction-band network.
    """
    T = paths.maturity
    S = paths.spot[:, :-1]
    tau = (T - paths.times[:-1])[None, :]
    log_m = np.log(S / K) / (sigma * np.sqrt(T))
    time_left = np.broadcast_to(tau / T, S.shape)
    features = np.stack([log_m, time_left], axis=-1)
    base = bs_call_delta(S, K, tau, sigma)[..., None]
    return HedgingEnv(
        prices=paths.spot[..., None],
        payoff=call_payoff(paths, K),
        costs=np.array([cost]),
        features=features,
        base=base,
        instrument_names=("spot",),
    )


def heston_call_env(paths: MarketPaths, model: Heston, K: float, cost: float = 0.0,
                    variance_swap: bool = False, variance_swap_notional: float = 100.0) -> HedgingEnv:
    """Short European call under Heston, hedged with spot (and a variance swap).

    Features: log-moneyness, time left, and instantaneous volatility relative
    to its long-run level. The variance swap pays
    ``variance_swap_notional * int_0^T v_s ds`` and is traded without costs.
    """
    p = model.params
    T = paths.maturity
    S = paths.spot[:, :-1]
    tau = (T - paths.times[:-1])[None, :]
    log_m = np.log(S / K) / (np.sqrt(p.theta * T))
    time_left = np.broadcast_to(tau / T, S.shape)
    vol = (np.sqrt(paths.var[:, :-1]) - np.sqrt(p.theta)) / np.sqrt(p.theta)
    features = np.stack([log_m, time_left, vol], axis=-1)

    prices = [paths.spot]
    costs = [cost]
    names = ["spot"]
    if variance_swap:
        prices.append(variance_swap_notional * model.variance_swap_price(paths))
        costs.append(0.0)
        names.append("variance_swap")
    return HedgingEnv(
        prices=np.stack(prices, axis=-1),
        payoff=call_payoff(paths, K),
        costs=np.array(costs),
        features=features,
        instrument_names=tuple(names),
    )
