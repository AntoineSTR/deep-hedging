"""Hedging environment and terminal P&L with proportional transaction costs.

Conventions
-----------
* A hedging problem has N rebalancing dates t_0, ..., t_{N-1} and maturity t_N.
* ``positions[:, i, k]`` is the quantity of instrument k held over [t_i, t_{i+1}).
* The book starts flat (position 0 before t_0) and is unwound at t_N.
* Trading q units of instrument k at price P costs ``costs[k] * |q| * P``.

The terminal P&L of a short claim Z hedged with positions delta is

    PnL = p0 - Z + sum_i delta_i . (P_{i+1} - P_i) - sum_{i=0}^{N} c . |delta_i - delta_{i-1}| P_i

with delta_{-1} = delta_N = 0. ``terminal_pnl`` returns it *without* the
premium p0, so that risk measures evaluated on it directly give the
indifference (required) premium.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class HedgingEnv:
    """Everything a hedging strategy needs, on a batch of simulated paths.

    Attributes
    ----------
    prices : (B, N+1, n) prices of the hedging instruments.
    payoff : (B,) payoff of the short claim at maturity.
    costs : (n,) proportional transaction cost of each instrument.
    features : (B, N, d) information available at each rebalancing date.
    base : (B, N, n) optional reference hedge (e.g. BS delta) used by the
        no-transaction-band policy.
    instrument_names : labels for plots and tables.
    """

    prices: np.ndarray
    payoff: np.ndarray
    costs: np.ndarray
    features: np.ndarray
    base: Optional[np.ndarray] = None
    instrument_names: Sequence[str] = field(default_factory=lambda: ("spot",))

    def __post_init__(self):
        self.costs = np.asarray(self.costs, dtype=float).reshape(-1)
        B, N1, n = self.prices.shape
        assert self.payoff.shape == (B,)
        assert self.costs.shape == (n,)
        assert self.features.shape[:2] == (B, N1 - 1)
        if self.base is not None:
            assert self.base.shape == (B, N1 - 1, n)

    @property
    def n_paths(self) -> int:
        return self.prices.shape[0]

    @property
    def n_steps(self) -> int:
        return self.prices.shape[1] - 1

    @property
    def n_instruments(self) -> int:
        return self.prices.shape[2]

    @property
    def n_features(self) -> int:
        return self.features.shape[2]

    def subset(self, idx) -> "HedgingEnv":
        """Restrict the environment to a subset of paths."""
        return HedgingEnv(
            prices=self.prices[idx],
            payoff=self.payoff[idx],
            costs=self.costs,
            features=self.features[idx],
            base=None if self.base is None else self.base[idx],
            instrument_names=self.instrument_names,
        )


def _trades(positions: np.ndarray) -> np.ndarray:
    """Trades at t_0..t_N, shape (B, N+1, n), including the final unwind."""
    zeros = np.zeros_like(positions[:, :1, :])
    before = np.concatenate([zeros, positions], axis=1)
    after = np.concatenate([positions, zeros], axis=1)
    return after - before


def terminal_pnl(positions: np.ndarray, env: HedgingEnv):
    """Terminal P&L (premium excluded) and total transaction costs per path.

    Parameters
    ----------
    positions : (B, N, n) holdings over each hedging interval.
    env : HedgingEnv

    Returns
    -------
    pnl : (B,)  -Z + trading gains - transaction costs
    costs : (B,) transaction costs paid
    """
    gains = np.sum(positions * np.diff(env.prices, axis=1), axis=(1, 2))
    trade_value = np.abs(_trades(positions)) * env.prices * env.costs
    costs = np.sum(trade_value, axis=(1, 2))
    return gains - env.payoff - costs, costs


def pnl_gradient(positions: np.ndarray, env: HedgingEnv) -> np.ndarray:
    """Partial derivative of :func:`terminal_pnl` w.r.t. each position, (B, N, n).

    d PnL / d delta_i = (P_{i+1} - P_i) - c P_i sign(trade_i) + c P_{i+1} sign(trade_{i+1})
    (the cost term uses a subgradient at zero trades).
    """
    sign = np.sign(_trades(positions))                      # (B, N+1, n)
    cost_grad = env.costs * env.prices * sign               # d cost_k / d delta_k
    return np.diff(env.prices, axis=1) - cost_grad[:, :-1, :] + cost_grad[:, 1:, :]


def turnover(positions: np.ndarray) -> np.ndarray:
    """Total absolute quantity traded per path and instrument, (B, n)."""
    return np.sum(np.abs(_trades(positions)), axis=1)
