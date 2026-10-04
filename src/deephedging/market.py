"""Market simulators.

Both models are simulated under the pricing measure with zero rates, so the
spot (and the variance swap in Heston) are martingales. Paths are returned on
the hedging grid t_0 = 0 < t_1 < ... < t_N = T.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pricing import HestonParams, expected_integrated_variance


@dataclass
class MarketPaths:
    """Simulated market states on the hedging grid.

    Attributes
    ----------
    times : (N+1,) hedging dates.
    spot : (B, N+1) underlying price.
    var : (B, N+1) instantaneous variance (Heston only).
    int_var : (B, N+1) realised integrated variance int_0^t v_s ds (Heston only).
    """

    times: np.ndarray
    spot: np.ndarray
    var: Optional[np.ndarray] = None
    int_var: Optional[np.ndarray] = None

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def n_steps(self) -> int:
        return self.spot.shape[1] - 1

    @property
    def maturity(self) -> float:
        return float(self.times[-1])


@dataclass
class GBM:
    """Black-Scholes model dS = mu S dt + sigma S dW."""

    sigma: float = 0.2
    S0: float = 100.0
    mu: float = 0.0

    def simulate(self, n_paths: int, maturity: float, n_steps: int, seed=None) -> MarketPaths:
        rng = np.random.default_rng(seed)
        dt = maturity / n_steps
        z = rng.standard_normal((n_paths, n_steps))
        log_inc = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        log_spot = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
        times = np.linspace(0.0, maturity, n_steps + 1)
        return MarketPaths(times=times, spot=self.S0 * np.exp(log_spot))


@dataclass
class Heston:
    """Heston stochastic-volatility model.

    dS = sqrt(v) S dW1,   dv = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt.

    Discretised with a full-truncation Euler scheme (Lord, Koekkoek & van Dijk,
    2010) and ``substeps`` sub-steps per hedging interval; the spot uses the
    exact log-Euler step given the frozen variance.
    """

    params: HestonParams = HestonParams()
    S0: float = 100.0
    substeps: int = 8

    def simulate(self, n_paths: int, maturity: float, n_steps: int, seed=None) -> MarketPaths:
        p = self.params
        rng = np.random.default_rng(seed)
        dt = maturity / (n_steps * self.substeps)
        sqrt_dt = np.sqrt(dt)
        corr = np.sqrt(1.0 - p.rho**2)

        log_s = np.zeros(n_paths)
        v = np.full(n_paths, p.v0)
        iv = np.zeros(n_paths)
        spot = np.empty((n_paths, n_steps + 1))
        var = np.empty((n_paths, n_steps + 1))
        int_var = np.empty((n_paths, n_steps + 1))
        spot[:, 0], var[:, 0], int_var[:, 0] = 1.0, p.v0, 0.0

        for i in range(n_steps):
            for _ in range(self.substeps):
                z_v = rng.standard_normal(n_paths)
                z_s = p.rho * z_v + corr * rng.standard_normal(n_paths)
                vp = np.maximum(v, 0.0)
                sv = np.sqrt(vp)
                log_s += -0.5 * vp * dt + sv * sqrt_dt * z_s
                iv += vp * dt
                v = v + p.kappa * (p.theta - vp) * dt + p.xi * sv * sqrt_dt * z_v
            spot[:, i + 1] = np.exp(log_s)
            var[:, i + 1] = np.maximum(v, 0.0)
            int_var[:, i + 1] = iv

        times = np.linspace(0.0, maturity, n_steps + 1)
        return MarketPaths(times=times, spot=self.S0 * spot, var=var, int_var=int_var)

    def variance_swap_price(self, paths: MarketPaths) -> np.ndarray:
        """Value process of a variance swap paying int_0^T v_s ds at T (no strike).

        V_t = int_0^t v_s ds + E[int_t^T v_s ds | v_t], a martingale under Heston.
        """
        tau = paths.maturity - paths.times
        future = expected_integrated_variance(paths.var, tau[None, :], self.params.kappa, self.params.theta)
        return paths.int_var + future
