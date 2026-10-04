"""Closed-form and semi-closed-form pricing used as benchmarks.

All functions assume zero interest rates and no dividends, so prices are
expressed in units of the money-market account.

Contents
--------
* Black-Scholes price and Greeks of a European call.
* Leland (1985) volatility adjustment for discrete hedging under costs.
* Whalley & Wilmott (1997) asymptotically optimal no-transaction band.
* Heston (1993) call price, delta and variance sensitivity by Fourier inversion
  (Gil-Pelaez), using the "little Heston trap" formulation of Albrecher et al.
  (2007) for numerical stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

_TAU_FLOOR = 1e-10


# ----------------------------------------------------------------------------
# Black-Scholes
# ----------------------------------------------------------------------------
def _d1_d2(S, K, tau, sigma):
    tau = np.maximum(tau, _TAU_FLOOR)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + 0.5 * vol**2) / vol
    return d1, d1 - vol


def bs_call_price(S, K, tau, sigma):
    """Black-Scholes price of a European call."""
    d1, d2 = _d1_d2(S, K, tau, sigma)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_call_delta(S, K, tau, sigma):
    """Black-Scholes delta of a European call."""
    d1, _ = _d1_d2(S, K, tau, sigma)
    return norm.cdf(d1)


def bs_call_gamma(S, K, tau, sigma):
    """Black-Scholes gamma of a European call."""
    d1, _ = _d1_d2(S, K, tau, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(np.maximum(tau, _TAU_FLOOR)))


def bs_call_vega(S, K, tau, sigma):
    """Black-Scholes vega (derivative w.r.t. sigma) of a European call."""
    d1, _ = _d1_d2(S, K, tau, sigma)
    return S * norm.pdf(d1) * np.sqrt(np.maximum(tau, _TAU_FLOOR))


# ----------------------------------------------------------------------------
# Transaction-cost benchmarks
# ----------------------------------------------------------------------------
def leland_sigma(sigma, cost, dt):
    """Leland (1985) adjusted volatility for a *short* option position.

    Rebalancing every ``dt`` with proportional cost ``cost`` per side
    (round trip ``2 * cost``) is equivalent, to first order, to hedging with
    the enlarged volatility  sigma^2 (1 + sqrt(2/pi) * 2 cost / (sigma sqrt(dt))).
    """
    leland_number = np.sqrt(2.0 / np.pi) * 2.0 * cost / (sigma * np.sqrt(dt))
    return sigma * np.sqrt(1.0 + leland_number)


def whalley_wilmott_half_width(S, K, tau, sigma, cost, risk_aversion):
    """Half-width of the Whalley & Wilmott (1997) no-transaction band.

    Asymptotically optimal for exponential utility with absolute risk
    aversion ``risk_aversion`` and small proportional costs ``cost``:
        H = (3/2 * cost * S * Gamma^2 / risk_aversion)^(1/3).
    The hedger only trades when the position leaves [delta - H, delta + H],
    and then only up to the nearest edge of the band.
    """
    gamma = bs_call_gamma(S, K, tau, sigma)
    return np.cbrt(1.5 * cost * S * gamma**2 / risk_aversion)


# ----------------------------------------------------------------------------
# Heston
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class HestonParams:
    """Parameters of the Heston model  dv = kappa (theta - v) dt + xi sqrt(v) dW."""

    kappa: float = 1.0   # mean-reversion speed
    theta: float = 0.04  # long-run variance
    xi: float = 0.5      # volatility of variance
    rho: float = -0.7    # spot/variance correlation
    v0: float = 0.04     # initial variance


def expected_integrated_variance(v, tau, kappa, theta):
    """E[ int_t^{t+tau} v_s ds | v_t = v ] under Heston (fair variance-swap leg)."""
    weight = -np.expm1(-kappa * np.asarray(tau, dtype=float)) / kappa
    return (np.asarray(v) - theta) * weight + theta * np.asarray(tau)


def d_expected_integrated_variance_dv(tau, kappa):
    """Derivative of :func:`expected_integrated_variance` with respect to v."""
    return -np.expm1(-kappa * np.asarray(tau, dtype=float)) / kappa


def _heston_exponents(u, tau, p: HestonParams):
    """Return (C, D) with E[exp(i u log(S_T / S_t)) | v_t] = exp(C + D v_t).

    ``u`` may be complex. Little-trap formulation (Albrecher et al., 2007).
    """
    iu = 1j * u
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta * beta + p.xi**2 * (iu + u * u))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * tau)
    one_minus_ge = 1.0 - g * e
    D = (beta - d) / p.xi**2 * (1.0 - e) / one_minus_ge
    C = p.kappa * p.theta / p.xi**2 * ((beta - d) * tau - 2.0 * np.log(one_minus_ge / (1.0 - g)))
    return C, D


# Gauss-Legendre nodes on [0, W] for the (rescaled) Fourier integrals.
_GL_W_MAX = 24.0
_GL_N = 192
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_GL_NODES = 0.5 * _GL_W_MAX * (_gl_x + 1.0)
_GL_WEIGHTS = 0.5 * _GL_W_MAX * _gl_w


def heston_call(S, v, tau, K, params: HestonParams, chunk_size: int = 4096):
    """Heston call price, delta and dPrice/dv by Fourier inversion.

    Parameters
    ----------
    S, v : array_like, shape (n,)
        Spot and instantaneous variance.
    tau : float
        Time to maturity (same for all entries).
    K : float
        Strike.

    Returns
    -------
    price, delta, dprice_dv : ndarray, shape (n,)

    Notes
    -----
    Gil-Pelaez:  C = S P1 - K P2  with
        P2 = 1/2 + 1/pi int_0^inf Re[ e^{i u m} psi(u)     / (i u) ] du
        P1 = 1/2 + 1/pi int_0^inf Re[ e^{i u m} psi(u - i) / (i u) ] du
    where m = log(S/K) and psi is the characteristic function of log(S_T/S_t).
    Delta is exactly P1, and dC/dv follows from d psi / dv = D psi.
    The integration variable is rescaled path by path by the expected total
    variance, u = w / sqrt(E[int v]), which keeps the quadrature accurate from
    long maturities down to the last hedging date.
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    v = np.broadcast_to(np.asarray(v, dtype=float), S.shape)
    v = np.maximum(v, 0.0)
    n = S.shape[0]

    if tau <= _TAU_FLOOR:
        itm = (S > K).astype(float)
        return np.maximum(S - K, 0.0), itm, np.zeros(n)

    price = np.empty(n)
    delta = np.empty(n)
    vega_v = np.empty(n)
    for start in range(0, n, chunk_size):
        sl = slice(start, start + chunk_size)
        s, vv = S[sl], v[sl]
        scale = np.sqrt(np.maximum(expected_integrated_variance(vv, tau, params.kappa, params.theta), 1e-12))
        u = _GL_NODES[None, :] / scale[:, None]                      # (c, m)
        m = np.log(s / K)[:, None]
        phase_over_iu = np.exp(1j * u * m) / (1j * u)

        C2, D2 = _heston_exponents(u, tau, params)
        C1, D1 = _heston_exponents(u - 1j, tau, params)
        psi2 = np.exp(C2 + D2 * vv[:, None])
        psi1 = np.exp(C1 + D1 * vv[:, None])

        w = _GL_WEIGHTS[None, :] / scale[:, None] / np.pi
        P1 = 0.5 + np.sum(w * np.real(phase_over_iu * psi1), axis=1)
        P2 = 0.5 + np.sum(w * np.real(phase_over_iu * psi2), axis=1)
        dP1 = np.sum(w * np.real(phase_over_iu * D1 * psi1), axis=1)
        dP2 = np.sum(w * np.real(phase_over_iu * D2 * psi2), axis=1)

        # More than 8 'standard deviations' from the strike: the option is
        # digital-like and the truncated quadrature is less accurate than the limit.
        far = (np.abs(m[:, 0]) / scale) > 8.0
        itm = (m[:, 0] > 0).astype(float)
        P1 = np.where(far, itm, np.clip(P1, 0.0, 1.0))
        P2 = np.where(far, itm, np.clip(P2, 0.0, 1.0))
        dP1 = np.where(far, 0.0, dP1)
        dP2 = np.where(far, 0.0, dP2)
        price[sl] = np.maximum(s * P1 - K * P2, np.maximum(s - K, 0.0))
        delta[sl] = P1
        vega_v[sl] = s * dP1 - K * dP2
    return price, delta, vega_v
