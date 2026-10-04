"""Convex risk measures used as training objectives, and evaluation metrics.

Both measures are *cash-invariant*: rho(X + c) = rho(X) - c. Evaluated on
the hedged P&L without premium, rho is therefore the premium the hedger needs
to charge to be indifferent between selling the claim and doing nothing — the
indifference price of Buehler et al. (2019).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class EntropicRisk:
    """Entropic risk  rho(X) = (1/lambda) log E[exp(-lambda X)]  (exponential utility)."""

    name = "entropic"

    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = float(risk_aversion)

    def init_aux(self, pnl: np.ndarray) -> np.ndarray:
        return np.zeros(1)

    def __call__(self, pnl: np.ndarray) -> float:
        a = -self.risk_aversion * pnl
        m = a.max()
        return float((m + np.log(np.mean(np.exp(a - m)))) / self.risk_aversion)

    def value_and_grad(self, pnl: np.ndarray, aux: np.ndarray):
        """Return (rho, d rho / d pnl, d rho / d aux)."""
        a = -self.risk_aversion * pnl
        m = a.max()
        e = np.exp(a - m)
        value = (m + np.log(e.mean())) / self.risk_aversion
        return float(value), -e / e.sum(), np.zeros_like(aux)

    def __repr__(self):
        return f"EntropicRisk(risk_aversion={self.risk_aversion})"


class CVaR:
    """Conditional Value-at-Risk (expected shortfall) of the loss -X at level alpha.

    Training uses the Rockafellar-Uryasev representation
        CVaR_alpha(X) = min_w  w + E[(-X - w)^+] / (1 - alpha),
    where the auxiliary variable w (the VaR) is learned jointly.
    """

    name = "cvar"

    def __init__(self, alpha: float = 0.95):
        self.alpha = float(alpha)

    def init_aux(self, pnl: np.ndarray) -> np.ndarray:
        return np.array([np.quantile(-pnl, self.alpha)])

    def __call__(self, pnl: np.ndarray) -> float:
        loss = np.sort(-pnl)
        k = int(np.ceil((1.0 - self.alpha) * loss.size))
        return float(loss[-k:].mean())

    def value_and_grad(self, pnl: np.ndarray, aux: np.ndarray):
        w = aux[0]
        excess = -pnl - w
        tail = excess > 0
        scale = 1.0 / ((1.0 - self.alpha) * pnl.size)
        value = w + scale * np.sum(np.maximum(excess, 0.0))
        d_pnl = -scale * tail.astype(float)
        d_w = np.array([1.0 - scale * tail.sum()])
        return float(value), d_pnl, d_w

    def __repr__(self):
        return f"CVaR(alpha={self.alpha})"


def summarize(pnl_ex_premium: np.ndarray, costs: np.ndarray, turnover: np.ndarray,
              premium: float, risk_aversion: float = 1.0, alpha: float = 0.95) -> Dict[str, float]:
    """Evaluation metrics of a hedging strategy.

    ``pnl_ex_premium`` is the terminal P&L without premium; ``premium`` is the
    reference model price added back for the P&L statistics.
    """
    pnl = pnl_ex_premium + premium
    loss = -pnl
    var = float(np.quantile(loss, alpha))
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
        "var95": var,
        "cvar95": CVaR(alpha)(pnl),
        "mean_costs": float(costs.mean()),
        "turnover": float(turnover.mean()),
        "indifference_price": EntropicRisk(risk_aversion)(pnl_ex_premium),
    }
