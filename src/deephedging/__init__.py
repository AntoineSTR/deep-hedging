"""Deep hedging of options under transaction costs and stochastic volatility."""

from .deep_hedger import DeepHedger
from .hedging import HedgingEnv, terminal_pnl, turnover
from .market import GBM, Heston, MarketPaths
from .pricing import HestonParams
from .risk import CVaR, EntropicRisk, summarize

__all__ = [
    "DeepHedger", "HedgingEnv", "terminal_pnl", "turnover",
    "GBM", "Heston", "MarketPaths", "HestonParams",
    "CVaR", "EntropicRisk", "summarize",
]

__version__ = "1.0.0"
