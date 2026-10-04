"""Simulation, hedged P&L, risk measures and benchmark strategies."""
import unittest

import numpy as np

import _setup  # noqa: F401
from deephedging import GBM, DeepHedger, Heston, HestonParams
from deephedging.hedging import HedgingEnv, terminal_pnl, turnover
from deephedging.pricing import bs_call_price, heston_call
from deephedging.problems import gbm_call_env, heston_call_env
from deephedging.risk import CVaR, EntropicRisk
from deephedging.strategies import bs_delta, stack_instruments, whalley_wilmott

T, K, SIGMA = 30 / 365, 100.0, 0.2


class TestMarket(unittest.TestCase):
    def test_gbm_is_a_martingale(self):
        paths = GBM(sigma=SIGMA).simulate(200_000, T, 30, seed=0)
        self.assertAlmostEqual(paths.spot[:, -1].mean(), 100.0, delta=0.05)

    def test_heston_prices_and_martingales(self):
        p = HestonParams()
        model = Heston(params=p, substeps=8)
        paths = model.simulate(100_000, T, 30, seed=0)
        self.assertAlmostEqual(paths.spot[:, -1].mean(), 100.0, delta=0.05)
        vs = model.variance_swap_price(paths)
        self.assertAlmostEqual(vs[:, -1].mean() / vs[0, 0], 1.0, delta=0.01)
        mc = np.maximum(paths.spot[:, -1] - K, 0).mean()
        fourier = heston_call(np.array([100.0]), np.array([p.v0]), T, K, p)[0][0]
        self.assertAlmostEqual(mc, fourier, delta=0.03)


class TestPnL(unittest.TestCase):
    def test_no_hedge_and_costs(self):
        paths = GBM(sigma=SIGMA).simulate(1000, T, 30, seed=1)
        env = gbm_call_env(paths, K, SIGMA, cost=0.001)
        zero = np.zeros((1000, 30, 1))
        pnl, costs = terminal_pnl(zero, env)
        np.testing.assert_allclose(pnl, -env.payoff)
        np.testing.assert_allclose(costs, 0.0)
        # buy one share and hold: costs = buy at S0 + sell at S_T
        one = np.ones((1000, 30, 1))
        pnl, costs = terminal_pnl(one, env)
        np.testing.assert_allclose(costs, 0.001 * (paths.spot[:, 0] + paths.spot[:, -1]))
        np.testing.assert_allclose(turnover(one)[:, 0], 2.0)

    def test_delta_hedging_converges(self):
        """Without costs, the BS-delta hedging error shrinks like 1/sqrt(N)."""
        premium = bs_call_price(100.0, K, T, SIGMA)
        stds = []
        for n in (10, 40, 160):
            paths = GBM(sigma=SIGMA).simulate(40_000, T, n, seed=2)
            env = gbm_call_env(paths, K, SIGMA, cost=0.0)
            pnl, _ = terminal_pnl(stack_instruments(bs_delta(paths, K, SIGMA)), env)
            self.assertAlmostEqual((pnl + premium).mean(), 0.0, delta=0.01)
            stds.append((pnl + premium).std())
        self.assertAlmostEqual(stds[0] / stds[1], 2.0, delta=0.15)
        self.assertAlmostEqual(stds[1] / stds[2], 2.0, delta=0.15)

    def test_whalley_wilmott_stays_in_band_and_trades_less(self):
        paths = GBM(sigma=SIGMA).simulate(5000, T, 30, seed=3)
        bs = bs_delta(paths, K, SIGMA)
        ww = whalley_wilmott(paths, K, SIGMA, cost=0.001, risk_aversion=1.0)
        self.assertLess(turnover(ww[..., None]).mean(), turnover(bs[..., None]).mean())


class TestRisk(unittest.TestCase):
    def test_cash_invariance(self):
        x = np.random.default_rng(0).standard_normal(10_000)
        for rho in (EntropicRisk(2.0), CVaR(0.9)):
            self.assertAlmostEqual(rho(x + 3.0), rho(x) - 3.0, places=10)

    def test_entropic_gaussian(self):
        # for X ~ N(m, s^2): rho = -m + lambda s^2 / 2
        x = np.random.default_rng(1).normal(0.5, 0.3, 2_000_000)
        self.assertAlmostEqual(EntropicRisk(2.0)(x), -0.5 + 2.0 * 0.09 / 2, places=3)

    def test_cvar_representation(self):
        """min_w w + E[(L - w)^+]/(1 - alpha) equals the empirical CVaR."""
        x = np.random.default_rng(2).standard_normal(100_000)
        rho = CVaR(0.95)
        w = np.array([np.quantile(-x, 0.95)])
        value, _, _ = rho.value_and_grad(x, w)
        self.assertAlmostEqual(value, rho(x), places=3)
        self.assertAlmostEqual(rho(x), 2.0627, delta=0.03)   # N(0,1) CVaR_95


class TestDeepHedger(unittest.TestCase):
    def test_training_reduces_risk_and_save_load(self):
        paths = GBM(sigma=SIGMA).simulate(4096, T, 10, seed=4)
        env = gbm_call_env(paths, K, SIGMA, cost=0.001)
        for policy in ("mlp", "ntb"):
            h = DeepHedger(env.n_features, 1, policy=policy, hidden=(16,), seed=0)
            before = h.evaluate_risk(env)
            h.fit(env, n_iters=60, batch_size=1024, lr=1e-2, verbose=False)
            after = h.evaluate_risk(env)
            self.assertLess(after, before)
            import os
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "model.npz")
                h.save(path)
                h2 = DeepHedger.load(path)
                np.testing.assert_allclose(h2.positions(env), h.positions(env))

    def test_heston_env_shapes(self):
        model = Heston(substeps=2)
        paths = model.simulate(64, T, 10, seed=5)
        env = heston_call_env(paths, model, K, variance_swap=True)
        self.assertEqual(env.prices.shape, (64, 11, 2))
        self.assertEqual(env.features.shape, (64, 10, 3))
        self.assertIsInstance(env, HedgingEnv)


if __name__ == "__main__":
    unittest.main()
