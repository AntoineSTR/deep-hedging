"""Finite-difference checks of the hand-written backpropagation."""
import unittest

import numpy as np

import _setup  # noqa: F401
from deephedging.deep_hedger import DeepHedger
from deephedging.hedging import HedgingEnv
from deephedging.nn import MLP
from deephedging.risk import CVaR, EntropicRisk


def random_env(B=64, N=5, n=2, d=3, cost=0.01, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(np.concatenate(
        [np.zeros((B, 1, n)), 0.05 * rng.standard_normal((B, N, n))], axis=1), axis=1))
    payoff = np.maximum(prices[:, -1, 0] - 100.0, 0.0)
    features = rng.standard_normal((B, N, d))
    base = rng.uniform(0.2, 0.8, size=(B, N, n))
    return HedgingEnv(prices, payoff, np.full(n, cost), features, base, tuple(f"x{k}" for k in range(n)))


def numerical_grad(f, params, eps=1e-6):
    out = []
    for p in params:
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            up = f()
            p[i] = old - eps
            down = f()
            p[i] = old
            g[i] = (up - down) / (2 * eps)
        out.append(g)
    return out


def max_rel_error(a_list, b_list):
    num = max(np.max(np.abs(a - b)) for a, b in zip(a_list, b_list))
    den = max(np.max(np.abs(b)) for b in b_list) + 1e-12
    return num / den


class TestMLP(unittest.TestCase):
    def test_mlp_backward(self):
        rng = np.random.default_rng(1)
        net = MLP((4, 7, 5, 3), rng, out_scale=1.0, dtype=np.float64)
        x = rng.standard_normal((10, 4))
        w = rng.standard_normal((10, 3))
        f = lambda: float(np.sum(net.forward(x)[0] * w))
        y, cache = net.forward(x)
        grads, dx = net.backward(cache, w)
        self.assertLess(max_rel_error(grads, numerical_grad(f, net.params)), 1e-6)
        xp = [x]
        f_x = lambda: float(np.sum(net.forward(xp[0])[0] * w))
        self.assertLess(max_rel_error([dx], numerical_grad(f_x, xp)), 1e-6)


class TestDeepHedgerGradients(unittest.TestCase):
    def check(self, policy, risk):
        env = random_env()
        model = DeepHedger(env.n_features, env.n_instruments, policy=policy,
                           hidden=(6, 5), risk=risk, seed=3, dtype=np.float64)
        # larger outputs so that the policy actually trades and clips
        model.net.weights[-1] *= 10.0
        if isinstance(risk, CVaR):
            from deephedging.hedging import terminal_pnl
            pnl, _ = terminal_pnl(model.positions(env), env)
            model.aux[:] = np.quantile(-pnl, 0.7)
        loss, grads = model.loss_and_grads(env)
        f = lambda: model.loss_and_grads(env)[0]
        num = numerical_grad(f, model.parameters)
        self.assertLess(max_rel_error(grads, num), 1e-5)

    def test_mlp_entropic(self):
        self.check("mlp", EntropicRisk(0.5))

    def test_mlp_cvar(self):
        self.check("mlp", CVaR(0.5))

    def test_ntb_entropic(self):
        self.check("ntb", EntropicRisk(0.5))

    def test_ntb_cvar(self):
        self.check("ntb", CVaR(0.5))


if __name__ == "__main__":
    unittest.main()
