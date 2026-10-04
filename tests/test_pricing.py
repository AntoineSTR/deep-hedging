"""Black-Scholes and Heston pricing."""
import unittest

import numpy as np
from scipy.integrate import quad

import _setup  # noqa: F401
from deephedging.pricing import (
    HestonParams,
    _heston_exponents,
    bs_call_delta,
    bs_call_gamma,
    bs_call_price,
    bs_call_vega,
    expected_integrated_variance,
    heston_call,
    leland_sigma,
    whalley_wilmott_half_width,
)

P = HestonParams(kappa=1.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)


class TestBlackScholes(unittest.TestCase):
    def test_known_value(self):
        # S = K = 100, sigma = 20%, 1 year, r = 0: C = 100 (2 N(0.1) - 1) = 7.9656
        self.assertAlmostEqual(bs_call_price(100.0, 100.0, 1.0, 0.2), 7.965567, places=5)

    def test_greeks_against_finite_differences(self):
        S, K, tau, sig, h = 103.0, 100.0, 0.2, 0.25, 1e-4
        fd_delta = (bs_call_price(S + h, K, tau, sig) - bs_call_price(S - h, K, tau, sig)) / (2 * h)
        fd_gamma = (bs_call_delta(S + h, K, tau, sig) - bs_call_delta(S - h, K, tau, sig)) / (2 * h)
        fd_vega = (bs_call_price(S, K, tau, sig + h) - bs_call_price(S, K, tau, sig - h)) / (2 * h)
        self.assertAlmostEqual(bs_call_delta(S, K, tau, sig), fd_delta, places=7)
        self.assertAlmostEqual(bs_call_gamma(S, K, tau, sig), fd_gamma, places=6)
        self.assertAlmostEqual(bs_call_vega(S, K, tau, sig), fd_vega, places=5)

    def test_cost_benchmarks(self):
        self.assertEqual(leland_sigma(0.2, 0.0, 1 / 365), 0.2)
        self.assertGreater(leland_sigma(0.2, 0.001, 1 / 365), 0.2)
        h1 = whalley_wilmott_half_width(100.0, 100.0, 0.1, 0.2, 0.001, 1.0)
        h2 = whalley_wilmott_half_width(100.0, 100.0, 0.1, 0.2, 0.008, 1.0)
        self.assertAlmostEqual(h2 / h1, 2.0, places=10)   # band scales as cost^(1/3)


class TestHeston(unittest.TestCase):
    def reference(self, S, v, tau, K=100.0):
        m = np.log(S / K)

        def integrand(u, shift):
            C, D = _heston_exponents(np.array(u - shift), tau, P)
            return np.real(np.exp(1j * u * m) * np.exp(C + D * v) / (1j * u))

        p1 = 0.5 + quad(integrand, 1e-10, np.inf, args=(1j,), limit=500)[0] / np.pi
        p2 = 0.5 + quad(integrand, 1e-10, np.inf, args=(0.0,), limit=500)[0] / np.pi
        return S * p1 - K * p2, p1

    def test_against_adaptive_quadrature(self):
        for tau in (30 / 365, 5 / 365):
            for S in (92.0, 100.0, 107.0):
                for v in (0.01, 0.04, 0.1):
                    price, delta, _ = heston_call(np.array([S]), np.array([v]), tau, 100.0, P)
                    ref_price, ref_delta = self.reference(S, v, tau)
                    self.assertAlmostEqual(price[0], ref_price, places=5)
                    self.assertAlmostEqual(delta[0], ref_delta, places=5)

    def test_greeks_against_finite_differences(self):
        S = np.array([95.0, 100.0, 105.0])
        v = np.array([0.03, 0.04, 0.06])
        tau = 20 / 365
        _, delta, dcdv = heston_call(S, v, tau, 100.0, P)
        h, hv = 1e-3, 1e-5
        fd_delta = (heston_call(S + h, v, tau, 100.0, P)[0] - heston_call(S - h, v, tau, 100.0, P)[0]) / (2 * h)
        fd_v = (heston_call(S, v + hv, tau, 100.0, P)[0] - heston_call(S, v - hv, tau, 100.0, P)[0]) / (2 * hv)
        np.testing.assert_allclose(delta, fd_delta, atol=1e-6)
        np.testing.assert_allclose(dcdv, fd_v, rtol=1e-5)

    def test_black_scholes_limit(self):
        # with (almost) no vol-of-vol, Heston is BS with the expected average variance
        q = HestonParams(kappa=1.0, theta=0.04, xi=1e-4, rho=0.0, v0=0.06)
        tau = 30 / 365
        sigma = np.sqrt(expected_integrated_variance(q.v0, tau, q.kappa, q.theta) / tau)
        price = heston_call(np.array([100.0]), np.array([q.v0]), tau, 100.0, q)[0][0]
        self.assertAlmostEqual(price, bs_call_price(100.0, 100.0, tau, sigma), places=6)


if __name__ == "__main__":
    unittest.main()
