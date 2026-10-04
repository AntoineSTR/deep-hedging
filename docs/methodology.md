# Methodology

This note documents every modelling and implementation choice in the
repository, from the hedging problem to the gradients used for training.

---

## 1. The hedging problem

A trader sells a European call with payoff $Z = (S_T - K)^+$ and receives the
premium $p_0$. She hedges with $n$ traded instruments with prices
$P_t = (P^1_t, \dots, P^n_t)$, rebalanced at dates
$0 = t_0 < t_1 < \dots < t_N = T$. Interest rates are zero.

A strategy is a sequence of holdings $\delta_0, \dots, \delta_{N-1}$, where
$\delta_i \in \mathbb{R}^n$ is held over $[t_i, t_{i+1})$ and may only depend on
information available at $t_i$. The book starts flat and is unwound at
maturity ($\delta_{-1} = \delta_N = 0$). With proportional costs $c_k$ per
instrument, the terminal profit and loss is

$$
\mathrm{PnL}(\delta) = p_0 - Z
+ \sum_{i=0}^{N-1} \delta_i \cdot (P_{t_{i+1}} - P_{t_i})
- \sum_{i=0}^{N} \sum_{k=1}^{n} c_k \, P^k_{t_i} \, \bigl|\delta^k_i - \delta^k_{i-1}\bigr| .
$$

In a complete, frictionless market with continuous trading, a replicating
strategy makes this P&L zero. With discrete rebalancing, transaction costs or
untraded risk factors, no strategy does, and the question becomes *which risk
to keep*.

## 2. Risk measures and the indifference price

Strategies are ranked by a convex risk measure $\rho$ of the hedged P&L. Both
measures implemented are **cash-invariant**, $\rho(X + c) = \rho(X) - c$, so

$$
\pi(Z) = \min_\delta \rho\bigl(-Z + (\delta \cdot \Delta P) - \text{costs}\bigr)
$$

is the premium that makes the trader indifferent between selling the option and
doing nothing: the **indifference price** of Buehler et al. (2019). It is the
single number used to compare strategies — lower is better — and it decomposes
into the model price, the expected transaction costs and a risk premium for the
unhedgeable risk.

**Entropic risk** (exponential utility with absolute risk aversion $\lambda$):

$$
\rho_\lambda(X) = \frac{1}{\lambda} \log \mathbb{E}\bigl[e^{-\lambda X}\bigr],
\qquad
\frac{\partial \rho_\lambda}{\partial X_b} = -\frac{e^{-\lambda X_b}}{\sum_j e^{-\lambda X_j}}
\ \text{(on a sample)}.
$$

For a Gaussian P&L, $\rho_\lambda = -\mathbb{E}[X] + \tfrac{\lambda}{2}\mathrm{Var}[X]$:
a mean-variance trade-off. All experiments use $\lambda = 1$ with
$S_0 = 100$.

**CVaR** (expected shortfall at level $\alpha$), trained through the
Rockafellar-Uryasev representation, where the VaR level $w$ is learned jointly
with the network:

$$
\mathrm{CVaR}_\alpha(X) = \min_{w \in \mathbb{R}} \; w + \frac{1}{1 - \alpha}\, \mathbb{E}\bigl[(-X - w)^+\bigr].
$$

## 3. Deep hedging

The strategy is a neural network evaluated at every date and trained to
minimise $\rho$ on simulated paths:

$$
\min_\theta \; \hat\rho_B\bigl(\mathrm{PnL}(\delta^\theta)\bigr),
$$

where $\hat\rho_B$ is the sample estimate of $\rho$ on a mini-batch of $B$ simulated paths.

Two policies are implemented.

**Recurrent MLP** (Buehler et al., 2019). The network sees the market state and
its own previous position:

$$
\delta_i = F_\theta\bigl(x_i, \delta_{i-1}\bigr).
$$

Feeding back $\delta_{i-1}$ is what allows it to learn *not to trade* when a
trade costs more than the risk it removes.

**No-Transaction Band Network** (Imaki et al., 2021). Under proportional
costs the optimal policy is known to be a band around a target hedge (Davis &
Norman, 1990; Whalley & Wilmott, 1997). The network therefore only learns the
band, around the Black-Scholes delta $\Delta^{BS}_i$:

$$
\ell_i = \Delta^{BS}_i - \operatorname{softplus}(a_\theta(x_i)), \quad
u_i = \Delta^{BS}_i + \operatorname{softplus}(b_\theta(x_i)), \quad
\delta_i = \operatorname{clip}(\delta_{i-1}, \ell_i, u_i).
$$

Because the band does not depend on $\delta_{i-1}$, the network is evaluated
for all dates in one batched pass, and the structure removes a whole class of
sub-optimal policies from the search space: in practice it converges in a few
hundred iterations where the recurrent MLP needs thousands. The output bias is initialised so that the initial band is
narrow (half-width $\approx 0.05$): training starts from the BS delta.

**Architecture and training.** Two hidden layers of 32 tanh units, linear
output, inputs normalised to $O(1)$ (log-moneyness in units of total
volatility, fraction of time left, and in Heston the volatility relative to its
long-run level). Adam with an exponentially decaying learning rate, mini-batches
of 8,192 paths drawn from 131,072 training paths, gradient-norm clipping at 10.
Evaluation is always on 131,072 independent test paths (32,768 under Heston,
where every benchmark needs Fourier pricing at every node).

## 4. Gradients, by hand

The repository does not use an automatic-differentiation library. Gradients are
derived and coded explicitly, and checked against finite differences in
`tests/test_gradients.py` for both policies and both risk measures.

**Through the P&L.** Writing $s_i = \operatorname{sign}(\delta_i - \delta_{i-1})$,

$$
\frac{\partial\, \mathrm{PnL}}{\partial \delta_i}
= (P_{t_{i+1}} - P_{t_i}) - c\, P_{t_i}\, s_i + c\, P_{t_{i+1}}\, s_{i+1},
$$

since $\delta_i$ enters the trade at $t_i$ and the trade at $t_{i+1}$.

**Through time (recurrent MLP).** $\delta_i$ also affects every later
position through the network input. Going backwards from $i = N-1$ to $0$:

$$
g_i = \frac{\partial \rho}{\partial \mathrm{PnL}}\, \frac{\partial\, \mathrm{PnL}}{\partial \delta_i} + \Bigl(\frac{\partial F_\theta}{\partial \delta_{i-1}}\Bigr)^{\!\top}_{i+1} g_{i+1},
$$

and $g_i$ is backpropagated through the network at date $i$; parameter
gradients are summed over dates.

**Through the clip (band network).** If $\delta_{i-1} < \ell_i$ the position is
$\ell_i$ and the gradient flows to $a_\theta$ with factor
$-\sigma(a_\theta)$ (the derivative of softplus is the sigmoid); if
$\delta_{i-1} > u_i$ it flows to $b_\theta$ with factor $\sigma(b_\theta)$;
inside the band it flows unchanged to $\delta_{i-1}$.

## 5. Benchmarks

**Black-Scholes delta.** $\Delta = N(d_1)$, rebalanced every day.

**Leland (1985).** Delta computed with the enlarged volatility

$$
\tilde\sigma^2 = \sigma^2\Bigl(1 + \sqrt{2/\pi}\, \frac{2c}{\sigma \sqrt{\Delta t}}\Bigr),
$$

which prices in, to first order, the costs of rebalancing every $\Delta t$.

**Whalley & Wilmott (1997).** Asymptotically optimal band for exponential
utility with small proportional costs. Do not trade while the position is in

$$
\Bigl[\Delta - H,\; \Delta + H\Bigr], \qquad
H = \Bigl(\frac{3}{2}\,\frac{c\, S\, \Gamma^2}{\lambda}\Bigr)^{1/3},
$$

and move to the nearest edge otherwise. It optimises the *same* objective as the
deep hedger, which makes it the natural benchmark — but it is derived for
continuous trading and vanishing costs.

**Heston model hedges.** Under
$dS = \sqrt{v} S\, dW^1$, $dv = \kappa(\theta - v)dt + \xi \sqrt{v}\, dW^2$,
$d\langle W^1, W^2\rangle = \rho\, dt$:

* *Heston delta* $C_S$: the model's sensitivity to spot.
* *Minimum-variance delta*: the spot position that minimises the
  instantaneous variance of the hedged book,
  $$
  \delta^{MV} = \frac{d\langle C, S\rangle}{d\langle S, S\rangle} = C_S + \frac{\rho\, \xi}{S}\, C_v .
  $$
  With $\rho < 0$, spot moves predict variance moves, and the correction hedges
  part of the vega exposure through the spot.
* *Delta-vega hedge* with a variance swap. The swap paying $\int_0^T v_s ds$
  has value $V_t = \int_0^t v_s ds + L(t, v_t)$ with
  $L(t, v) = (v - \theta)\frac{1 - e^{-\kappa(T - t)}}{\kappa} + \theta (T - t)$.
  Holding $C_S$ shares and $C_v / \partial_v L$ swaps replicates the option in
  continuous time: the market is complete.

**Fourier pricing.** Prices and Greeks under Heston use the Gil-Pelaez
inversion

$$
C = S P_1 - K P_2,\qquad
P_j = \frac12 + \frac1\pi \int_0^\infty \operatorname{Re}\Bigl[\frac{e^{iu\, \log(S/K)}\, \psi(u - i\,\mathbb{1}_{j=1})}{iu}\Bigr] du,
$$

with the characteristic function $\psi(u) = \exp(C(\tau, u) + D(\tau, u)\, v)$
in the "little Heston trap" form of Albrecher et al. (2007). Delta is exactly
$P_1$, and $C_v$ follows from $\partial_v \psi = D \psi$. The integration
variable is rescaled path by path by $\sqrt{L(t, v)}$ and integrated with 192
Gauss-Legendre nodes; the implementation matches adaptive quadrature to
$10^{-5}$ (see `tests/test_pricing.py`).

## 6. Simulation

* **Black-Scholes**: exact log-normal steps.
* **Heston**: full-truncation Euler (Lord, Koekkoek & van Dijk, 2010) with 8
  sub-steps per day; the realised integrated variance is accumulated on the
  sub-grid. Monte Carlo prices match the Fourier price within sampling error,
  and both the spot and the variance swap are martingales in the simulation
  (`tests/test_hedging.py`).

## 7. Metrics

All metrics are computed on test paths never seen in training. The P&L
statistics add back the model price (Black-Scholes or Heston) as premium.

| Metric | Definition |
|---|---|
| Mean / Std P&L | moments of the terminal P&L |
| CVaR 95% | mean loss in the worst 5% of paths |
| Avg. costs | mean transaction costs paid |
| Turnover | mean number of shares traded over the life of the option, per option |
| Indifference price | $\rho_\lambda$ of the P&L without premium: the premium the hedger must charge |

## References

* Albrecher, Mayer, Schoutens & Tistaert (2007). *The little Heston trap*. Wilmott Magazine.
* Buehler, Gonon, Teichmann & Wood (2019). *Deep hedging*. Quantitative Finance 19(8).
* Davis & Norman (1990). *Portfolio selection with transaction costs*. Mathematics of Operations Research 15(4).
* Heston (1993). *A closed-form solution for options with stochastic volatility*. Review of Financial Studies 6(2).
* Imaki, Imajo, Ito, Minami & Nakagawa (2021). *No-transaction band network: a neural network architecture for efficient deep hedging*. arXiv:2103.01775.
* Kingma & Ba (2015). *Adam: a method for stochastic optimization*. ICLR.
* Leland (1985). *Option pricing and replication with transactions costs*. Journal of Finance 40(5).
* Lord, Koekkoek & van Dijk (2010). *A comparison of biased simulation schemes for stochastic volatility models*. Quantitative Finance 10(2).
* Rockafellar & Uryasev (2000). *Optimization of conditional value-at-risk*. Journal of Risk 2.
* Whalley & Wilmott (1997). *An asymptotic analysis of an optimal hedging model for option pricing with transaction costs*. Mathematical Finance 7(3).
