# Deep hedging avec coûts de transaction et volatilité stochastique

![Python](https://img.shields.io/badge/python-3.9%2B-2a78d6)
![Dépendances](https://img.shields.io/badge/d%C3%A9pendances-numpy%20%7C%20scipy%20%7C%20matplotlib-52514e)
![Licence](https://img.shields.io/badge/licence-MIT-1baf7a)

*[English version](README.md)*

Un réseau de neurones apprend à couvrir une position courte sur une option en
minimisant le risque de son P&L couvert sur des trajectoires simulées, sans
formule de pricing, sans grecques et sans supposer un trading continu. Le tout
est codé de zéro en **NumPy**, rétropropagation à travers le temps comprise, et
comparé aux solutions classiques de la littérature.

![Bande de non-transaction apprise](results/figures/tc_no_trade_band.png)

## Résultats clés

Call à la monnaie vendu, maturité 30 jours, $S_0 = 100$, rééquilibrage
quotidien, 131 072 trajectoires de test indépendantes.

* **Coûts de transaction : une couverture moins chère que toutes les stratégies
  classiques.** Avec 10 bp de coûts proportionnels, la prime supplémentaire à
  demander en plus du prix Black-Scholes passe de **0,350** (delta BS) et
  **0,323** (bande de Whalley-Wilmott) à **0,288**, soit **−18 % par rapport au
  delta BS et −11 % par rapport à Whalley-Wilmott**. Les frais baissent de 30 %
  *et* la CVaR 95 % s'améliore de 8 %. Le gain augmente avec les coûts : −11 %,
  −18 % et −26 % par rapport au delta BS à 5, 10 et 20 bp.
* **Sans coûts : le réseau retrouve Black-Scholes.** Entraîné à partir de rien,
  sa couverture est en moyenne à **0,008 action** du delta BS et atteint le même
  prix d'indifférence (2,355 contre 2,354).
* **Volatilité stochastique (Heston) : il trouve la bonne couverture, pas celle
  du manuel.** Avec le sous-jacent comme seul instrument, le delta du modèle
  couvre mal (2,440) ; le réseau apprend le delta *à variance minimale*, corrigé
  de la corrélation (2,378 contre 2,376), sans qu'on le lui indique. Avec un
  variance swap, il égale la couverture parfaite delta-vega du modèle (2,309
  contre 2,307).

*Les prix sont des prix d'indifférence en utilité exponentielle (risque
entropique, aversion au risque égale à 1) : la prime qui rend le vendeur
indifférent entre vendre l'option et ne rien faire. Plus c'est bas, mieux c'est.
Détails dans la [méthodologie](docs/methodology.md) (en anglais).*

## Pourquoi ce problème

Black-Scholes dit de détenir Δ actions et de rééquilibrer en continu,
gratuitement. Un vrai desk rééquilibre de façon discrète, paie le spread
bid-ask à chaque trade et fait face à une volatilité qui bouge. Chacun de ces
points casse l'argument de réplication, et les corrections classiques (la
volatilité ajustée de Leland, les bandes de Whalley-Wilmott, les deltas de
modèle) sont asymptotiques ou propres à un modèle. Le deep hedging (Buehler,
Gonon, Teichmann et Wood, 2019, développé avec J.P. Morgan) transforme la
question en un problème d'optimisation qu'on peut résoudre pour n'importe quel
simulateur de marché, n'importe quel ensemble d'instruments et n'importe quelles
frictions.

## Contenu du dépôt

| | |
|---|---|
| **Deep hedger** | Politique MLP récurrente (Buehler et al.) et No-Transaction Band Network (Imaki et al., 2021) |
| **Entraînement** | Objectif risque entropique ou CVaR, Adam, rétropropagation à travers le temps codée à la main, gradients vérifiés |
| **Marchés** | Black-Scholes ; Heston (schéma d'Euler full truncation) avec variance swap négociable |
| **Références** | Delta BS, delta de Leland, bande de Whalley-Wilmott, delta Heston, delta à variance minimale, delta-vega |
| **Pricing** | Prix, delta et vega Heston par inversion de Fourier (Gil-Pelaez, « little Heston trap »), précision 1e-5 |
| **Qualité** | 21 tests unitaires (vérification des gradients, pricing contre quadrature, tests de martingale et de convergence), intégration continue GitHub Actions |

## Résultats

### 1. Test de validité : retrouver Black-Scholes

Sans coûts, la couverture optimale est (presque) le delta BS. Le réseau ne voit
jamais la formule.

![Couverture apprise et delta BS](results/figures/gbm_frictionless_hedge.png)

| Stratégie | P&L moyen | Écart-type | CVaR 95 % | Volume traité | Prix d'indifférence |
|---|---:|---:|---:|---:|---:|
| Pas de couverture | −0,019 | 3,478 | 10,172 | 0,00 | 17,103 |
| Delta Black-Scholes | 0,000 | 0,357 | 0,828 | 2,71 | **2,354** |
| Deep hedger (MLP récurrent) | 0,000 | 0,366 | 0,801 | 2,70 | 2,355 |

La dispersion restante du P&L vient de l'erreur de discrétisation d'une
couverture quotidienne, qu'aucune stratégie ne peut supprimer.

### 2. Coûts de transaction : apprendre quand ne pas trader

Avec des coûts proportionnels, la politique optimale est une **bande de
non-transaction** : ne rien faire tant que la couverture est « assez proche »,
sinon trader jusqu'au bord de la bande. Le réseau à bande apprend directement
sa forme.

| Coût | Delta BS | Leland | Whalley-Wilmott | **Deep hedger** | vs delta BS | vs meilleure classique |
|---:|---:|---:|---:|---:|---:|---:|
| 5 bp | 0,207 | 0,204 | 0,213 | **0,185** | −11 % | −9 % |
| 10 bp | 0,350 | 0,339 | 0,323 | **0,288** | −18 % | −11 % |
| 20 bp | 0,640 | 0,604 | 0,528 | **0,473** | −26 % | −10 % |

*Prime supplémentaire nécessaire pour couvrir, au-delà du prix BS (prix d'indifférence − 2,287).*

![Coût de couverture selon les coûts de transaction](results/figures/tc_price_vs_cost.png)

Ce que le réseau a appris, comparé à la théorie :

* **Une bande autour du delta, comme le prédit la théorie** (graphique du haut).
  Elle est plus étroite que celle de Whalley-Wilmott. Explication probable :
  cette formule est un résultat asymptotique pour une surveillance continue et
  des coûts infinitésimaux, alors qu'ici la couverture n'est ajustée qu'une fois
  par jour, donc s'éloigner du delta est plus risqué.
* **Moins de trades, moins de frais, des queues plus fines.** À 10 bp, il trade
  30 % de moins que le delta BS, paie 30 % de frais en moins et a pourtant une
  meilleure CVaR (1,097 contre 1,188). Whalley-Wilmott réduit encore plus les
  frais, mais le paie en risque (écart-type 0,56 contre 0,44).

![Positions le long d'une trajectoire](results/figures/tc_sample_path.png)

Métriques complètes à 10 bp (tous les niveaux de coûts dans [`results/tables`](results/tables/gbm_transaction_costs.md)) :

| Stratégie | P&L moyen | Écart-type | CVaR 95 % | Frais moyens | Volume traité | Prix d'indifférence |
|---|---:|---:|---:|---:|---:|---:|
| Delta Black-Scholes | −0,273 | 0,376 | 1,188 | 0,274 | 2,71 | 2,637 |
| Delta de Leland | −0,268 | 0,372 | 1,113 | 0,268 | 2,66 | 2,626 |
| Bande de Whalley-Wilmott | −0,165 | 0,562 | 1,334 | 0,166 | 1,63 | 2,610 |
| Deep hedger (réseau à bande) | −0,192 | 0,441 | 1,097 | 0,192 | 1,90 | **2,575** |
| Deep hedger (MLP récurrent) | −0,210 | 0,433 | 1,120 | 0,210 | 2,08 | 2,591 |

**L'architecture compte.** Le réseau à bande est à moins de 0,001 de son risque
final après 300 itérations (moins d'une minute sur un processeur à 2 cœurs) ; le
MLP récurrent générique est entraîné sur 3 000 itérations et finit pourtant plus
haut (2,591 contre 2,575). Intégrer au réseau la structure connue de la solution
fonctionne mieux que lui demander de la redécouvrir.

### 3. Volatilité stochastique : Heston

Paramètres : $\kappa = 1$, $\theta = 0.04$, $\xi = 0.5$, $\rho = -0.7$,
$v_0 = 0.04$. Prix Heston du call : 2,237. Évaluation sur 32 768 trajectoires
de test (les stratégies de modèle demandent un pricing de Fourier à chaque nœud).

| Stratégie | Instruments | Écart-type | CVaR 95 % | Prix d'indifférence |
|---|---|---:|---:|---:|
| Delta Black-Scholes (σ = 20 %) | sous-jacent | 0,534 | 1,329 | 2,415 |
| Delta Heston | sous-jacent | 0,597 | 1,356 | 2,440 |
| Delta à variance minimale | sous-jacent | 0,492 | 1,171 | 2,376 |
| Deep hedger | sous-jacent | 0,502 | 1,159 | 2,378 |
| Delta-vega | sous-jacent + variance swap | 0,355 | 0,842 | **2,307** |
| Deep hedger | sous-jacent + variance swap | 0,374 | 0,815 | 2,309 |

* **Sous-jacent seul (marché incomplet).** Avec $\rho = -0.7$, le sous-jacent
  baisse quand la volatilité monte : une partie du risque de vega peut donc être
  couverte avec le sous-jacent. Le delta du modèle l'ignore et fait moins bien
  qu'un simple delta BS. Le delta à variance minimale $C_S + \rho\xi C_v / S$
  corrige cet effet, et le réseau apprend cette correction à partir des seules
  données.
* **Sous-jacent + variance swap (marché complet).** Le réseau apprend à détenir
  le variance swap et égale la stratégie de réplication du modèle, avec une
  queue légèrement meilleure.

![Couverture Heston apprise](results/figures/heston_hedge.png)

![Distributions du P&L sous Heston](results/figures/heston_pnl.png)

La couverture apprise suit de près le delta à variance minimale là où les
trajectoires sont nombreuses ; elle s'en écarte dans les zones rarement
visitées (volatilité élevée, option très dans la monnaie), où les données
d'entraînement sont rares.

## Fonctionnement

1. **Simuler** 131 072 trajectoires du marché (Black-Scholes ou Heston).
2. **Couvrir** chaque trajectoire avec le réseau : à chaque date, il voit la
   log-moneyness, le temps restant (et la volatilité sous Heston) et donne la
   position.
3. **Noter** le P&L final, frais compris, avec une mesure de risque convexe
   (risque entropique ou CVaR). Comme les deux sont invariantes par ajout de
   cash, la valeur optimale est le prix d'indifférence.
4. **Entraîner** par descente de gradient sur ce risque, en propageant les
   gradients à travers le P&L, les frais et le temps.

Le réseau à bande produit une bande $[\Delta - a,\ \Delta + b]$ autour du delta
BS et garde la position précédente, ramenée dans la bande :
$\delta_t = \mathrm{clip}(\delta_{t-1}, \Delta_t - a_t, \Delta_t + b_t)$.

Toutes les démonstrations (P&L, mesures de risque, gradients à travers le temps
et à travers le clip, Leland, Whalley-Wilmott, delta à variance minimale,
pricing de Fourier) sont dans [**docs/methodology.md**](docs/methodology.md).

## Démarrage rapide

Depuis la racine du dépôt :

```bash
pip install -e .                              # numpy, scipy, matplotlib
python -m unittest discover -s tests          # 21 tests, ~5 secondes
python experiments/run_all.py --quick         # test rapide de toutes les expériences, ~1 minute
python experiments/run_all.py                 # reproduit tous les résultats, ~40 minutes sur 2 cœurs
```

Chaque expérience peut aussi être lancée seule (`experiments/gbm_frictionless.py`,
`gbm_transaction_costs.py`, `heston_stochastic_vol.py`). L'option `--quick`
écrit dans `results/quick/` et n'écrase jamais les résultats publiés.

Dans ton propre code :

```python
from deephedging import GBM, DeepHedger, EntropicRisk, terminal_pnl
from deephedging.problems import gbm_call_env

T, K, sigma, cost = 30 / 365, 100.0, 0.2, 0.001          # call ATM 30 jours, coûts 10 bp
market = GBM(sigma=sigma, S0=100.0)
train = gbm_call_env(market.simulate(2**16, T, 30, seed=1), K, sigma, cost)
test = gbm_call_env(market.simulate(2**16, T, 30, seed=2), K, sigma, cost)

hedger = DeepHedger(train.n_features, n_instruments=1, policy="ntb",
                    risk=EntropicRisk(risk_aversion=1.0))
hedger.fit(train, n_iters=300)                            # moins d'une minute

pnl, costs = terminal_pnl(hedger.positions(test), test)
print(f"indifference price: {hedger.risk(pnl):.3f}   average costs: {costs.mean():.3f}")
```

## Structure du dépôt

```
src/deephedging/
  deep_hedger.py    politiques MLP récurrente et à bande, boucle d'entraînement
  nn.py             MLP avec rétropropagation codée à la main, Adam
  hedging.py        environnement de couverture, P&L avec frais et son gradient
  risk.py           risque entropique, CVaR, métriques d'évaluation
  market.py         simulateurs Black-Scholes et Heston, variance swap
  pricing.py        grecques BS, Leland, Whalley-Wilmott, pricing Heston par Fourier
  strategies.py     stratégies de couverture classiques
  problems.py       transforme les trajectoires simulées en environnements de couverture
  plotting.py       style des graphiques
experiments/        les trois expériences et run_all.py
tests/              tests unitaires, dont la vérification des gradients par différences finies
results/            graphiques, tableaux (Markdown), métriques (JSON), réseaux entraînés
docs/methodology.md les maths derrière chaque composant (en anglais)
```

## Notes d'implémentation

* **Aucune librairie de deep learning.** Le réseau, la rétropropagation à
  travers le temps et l'optimiseur sont écrits en NumPy et vérifiés par
  différences finies à 1e-5 près en erreur relative. L'entraînement tourne en
  float32, les tests en float64.
* **Comparaison équitable.** Toutes les stratégies sont évaluées sur les mêmes
  trajectoires de test, avec le même modèle de frais (y compris le débouclage de
  la couverture à maturité) et la même mesure de risque. La bande de
  Whalley-Wilmott utilise la même aversion au risque que l'objectif du réseau.
* **Reproductible.** Graines fixées, jeux d'entraînement et de test
  indépendants, et chaque chiffre de ce README est produit par
  `experiments/run_all.py` (valeurs brutes dans `results/*.json`).

## Limites et pistes

* Le marché est simulé : les résultats ne valent que ce que vaut le simulateur.
  Extensions naturelles : volatilité rough ou locale-stochastique, sauts, ou un
  modèle génératif calibré sur des données historiques.
* Une seule option à la fois. Un portefeuille d'options, ou des payoffs
  path-dependent (barrières, asiatiques), ne change que le payoff et les
  variables d'entrée.
* Les coûts sont proportionnels. Des coûts fixes, de l'impact de marché ou un
  spread bid-ask sur le variance swap s'intègrent dans le même cadre.

## Références

* Buehler, Gonon, Teichmann & Wood (2019). Deep hedging. *Quantitative Finance*.
* Imaki, Imajo, Ito, Minami & Nakagawa (2021). No-transaction band network. arXiv:2103.01775.
* Whalley & Wilmott (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs. *Mathematical Finance*.
* Leland (1985). Option pricing and replication with transactions costs. *Journal of Finance*.
* Heston (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*.

Liste complète dans [docs/methodology.md](docs/methodology.md#references).

## Auteur

**Antoine Streichenberger**, MSc Financial Engineering, ESILV.
Sous licence MIT.
