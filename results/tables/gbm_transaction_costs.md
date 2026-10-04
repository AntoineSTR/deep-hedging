### Proportional cost 5 bp

| Strategy | Mean P&L | Std P&L | CVaR 95% | Avg. costs | Turnover | Indifference price |
|---|---:|---:|---:|---:|---:|---:|
| Black-Scholes delta | -0.137 | 0.364 | 1.006 | 0.137 | 2.71 | 2.495 |
| Leland delta | -0.135 | 0.363 | 0.970 | 0.135 | 2.68 | 2.491 |
| Whalley-Wilmott band | -0.087 | 0.501 | 1.146 | 0.088 | 1.73 | 2.500 |
| Deep hedger (band network) | -0.107 | 0.397 | 0.940 | 0.107 | 2.12 | **2.472** |

### Proportional cost 10 bp

| Strategy | Mean P&L | Std P&L | CVaR 95% | Avg. costs | Turnover | Indifference price |
|---|---:|---:|---:|---:|---:|---:|
| Black-Scholes delta | -0.273 | 0.376 | 1.188 | 0.274 | 2.71 | 2.637 |
| Leland delta | -0.268 | 0.372 | 1.113 | 0.268 | 2.66 | 2.626 |
| Whalley-Wilmott band | -0.165 | 0.562 | 1.334 | 0.166 | 1.63 | 2.610 |
| Deep hedger (band network) | -0.192 | 0.441 | 1.097 | 0.192 | 1.90 | **2.575** |
| Deep hedger (recurrent MLP) | -0.210 | 0.433 | 1.120 | 0.210 | 2.08 | 2.591 |

### Proportional cost 20 bp

| Strategy | Mean P&L | Std P&L | CVaR 95% | Avg. costs | Turnover | Indifference price |
|---|---:|---:|---:|---:|---:|---:|
| Black-Scholes delta | -0.547 | 0.409 | 1.560 | 0.547 | 2.71 | 2.927 |
| Leland delta | -0.525 | 0.392 | 1.399 | 0.525 | 2.60 | 2.892 |
| Whalley-Wilmott band | -0.313 | 0.657 | 1.648 | 0.313 | 1.54 | 2.815 |
| Deep hedger (band network) | -0.343 | 0.516 | 1.379 | 0.344 | 1.70 | **2.760** |
