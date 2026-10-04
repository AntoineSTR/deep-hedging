"""Deep hedging (Buehler, Gonon, Teichmann & Wood, 2019) in NumPy.

Two policy architectures are available:

``policy="mlp"`` — the original recurrent deep hedger
    delta_i = F_theta(features_i, delta_{i-1}).
    Feeding back the previous position lets the network learn not to trade
    when trading is not worth its cost. Gradients are propagated backwards
    through time across this recursion.

``policy="ntb"`` — No-Transaction Band Network (Imaki, Imajo, Ito, Minami &
    Nakagawa, 2021). The network outputs a band around a reference hedge
    (here the Black-Scholes delta) and the position is the previous one
    clipped into the band:
        lower_i = base_i - softplus(a_i),  upper_i = base_i + softplus(b_i),
        delta_i = clip(delta_{i-1}, lower_i, upper_i).
    The band does not depend on the position, so the network is evaluated
    for all dates in a single batched pass; this trains much faster and
    encodes the known structure of the optimal policy under proportional costs.

Training minimises a convex risk measure of the terminal hedged P&L
(entropic risk by default) with Adam on mini-batches of simulated paths.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .hedging import HedgingEnv, pnl_gradient, terminal_pnl
from .nn import MLP, Adam
from .risk import CVaR, EntropicRisk


def _softplus(x):
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class DeepHedger:
    """Neural hedging policy trained by risk minimisation."""

    def __init__(self, n_features: int, n_instruments: int, policy: str = "mlp",
                 hidden: Sequence[int] = (32, 32), risk=None, seed: int = 0,
                 dtype=np.float32):
        if policy not in ("mlp", "ntb"):
            raise ValueError("policy must be 'mlp' or 'ntb'")
        self.n_features = n_features
        self.n_instruments = n_instruments
        self.policy = policy
        self.hidden = tuple(hidden)
        self.risk = risk if risk is not None else EntropicRisk(1.0)
        self.seed = seed
        rng = np.random.default_rng(seed)
        if policy == "mlp":
            sizes = (n_features + n_instruments, *self.hidden, n_instruments)
        else:
            sizes = (n_features, *self.hidden, 2 * n_instruments)
        self.dtype = np.dtype(dtype)
        self.net = MLP(sizes, rng, dtype=self.dtype)
        if policy == "ntb":
            # Start from a narrow band (softplus(-3) ~ 0.05) so that the
            # initial policy is close to the reference hedge.
            self.net.biases[-1][:] = -3.0
        self.aux = np.zeros(1)          # auxiliary risk variable (VaR level for CVaR)
        self._aux_initialised = False
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ forward
    def _forward(self, env: HedgingEnv, keep_cache: bool):
        if self.policy == "mlp":
            return self._forward_mlp(env, keep_cache)
        return self._forward_ntb(env, keep_cache)

    def _forward_mlp(self, env, keep_cache):
        B, N, n = env.n_paths, env.n_steps, env.n_instruments
        positions = np.empty((B, N, n))
        prev = np.zeros((B, n))
        caches = []
        for i in range(N):
            x = np.concatenate([env.features[:, i, :], prev], axis=1)
            y, cache = self.net.forward(x)
            positions[:, i, :] = y
            prev = y
            if keep_cache:
                caches.append(cache)
        return positions, caches

    def _forward_ntb(self, env, keep_cache):
        if env.base is None:
            raise ValueError("the 'ntb' policy needs env.base (reference hedge)")
        B, N, n = env.n_paths, env.n_steps, env.n_instruments
        out, cache = self.net.forward(env.features.reshape(B * N, -1))
        out = out.reshape(B, N, 2 * n)
        a_low, a_up = out[..., :n], out[..., n:]
        lower = env.base - _softplus(a_low)
        upper = env.base + _softplus(a_up)
        positions = np.empty((B, N, n))
        below = np.empty((B, N, n), dtype=bool)
        above = np.empty((B, N, n), dtype=bool)
        prev = np.zeros((B, n))
        for i in range(N):
            below[:, i] = prev < lower[:, i]
            above[:, i] = prev > upper[:, i]
            prev = np.clip(prev, lower[:, i], upper[:, i])
            positions[:, i] = prev
        ctx = (cache, a_low, a_up, below, above) if keep_cache else None
        return positions, ctx

    def positions(self, env: HedgingEnv, chunk_size: int = 32768) -> np.ndarray:
        """Hedge positions (B, N, n) of the trained policy."""
        out = []
        for start in range(0, env.n_paths, chunk_size):
            sub = env.subset(slice(start, start + chunk_size))
            out.append(self._forward(sub, keep_cache=False)[0])
        return np.concatenate(out, axis=0)

    def act(self, features: np.ndarray, prev: np.ndarray, base: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate the policy at a single date.

        features : (B, d), prev : (B, n) previous holdings, base : (B, n) reference
        hedge (``ntb`` only). Returns the new holdings (B, n).
        """
        if self.policy == "mlp":
            return self.net.forward(np.concatenate([features, prev], axis=1))[0].astype(float)
        out, _ = self.net.forward(features)
        n = self.n_instruments
        lower = base - _softplus(out[:, :n])
        upper = base + _softplus(out[:, n:])
        return np.clip(prev, lower, upper)

    def band(self, features: np.ndarray, base: np.ndarray):
        """Edges of the learned no-transaction band (``ntb`` policy only).

        features : (..., d), base : (..., n). Returns (lower, upper), each (..., n).
        """
        if self.policy != "ntb":
            raise ValueError("band() is only defined for the 'ntb' policy")
        n = self.n_instruments
        out, _ = self.net.forward(features.reshape(-1, features.shape[-1]))
        out = out.astype(float).reshape(*features.shape[:-1], 2 * n)
        return base - _softplus(out[..., :n]), base + _softplus(out[..., n:])

    # ----------------------------------------------------------------- backward
    def loss_and_grads(self, env: HedgingEnv):
        """Risk of the terminal P&L on ``env`` and its gradient w.r.t. all parameters.

        Returns (loss, grads) where grads matches ``self.parameters``.
        """
        positions, ctx = self._forward(env, keep_cache=True)
        pnl, _ = terminal_pnl(positions, env)
        loss, d_pnl, d_aux = self.risk.value_and_grad(pnl, self.aux)
        d_pos = pnl_gradient(positions, env) * d_pnl[:, None, None]

        if self.policy == "mlp":
            grads = [np.zeros_like(p) for p in self.net.params]
            carry = np.zeros((env.n_paths, env.n_instruments))
            d_in = env.n_features
            for i in range(env.n_steps - 1, -1, -1):
                g_i, dx = self.net.backward(ctx[i], d_pos[:, i, :] + carry)
                for acc, g in zip(grads, g_i):
                    acc += g
                carry = dx[:, d_in:]
        else:
            cache, a_low, a_up, below, above = ctx
            inside = ~(below | above)
            d_lower = np.zeros_like(d_pos)
            d_upper = np.zeros_like(d_pos)
            carry = np.zeros((env.n_paths, env.n_instruments))
            for i in range(env.n_steps - 1, -1, -1):
                d = d_pos[:, i, :] + carry
                d_lower[:, i] = d * below[:, i]
                d_upper[:, i] = d * above[:, i]
                carry = d * inside[:, i]
            d_out = np.concatenate([-d_lower * _sigmoid(a_low), d_upper * _sigmoid(a_up)], axis=-1)
            grads, _ = self.net.backward(cache, d_out.reshape(env.n_paths * env.n_steps, -1))
        return loss, grads + [d_aux]

    @property
    def parameters(self) -> List[np.ndarray]:
        return self.net.params + [self.aux]

    # ----------------------------------------------------------------- training
    def fit(self, env: HedgingEnv, n_iters: int = 2000, batch_size: int = 8192,
            lr: float = 3e-3, lr_end: float = 1e-4, clip_norm: float = 10.0,
            seed: int = 0, log_every: int = 100, verbose: bool = True,
            valid_env: Optional[HedgingEnv] = None) -> "DeepHedger":
        """Train with Adam and an exponentially decaying learning rate."""
        rng = np.random.default_rng(seed)
        batch_size = min(batch_size, env.n_paths)
        if not self._aux_initialised:
            sub = env.subset(rng.choice(env.n_paths, batch_size, replace=False))
            pnl, _ = terminal_pnl(self._forward(sub, keep_cache=False)[0], sub)
            self.aux[:] = self.risk.init_aux(pnl)
            self._aux_initialised = True
        opt = Adam(self.parameters, lr=lr)
        start = time.time()
        for it in range(1, n_iters + 1):
            lr_t = lr * (lr_end / lr) ** ((it - 1) / max(n_iters - 1, 1))
            idx = rng.choice(env.n_paths, batch_size, replace=False)
            loss, grads = self.loss_and_grads(env.subset(idx))
            norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
            if norm > clip_norm:
                grads = [g * (clip_norm / norm) for g in grads]
            opt.step(grads, lr_t)
            if it % log_every == 0 or it == n_iters:
                record = {"iter": it, "train_loss": loss, "lr": lr_t, "seconds": time.time() - start}
                if valid_env is not None:
                    record["valid_loss"] = self.evaluate_risk(valid_env)
                self.history.append(record)
                if verbose:
                    msg = f"  iter {it:5d}  loss {loss: .5f}"
                    if valid_env is not None:
                        msg += f"  valid {record['valid_loss']: .5f}"
                    print(msg + f"  ({record['seconds']:.0f}s)", flush=True)
        return self

    def evaluate_risk(self, env: HedgingEnv) -> float:
        pnl, _ = terminal_pnl(self.positions(env), env)
        return self.risk(pnl)

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        meta = {
            "n_features": self.n_features, "n_instruments": self.n_instruments,
            "policy": self.policy, "hidden": list(self.hidden), "seed": self.seed,
            "dtype": self.dtype.name,
            "risk": {"name": self.risk.name,
                     "risk_aversion": getattr(self.risk, "risk_aversion", None),
                     "alpha": getattr(self.risk, "alpha", None)},
        }
        np.savez(path, meta=json.dumps(meta), aux=self.aux, **self.net.state())

    @classmethod
    def load(cls, path: str) -> "DeepHedger":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        r = meta["risk"]
        risk = EntropicRisk(r["risk_aversion"]) if r["name"] == "entropic" else CVaR(r["alpha"])
        model = cls(meta["n_features"], meta["n_instruments"], meta["policy"],
                    meta["hidden"], risk, meta["seed"], meta.get("dtype", "float32"))
        model.net.load_state(data)
        model.aux[:] = data["aux"]
        model._aux_initialised = True
        return model
