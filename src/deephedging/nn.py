"""A minimal fully-connected network with hand-written backpropagation, and Adam.

Written in plain NumPy on purpose: the whole training loop of the deep
hedger, including backpropagation through time, is explicit and verified by
finite-difference gradient checks (see ``tests/test_gradients.py``).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class MLP:
    """Multi-layer perceptron with tanh hidden layers and a linear output.

    ``forward`` returns the output and a cache; ``backward`` takes the cache and
    dL/d(output) and returns (parameter gradients, dL/d(input)).
    """

    def __init__(self, sizes: Sequence[int], rng: np.random.Generator, out_scale: float = 0.1,
                 dtype=np.float32):
        self.sizes = tuple(int(s) for s in sizes)
        self.dtype = np.dtype(dtype)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        n_layers = len(self.sizes) - 1
        for k, (fan_in, fan_out) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            scale = np.sqrt(1.0 / fan_in) * (out_scale if k == n_layers - 1 else 1.0)
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(self.dtype))
            self.biases.append(np.zeros(fan_out, dtype=self.dtype))

    @property
    def params(self) -> List[np.ndarray]:
        out = []
        for W, b in zip(self.weights, self.biases):
            out += [W, b]
        return out

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        x = np.asarray(x, dtype=self.dtype)
        cache = [x]
        h = x
        last = len(self.weights) - 1
        for k, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            if k < last:
                h = np.tanh(z)
                cache.append(h)
            else:
                h = z
        return h, cache

    def backward(self, cache: List[np.ndarray], dy: np.ndarray):
        grads: List[np.ndarray] = [None] * (2 * len(self.weights))
        g = np.asarray(dy, dtype=self.dtype)
        for k in range(len(self.weights) - 1, -1, -1):
            h_in = cache[k]
            grads[2 * k] = h_in.T @ g
            grads[2 * k + 1] = g.sum(axis=0)
            g = g @ self.weights[k].T
            if k > 0:
                h = cache[k]
                g *= 1.0 - h * h
        return grads, g

    def state(self) -> dict:
        return {f"p{i}": p for i, p in enumerate(self.params)}

    def load_state(self, state: dict) -> None:
        params = self.params
        for i, p in enumerate(params):
            p[...] = state[f"p{i}"]


class Adam:
    """Adam optimiser (Kingma & Ba, 2015) acting in place on a list of arrays."""

    def __init__(self, params: List[np.ndarray], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray], lr: float = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
