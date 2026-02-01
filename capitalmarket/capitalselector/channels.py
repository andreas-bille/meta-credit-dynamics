
from __future__ import annotations

from typing import Tuple
import numpy as np

from .core import Channel
from .broker import PhaseCChannel


class DummyChannel(Channel):
    """
    Semantikfreier Kanal ohne eigene Dynamik.
    Gibt immer (r=0.0, c=0.0) zurück – nützlich, um nur die
    Simplex-Dimension K des Selectors festzulegen, ohne echte Sub-Channels.
    """

    def step(self, weight: float) -> Tuple[float, float]:
        # Kein Rückfluss, keine Kosten – Gewicht wird ignoriert
        return 0.0, 0.0


# -----------------------------
# Phase-C synthetic channels
# -----------------------------

class GaussianExplorer(PhaseCChannel):
    """Synthetic explorer producing Gaussian returns with optional delay.

    Parameters
    ----------
    mu : float
        Mean return per step.
    sigma : float
        Std dev of return per step.
    cost : float
        Constant cost per step.
    alive_prob : float
        Probability of staying alive per step.
    dt : float
        Constant dt returned each step.
    """

    def __init__(self, *, mu: float, sigma: float, cost: float = 0.0, alive_prob: float = 1.0, dt: float = 1.0, seed: int | None = None):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.cost = float(cost)
        self.alive_prob = float(alive_prob)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.alive = True

    def step(self, weight: float):
        if not self.alive:
            return 0.0, 0.0, False, self.dt
        r = float(self.rng.normal(self.mu, self.sigma)) * float(weight)
        c = float(self.cost) * float(weight)
        if self.rng.random() > self.alive_prob:
            self.alive = False
        return r, c, self.alive, self.dt


class TailRiskExplorer(PhaseCChannel):
    """Explorer with occasional large negative tail events.

    With probability p_tail, return is sampled from N(mu_tail, sigma_tail),
    otherwise from N(mu, sigma).
    """

    def __init__(
        self,
        *,
        mu: float,
        sigma: float,
        p_tail: float = 0.05,
        mu_tail: float = -5.0,
        sigma_tail: float = 1.0,
        cost: float = 0.0,
        alive_prob: float = 1.0,
        dt: float = 1.0,
        seed: int | None = None,
    ):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.p_tail = float(p_tail)
        self.mu_tail = float(mu_tail)
        self.sigma_tail = float(sigma_tail)
        self.cost = float(cost)
        self.alive_prob = float(alive_prob)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.alive = True

    def step(self, weight: float):
        if not self.alive:
            return 0.0, 0.0, False, self.dt
        if self.rng.random() < self.p_tail:
            base = float(self.rng.normal(self.mu_tail, self.sigma_tail))
        else:
            base = float(self.rng.normal(self.mu, self.sigma))
        r = base * float(weight)
        c = float(self.cost) * float(weight)
        if self.rng.random() > self.alive_prob:
            self.alive = False
        return r, c, self.alive, self.dt


class DeterministicExplorer(PhaseCChannel):
    """Deterministic explorer: constant return and cost."""
    def __init__(self, *, r: float, c: float = 0.0, dt: float = 1.0):
        self.r = float(r)
        self.c = float(c)
        self.dt = float(dt)
        self.alive = True

    def step(self, weight: float):
        if not self.alive:
            return 0.0, 0.0, False, self.dt
        return self.r * float(weight), self.c * float(weight), True, self.dt
