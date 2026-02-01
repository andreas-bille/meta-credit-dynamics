from __future__ import annotations

import numpy as np
from .stats import EWMAStats
from .rebirth import RebirthPolicy

from abc import ABC, abstractmethod
from typing import Tuple


class Channel(ABC):
    """
    Minimaler ökonomischer Kanal:
    nimmt Kapitalgewicht und liefert (r, c) zurück.
    """

    @abstractmethod
    def step(self, weight: float) -> Tuple[float, float]:
        pass


class CapitalSelector(Channel):
    """
    Der kanonische, stackbare CapitalSelector.
    """

    def __init__(
        self,
        *,
        wealth: float,
        rebirth_threshold: float,
        stats: EWMAStats,
        reweight_fn,
        kind: str = "entrepreneur",
        rebirth_policy: RebirthPolicy | None = None,
        channels: list[Channel] | None = None,
    ):
        self.wealth = wealth
        self.rebirth_threshold = rebirth_threshold
        self.stats = stats
        self.reweight_fn = reweight_fn
        self.kind = kind
        self.rebirth_policy = rebirth_policy

        self.channels = channels or []
        self.K = len(self.channels)

        self.w = np.ones(self.K) / self.K if self.K > 0 else None

        self._last_r = 0.0
        self._last_c = 0.0

    # ---------- Allocation ----------

    def allocate(self) -> np.ndarray:
        return None if self.w is None else self.w.copy()

    # ---------- Channel Interface ----------

    def step(self, weight: float) -> tuple[float, float]:
        """
        Exportiert diesen Selector als Kanal.
        """
        return weight * self._last_r, weight * self._last_c

    # ---------- Stack Step ----------

    def stack_step(self):
        if not self.channels:
            return

        rs, cs = [], []
        w = self.allocate()

        for wi, ch in zip(w, self.channels):
            r_i, c_i = ch.step(wi)
            rs.append(r_i)
            cs.append(c_i)

        self.feedback(sum(rs), sum(cs))

    # ---------- Feedback ----------


    def feedback(self, r: float, c: float):
        """Scalar feedback for standalone selectors (K==0).

        For stacked selectors with sub-channels, use `feedback_vector(r_vec, c)`.
        """
        r = float(r); c = float(c)
        self._last_r = r
        self._last_c = c
        self.wealth += r - c
        self.stats.update(r)
        if self.wealth < self.rebirth_threshold:
            self.rebirth()

    def feedback_vector(self, r_vec: np.ndarray, c: float):
        r_total = r_vec.sum()

        self._last_r = r_total
        self._last_c = c
        self.wealth += r_total - c

        self.stats.update(r_total)

        adv = r_vec - self.stats.mu   # ⚠️ jetzt Vektor!
        self.w = self.reweight_fn(self.w, adv)

        if self.wealth < self.rebirth_threshold:
            self.rebirth()

    # ---------- Rebirth ----------

    def rebirth(self):
        if self.rebirth_policy:
            self.rebirth_policy.on_rebirth(self)

        self.wealth = max(self.wealth, self.rebirth_threshold)
        if self.w is not None:
            self.w = np.ones(self.K) / self.K

    # ---------- Introspection ----------

    def state(self):
        return {
            "wealth": self.wealth,
            "kind": self.kind,
            "mu": self.stats.mu,
            "var": self.stats.var,
            "weights": None if self.w is None else self.w.copy(),
        }