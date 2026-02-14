from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class FlipClusterWorld:
    """Deterministic world with a single regime flip at flip_time."""

    flip_time: int = 250
    shock_start: int = 250
    shock_end: int = 260

    def __post_init__(self) -> None:
        self._r_pre = np.array([0.05, 0.05, 0.0, -0.03, -0.03], dtype=float)
        self._r_post = np.array([-0.03, -0.03, 0.0, 0.05, 0.05], dtype=float)
        self._r_shock = np.array([-2.0, -2.0, 0.0, -2.0, -2.0], dtype=float)

    def step(self, t: int) -> Dict[str, Any]:
        t = int(t)
        if self.shock_start <= t < self.shock_end:
            r_vec = self._r_shock
        else:
            r_vec = self._r_pre if t < int(self.flip_time) else self._r_post
        return {"r": r_vec.copy(), "c": 0.0}
