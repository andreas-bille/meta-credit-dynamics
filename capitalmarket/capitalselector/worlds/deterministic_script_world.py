from __future__ import annotations

from typing import Dict, Any
import numpy as np


class DeterministicScriptWorld:
    """Deterministic world with fixed scripted returns (Profile A)."""

    def __init__(self, *, r: list[float] | None = None, c: float = 0.0):
        self._r = np.asarray(r if r is not None else [0.01, 0.0, -0.01], dtype=float)
        self._c = float(c)

    def step(self, t: int) -> Dict[str, Any]:
        _ = t  # unused; world is time-invariant
        return {"r": self._r, "c": self._c}
