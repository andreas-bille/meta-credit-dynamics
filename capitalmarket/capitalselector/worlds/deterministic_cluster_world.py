from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class DeterministicClusterWorld:
    """Deterministic world with a persistent clustered return signal."""

    r_vec: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.r_vec is None:
            self.r_vec = np.array([0.02, 0.02, 0.0, -0.01, -0.01], dtype=float)
        else:
            self.r_vec = np.asarray(self.r_vec, dtype=float)

    def step(self, t: int) -> Dict[str, Any]:
        _ = t
        return {"r": self.r_vec.copy(), "c": 0.0}
