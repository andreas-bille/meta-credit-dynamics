from __future__ import annotations

from typing import Protocol, Dict, Any
import numpy as np


class World(Protocol):
    """World provides exogenous signals (Profile A).

    Contract: step(t) -> { r: array[float], c: float }
    """

    def step(self, t: int) -> Dict[str, Any]:
        ...


class Curriculum(Protocol):
    """Curriculum provides a sequence of worlds (state-agnostic)."""

    def next(self, t: int) -> World:
        ...


class Teacher(Protocol):
    """Teacher provides configuration authority, not state intervention."""

    def configure(self, run_id: str, profile: str, mode: str, params: Dict[str, Any]) -> None:
        ...


def validate_world_output(out: Dict[str, Any]) -> tuple[np.ndarray, float]:
    """Validate and normalize World output.

    Returns (r_vec, c_total). Raises ValueError on invalid shape/types.
    """
    if "r" not in out or "c" not in out:
        raise ValueError("World.step must return keys: r, c")
    r = np.asarray(out["r"], dtype=float)
    c = float(out["c"])
    if r.ndim != 1:
        raise ValueError("World r must be a 1D array")
    return r, c
