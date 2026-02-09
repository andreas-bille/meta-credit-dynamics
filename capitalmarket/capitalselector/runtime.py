from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from .config import ProfileAConfig
from .interfaces import World, Curriculum, Teacher, validate_world_output
from .builder import CapitalSelectorBuilder


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str = "A"
    freeze: bool = False
    mode: str = "A"


def run(
    *,
    world: World,
    steps: int,
    config: RuntimeConfig | None = None,
    profile: ProfileAConfig | None = None,
) -> Dict[str, Any]:
    """Canonical runtime entry point (Profile A).

    This is a minimal runner for deterministic Profile A semantics.
    """
    cfg = config or RuntimeConfig()
    if cfg.profile != "A":
        raise ValueError("Only Profile A is supported in v1")

    prof = profile or ProfileAConfig()

    # initialize selector from Profile A defaults
    selector = CapitalSelectorBuilder().with_K(0).build()

    trace = []
    history = []
    for t in range(int(steps)):
        out = world.step(t)
        r_vec, c_total = validate_world_output(out)
        if selector.w is None or len(selector.w) != len(r_vec):
            selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
            selector.K = len(r_vec)
        selector.feedback_vector(r_vec, c_total, trace=None, freeze=cfg.freeze)
        history.append(selector.state())
        trace.append("step")

    return {"history": history, "trace": trace}
