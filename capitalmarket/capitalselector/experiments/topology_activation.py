from __future__ import annotations

import os
from typing import Dict, Optional

from ..broker import PhaseCChannel, Broker
from ..stack import StackManager, StackChannel


class _StepwisePhaseCChannel(PhaseCChannel):
    def __init__(self) -> None:
        self._r = 0.0
        self._c = 0.0
        self._alive = True
        self._dt = 1.0

    def set_current(self, r: float, c: float = 0.0, *, alive: bool = True, dt: float = 1.0) -> None:
        self._r = float(r)
        self._c = float(c)
        self._alive = bool(alive)
        self._dt = float(dt)

    def step(self, weight: float):
        return self._r * float(weight), self._c * float(weight), self._alive, self._dt


def topology_enabled(enable_topology: Optional[bool] = None) -> bool:
    if enable_topology is not None:
        return bool(enable_topology)
    return str(os.getenv("CAPM_ENABLE_TOPOLOGY", "")).lower() in {"1", "true", "yes"}


def ensure_topology_state(state: Dict[str, object], k: int) -> None:
    if "base_channels" in state:
        return
    base_channels = {f"ch_{i}": _StepwisePhaseCChannel() for i in range(int(k))}
    state["base_channels"] = base_channels
    state["channels"] = dict(base_channels)
    state["broker"] = Broker()


def update_topology_state(state: Dict[str, object], r_vec, stack_manager: StackManager) -> None:
    base_channels: Dict[str, _StepwisePhaseCChannel] = state["base_channels"]  # type: ignore[assignment]
    channels: Dict[str, PhaseCChannel] = state["channels"]  # type: ignore[assignment]
    broker: Broker = state["broker"]  # type: ignore[assignment]

    for idx, (eid, ch) in enumerate(base_channels.items()):
        r_val = float(r_vec[idx]) if idx < len(r_vec) else 0.0
        ch.set_current(r_val, 0.0)

    for eid, ch in base_channels.items():
        r, c, alive, dt = ch.step(1.0)
        broker.observe(eid, r, c, alive, dt)

    broker.update_correlations(base_channels.keys())

    for st in list(stack_manager.stacks.values()):
        if isinstance(st, StackChannel):
            st.step(1.0)

    stack_manager.maintain(channels)
    stack_manager.try_form_stack(broker, channels)
