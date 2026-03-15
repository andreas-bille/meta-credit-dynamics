from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .flow_contracts import InputFlowState, OutputFlowPlan


class StructuralPolicy(Protocol):
    def propose_flow_plan(self, state: InputFlowState) -> OutputFlowPlan:
        ...


@dataclass(frozen=True)
class EmptyStructuralPolicy:
    """Deterministic no-op policy for integration defaults and tests."""

    def propose_flow_plan(self, state: InputFlowState) -> OutputFlowPlan:
        return OutputFlowPlan(tau=int(state.tau), edges=tuple())