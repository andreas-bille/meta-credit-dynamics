from __future__ import annotations

import math
import numpy as np

from .flow_contracts import InputFlowState, OutputFlowPlan


def _raise(message: str) -> None:
    raise ValueError(message)


def _validate_state_dimensions(state: InputFlowState) -> tuple[int, int]:
    n_inputs = int(state.dimensions.n_inputs)
    m_outputs = int(state.dimensions.m_outputs)
    if n_inputs <= 0:
        _raise("invalid flow dimensions: n_inputs must be > 0")
    if m_outputs <= 0:
        _raise("flow state dimension mismatch")

    if len(state.returns_by_channel) != n_inputs:
        _raise("flow state dimension mismatch")
    if len(state.due_by_channel) != n_inputs:
        _raise("flow state dimension mismatch")
    if len(state.available_input_by_channel) != n_inputs:
        _raise("flow state dimension mismatch")
    return n_inputs, m_outputs


def _validate_state_finite(state: InputFlowState) -> None:
    if not math.isfinite(float(state.liquidity)):
        _raise("flow state contains non-finite values")
    arrays = (
        np.asarray(state.returns_by_channel, dtype=float),
        np.asarray(state.due_by_channel, dtype=float),
        np.asarray(state.available_input_by_channel, dtype=float),
    )
    for arr in arrays:
        if np.any(~np.isfinite(arr)):
            _raise("flow state contains non-finite values")


def validate_flow_plan(state: InputFlowState, plan: OutputFlowPlan, eps: float = 1e-9) -> None:
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or float(eps) < 0.0:
        _raise("invalid flow validator eps")

    n_inputs, m_outputs = _validate_state_dimensions(state)
    _validate_state_finite(state)

    tol = float(eps)
    if int(plan.tau) != int(state.tau):
        _raise("flow plan tau mismatch")

    available = np.asarray(state.available_input_by_channel, dtype=float).reshape(-1)
    if np.any(available < -tol):
        _raise("flow state dimension mismatch")

    outgoing = np.zeros(n_inputs, dtype=float)

    for edge in plan.edges:
        src = int(edge.src_channel)
        dst = int(edge.dst_channel)
        amount = float(edge.amount)
        maturity_tau = int(edge.maturity_tau)

        if src < 0 or src >= n_inputs or dst < 0 or dst >= m_outputs:
            _raise("invalid channel index for F[n,m] mapping")
        if not math.isfinite(amount):
            _raise("flow edge amount must be finite")
        if amount <= 0.0:
            _raise("plan edge amount must be > 0")
        if maturity_tau <= int(state.tau):
            _raise("flow edge maturity_tau must be in the future")

        outgoing[src] += amount

    for src in range(n_inputs):
        if outgoing[src] > available[src] + tol:
            _raise("flow conservation violated")