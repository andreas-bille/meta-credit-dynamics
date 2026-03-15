from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import numpy as np


@dataclass(frozen=True)
class FlowDimensions:
    n_inputs: int
    m_outputs: int


@dataclass(frozen=True)
class InputFlowState:
    tau: int
    dimensions: FlowDimensions
    liquidity: float
    due_by_channel: tuple[float, ...]
    returns_by_channel: tuple[float, ...]
    claim_ledger: Any
    offers: tuple[Any, ...]
    available_input_by_channel: tuple[float, ...]


@dataclass(frozen=True)
class PlanEdge:
    src_channel: int
    dst_channel: int
    amount: float
    maturity_tau: int


@dataclass(frozen=True)
class OutputFlowPlan:
    tau: int
    edges: tuple[PlanEdge, ...]


@dataclass(frozen=True)
class PlanSettlementRecord:
    edge_id: str
    src_channel: int
    dst_channel: int
    amount: float
    maturity_tau: int


@dataclass(frozen=True)
class PlanSettlementInput:
    tau: int
    edge_records: list[PlanSettlementRecord]


@dataclass(frozen=True)
class BookingRecord:
    edge_id: str
    tau: int
    src_channel: int
    dst_channel: int
    maturity_tau: int
    effect_kind: str
    amount_delta: float


def validate_flow_dimensions(dimensions: FlowDimensions) -> None:
    n_inputs = int(dimensions.n_inputs)
    m_outputs = int(dimensions.m_outputs)
    if n_inputs <= 0:
        raise ValueError("invalid flow dimensions: n_inputs must be > 0")
    if m_outputs <= 0:
        raise ValueError("flow state dimension mismatch")


def _ensure_finite(value: float) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("flow state contains non-finite values")
    return out


def _normalize_channel_vector(values: Iterable[float] | None, expected_len: int) -> np.ndarray:
    if expected_len <= 0:
        raise ValueError("invalid flow dimensions: n_inputs must be > 0")
    if values is None:
        return np.zeros(expected_len, dtype=float)

    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.shape[0] != expected_len:
        raise ValueError("flow state dimension mismatch")
    if np.any(~np.isfinite(arr)):
        raise ValueError("flow state contains non-finite values")
    return arr


def _derive_due_by_channel(
    *,
    input_events: Mapping[str, Any],
    due_obligations: list[dict[str, Any]],
    n_inputs: int,
) -> np.ndarray:
    explicit_due = input_events.get("due_by_channel")
    if explicit_due is not None:
        return np.maximum(0.0, _normalize_channel_vector(explicit_due, n_inputs))

    due = np.zeros(n_inputs, dtype=float)
    for obligation in due_obligations:
        amount = float(obligation.get("amount_due", 0.0))
        if not np.isfinite(amount):
            raise ValueError("flow state contains non-finite values")
        if amount <= 0.0:
            continue

        channel_raw = obligation.get("src_channel", obligation.get("debtor_channel", obligation.get("channel_id")))
        if channel_raw is None:
            due += amount / float(n_inputs)
            continue

        channel_idx = int(channel_raw)
        if channel_idx < 0 or channel_idx >= n_inputs:
            raise ValueError("flow state dimension mismatch")
        due[channel_idx] += amount
    return due


def build_input_flow_state(
    *,
    state: Any,
    input_events: Mapping[str, Any],
    tau: int,
    due_obligations: list[dict[str, Any]],
    due_returns: Mapping[str, Any],
    flow_dimensions: FlowDimensions,
) -> InputFlowState:
    validate_flow_dimensions(flow_dimensions)
    n_inputs = int(flow_dimensions.n_inputs)

    liquidity = _ensure_finite(getattr(state, "liquidity", getattr(state, "wealth", 0.0)))

    returns_vec = _normalize_channel_vector(due_returns.get("r_vec", input_events.get("r_vec")), n_inputs)
    due_vec = _derive_due_by_channel(
        input_events=input_events,
        due_obligations=due_obligations,
        n_inputs=n_inputs,
    )

    if np.any(~np.isfinite(due_vec)) or np.any(~np.isfinite(returns_vec)):
        raise ValueError("flow state contains non-finite values")

    liquidity_component = np.full(n_inputs, liquidity / float(n_inputs), dtype=float)
    available = np.maximum(0.0, returns_vec - due_vec + liquidity_component)

    offers = tuple(getattr(state, "offers", []) or [])

    return InputFlowState(
        tau=int(tau),
        dimensions=flow_dimensions,
        liquidity=float(liquidity),
        due_by_channel=tuple(float(x) for x in due_vec),
        returns_by_channel=tuple(float(x) for x in returns_vec),
        claim_ledger=getattr(state, "claim_ledger", None),
        offers=offers,
        available_input_by_channel=tuple(float(x) for x in available),
    )


def build_plan_settlement_input(plan: OutputFlowPlan) -> PlanSettlementInput:
    records: list[PlanSettlementRecord] = []
    tau = int(plan.tau)
    for index, edge in enumerate(plan.edges):
        mapped_maturity_tau = int(edge.maturity_tau)
        if mapped_maturity_tau != int(edge.maturity_tau):
            raise ValueError("maturity_tau collapsed during settlement mapping")
        records.append(
            PlanSettlementRecord(
                edge_id=f"edge:{tau}:{index}",
                src_channel=int(edge.src_channel),
                dst_channel=int(edge.dst_channel),
                amount=float(edge.amount),
                maturity_tau=mapped_maturity_tau,
            )
        )
    return PlanSettlementInput(tau=tau, edge_records=records)


def validate_plan_settlement_input(plan_input: PlanSettlementInput) -> None:
    if not isinstance(plan_input, PlanSettlementInput):
        raise ValueError("invalid plan settlement input schema")

    tau = int(plan_input.tau)
    if tau < 0:
        raise ValueError("invalid plan settlement input schema")

    if not isinstance(plan_input.edge_records, list):
        raise ValueError("invalid plan settlement input schema")

    for index, record in enumerate(plan_input.edge_records):
        if not isinstance(record, PlanSettlementRecord):
            raise ValueError("invalid plan settlement input schema")
        if getattr(record, "src_channel", None) is None or getattr(record, "dst_channel", None) is None:
            raise ValueError("dst_channel not propagated to settlement")
        expected_edge_id = f"edge:{tau}:{index}"
        if str(record.edge_id) != expected_edge_id:
            raise ValueError("invalid deterministic edge id")
        if not np.isfinite(float(record.amount)):
            raise ValueError("invalid plan settlement input schema")
        if int(record.maturity_tau) <= tau:
            raise ValueError("invalid plan settlement input schema")


def validate_booking_record_schema(records: list[BookingRecord]) -> None:
    if not isinstance(records, list):
        raise ValueError("invalid booking record schema")
    for record in records:
        if not isinstance(record, BookingRecord):
            raise ValueError("invalid booking record schema")
        if not isinstance(record.edge_id, str) or not record.edge_id:
            raise ValueError("invalid booking record schema")
        if getattr(record, "dst_channel", None) is None:
            raise ValueError("booking record missing dst_channel")
        if getattr(record, "maturity_tau", None) is None:
            raise ValueError("booking record missing maturity_tau")
        if not isinstance(record.effect_kind, str) or not record.effect_kind:
            raise ValueError("invalid booking record schema")
        if not np.isfinite(float(record.amount_delta)):
            raise ValueError("invalid booking record schema")