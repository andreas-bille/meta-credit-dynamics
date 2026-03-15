from __future__ import annotations

from typing import Any, Callable, Mapping
import numpy as np

from .flow_contracts import (
    BookingRecord,
    FlowDimensions,
    OutputFlowPlan,
    PlanSettlementInput,
    build_input_flow_state,
    build_plan_settlement_input,
    validate_booking_record_schema,
    validate_flow_dimensions,
    validate_plan_settlement_input,
)
from .flow_validator import validate_flow_plan
from .phase_i_events import psi, update_mu_from_events, update_rho_from_events
from .settlement import extract_due_obligations_at_tau, settle_due_claims_at_tau
from .accounting_contract import assert_selector_accounting_contract


HookLike = Mapping[str, Callable[..., None]] | Any | None


def _call_hook(hooks: HookLike, name: str, *args: Any) -> None:
    if hooks is None:
        return
    if isinstance(hooks, Mapping):
        fn = hooks.get(name)
        if callable(fn):
            fn(*args)
        return
    fn = getattr(hooks, name, None)
    if callable(fn):
        fn(*args)


def _default_due_extractor(state: Any, input_events: Mapping[str, Any], tau: int):
    r_vec = np.asarray(input_events.get("r_vec", []), dtype=float)
    due_returns = {
        "r_vec": r_vec,
        "total": float(r_vec.sum()),
    }
    due_obligations = extract_due_obligations_at_tau(state, input_events, tau)
    return due_obligations, due_returns


def _default_returns_booker(state: Any, due_returns: Mapping[str, Any], tau: int, input_events: Mapping[str, Any]):
    liquidity_before = float(state.wealth)
    total_returns = float(due_returns.get("total", 0.0))
    # Phase-I design invariant:
    # returns are booked independent of selector weights.
    # Policy affects only internal attribution/state.
    # Closed-loop coupling is introduced in Phase-II.
    state.wealth = liquidity_before + total_returns
    state.liquidity = float(state.wealth)
    state._last_r = total_returns
    state._last_c = float(input_events.get("c_total", 0.0))
    return liquidity_before, float(state.wealth)


def _default_settlement_processor(state: Any, due_obligations: list[dict[str, float]], tau: int):
    _, _, settlement_result = settle_due_claims_at_tau(
        state,
        tau,
        rng=None,
        config=getattr(state, "settlement_config", None),
        due_obligations=due_obligations,
    )
    return settlement_result


def _default_wealth_computer(state: Any, settlement_result: Mapping[str, Any], tau: int):
    obligations_after = settlement_result.get("obligations_after", [])
    due_total = float(sum(float(item.get("amount_due", 0.0)) for item in obligations_after))
    state.wealth = float(state.wealth) - due_total
    state.liquidity = float(state.wealth)
    return float(state.wealth)


def _default_dead_decider(state: Any, wealth_value: float, tau: int) -> bool:
    settlement_failed = bool(getattr(state, "_last_settlement_failed", False))
    return settlement_failed or float(wealth_value) < 0.0


def _default_offer_publisher(state: Any, due_returns: Mapping[str, Any], input_events: Mapping[str, Any], tau: int):
    r_vec = np.asarray(due_returns.get("r_vec", []), dtype=float)
    c_total = float(input_events.get("c_total", 0.0))

    _, _, pi_total, pi_vec = state.compute_pi(r_vec, c_total)
    state.stats.update(pi_total)

    adv = state.compute_advantage(pi_vec)
    state.w = state.reweight_fn(state.w, adv)

    state._enforce_invariants()
    return []


def _resolve_structural_policy(policy: Mapping[str, Any]) -> Any:
    return policy.get("structural_policy")


def _resolve_flow_dimensions(policy: Mapping[str, Any], state: Any) -> FlowDimensions:
    _ = state
    raw = policy.get("flow_dimensions")
    if not isinstance(raw, FlowDimensions):
        raise ValueError("flow state dimension mismatch")

    dims = raw

    validate_flow_dimensions(dims)
    return dims


def _sum_obligations_nominal(obligations: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in obligations:
        amount = float(item.get("amount_due", 0.0))
        if not np.isfinite(amount):
            raise ValueError("flow state contains non-finite values")
        if amount > 0.0:
            total += amount
    return float(total)


def _sum_open_claim_ledger_nominal(state: Any) -> float:
    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    if ledger is None or process_id is None:
        return 0.0
    total = 0.0
    for claim in ledger.claims_for_process(process_id):
        if ledger.get_status(claim.claim_id) != "open":
            continue
        total += float(claim.nominal)
    return float(total)


def _claim_sequence_from_id(claim_id: str) -> int:
    parts = str(claim_id).split(":")
    if len(parts) < 3 or parts[-2] != "claim":
        raise ValueError("non-deterministic claim id generation")
    try:
        return int(parts[-1])
    except ValueError as exc:
        raise ValueError("non-deterministic claim id generation") from exc


def _claim_channel_meta(state: Any) -> dict[str, dict[str, int | str]]:
    current = getattr(state, "_claim_channel_meta", None)
    if current is None:
        current = {}
        state._claim_channel_meta = current
    if not isinstance(current, dict):
        raise ValueError("dst_channel not propagated to settlement")
    return current


def _apply_plan_settlement_input(
    *,
    state: Any,
    due_obligations: list[dict[str, Any]],
    plan_settlement_input: PlanSettlementInput,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    validate_plan_settlement_input(plan_settlement_input)
    merged = list(due_obligations)
    edge_to_claim_id: dict[str, str] = {}
    claim_channel_meta = _claim_channel_meta(state)
    prev_claim_seq = -1

    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    generation_id = int(getattr(state, "generation_id", 0))
    if ledger is None or process_id is None:
        raise ValueError("invalid plan settlement input schema")

    ordered_records = sorted(
        list(plan_settlement_input.edge_records),
        key=lambda item: (int(item.dst_channel), int(item.src_channel), str(item.edge_id)),
    )

    for record in ordered_records:
        if int(record.maturity_tau) <= int(plan_settlement_input.tau):
            raise ValueError("invalid plan settlement input schema")
        claim = ledger.create_claim(
            process_id=process_id,
            generation_id=generation_id,
            created_tau=int(plan_settlement_input.tau),
            creditor_id=f"channel-{int(record.src_channel)}",
            debtor_id=str(process_id),
            nominal=float(record.amount),
            maturity_tau=int(record.maturity_tau),
            src_channel=int(record.src_channel),
            dst_channel=int(record.dst_channel),
            claim_type="flow_plan_edge",
            source_offer_id=str(record.edge_id),
            drawn_principal=float(record.amount),
        )
        if int(claim.maturity_tau) != int(record.maturity_tau):
            raise ValueError("maturity_tau collapsed during settlement mapping")
        claim_seq = _claim_sequence_from_id(str(claim.claim_id))
        if claim_seq <= prev_claim_seq:
            raise ValueError("non-deterministic claim id generation")
        prev_claim_seq = claim_seq

        edge_to_claim_id[str(record.edge_id)] = str(claim.claim_id)
        claim_channel_meta[str(claim.claim_id)] = {
            "src_channel": int(record.src_channel),
            "dst_channel": int(record.dst_channel),
            "edge_id": str(record.edge_id),
        }
    return merged, edge_to_claim_id


def _assert_maturity_mapping_not_collapsed(*, flow_plan: OutputFlowPlan, plan_settlement_input: PlanSettlementInput) -> None:
    if len(flow_plan.edges) != len(plan_settlement_input.edge_records):
        raise ValueError("invalid plan settlement input schema")
    for index, edge in enumerate(flow_plan.edges):
        record = plan_settlement_input.edge_records[index]
        if int(edge.maturity_tau) != int(record.maturity_tau):
            raise ValueError("maturity_tau collapsed during settlement mapping")


def _event_claim_id(event: Any) -> str | None:
    if isinstance(event, Mapping):
        raw = event.get("claim_id")
    else:
        raw = getattr(event, "claim_id", None)
    return None if raw is None else str(raw)


def _event_cash_paid(event: Any) -> float:
    if isinstance(event, Mapping):
        raw = event.get("cash_paid", 0.0)
    else:
        raw = getattr(event, "cash_paid", 0.0)
    return float(raw)


def _build_booking_records(
    *,
    plan_settlement_input: PlanSettlementInput,
    edge_to_claim_id: Mapping[str, str],
    settlement_result: Mapping[str, Any],
) -> list[BookingRecord]:
    events = list(settlement_result.get("events", []) or [])
    unresolved = list(settlement_result.get("obligations_after", []) or [])

    event_by_claim: dict[str, Any] = {}
    for event in events:
        claim_id = _event_claim_id(event)
        if claim_id is None:
            continue
        event_by_claim[claim_id] = event

    unresolved_by_claim: dict[str, float] = {}
    for item in unresolved:
        claim_id = item.get("claim_id")
        if claim_id is None:
            continue
        unresolved_by_claim[str(claim_id)] = float(item.get("amount_due", 0.0))

    records: list[BookingRecord] = []
    for edge in plan_settlement_input.edge_records:
        edge_id = str(edge.edge_id)
        claim_id = edge_to_claim_id.get(edge_id)
        if claim_id is None:
            raise ValueError("plan edge produced no booking effect")

        event = event_by_claim.get(claim_id)
        cash_paid = 0.0 if event is None else _event_cash_paid(event)

        if cash_paid > 0.0:
            effect_kind = "cash_paid"
            amount_delta = float(cash_paid)
        elif claim_id in unresolved_by_claim and float(unresolved_by_claim[claim_id]) > 0.0:
            effect_kind = "obligation_carry"
            amount_delta = float(unresolved_by_claim[claim_id])
        elif claim_id is not None:
            # Newly created non-due claims must still produce deterministic booking trace.
            effect_kind = "obligation_carry"
            amount_delta = float(edge.amount)
        else:
            raise ValueError("plan edge produced no booking effect")

        records.append(
            BookingRecord(
                edge_id=edge_id,
                tau=int(plan_settlement_input.tau),
                src_channel=int(edge.src_channel),
                dst_channel=int(edge.dst_channel),
                maturity_tau=int(edge.maturity_tau),
                effect_kind=effect_kind,
                amount_delta=float(amount_delta),
            )
        )

    validate_booking_record_schema(records)
    return records


def _ensure_material_booking_effect(
    *,
    pre_obligations_nominal: float,
    post_obligations_nominal: float,
    pre_claim_open_nominal: float,
    post_claim_open_nominal: float,
    cash_paid_total: float,
) -> None:
    delta_obligations = float(post_obligations_nominal) - float(pre_obligations_nominal)
    delta_claim_open = float(post_claim_open_nominal) - float(pre_claim_open_nominal)
    if delta_obligations > 0.0:
        return
    if delta_claim_open > 0.0:
        return
    if float(cash_paid_total) > 0.0:
        return
    raise ValueError("plan produced no material booking effect")


def _propose_and_validate_flow_plan(
    *,
    state: Any,
    input_events: Mapping[str, Any],
    tau: int,
    due_obligations: list[dict[str, Any]],
    due_returns: Mapping[str, Any],
    flow_dimensions: FlowDimensions,
    structural_policy: Any,
    eps: float,
) -> tuple[Any, OutputFlowPlan]:
    flow_state = build_input_flow_state(
        state=state,
        input_events=input_events,
        tau=int(tau),
        due_obligations=due_obligations,
        due_returns=due_returns,
        flow_dimensions=flow_dimensions,
    )

    if callable(getattr(structural_policy, "propose_flow_plan", None)):
        plan = structural_policy.propose_flow_plan(flow_state)
    elif callable(structural_policy):
        plan = structural_policy(flow_state)
    else:
        raise ValueError("structural_policy must be callable or implement propose_flow_plan")

    if not isinstance(plan, OutputFlowPlan):
        raise ValueError("structural_policy must return OutputFlowPlan")

    validate_flow_plan(flow_state, plan, eps=eps)
    return flow_state, plan


def _build_flow_diagnostics(flow_plan: OutputFlowPlan) -> dict[str, Any]:
    edge_count = int(len(flow_plan.edges))
    aggregate_magnitude = 0.0
    dst_counts: dict[str, int] = {}

    for edge in flow_plan.edges:
        amount = float(edge.amount)
        if not np.isfinite(amount):
            raise ValueError("flow state contains non-finite values")
        aggregate_magnitude += abs(amount)

        dst_key = str(int(edge.dst_channel))
        dst_counts[dst_key] = int(dst_counts.get(dst_key, 0)) + 1

    if edge_count <= 0:
        distribution: dict[str, float] = {}
    else:
        ordered = sorted(dst_counts.items(), key=lambda item: int(item[0]))
        distribution = {str(key): float(count) / float(edge_count) for key, count in ordered}

    return {
        "plan_edge_count": int(edge_count),
        "aggregate_flow_magnitude": float(aggregate_magnitude),
        "dst_channel_distribution": dict(distribution),
    }


def step_at_tau(
    state: Any,
    input_events: Mapping[str, Any],
    policy: Mapping[str, Callable[..., Any]] | None,
    tau: int,
    hooks: HookLike = None,
):
    """Execute one explicit Phase-H semantic step at time tau.

    Ordered phases:
      1) due extraction
      2) returns booking
      3) settlement
      4) wealth computation
      5) dead decision
      6) offer publication (only if alive)
    """

    assert_selector_accounting_contract(state, tau=int(tau), stage="pre")

    freeze = bool(input_events.get("freeze", False))
    if freeze:
        state._last_flow_diagnostics = None
        state._enforce_invariants()
        assert_selector_accounting_contract(state, tau=int(tau), stage="post")
        next_tau = int(tau) + 1
        return state, [], next_tau

    policy = policy or {}
    due_extractor = policy.get("due_extractor", _default_due_extractor)
    returns_booker = policy.get("returns_booker", _default_returns_booker)
    settlement_processor = policy.get("settlement_processor", _default_settlement_processor)
    wealth_computer = policy.get("wealth_computer", _default_wealth_computer)
    dead_decider = policy.get("dead_decider", _default_dead_decider)
    offer_publisher = policy.get("offer_publisher", _default_offer_publisher)
    next_tau_fn = policy.get("next_tau_fn")
    structural_policy = _resolve_structural_policy(policy)
    flow_validator_eps = float(policy.get("flow_validator_eps", 1e-9))
    state._last_flow_diagnostics = None

    due_obligations, due_returns = due_extractor(state, input_events, tau)
    _call_hook(hooks, "on_due_extracted", due_obligations, due_returns)

    due_obligations_for_settlement = list(due_obligations)
    flow_state = None
    flow_plan = None
    plan_settlement_input = None
    flow_hardening_summary = None
    edge_to_claim_id: dict[str, str] = {}
    pre_obligations_nominal = _sum_obligations_nominal(due_obligations)
    pre_claim_open_nominal = _sum_open_claim_ledger_nominal(state)

    if structural_policy is not None:
        flow_dimensions = _resolve_flow_dimensions(policy, state)
        flow_state, flow_plan = _propose_and_validate_flow_plan(
            state=state,
            input_events=input_events,
            tau=int(tau),
            due_obligations=due_obligations,
            due_returns=due_returns,
            flow_dimensions=flow_dimensions,
            structural_policy=structural_policy,
            eps=flow_validator_eps,
        )
        plan_settlement_input = build_plan_settlement_input(flow_plan)
        _assert_maturity_mapping_not_collapsed(flow_plan=flow_plan, plan_settlement_input=plan_settlement_input)
        validate_plan_settlement_input(plan_settlement_input)
        due_obligations_for_settlement, edge_to_claim_id = _apply_plan_settlement_input(
            state=state,
            due_obligations=due_obligations,
            plan_settlement_input=plan_settlement_input,
        )
        _call_hook(hooks, "on_flow_plan_validated", flow_state, flow_plan)

        if bool(policy.get("flow_diagnostics_enabled", False)):
            diagnostics = _build_flow_diagnostics(flow_plan)
            state._last_flow_diagnostics = dict(diagnostics)
            _call_hook(hooks, "on_flow_diagnostics", diagnostics)

    liquidity_before, liquidity_after = returns_booker(state, due_returns, tau, input_events)
    _call_hook(hooks, "on_returns_booked", liquidity_before, liquidity_after, due_returns)

    settlement_result = settlement_processor(state, due_obligations_for_settlement, tau)

    booking_records: list[BookingRecord] = []
    if structural_policy is not None and plan_settlement_input is not None:
        booking_records = _build_booking_records(
            plan_settlement_input=plan_settlement_input,
            edge_to_claim_id=edge_to_claim_id,
            settlement_result=settlement_result,
        )
        post_obligations_nominal = _sum_obligations_nominal(list(settlement_result.get("obligations_after", []) or []))
        post_claim_open_nominal = _sum_open_claim_ledger_nominal(state)
        cash_paid_total = float(settlement_result.get("settled_amount", 0.0))

        _ensure_material_booking_effect(
            pre_obligations_nominal=pre_obligations_nominal,
            post_obligations_nominal=post_obligations_nominal,
            pre_claim_open_nominal=pre_claim_open_nominal,
            post_claim_open_nominal=post_claim_open_nominal,
            cash_paid_total=cash_paid_total,
        )

        state._last_plan_settlement_input = plan_settlement_input
        state._last_booking_records = booking_records
        flow_hardening_summary = {
            "bounds_accept_mask": [True for _ in booking_records],
            "available_input_by_channel": [] if flow_state is None else list(flow_state.available_input_by_channel),
            "sum_obligations_nominal_after_plan": float(post_obligations_nominal),
            "sum_claim_ledger_open_nominal_after_plan": float(post_claim_open_nominal),
            "cash_paid_total": float(cash_paid_total),
            "cash_paid_by_dst_channel": dict(settlement_result.get("cash_paid_by_dst_channel", {}) or {}),
            "obligations_grouped_by_dst_channel": dict(settlement_result.get("obligations_grouped_by_dst_channel", {}) or {}),
            "claim_ledger_grouped_by_dst_channel": dict(settlement_result.get("claim_ledger_grouped_by_dst_channel", {}) or {}),
            "dead_flag": False,
        }

    _call_hook(hooks, "on_settlement_completed", settlement_result)

    phase_i_events = psi(
        state=state,
        tau=int(tau),
        input_events=input_events,
        due_returns=due_returns,
        due_obligations=due_obligations_for_settlement,
        settlement_result=settlement_result,
    )
    update_mu_from_events(state, phase_i_events)
    update_rho_from_events(state, phase_i_events)
    state._last_phase_i_events = phase_i_events

    wealth_value = float(wealth_computer(state, settlement_result, tau))
    _call_hook(hooks, "on_wealth_computed", wealth_value)

    is_dead = bool(dead_decider(state, wealth_value, tau))
    state.dead = bool(is_dead)
    if state.dead:
        if getattr(state, "tau_dead", None) is None:
            state.tau_dead = int(tau)
    else:
        state.tau_dead = None
    _call_hook(hooks, "on_dead_decided", is_dead, tau if is_dead else None)

    if flow_hardening_summary is not None and plan_settlement_input is not None:
        flow_hardening_summary["dead_flag"] = bool(is_dead)
        state._last_flow_hardening_summary = flow_hardening_summary
        _call_hook(
            hooks,
            "on_flow_plan_booked",
            plan_settlement_input,
            booking_records,
            flow_hardening_summary,
        )

    assert_selector_accounting_contract(state, tau=int(tau), stage="post")

    output_offers = []
    if not is_dead and structural_policy is None:
        published_offers = offer_publisher(state, due_returns, input_events, tau)
        if published_offers:
            output_offers.extend(list(published_offers))
    _call_hook(hooks, "on_offers_published", output_offers)

    next_tau = next_tau_fn(state, tau) if callable(next_tau_fn) else int(tau) + 1
    return state, output_offers, int(next_tau)
