from __future__ import annotations

import os
from typing import Any, Mapping
import numpy as np
import torch

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
from .cuda_state import DeviceState, to_device_state
from .kernel_semantics_cuda import batch_core_step
from .accounting_contract import assert_selector_accounting_contract
from .ledger import ClaimLedger
from .settlement import extract_due_obligations_at_tau


_PUBLISH_POLICY_BY_PROFILE = {
    "test": "full",
    "parity": "full",
    "benchmark": "minimal",
    "masssim": "minimal",
    "prod": "minimal",
    "runtime": "minimal",
}


def _resolve_publish_policy() -> tuple[str, str]:
    profile = str(os.environ.get("CAPM_CUDA_PROFILE", "test")).strip().lower()
    if profile not in _PUBLISH_POLICY_BY_PROFILE:
        allowed_profiles = ", ".join(sorted(_PUBLISH_POLICY_BY_PROFILE.keys()))
        raise ValueError(f"invalid CAPM_CUDA_PROFILE='{profile}', expected one of: {allowed_profiles}")

    explicit_policy = os.environ.get("CAPM_CUDA_PUBLISH_POLICY")
    if explicit_policy is None:
        return _PUBLISH_POLICY_BY_PROFILE[profile], profile

    policy = str(explicit_policy).strip().lower()
    if policy not in {"minimal", "full"}:
        raise ValueError("invalid CAPM_CUDA_PUBLISH_POLICY, expected 'minimal' or 'full'")
    return policy, profile


def _resolve_structural_policy(policy: Any) -> Any:
    if isinstance(policy, Mapping):
        return policy.get("structural_policy")
    return getattr(policy, "structural_policy", None)


def _build_due_returns(input_events: Mapping[str, Any]) -> Mapping[str, Any]:
    r_vec = np.asarray(input_events.get("r_vec", []), dtype=float).reshape(-1)
    return {
        "r_vec": r_vec,
        "total": float(r_vec.sum()),
    }


def _default_due_extractor(state: Any, input_events: Mapping[str, Any], tau: int):
    due_returns = _build_due_returns(input_events)
    due_obligations = extract_due_obligations_at_tau(state, input_events, tau)
    return due_obligations, due_returns


def _claim_sequence_from_id(claim_id: str) -> int:
    parts = str(claim_id).split(":")
    if len(parts) < 3 or parts[-2] != "claim":
        raise ValueError("non-deterministic claim id generation")
    try:
        return int(parts[-1])
    except ValueError as exc:
        raise ValueError("non-deterministic claim id generation") from exc


def _claim_channel_meta(selector: Any) -> dict[str, dict[str, int | str]]:
    current = getattr(selector, "_claim_channel_meta", None)
    if current is None:
        current = {}
        selector._claim_channel_meta = current
    if not isinstance(current, dict):
        raise ValueError("dst_channel not propagated to settlement")
    return current


def _dst_bucket(obligation: Mapping[str, Any]) -> str:
    raw = obligation.get("dst_channel")
    if raw is None:
        return "none"
    return str(int(raw))


def _claim_id_by_target(ledger: ClaimLedger, process_id: int | str) -> dict[int, str]:
    return {int(idx): str(claim.claim_id) for idx, claim in enumerate(ledger.claims_for_process(process_id))}


def _claim_dst_bucket_by_target(selector: Any) -> dict[int, str]:
    ledger = getattr(selector, "claim_ledger", None)
    process_id = getattr(selector, "process_id", None)
    if not isinstance(ledger, ClaimLedger) or process_id is None:
        raise ValueError("cuda claim state drift detected")

    claim_channel_meta = _claim_channel_meta(selector)
    claim_id_by_target = _claim_id_by_target(ledger, process_id)
    out: dict[int, str] = {}
    for target_idx, claim_id in claim_id_by_target.items():
        meta = claim_channel_meta.get(str(claim_id), {})
        if meta.get("dst_channel") is not None:
            out[int(target_idx)] = str(int(meta["dst_channel"]))
            continue
        claim = ledger.get_claim(str(claim_id))
        claim_dst = int(getattr(claim, "dst_channel", -1))
        out[int(target_idx)] = "none" if claim_dst < 0 else str(claim_dst)
    return out


def _device_open_claims_by_target(state: DeviceState) -> dict[int, tuple[float, int]]:
    active = state.claim_active_mask[0]
    target = state.claim_target[0]
    amount = state.claim_amount[0]
    maturity = state.claim_maturity_tau[0]
    slots = torch.nonzero(active, as_tuple=False).flatten().tolist()

    by_target: dict[int, tuple[float, int]] = {}
    for slot in slots:
        target_idx = int(target[slot].item())
        if target_idx < 0 or target_idx in by_target:
            raise ValueError("cuda claim state drift detected")
        nominal = float(amount[slot].item())
        maturity_tau = int(maturity[slot].item())
        if not np.isfinite(nominal) or nominal < 0.0:
            raise ValueError("cuda claim state drift detected")
        by_target[target_idx] = (nominal, maturity_tau)
    return by_target


def _host_open_claims_by_target(ledger: ClaimLedger, process_id: int | str) -> dict[int, tuple[float, int]]:
    claims = ledger.claims_for_process(process_id)
    claim_count = len(claims)

    nominal = ledger._nominal_by_process[process_id]
    maturity = ledger._maturity_by_process[process_id]
    open_mask = ledger._open_mask_by_process[process_id]

    if int(nominal.shape[0]) != claim_count or int(maturity.shape[0]) != claim_count or int(open_mask.shape[0]) != claim_count:
        raise ValueError("cuda claim state drift detected")

    out: dict[int, tuple[float, int]] = {}
    for idx in range(claim_count):
        if not bool(open_mask[idx].item()):
            continue
        out[int(idx)] = (float(nominal[idx].item()), int(maturity[idx].item()))
    return out


def _assert_no_claim_state_drift(*, selector: Any, state: DeviceState) -> None:
    ledger = getattr(selector, "claim_ledger", None)
    process_id = getattr(selector, "process_id", None)
    if not isinstance(ledger, ClaimLedger) or process_id is None:
        raise ValueError("cuda claim state drift detected")

    host_open = _host_open_claims_by_target(ledger, process_id)
    device_open = _device_open_claims_by_target(state)

    if set(host_open.keys()) != set(device_open.keys()):
        raise ValueError("cuda claim state drift detected")

    for target_idx in sorted(host_open.keys()):
        host_amount, host_maturity = host_open[target_idx]
        device_amount, device_maturity = device_open[target_idx]
        if abs(float(host_amount) - float(device_amount)) > 1e-12:
            raise ValueError("cuda claim state drift detected")
        if int(host_maturity) != int(device_maturity):
            raise ValueError("cuda claim state drift detected")


def _sync_host_claim_view_from_device(
    *,
    selector: Any,
    selector_id: int,
    state_pre: DeviceState,
    state: DeviceState,
    out: Mapping[str, Any],
    tau: int,
    imported_claims_by_selector: dict[int, int],
) -> None:
    ledger = getattr(selector, "claim_ledger", None)
    process_id = getattr(selector, "process_id", None)
    if not isinstance(ledger, ClaimLedger) or process_id is None:
        raise ValueError("cuda claim state drift detected")

    claim_slot_mask = out.get("claim_slot_mask")
    claim_slot_remainder = out.get("claim_slot_remainder")
    claim_slot_unresolved_mask = out.get("claim_slot_unresolved_mask")
    if (
        not isinstance(claim_slot_mask, torch.Tensor)
        or not isinstance(claim_slot_remainder, torch.Tensor)
        or not isinstance(claim_slot_unresolved_mask, torch.Tensor)
    ):
        raise ValueError("cuda claim state drift detected")

    due_slots = torch.nonzero(claim_slot_mask[0], as_tuple=False).flatten().tolist()
    claim_id_by_target = _claim_id_by_target(ledger, process_id)
    claim_channel_meta = _claim_channel_meta(selector)

    rewrite_target_map: dict[int, int] = {}
    close_targets: set[int] = set()
    eps = 1e-12

    for slot in due_slots:
        target_idx = int(state_pre.claim_target[0, slot].item())
        if target_idx < 0:
            raise ValueError("cuda claim state drift detected")

        claim_id = claim_id_by_target.get(target_idx)
        if claim_id is None:
            raise ValueError("cuda claim state drift detected")

        if bool(claim_slot_unresolved_mask[0, slot].item()):
            # Rejected + insufficient cash remains open at the same target.
            continue

        remainder = float(claim_slot_remainder[0, slot].item())
        if not np.isfinite(remainder) or remainder < 0.0:
            raise ValueError("cuda claim state drift detected")

        if remainder > eps:
            settlement_cfg = dict(getattr(selector, "settlement_config", {}) or {})
            child_claim = ledger.rewrite_claim(
                claim_id=str(claim_id),
                generation_id=int(getattr(selector, "generation_id", 0)),
                closed_at=int(tau),
                nominal=float(remainder),
                maturity_tau=int(tau) + int(settlement_cfg.get("future_maturity_offset", 1)),
            )
            child_target = int(ledger._claim_slot_by_id.get(str(child_claim.claim_id), -1))
            if child_target < 0:
                raise ValueError("cuda claim state drift detected")
            parent_meta = claim_channel_meta.get(str(claim_id), {})
            if "src_channel" in parent_meta and "dst_channel" in parent_meta:
                claim_channel_meta[str(child_claim.claim_id)] = {
                    "src_channel": int(parent_meta["src_channel"]),
                    "dst_channel": int(parent_meta["dst_channel"]),
                    "edge_id": str(parent_meta.get("edge_id", "")),
                }
            rewrite_target_map[int(target_idx)] = int(child_target)
            continue

        # Fully consumed due claim.
        if ledger.get_status(str(claim_id)) == "open":
            ledger.close_claim(claim_id=str(claim_id), closed_at=int(tau), status="consumed")
        close_targets.add(int(target_idx))

    # Remap rewritten active claims to newly created child targets so next due extraction
    # references the same claim identities as CPU path.
    for parent_target, child_target in rewrite_target_map.items():
        mask = state.claim_active_mask[0] & (state.claim_target[0] == int(parent_target))
        if int(mask.sum().item()) != 1:
            raise ValueError("cuda claim state drift detected")
        state.claim_target[0, mask] = int(child_target)
        if state.claim_generation_id is not None:
            state.claim_generation_id[0, mask] = int(getattr(selector, "generation_id", 0))
        if state.claim_parent_id is not None:
            state.claim_parent_id[0, mask] = int(parent_target)

    # Ensure closed targets are no longer active on device.
    for parent_target in close_targets:
        still_active = bool(torch.any(state.claim_active_mask[0] & (state.claim_target[0] == int(parent_target))).item())
        if still_active:
            raise ValueError("cuda claim state drift detected")

    # Claims created by rewrite are already represented by remapped device slots.
    imported_claims_by_selector[selector_id] = int(len(ledger.claims_for_process(process_id)))


def _extract_due_obligations_from_device_state(
    *,
    selector: Any,
    state: DeviceState,
    input_events: Mapping[str, Any],
    tau: int,
) -> list[dict[str, Any]]:
    ledger = getattr(selector, "claim_ledger", None)
    process_id = getattr(selector, "process_id", None)
    if not isinstance(ledger, ClaimLedger) or process_id is None:
        raise ValueError("cuda claim state drift detected")

    claim_id_by_target = _claim_id_by_target(ledger, process_id)
    claim_channel_meta = _claim_channel_meta(selector)
    obligations: list[dict[str, Any]] = []

    c_total = float(input_events.get("c_total", 0.0))
    if c_total > 0.0:
        obligations.append(
            {
                "kind": "legacy_cash_due",
                "claim_id": None,
                "amount_due": float(c_total),
                "due_time": int(tau),
            }
        )

    active = state.claim_active_mask[0]
    maturity = state.claim_maturity_tau[0]
    due_slots = torch.nonzero(active & (maturity <= int(tau)), as_tuple=False).flatten().tolist()

    for slot in due_slots:
        target_idx = int(state.claim_target[0, slot].item())
        amount_due = float(state.claim_amount[0, slot].item())
        if target_idx < 0 or not np.isfinite(amount_due):
            raise ValueError("cuda claim state drift detected")
        if amount_due <= 0.0:
            continue

        claim_id = claim_id_by_target.get(target_idx)
        if claim_id is None:
            raise ValueError("cuda claim state drift detected")

        claim = ledger.get_claim(claim_id)
        obligations.append(
            {
                "kind": "claim_due",
                "claim_id": str(claim_id),
                "amount_due": float(amount_due),
                "due_time": int(tau),
                "maturity_tau": int(claim.maturity_tau),
                "created_tau": int(getattr(claim, "created_tau", 0)),
                "debtor_id": claim.debtor_id,
                "creditor_id": claim.creditor_id,
            }
        )

        meta = claim_channel_meta.get(str(claim_id))
        if meta is not None:
            obligations[-1]["src_channel"] = int(meta["src_channel"])
            obligations[-1]["dst_channel"] = int(meta["dst_channel"])
            obligations[-1]["edge_id"] = str(meta.get("edge_id", ""))
        elif int(getattr(claim, "src_channel", -1)) >= 0 and int(getattr(claim, "dst_channel", -1)) >= 0:
            obligations[-1]["src_channel"] = int(getattr(claim, "src_channel"))
            obligations[-1]["dst_channel"] = int(getattr(claim, "dst_channel"))
        elif str(claim.claim_type) == "flow_plan_edge":
            raise ValueError("dst_channel not propagated to settlement")

    obligations.sort(key=lambda item: (int(item.get("due_time", tau)), str(item.get("claim_id") or ""), str(item.get("kind", ""))))
    return obligations


def _build_cuda_settlement_result(
    *,
    due_obligations: list[dict[str, Any]],
    claim_slot_remainder: torch.Tensor,
    claim_slot_cash_paid: torch.Tensor,
    claim_slot_mask: torch.Tensor,
    claim_slot_unresolved_mask: torch.Tensor,
    state_pre: DeviceState,
    claim_id_by_target: Mapping[int, str],
    claim_dst_bucket_by_target: Mapping[int, str],
    tau: int,
    legacy_cash_paid_total: float,
    legacy_unresolved_total: float,
) -> Mapping[str, Any]:
    if claim_slot_remainder.shape != state_pre.claim_amount.shape:
        raise ValueError("cuda claim state drift detected")
    if claim_slot_cash_paid.shape != state_pre.claim_amount.shape:
        raise ValueError("cuda claim state drift detected")
    if claim_slot_mask.shape != state_pre.claim_active_mask.shape:
        raise ValueError("cuda claim state drift detected")
    if claim_slot_unresolved_mask.shape != state_pre.claim_active_mask.shape:
        raise ValueError("cuda claim state drift detected")

    unresolved: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for item in due_obligations:
        if item.get("dst_channel") is None:
            continue
        if item.get("src_channel") is None or item.get("dst_channel") is None:
            raise ValueError("dst_channel not propagated to settlement")
        if item.get("claim_id") is None or item.get("amount_due") is None or item.get("maturity_tau") is None:
            raise ValueError("invalid plan settlement input schema")

    claim_cash_by_id: dict[str, float] = {}
    claim_unresolved_by_id: dict[str, float] = {}
    cash_paid_by_dst_channel: dict[str, float] = {}
    obligations_grouped_by_dst_channel: dict[str, float] = {}

    # Rebuild per-claim summaries directly from device slots (deterministic slot order).
    due_slots = torch.nonzero(claim_slot_mask[0], as_tuple=False).flatten().tolist()
    for slot in due_slots:
        target_idx = int(state_pre.claim_target[0, slot].item())
        claim_id = claim_id_by_target.get(target_idx)
        if claim_id is None:
            raise ValueError("cuda claim state drift detected")

        unresolved_amt = float(claim_slot_remainder[0, slot].item())
        paid_amt = float(claim_slot_cash_paid[0, slot].item())
        if not np.isfinite(unresolved_amt) or not np.isfinite(paid_amt):
            raise ValueError("cuda claim state drift detected")
        if unresolved_amt < 0.0 or paid_amt < 0.0:
            raise ValueError("cuda claim state drift detected")

        claim_cash_by_id[claim_id] = claim_cash_by_id.get(claim_id, 0.0) + paid_amt
        if bool(claim_slot_unresolved_mask[0, slot].item()) and unresolved_amt > 0.0:
            claim_unresolved_by_id[claim_id] = claim_unresolved_by_id.get(claim_id, 0.0) + unresolved_amt

    if float(legacy_unresolved_total) > 0.0:
        unresolved.append(
            {
                "kind": "legacy_cash_due",
                "claim_id": None,
                "amount_due": float(legacy_unresolved_total),
                "due_time": int(tau),
            }
        )
        obligations_grouped_by_dst_channel["none"] = obligations_grouped_by_dst_channel.get("none", 0.0) + float(legacy_unresolved_total)

    settled_amount = max(0.0, float(legacy_cash_paid_total))
    if float(legacy_cash_paid_total) > 0.0:
        cash_paid_by_dst_channel["none"] = cash_paid_by_dst_channel.get("none", 0.0) + float(legacy_cash_paid_total)

    due_slot_to_dst: dict[int, str] = {}
    for slot in due_slots:
        target_idx = int(state_pre.claim_target[0, slot].item())
        claim_id = claim_id_by_target.get(target_idx)
        if claim_id is None:
            raise ValueError("cuda claim state drift detected")
        dst_key = str(claim_dst_bucket_by_target.get(int(target_idx), "none"))
        for item in due_obligations:
            if str(item.get("claim_id") or "") == str(claim_id):
                dst_key = _dst_bucket(item)
                break
        due_slot_to_dst[int(slot)] = dst_key

    claim_ledger_grouped_by_dst_channel: dict[str, float] = {}
    for slot in torch.nonzero(state_pre.claim_active_mask[0], as_tuple=False).flatten().tolist():
        if int(slot) in due_slot_to_dst:
            amount = float(claim_slot_remainder[0, slot].item())
            due_key = due_slot_to_dst[int(slot)]
        else:
            amount = float(state_pre.claim_amount[0, slot].item())
            target_idx = int(state_pre.claim_target[0, slot].item())
            due_key = str(claim_dst_bucket_by_target.get(int(target_idx), "none"))
        if amount > 0.0:
            claim_ledger_grouped_by_dst_channel[due_key] = claim_ledger_grouped_by_dst_channel.get(due_key, 0.0) + amount

    for item in due_obligations:
        claim_id = item.get("claim_id")
        if claim_id is None:
            continue

        claim_key = str(claim_id)
        amount_due = float(item.get("amount_due", 0.0))
        unresolved_amt = float(claim_unresolved_by_id.get(claim_key, 0.0))
        paid_amt = float(claim_cash_by_id.get(claim_key, 0.0))

        if unresolved_amt > amount_due + 1e-9:
            raise ValueError("cuda claim state drift detected")

        if unresolved_amt > 0.0:
            dst_key = _dst_bucket(item)
            unresolved.append(
                {
                    "kind": str(item.get("kind", "claim_due")),
                    "claim_id": claim_key,
                    "amount_due": float(unresolved_amt),
                    "due_time": int(item.get("due_time", tau)),
                    "debtor_id": item.get("debtor_id"),
                    "creditor_id": item.get("creditor_id"),
                    "src_channel": item.get("src_channel"),
                    "dst_channel": item.get("dst_channel"),
                }
            )
            obligations_grouped_by_dst_channel[dst_key] = obligations_grouped_by_dst_channel.get(dst_key, 0.0) + float(unresolved_amt)

        dst_key = _dst_bucket(item)
        if paid_amt > 0.0:
            cash_paid_by_dst_channel[dst_key] = cash_paid_by_dst_channel.get(dst_key, 0.0) + float(paid_amt)

        events.append(
            {
                "claim_id": claim_key,
                "cash_paid": float(paid_amt),
            }
        )
        settled_amount += max(0.0, float(paid_amt))

    settlement_failed = bool(float(legacy_unresolved_total) > 0.0 or any(float(v) > 0.0 for v in claim_unresolved_by_id.values()))

    return {
        "obligations_after": unresolved,
        "settled_amount": float(settled_amount),
        "settlement_failed": settlement_failed,
        "events": events,
        "cash_paid_by_dst_channel": cash_paid_by_dst_channel,
        "obligations_grouped_by_dst_channel": obligations_grouped_by_dst_channel,
        "claim_ledger_grouped_by_dst_channel": claim_ledger_grouped_by_dst_channel,
    }


def _call_hook(hooks: Any, name: str, *args: Any) -> None:
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


def _resolve_flow_dimensions(policy: Any) -> FlowDimensions:
    raw = None
    if isinstance(policy, Mapping):
        raw = policy.get("flow_dimensions")
    elif policy is not None:
        raw = getattr(policy, "flow_dimensions", None)

    if not isinstance(raw, FlowDimensions):
        raise ValueError("flow state dimension mismatch")

    validate_flow_dimensions(raw)
    return raw


def _sum_obligations_nominal(obligations: list[dict[str, Any]]) -> float:
    total = 0.0
    for item in obligations:
        amount = float(item.get("amount_due", 0.0))
        if not np.isfinite(amount):
            raise ValueError("flow state contains non-finite values")
        if amount > 0.0:
            total += amount
    return float(total)


def _propose_and_validate_flow_plan(
    *,
    selector: Any,
    input_events: Mapping[str, Any],
    tau: int,
    due_obligations: list[dict[str, Any]],
    due_returns: Mapping[str, Any],
    policy: Any,
    structural_policy: Any,
    eps: float,
):
    flow_dimensions = _resolve_flow_dimensions(policy)
    flow_state = build_input_flow_state(
        state=selector,
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

    validate_flow_plan(flow_state, plan, eps=float(eps))
    return flow_state, plan


def _sum_open_claim_nominal_device(state: DeviceState) -> float:
    active = state.claim_active_mask[0]
    if active.numel() == 0:
        return 0.0
    total = state.claim_amount[0][active].sum()
    return float(total.item())


def _ensure_material_booking_effect(
    *,
    pre_obligations_nominal: float,
    post_obligations_nominal: float,
    pre_claim_open_nominal: float,
    post_claim_open_nominal: float,
    cash_paid_total: float,
) -> None:
    if float(post_obligations_nominal) - float(pre_obligations_nominal) > 0.0:
        return
    if float(post_claim_open_nominal) - float(pre_claim_open_nominal) > 0.0:
        return
    if float(cash_paid_total) > 0.0:
        return
    raise ValueError("plan produced no material booking effect")


def _apply_plan_settlement_input_cuda(*, selector: Any, plan_settlement_input: PlanSettlementInput) -> dict[str, str]:
    validate_plan_settlement_input(plan_settlement_input)

    ledger = getattr(selector, "claim_ledger", None)
    process_id = getattr(selector, "process_id", None)
    generation_id = int(getattr(selector, "generation_id", 0))
    if ledger is None or process_id is None:
        raise ValueError("invalid plan settlement input schema")

    edge_to_claim_id: dict[str, str] = {}
    claim_channel_meta = _claim_channel_meta(selector)
    prev_claim_seq = -1
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
    return edge_to_claim_id


def _build_cuda_booking_records(
    *,
    plan_settlement_input: PlanSettlementInput,
    edge_to_claim_id: Mapping[str, str],
    settlement_result: Mapping[str, Any],
) -> list[BookingRecord]:
    events = list(settlement_result.get("events", []) or [])
    unresolved = list(settlement_result.get("obligations_after", []) or [])

    event_by_claim: dict[str, Mapping[str, Any]] = {}
    for event in events:
        claim_id = event.get("claim_id")
        if claim_id is None:
            continue
        event_by_claim[str(claim_id)] = event

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

        event = event_by_claim.get(str(claim_id))
        cash_paid = 0.0 if event is None else float(event.get("cash_paid", 0.0))
        unresolved_amount = float(unresolved_by_claim.get(str(claim_id), 0.0))

        if cash_paid > 0.0:
            effect_kind = "cash_paid"
            amount_delta = cash_paid
        elif unresolved_amount > 0.0:
            effect_kind = "obligation_carry"
            amount_delta = unresolved_amount
        elif claim_id is not None:
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


def _assert_maturity_mapping_not_collapsed(*, flow_plan: OutputFlowPlan, plan_settlement_input: PlanSettlementInput) -> None:
    if len(flow_plan.edges) != len(plan_settlement_input.edge_records):
        raise ValueError("invalid plan settlement input schema")
    for index, edge in enumerate(flow_plan.edges):
        record = plan_settlement_input.edge_records[index]
        if int(edge.maturity_tau) != int(record.maturity_tau):
            raise ValueError("maturity_tau collapsed during settlement mapping")

class CudaCore:
    """CUDA backend bound to Phase-H event-order semantics.

    This keeps CPU as semantic oracle and mirrors the same ordered phases.
    """

    def __init__(self, *, hooks=None, policy=None, start_tau: int = 0, device: str | torch.device = "cuda"):
        self._tau = int(start_tau)
        self._hooks = hooks
        self._policy = policy
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._mode = str(os.environ.get("CAPM_MODE", "deterministic"))
        self._publish_policy, self._publish_profile = _resolve_publish_policy()
        self._state_by_selector: dict[int, DeviceState] = {}
        self._imported_claims_by_selector: dict[int, int] = {}
        self._scalar_cache_by_selector: dict[int, dict[str, torch.Tensor]] = {}
        self._metrics = {
            "effective_backend": "cuda",
            "CAPM_MODE": self._mode,
            "CAPM_CUDA_PROFILE": self._publish_profile,
            "CAPM_CUDA_PUBLISH_POLICY": self._publish_policy,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_ops_count": 0,
            "cpu_fallback_used": 0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "overflow_events": 0,
            "steps": 0,
        }
        self._snapshot_stride = int(os.environ.get("CAPM_CUDA_SNAPSHOT_EVERY", "1000"))

    def step(self, selector, r_vec, c_total, *, freeze: bool) -> None:
        self.step_with_tau(selector, r_vec, c_total, freeze=freeze, tau=self._tau)
        self._tau += 1

    def step_with_tau(self, selector: Any, r_vec, c_total, *, freeze: bool, tau: int) -> None:
        structural_policy = _resolve_structural_policy(self._policy)
        policy_payload = self._policy if isinstance(self._policy, Mapping) else {}
        due_extractor = policy_payload.get("due_extractor", _default_due_extractor)
        flow_validator_eps = float(policy_payload.get("flow_validator_eps", 1e-9))

        if self._device.type != "cuda":
            raise RuntimeError("CudaCore requires a cuda device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is False")

        assert_selector_accounting_contract(selector, tau=int(tau), stage="pre")

        selector_id = id(selector)
        state = self._state_by_selector.get(selector_id)
        if state is None:
            ledger = getattr(selector, "claim_ledger", None)
            if not isinstance(ledger, ClaimLedger):
                raise ValueError("selector.claim_ledger must be ClaimLedger for CUDA path")
            max_claims = int(ledger.max_claims_per_process)
            state = to_device_state(
                selector,
                device=self._device,
                max_claims_per_process=max_claims,
            )
            self._state_by_selector[selector_id] = state
            self._imported_claims_by_selector[selector_id] = 0
            self._scalar_cache_by_selector[selector_id] = {
                "returns_total": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "c_total": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "lambda_cash_share": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "stats_beta": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "accept_by_default": torch.zeros((), device=self._device, dtype=torch.bool),
                "future_maturity_offset": torch.zeros((), device=self._device, dtype=torch.int32),
                "phase_i_beta": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
                "phase_i_beta_r": torch.zeros((), device=self._device, dtype=state.wealth.dtype),
            }

        r_arr = np.asarray(r_vec, dtype=float)
        if hasattr(selector, "ensure_channel_state"):
            selector.ensure_channel_state(len(r_arr))
        elif selector.w is None or len(selector.w) != len(r_arr):
            selector.w = np.ones(len(r_arr), dtype=float) / max(1, len(r_arr))
            selector.K = len(r_arr)

        input_events = {
            "r_vec": r_arr,
            "c_total": float(c_total),
            "freeze": bool(freeze),
        }

        state_prev = state
        state = self._ingest_new_claims(selector=selector, state=state)
        state.validate_shapes()
        state.validate_dtypes()
        state.validate_device(expected_backend="cuda")
        state.validate_determinism_ready()

        if structural_policy is not None:
            _assert_no_claim_state_drift(selector=selector, state=state)
            due_returns = _build_due_returns(input_events)
            due_obligations = _extract_due_obligations_from_device_state(
                selector=selector,
                state=state,
                input_events=input_events,
                tau=int(tau),
            )
        else:
            due_obligations, due_returns = due_extractor(selector, input_events, int(tau))
        _call_hook(self._hooks, "on_due_extracted", due_obligations, due_returns)

        flow_state = None
        plan_settlement_input = None
        due_obligations_for_settlement = list(due_obligations)
        edge_to_claim_id: dict[str, str] = {}
        pre_obligations_nominal = 0.0
        pre_claim_open_nominal = 0.0
        if structural_policy is not None:
            pre_obligations_nominal = _sum_obligations_nominal(list(due_obligations))
            pre_claim_open_nominal = _sum_open_claim_nominal_device(state)
            flow_state, flow_plan = _propose_and_validate_flow_plan(
                selector=selector,
                input_events=input_events,
                tau=int(tau),
                due_obligations=list(due_obligations),
                due_returns=due_returns,
                policy=self._policy,
                structural_policy=structural_policy,
                eps=flow_validator_eps,
            )
            _call_hook(self._hooks, "on_flow_plan_validated", flow_state, flow_plan)
            plan_settlement_input = build_plan_settlement_input(flow_plan)
            _assert_maturity_mapping_not_collapsed(flow_plan=flow_plan, plan_settlement_input=plan_settlement_input)
            validate_plan_settlement_input(plan_settlement_input)
            edge_to_claim_id = _apply_plan_settlement_input_cuda(
                selector=selector,
                plan_settlement_input=plan_settlement_input,
            )
            state = self._ingest_new_claims(selector=selector, state=state)
            state.validate_shapes()
            state.validate_dtypes()
            state.validate_device(expected_backend="cuda")
            state.validate_determinism_ready()
            _assert_no_claim_state_drift(selector=selector, state=state)

            due_obligations_for_settlement = _extract_due_obligations_from_device_state(
                selector=selector,
                state=state,
                input_events=input_events,
                tau=int(tau),
            )
            created_claim_ids = set(str(value) for value in edge_to_claim_id.values())
            if created_claim_ids:
                for obligation in due_obligations_for_settlement:
                    claim_id = obligation.get("claim_id")
                    if claim_id is None:
                        continue
                    if str(claim_id) in created_claim_ids:
                        obligation["kind"] = "flow_plan_edge_due"

        claim_slot_dst_channel = torch.full_like(state.claim_target, -1)
        enable_dst_partition = bool(structural_policy is not None)
        if enable_dst_partition:
            claim_id_by_target_pre = _claim_id_by_target(selector.claim_ledger, selector.process_id)
            dst_by_claim_id: dict[str, int] = {}
            for obligation in due_obligations_for_settlement:
                if obligation.get("dst_channel") is None:
                    continue
                claim_id = obligation.get("claim_id")
                if claim_id is None:
                    raise ValueError("invalid plan settlement input schema")
                if obligation.get("src_channel") is None or obligation.get("dst_channel") is None:
                    raise ValueError("dst_channel not propagated to settlement")
                if obligation.get("maturity_tau") is None or obligation.get("amount_due") is None:
                    raise ValueError("invalid plan settlement input schema")
                dst_by_claim_id[str(claim_id)] = int(obligation["dst_channel"])

            due_slots = torch.nonzero(
                state.claim_active_mask[0] & (state.claim_maturity_tau[0] <= int(tau)),
                as_tuple=False,
            ).flatten().tolist()
            for slot in due_slots:
                target_idx = int(state.claim_target[0, slot].item())
                claim_id = claim_id_by_target_pre.get(target_idx)
                if claim_id is None:
                    raise ValueError("cuda claim state drift detected")
                if str(claim_id) in dst_by_claim_id:
                    claim_slot_dst_channel[0, slot] = int(dst_by_claim_id[str(claim_id)])

        returns_vec_arr = np.asarray(due_returns.get("r_vec", r_arr), dtype=float).reshape(-1)
        returns_total_value = float(due_returns.get("total", float(returns_vec_arr.sum())))
        scalar_cache = self._scalar_cache_by_selector[selector_id]
        returns_total = scalar_cache["returns_total"].fill_(returns_total_value)
        c_total_tensor = scalar_cache["c_total"].fill_(float(c_total))
        returns_vec_tensor = torch.as_tensor(returns_vec_arr, device=self._device, dtype=state.wealth.dtype).unsqueeze(0)
        self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int((returns_total.element_size() + c_total_tensor.element_size()))
        self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(returns_vec_tensor.numel() * returns_vec_tensor.element_size())

        settlement_cfg = dict(getattr(selector, "settlement_config", {}) or {})
        lambda_cash_share = float(settlement_cfg.get("lambda_cash_share", getattr(selector, "lambda_cash_share", 0.5)))
        accept_by_default = bool(settlement_cfg.get("accept_by_default", True))
        lambda_cash_share_t = scalar_cache["lambda_cash_share"].fill_(lambda_cash_share)
        accept_by_default_t = scalar_cache["accept_by_default"].fill_(accept_by_default)
        stats_beta_t = scalar_cache["stats_beta"].fill_(float(getattr(selector.stats, "beta", 0.0)))
        future_maturity_offset_t = scalar_cache["future_maturity_offset"].fill_(int(settlement_cfg.get("future_maturity_offset", 1)))
        phase_i_beta_t = scalar_cache["phase_i_beta"].fill_(float(getattr(selector, "beta_term", getattr(selector.stats, "beta", 0.0))))
        phase_i_beta_r_t = scalar_cache["phase_i_beta_r"].fill_(float(getattr(selector, "beta_r", getattr(selector.stats, "beta", 0.0))))

        out = batch_core_step(
            state,
            input_events={
                "returns_total": returns_total,
                "returns_vec": returns_vec_tensor,
                "c_total": c_total_tensor,
                "freeze": bool(freeze),
                "lambda_cash_share": lambda_cash_share_t,
                "accept_by_default": accept_by_default_t,
                "stats_beta": stats_beta_t,
                "phase_i_beta": phase_i_beta_t,
                "phase_i_beta_r": phase_i_beta_r_t,
                "future_maturity_offset": future_maturity_offset_t,
                "enable_dst_partition": enable_dst_partition,
                "claim_slot_dst_channel": claim_slot_dst_channel,
            },
            tau=int(tau),
        )
        state_next = out["state"]
        self._metrics["cuda_ops_count"] = int(self._metrics["cuda_ops_count"]) + int(out.get("cuda_ops_count", 0))

        if structural_policy is not None:
            claim_slot_remainder = out.get("claim_slot_remainder")
            claim_slot_cash_paid = out.get("claim_slot_cash_paid")
            claim_slot_mask = out.get("claim_slot_mask")
            claim_slot_unresolved_mask = out.get("claim_slot_unresolved_mask")
            if (
                not isinstance(claim_slot_remainder, torch.Tensor)
                or not isinstance(claim_slot_cash_paid, torch.Tensor)
                or not isinstance(claim_slot_mask, torch.Tensor)
                or not isinstance(claim_slot_unresolved_mask, torch.Tensor)
            ):
                raise ValueError("cuda claim state drift detected")

            claim_id_by_target = _claim_id_by_target(selector.claim_ledger, selector.process_id)
            claim_dst_bucket_by_target = _claim_dst_bucket_by_target(selector)
            settlement_result = _build_cuda_settlement_result(
                due_obligations=due_obligations_for_settlement,
                claim_slot_remainder=claim_slot_remainder,
                claim_slot_cash_paid=claim_slot_cash_paid,
                claim_slot_mask=claim_slot_mask,
                claim_slot_unresolved_mask=claim_slot_unresolved_mask,
                state_pre=state,
                claim_id_by_target=claim_id_by_target,
                claim_dst_bucket_by_target=claim_dst_bucket_by_target,
                tau=int(tau),
                legacy_cash_paid_total=float(out["legacy_cash_paid"][0].item()),
                legacy_unresolved_total=float(out["legacy_unresolved"][0].item()),
            )
        else:
            legacy_unresolved_total = float(out["legacy_unresolved"][0].item())
            claim_unresolved_total = float(out["claim_unresolved"][0].item())
            obligations_after: list[dict[str, Any]] = []
            if legacy_unresolved_total > 0.0:
                obligations_after.append(
                    {
                        "kind": "legacy_cash_due",
                        "claim_id": None,
                        "amount_due": legacy_unresolved_total,
                        "due_time": int(tau),
                    }
                )
            if claim_unresolved_total > 0.0:
                obligations_after.append(
                    {
                        "kind": "claim_due",
                        "claim_id": None,
                        "amount_due": claim_unresolved_total,
                        "due_time": int(tau),
                    }
                )
            settlement_result = {
                "obligations_after": obligations_after,
                "settled_amount": float(out["legacy_cash_paid"][0].item() + out["claim_cash_paid"][0].item()),
                "settlement_failed": bool((legacy_unresolved_total + claim_unresolved_total) > 0.0),
                "events": [],
            }

        selector._last_settlement_result = dict(settlement_result)
        _call_hook(self._hooks, "on_settlement_completed", settlement_result)

        self._publish_selector_runtime(
            selector=selector,
            state_prev=state_prev,
            state=state_next,
            out=out,
            r_vec=r_arr,
            c_total=float(c_total),
            freeze=bool(freeze),
            tau=int(tau),
            allow_legacy_offer_publication=(structural_policy is None),
        )

        if structural_policy is not None:
            _sync_host_claim_view_from_device(
                selector=selector,
                selector_id=selector_id,
                state_pre=state,
                state=state_next,
                out=out,
                tau=int(tau),
                imported_claims_by_selector=self._imported_claims_by_selector,
            )
            _assert_no_claim_state_drift(selector=selector, state=state_next)

        if structural_policy is not None and not bool(freeze) and plan_settlement_input is not None:
            claim_cash_paid_total = float(out["claim_cash_paid"][0].item())
            claim_unresolved_total = float(out["claim_unresolved"][0].item())
            legacy_cash_paid_total = float(out["legacy_cash_paid"][0].item())
            legacy_remainder_total = float(out["legacy_unresolved"][0].item())

            post_obligations_nominal = float(claim_unresolved_total + legacy_remainder_total)
            post_claim_open_nominal = _sum_open_claim_nominal_device(state_next)
            cash_paid_total = float(claim_cash_paid_total + legacy_cash_paid_total)

            _ensure_material_booking_effect(
                pre_obligations_nominal=pre_obligations_nominal,
                post_obligations_nominal=post_obligations_nominal,
                pre_claim_open_nominal=pre_claim_open_nominal,
                post_claim_open_nominal=post_claim_open_nominal,
                cash_paid_total=cash_paid_total,
            )

            booking_records = _build_cuda_booking_records(
                plan_settlement_input=plan_settlement_input,
                edge_to_claim_id=edge_to_claim_id,
                settlement_result=settlement_result,
            )
            selector._last_plan_settlement_input = plan_settlement_input
            selector._last_booking_records = booking_records
            summary = {
                "bounds_accept_mask": [True for _ in booking_records],
                "available_input_by_channel": [] if flow_state is None else list(flow_state.available_input_by_channel),
                "sum_obligations_nominal_after_plan": float(post_obligations_nominal),
                "sum_claim_ledger_open_nominal_after_plan": float(post_claim_open_nominal),
                "cash_paid_total": float(cash_paid_total),
                "cash_paid_by_dst_channel": dict(settlement_result.get("cash_paid_by_dst_channel", {}) or {}),
                "obligations_grouped_by_dst_channel": dict(settlement_result.get("obligations_grouped_by_dst_channel", {}) or {}),
                "claim_ledger_grouped_by_dst_channel": dict(settlement_result.get("claim_ledger_grouped_by_dst_channel", {}) or {}),
                "dead_flag": bool(getattr(selector, "dead", False)),
            }
            selector._last_flow_hardening_summary = summary
            _call_hook(self._hooks, "on_flow_plan_booked", plan_settlement_input, booking_records, summary)

        assert_selector_accounting_contract(selector, tau=int(tau), stage="post")

        self._state_by_selector[selector_id] = state_next
        self._metrics["steps"] = int(self._metrics["steps"]) + 1

    def metrics_snapshot(self) -> dict[str, Any]:
        return dict(self._metrics)

    def outstanding_claim_count(self, selector: Any) -> int:
        state = self._state_by_selector.get(id(selector))
        if state is None or state.claim_count is None:
            return 0
        return int(state.claim_count[0].item())

    def lifecycle_snapshot(self, selector: Any) -> dict[str, torch.Tensor]:
        state = self._state_by_selector.get(id(selector))
        if state is None:
            raise KeyError("selector state not initialized")

        process_id = int(getattr(selector, "process_id", 0))
        generation_id = int(getattr(selector, "generation_id", 0))
        return {
            "wealth": state.wealth,
            "dead_mask": state.dead_mask if state.dead_mask is not None else torch.zeros_like(state.wealth, dtype=torch.bool),
            "process_id": torch.as_tensor([process_id], device=state.device, dtype=torch.int64),
            "generation_id": torch.as_tensor([generation_id], device=state.device, dtype=torch.int64),
        }

    def _ingest_new_claims(self, *, selector: Any, state: DeviceState) -> DeviceState:
        ledger = getattr(selector, "claim_ledger", None)
        if not isinstance(ledger, ClaimLedger):
            raise ValueError("selector.claim_ledger must be ClaimLedger")

        selector_id = id(selector)
        process_id = int(getattr(selector, "process_id", 0))
        imported = int(self._imported_claims_by_selector.get(selector_id, 0))
        batch = ledger.claim_tensor_batch_for_process(
            process_id=process_id,
            start_index=imported,
            device=state.device,
            float_dtype=state.claim_amount.dtype,
        )

        batch_len = int(batch["batch_len"])
        if batch_len <= 0:
            return state

        imported_after = imported + batch_len
        self._imported_claims_by_selector[selector_id] = imported_after

        open_mask = batch["is_open"]
        if open_mask.numel() == 0:
            return state

        open_amount = batch["nominal"][open_mask]
        if open_amount.numel() == 0:
            return state

        open_target = batch["claim_target"][open_mask]
        open_maturity = batch["maturity_tau"][open_mask]
        open_generation = batch["generation_id"][open_mask]

        free_mask = ~state.claim_active_mask[0]
        free_slots = torch.nonzero(free_mask, as_tuple=False).flatten()

        requested = int(open_amount.numel())
        available = int(free_slots.numel())
        insert_count = min(requested, available)

        if requested > available:
            if self._mode == "deterministic":
                raise RuntimeError(
                    f"claim capacity exceeded in deterministic mode: process={process_id}, max_claims_per_process={state.max_claims_per_process}"
                )
            self._metrics["overflow_events"] = int(self._metrics["overflow_events"]) + int(requested - available)

        if insert_count <= 0:
            return state

        insert_slots = free_slots[:insert_count]

        state.claim_amount[0, insert_slots] = open_amount[:insert_count]
        state.claim_interest[0, insert_slots] = torch.zeros((insert_count,), device=state.device, dtype=state.claim_interest.dtype)
        state.claim_target[0, insert_slots] = open_target[:insert_count].to(dtype=state.claim_target.dtype)
        state.claim_maturity_tau[0, insert_slots] = open_maturity[:insert_count].to(dtype=state.claim_maturity_tau.dtype)
        state.claim_active_mask[0, insert_slots] = True

        if state.claim_generation_id is not None:
            state.claim_generation_id[0, insert_slots] = open_generation[:insert_count].to(dtype=state.claim_generation_id.dtype)
        if state.claim_parent_id is not None:
            state.claim_parent_id[0, insert_slots] = torch.full(
                (insert_count,),
                -1,
                device=state.device,
                dtype=state.claim_parent_id.dtype,
            )
        if state.claim_count is not None:
            state.claim_count[0] = state.claim_active_mask[0].sum().to(dtype=state.claim_count.dtype)

        self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(
            (insert_count * state.claim_amount.element_size())
            + (insert_count * state.claim_target.element_size())
            + (insert_count * state.claim_maturity_tau.element_size())
            + (insert_count * state.claim_active_mask.element_size())
        )

        return state

    def _publish_selector_runtime(
        self,
        *,
        selector: Any,
        state_prev: DeviceState,
        state: DeviceState,
        out: dict[str, Any],
        r_vec: np.ndarray,
        c_total: float,
        freeze: bool,
        tau: int,
        allow_legacy_offer_publication: bool,
    ) -> None:
        if self._publish_policy == "minimal":
            self._publish_selector_runtime_minimal(
                selector=selector,
                state_prev=state_prev,
                state=state,
                out=out,
                r_vec=r_vec,
                c_total=c_total,
                freeze=freeze,
                tau=tau,
                allow_legacy_offer_publication=allow_legacy_offer_publication,
            )
            return

        self._publish_selector_runtime_full(
            selector=selector,
            state_prev=state_prev,
            state=state,
            out=out,
            r_vec=r_vec,
            c_total=c_total,
            freeze=freeze,
            tau=tau,
            allow_legacy_offer_publication=allow_legacy_offer_publication,
        )

    def _publish_selector_runtime_minimal(
        self,
        *,
        selector: Any,
        state_prev: DeviceState,
        state: DeviceState,
        out: dict[str, Any],
        r_vec: np.ndarray,
        c_total: float,
        freeze: bool,
        tau: int,
        allow_legacy_offer_publication: bool,
    ) -> None:
        # Minimal Sync Surface (System Contract)
        # Required by CPU meta/control path only:
        # - selector.wealth: fitness accumulation in PopulationManager
        # - selector.stats.mu: CPU reweight advantage (pi_vec - mu)
        # - offer_publication_mask: determines whether CPU reweight is executed
        # - settlement_failed: deterministic dead/settlement decisions
        # - selector.w (full vector) only when offer_publication_mask is true
        #   because compute_pi/reweight_fn are CPU-side today.
        # Everything else remains device-resident in minimal mode.
        scalar_sync = torch.stack(
            [
                state.wealth[0],
                state.mean[0],
                out["offer_publication_mask"][0].to(dtype=state.wealth.dtype),
                out["settlement_failed"][0].to(dtype=state.wealth.dtype),
            ]
        ).detach().cpu()

        selector.wealth = float(scalar_sync[0])
        selector.liquidity = float(selector.wealth)
        selector.stats.mu = float(scalar_sync[1])

        settlement_failed = bool(scalar_sync[3])
        selector._last_settlement_failed = settlement_failed
        selector.dead = settlement_failed or float(selector.wealth) < 0.0
        if selector.dead:
            if getattr(selector, "tau_dead", None) is None:
                selector.tau_dead = int(tau)
        else:
            selector.tau_dead = None

        self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(4 * state.wealth.element_size())

        if freeze:
            return

        if str(getattr(selector, "selector_policy", "myopic")) in {"term_aware", "term_risk"}:
            self._sync_selector_phase_i_state(selector=selector, state=state)

        offer_mask = bool(scalar_sync[2])
        if allow_legacy_offer_publication and offer_mask and not bool(selector.dead):
            w_host = state.weights[0].detach().cpu().numpy().astype(float, copy=True)
            selector.w = w_host
            selector.K = int(w_host.shape[0])
            self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(
                state.weights[0].numel() * state.weights[0].element_size()
            )

            _, _, _, pi_vec = selector.compute_pi(r_vec, float(c_total))
            adv = selector.compute_advantage(np.asarray(pi_vec, dtype=float))
            selector.w = selector.reweight_fn(np.asarray(selector.w, dtype=float), adv)
            selector._enforce_invariants()
            state.weights[0] = torch.as_tensor(selector.w, device=state.device, dtype=state.weights.dtype)
            self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(
                state.weights[0].numel() * state.weights[0].element_size()
            )

        stride = max(1, int(getattr(selector, "cuda_snapshot_every", self._snapshot_stride)))
        if int(tau) % stride == 0:
            selector._cuda_state_snapshot = {
                "claim_count": int(state.claim_count[0].item()) if state.claim_count is not None else 0,
                "active_due": int(state.due_mask.sum().item()) if state.due_mask is not None else 0,
            }

    def _publish_selector_runtime_full(
        self,
        *,
        selector: Any,
        state_prev: DeviceState,
        state: DeviceState,
        out: dict[str, Any],
        r_vec: np.ndarray,
        c_total: float,
        freeze: bool,
        tau: int,
        allow_legacy_offer_publication: bool,
    ) -> None:
        stats_vec = torch.stack(
            [
                state.wealth[0],
                state.mean[0],
                state.var[0],
                state.drawdown[0],
                state.cum_pi[0],
                state.peak_cum_pi[0],
                out["legacy_cash_paid"][0],
                out["legacy_unresolved"][0],
            ]
        ).detach().cpu().tolist()

        selector.wealth = float(stats_vec[0])
        selector.liquidity = float(selector.wealth)
        selector.stats.mu = float(stats_vec[1])
        selector.stats.var = float(stats_vec[2])
        selector.stats.dd = float(stats_vec[3])
        selector.stats.cum_pi = float(stats_vec[4])
        selector.stats.peak_cum_pi = float(stats_vec[5])
        selector._last_r = float(stats_vec[6])
        selector._last_c = float(stats_vec[7])

        w_host = state.weights[0].detach().cpu().numpy().astype(float, copy=True)
        selector.w = w_host
        selector.K = int(w_host.shape[0])

        bool_vec = torch.stack(
            [
                state.dead_mask[0] if state.dead_mask is not None else torch.as_tensor(bool(getattr(selector, "dead", False)), device=state.device),
                out["settlement_failed"][0],
            ]
        ).detach().cpu().tolist()
        selector.dead = bool(bool_vec[0])
        selector._last_settlement_failed = bool(bool_vec[1])
        if selector.dead:
            if getattr(selector, "tau_dead", None) is None:
                selector.tau_dead = int(tau)
        else:
            selector.tau_dead = None

        self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(
            (9 * state.wealth.element_size()) + state.weights[0].numel() * state.weights[0].element_size()
        )

        if freeze:
            return

        if str(getattr(selector, "selector_policy", "myopic")) in {"term_aware", "term_risk"}:
            self._sync_selector_phase_i_state(selector=selector, state=state)

        offer_mask = bool(out["offer_publication_mask"][0].item())
        if allow_legacy_offer_publication and offer_mask and not bool(selector.dead):
            _, _, _, pi_vec = selector.compute_pi(np.asarray(r_vec, dtype=float), float(c_total))
            adv = selector.compute_advantage(np.asarray(pi_vec, dtype=float))
            selector.w = selector.reweight_fn(np.asarray(selector.w, dtype=float), adv)
            selector._enforce_invariants()
            state.weights[0] = torch.as_tensor(selector.w, device=state.device, dtype=state.weights.dtype)
            self._metrics["h2d_bytes"] = int(self._metrics["h2d_bytes"]) + int(state.weights[0].numel() * state.weights[0].element_size())

        stride = max(1, int(getattr(selector, "cuda_snapshot_every", self._snapshot_stride)))
        if int(tau) % stride == 0:
            selector._cuda_state_snapshot = {
                "claim_count": int(state.claim_count[0].item()) if state.claim_count is not None else 0,
                "active_due": int(state.due_mask.sum().item()) if state.due_mask is not None else 0,
            }

    def _sync_selector_phase_i_state(self, *, selector: Any, state: DeviceState) -> None:
        mu_host = state.mu_term[0].detach().cpu().numpy().astype(float, copy=True)
        rho_host = state.rho[0].detach().cpu().numpy().astype(float, copy=True)

        selector.mu_term = mu_host
        selector.rho = rho_host
        selector.horizon_count = int(mu_host.shape[1]) if mu_host.ndim == 2 else int(getattr(selector, "horizon_count", 0))

        self._metrics["d2h_bytes"] = int(self._metrics["d2h_bytes"]) + int(
            state.mu_term[0].numel() * state.mu_term[0].element_size()
            + state.rho[0].numel() * state.rho[0].element_size()
        )
