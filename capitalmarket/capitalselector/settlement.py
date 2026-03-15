from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .ledger import ClaimCapacityExceeded


class SettlementStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SettlementEvent:
    claim_id: str | None
    status: SettlementStatus
    cash_paid: float
    new_claim_ids: tuple[str, ...]
    reason: str | None = None


def _sorted_offers(state: Any):
    offers = list(getattr(state, "offers", []) or [])
    return sorted(offers, key=lambda item: str(item.offer_id))


def _materialize_repayment_claims_from_expired_offers(state: Any, tau: int) -> None:
    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    if ledger is None or process_id is None:
        return

    seen = set(getattr(state, "_processed_offer_ids", set()))
    generation_id = int(getattr(state, "generation_id", 0))

    for offer in _sorted_offers(state):
        offer_id = str(offer.offer_id)
        if offer_id in seen:
            continue

        payload = dict(offer.payload or {})
        borrow_window_end_tau = int(payload.get("borrow_window_end_tau", tau))
        if int(tau) <= borrow_window_end_tau:
            continue

        drawn_principal = float(payload.get("drawn_principal", 0.0))
        if drawn_principal > 0.0:
            ledger.create_claim(
                process_id=process_id,
                generation_id=generation_id,
                created_tau=int(tau),
                creditor_id=str(payload.get("creditor_id", "creditor")),
                debtor_id=str(payload.get("debtor_id", process_id)),
                nominal=drawn_principal,
                maturity_tau=int(payload.get("repayment_due_tau", tau)),
                src_channel=-1,
                dst_channel=-1,
                claim_type="repayment",
                source_offer_id=offer_id,
                drawn_principal=drawn_principal,
            )
        seen.add(offer_id)

    state._processed_offer_ids = seen


def _claim_channel_meta(state: Any) -> dict[str, dict[str, int | str]]:
    current = getattr(state, "_claim_channel_meta", None)
    if current is None:
        current = {}
        state._claim_channel_meta = current
    if not isinstance(current, dict):
        raise ValueError("dst_channel not propagated to settlement")
    return current


def _dst_bucket(obligation: Mapping[str, Any]) -> str:
    raw = obligation.get("dst_channel")
    if raw is None:
        return "none"
    return str(int(raw))


def _group_obligations_by_dst(obligations: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for obligation in obligations:
        amount_due = float(obligation.get("amount_due", 0.0))
        if amount_due <= 0.0:
            continue
        key = _dst_bucket(obligation)
        grouped[key] = grouped.get(key, 0.0) + amount_due
    return grouped


def _group_open_claim_ledger_by_dst(state: Any) -> dict[str, float]:
    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    if ledger is None or process_id is None:
        return {}

    claim_channel_meta = _claim_channel_meta(state)
    grouped: dict[str, float] = {}
    for claim in ledger.claims_for_process(process_id):
        if ledger.get_status(claim.claim_id) != "open":
            continue
        meta = claim_channel_meta.get(str(claim.claim_id), {})
        key = "none" if meta.get("dst_channel") is None else str(int(meta["dst_channel"]))
        grouped[key] = grouped.get(key, 0.0) + float(claim.nominal)
    return grouped


def extract_due_obligations_at_tau(state: Any, input_events: Mapping[str, Any], tau: int):
    _materialize_repayment_claims_from_expired_offers(state, tau)
    claim_channel_meta = _claim_channel_meta(state)

    obligations: list[dict[str, Any]] = []
    c_total = float(input_events.get("c_total", 0.0))
    if c_total > 0.0:
        obligations.append(
            {
                "kind": "legacy_cash_due",
                "claim_id": None,
                "amount_due": c_total,
                "due_time": int(tau),
            }
        )

    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    if ledger is not None and process_id is not None:
        for claim in ledger.claims_for_process(process_id):
            if ledger.get_status(claim.claim_id) != "open":
                continue
            if int(claim.maturity_tau) > int(tau):
                continue
            if claim.claim_type == "repayment" and float(claim.drawn_principal) <= 0.0:
                continue

            obligations.append(
                {
                    "kind": "claim_due",
                    "claim_id": claim.claim_id,
                    "amount_due": float(claim.nominal),
                    "due_time": int(claim.maturity_tau),
                    "maturity_tau": int(claim.maturity_tau),
                    "created_tau": int(getattr(claim, "created_tau", 0)),
                    "debtor_id": claim.debtor_id,
                    "creditor_id": claim.creditor_id,
                }
            )

            meta = claim_channel_meta.get(str(claim.claim_id))
            if meta is not None:
                obligations[-1]["src_channel"] = int(meta["src_channel"])
                obligations[-1]["dst_channel"] = int(meta["dst_channel"])
            elif int(getattr(claim, "src_channel", -1)) >= 0 and int(getattr(claim, "dst_channel", -1)) >= 0:
                obligations[-1]["src_channel"] = int(getattr(claim, "src_channel"))
                obligations[-1]["dst_channel"] = int(getattr(claim, "dst_channel"))
            elif str(claim.claim_type) == "flow_plan_edge":
                raise ValueError("dst_channel not propagated to settlement")

    obligations.sort(key=lambda item: (int(item.get("due_time", tau)), str(item.get("claim_id") or ""), str(item.get("kind", ""))))
    return obligations


def settle_due_claims_at_tau(state: Any, tau: int, rng: Any = None, config: Mapping[str, Any] | None = None, due_obligations=None):
    cfg = dict(config or {})
    lambda_cash_share = float(cfg.get("lambda_cash_share", getattr(state, "lambda_cash_share", 0.5)))
    lambda_cash_share = max(0.0, min(1.0, lambda_cash_share))
    maturity_offset = int(cfg.get("future_maturity_offset", 1))
    accept_by_default = bool(cfg.get("accept_by_default", True))

    if due_obligations is None:
        due_obligations = extract_due_obligations_at_tau(state, {"c_total": 0.0}, tau)

    claim_channel_meta = _claim_channel_meta(state)
    requires_dst_propagation = any(item.get("dst_channel") is not None for item in due_obligations)
    if requires_dst_propagation:
        for obligation in due_obligations:
            if obligation.get("dst_channel") is None:
                continue
            if obligation.get("src_channel") is None or obligation.get("dst_channel") is None:
                raise ValueError("dst_channel not propagated to settlement")
            if obligation.get("claim_id") is None or obligation.get("amount_due") is None or obligation.get("maturity_tau") is None:
                raise ValueError("invalid plan settlement input schema")

    events: list[SettlementEvent] = []
    unresolved: list[dict[str, Any]] = []
    settlement_failed = False
    settled_amount = 0.0
    cash_paid_by_dst: dict[str, float] = {}

    ledger = getattr(state, "claim_ledger", None)
    process_id = getattr(state, "process_id", None)
    generation_id = int(getattr(state, "generation_id", 0))

    ordered_obligations: list[tuple[str, dict[str, Any]]] = []

    # Preserve existing ordering for obligations without dst-channel metadata.
    for obligation in due_obligations:
        if obligation.get("dst_channel") is not None:
            continue
        ordered_obligations.append((_dst_bucket(obligation), obligation))

    # Apply dst-channel partitioning deterministically for all obligations that carry dst metadata.
    grouped_dst_obligations: dict[str, list[dict[str, Any]]] = {}
    for obligation in due_obligations:
        if obligation.get("dst_channel") is None:
            continue
        key = _dst_bucket(obligation)
        grouped_dst_obligations.setdefault(key, []).append(obligation)

    ordered_flow_group_keys = sorted(grouped_dst_obligations.keys(), key=lambda item: int(item))
    for group_key in ordered_flow_group_keys:
        rows = sorted(
            grouped_dst_obligations[group_key],
            key=lambda item: (
                str(item.get("claim_id") or ""),
                str(item.get("kind") or ""),
                int(item.get("due_time", tau)),
                str(item.get("edge_id") or ""),
            ),
        )
        for obligation in rows:
            ordered_obligations.append((group_key, obligation))

    for group_key, obligation in ordered_obligations:
            amount_due = float(obligation.get("amount_due", 0.0))
            claim_id = obligation.get("claim_id")
            force_reject = bool(obligation.get("force_reject", False))
            maturity_tau = int(obligation.get("maturity_tau", obligation.get("due_time", tau)))

            # Non-due claims must not be mutated by settlement.
            if claim_id is not None and int(tau) < maturity_tau:
                continue

            if amount_due <= 0.0:
                continue

            available_cash = max(0.0, float(getattr(state, "wealth", 0.0)))
            cash_part = min(available_cash, lambda_cash_share * amount_due)
            remainder = max(0.0, amount_due - cash_part)

            proposal_status = SettlementStatus.PROPOSED
            if force_reject:
                accepted = False
            else:
                accepted = bool(accept_by_default)

            if accepted and remainder > 0.0 and (ledger is None or process_id is None or claim_id is None):
                accepted = False

            new_claim_ids: list[str] = []

            if accepted and remainder > 0.0:
                try:
                    rewritten = ledger.rewrite_claim(
                        claim_id=str(claim_id),
                        generation_id=generation_id,
                        closed_at=int(tau),
                        nominal=float(remainder),
                        maturity_tau=int(tau) + maturity_offset,
                    )
                    new_claim_ids.append(rewritten.claim_id)
                    parent_meta = claim_channel_meta.get(str(claim_id), {})
                    if "src_channel" in parent_meta and "dst_channel" in parent_meta:
                        claim_channel_meta[str(rewritten.claim_id)] = {
                            "src_channel": int(parent_meta["src_channel"]),
                            "dst_channel": int(parent_meta["dst_channel"]),
                            "edge_id": str(parent_meta.get("edge_id", "")),
                        }
                except ClaimCapacityExceeded:
                    accepted = False

            if accepted:
                proposal_status = SettlementStatus.ACCEPTED
                state.wealth = float(state.wealth) - float(cash_part)
                settled_amount += float(cash_part)
                cash_paid_by_dst[group_key] = cash_paid_by_dst.get(group_key, 0.0) + float(cash_part)

                if remainder <= 0.0 and claim_id is not None and ledger is not None:
                    ledger.close_claim(claim_id=str(claim_id), closed_at=int(tau), status="consumed")

                events.append(
                    SettlementEvent(
                        claim_id=None if claim_id is None else str(claim_id),
                        status=proposal_status,
                        cash_paid=float(cash_part),
                        new_claim_ids=tuple(new_claim_ids),
                        reason=None,
                    )
                )
                continue

            proposal_status = SettlementStatus.REJECTED
            if float(getattr(state, "wealth", 0.0)) >= amount_due:
                state.wealth = float(state.wealth) - amount_due
                settled_amount += amount_due
                cash_paid_by_dst[group_key] = cash_paid_by_dst.get(group_key, 0.0) + float(amount_due)
                if claim_id is not None and ledger is not None:
                    ledger.close_claim(claim_id=str(claim_id), closed_at=int(tau), status="consumed")
                events.append(
                    SettlementEvent(
                        claim_id=None if claim_id is None else str(claim_id),
                        status=proposal_status,
                        cash_paid=float(amount_due),
                        new_claim_ids=tuple(),
                        reason="rejected_but_paid_full_cash",
                    )
                )
            else:
                settlement_failed = True
                unresolved.append(obligation)
                events.append(
                    SettlementEvent(
                        claim_id=None if claim_id is None else str(claim_id),
                        status=proposal_status,
                        cash_paid=0.0,
                        new_claim_ids=tuple(),
                        reason="rejected_and_insufficient_cash",
                    )
                )

    state._last_settlement_failed = bool(settlement_failed)
    state._last_settlement_events = events
    obligations_grouped_by_dst = _group_obligations_by_dst(unresolved)
    claim_ledger_grouped_by_dst = _group_open_claim_ledger_by_dst(state)
    return state, {getattr(state, "process_id", 0): bool(settlement_failed)}, {
        "obligations_after": unresolved,
        "settled_amount": float(settled_amount),
        "settlement_failed": bool(settlement_failed),
        "events": events,
        "cash_paid_by_dst_channel": cash_paid_by_dst,
        "obligations_grouped_by_dst_channel": obligations_grouped_by_dst,
        "claim_ledger_grouped_by_dst_channel": claim_ledger_grouped_by_dst,
    }
