from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .claims import Offer
from .ledger import ClaimLedger


_EPS = 1e-12


class SelectorAccountingContractError(ValueError):
    """Raised when selector accounting invariants are violated deterministically."""


@dataclass(frozen=True)
class SelectorAccountingState:
    tau: int
    process_id: int | str
    generation_id: int
    liquidity: float
    wealth: float
    claim_ledger: ClaimLedger
    offers: list[Offer]
    dead_flag: bool
    tau_dead: int | None
    settlement_failed: bool


def selector_accounting_state_from_selector(selector: Any, *, tau: int) -> SelectorAccountingState:
    required = (
        "generation_id",
        "liquidity",
        "wealth",
        "claim_ledger",
        "offers",
        "dead",
        "dead_flag",
        "tau_dead",
        "_last_settlement_failed",
    )
    missing = [name for name in required if not hasattr(selector, name)]
    if missing:
        joined = ", ".join(sorted(missing))
        raise SelectorAccountingContractError(f"selector missing required accounting fields: {joined}")

    ledger = getattr(selector, "claim_ledger")
    if not isinstance(ledger, ClaimLedger):
        raise SelectorAccountingContractError("claim_ledger must be a ClaimLedger instance")

    offers = list(getattr(selector, "offers") or [])
    for idx, item in enumerate(offers):
        if not isinstance(item, Offer):
            raise SelectorAccountingContractError(f"offers[{idx}] must be Offer, got {type(item).__name__}")

    process_id = getattr(selector, "process_id", 0)
    dead_value = bool(getattr(selector, "dead"))
    dead_flag_value = bool(getattr(selector, "dead_flag"))
    if dead_value != dead_flag_value:
        raise SelectorAccountingContractError("selector dead/dead_flag mismatch")

    return SelectorAccountingState(
        tau=int(tau),
        process_id=process_id,
        generation_id=int(getattr(selector, "generation_id")),
        liquidity=float(getattr(selector, "liquidity")),
        wealth=float(getattr(selector, "wealth")),
        claim_ledger=ledger,
        offers=offers,
        dead_flag=dead_flag_value,
        tau_dead=getattr(selector, "tau_dead"),
        settlement_failed=bool(getattr(selector, "_last_settlement_failed")),
    )


def validate_selector_accounting_state(state: SelectorAccountingState) -> None:
    tau = int(state.tau)

    if state.liquidity < -_EPS and not bool(state.dead_flag):
        raise SelectorAccountingContractError(
            f"negative liquidity is not allowed while alive: {state.liquidity}"
        )

    if abs(float(state.wealth) - float(state.liquidity)) > 1e-9:
        raise SelectorAccountingContractError(
            "wealth consistency violated: wealth must equal liquidity in local accounting state"
        )

    implied_dead = bool(state.settlement_failed or float(state.wealth) < 0.0)
    if implied_dead and not bool(state.dead_flag):
        raise SelectorAccountingContractError("dead criterion violated: settlement failure/negative wealth requires dead_flag")
    if bool(state.dead_flag) and not implied_dead:
        raise SelectorAccountingContractError("dead criterion violated: dead_flag requires settlement failure or negative wealth")

    if state.tau_dead is None and bool(state.dead_flag):
        raise SelectorAccountingContractError("dead state violated: tau_dead must be set when dead_flag is True")
    if state.tau_dead is not None:
        tau_dead = int(state.tau_dead)
        if not bool(state.dead_flag):
            raise SelectorAccountingContractError("dead state violated: tau_dead set while dead_flag is False")
        if tau_dead > tau:
            raise SelectorAccountingContractError("dead state violated: tau_dead cannot lie in the future")

    for idx, offer in enumerate(state.offers):
        payload = dict(offer.payload or {})
        drawn_principal = float(payload.get("drawn_principal", 0.0))
        if drawn_principal < -_EPS:
            raise SelectorAccountingContractError(f"offers[{idx}] has negative drawn_principal")

        borrow_window_end_tau = int(payload.get("borrow_window_end_tau", tau))
        repayment_due_tau = int(payload.get("repayment_due_tau", borrow_window_end_tau + 1))
        if repayment_due_tau <= borrow_window_end_tau:
            raise SelectorAccountingContractError(
                "offer maturity ordering violated: repayment_due_tau must be > borrow_window_end_tau"
            )

    ledger = state.claim_ledger
    claims_for_process = list(ledger.claims_for_process(state.process_id))

    rewritten_notional_by_parent: dict[str, float] = {}
    for claim in claims_for_process:
        nominal = float(claim.nominal)
        if nominal < -_EPS:
            raise SelectorAccountingContractError(
                f"negative claim nominal is not allowed: claim_id={claim.claim_id}, nominal={nominal}"
            )

        status = str(ledger.get_status(claim.claim_id))
        if status == "open" and int(claim.maturity_tau) < tau:
            raise SelectorAccountingContractError(
                f"maturity ordering violated: open claim {claim.claim_id} has maturity_tau < tau"
            )

        if claim.claim_type == "repayment" and float(claim.drawn_principal) <= 0.0 and status == "open":
            raise SelectorAccountingContractError(
                f"repayment claim without draw is not allowed: claim_id={claim.claim_id}"
            )

        parent_id = claim.parent_claim_id
        if parent_id is None:
            continue

        parent_closed_at = ledger.get_closed_at(parent_id)
        if parent_closed_at is None:
            raise SelectorAccountingContractError(
                f"rewrite consistency violated: parent claim {parent_id} not closed before rewrite"
            )
        if int(claim.maturity_tau) <= int(parent_closed_at):
            raise SelectorAccountingContractError(
                "illegal maturity ordering in rewrite: rewritten claim maturity must be strictly after parent close"
            )

        rewritten_notional_by_parent[parent_id] = rewritten_notional_by_parent.get(parent_id, 0.0) + nominal

    for parent_id, rewritten_total in rewritten_notional_by_parent.items():
        parent = ledger.get_claim(parent_id)
        parent_nominal = float(parent.nominal)
        if rewritten_total > parent_nominal + 1e-9:
            raise SelectorAccountingContractError(
                "implicit money creation through rewrites is not allowed: "
                f"parent={parent_id}, rewritten_total={rewritten_total}, parent_nominal={parent_nominal}"
            )


def assert_selector_accounting_contract(selector: Any, *, tau: int, stage: str) -> None:
    try:
        state = selector_accounting_state_from_selector(selector, tau=int(tau))
        validate_selector_accounting_state(state)
    except SelectorAccountingContractError as exc:
        raise SelectorAccountingContractError(f"accounting contract violation ({stage}) at tau={int(tau)}: {exc}") from exc
