# Phase G – Concept v0

Status: Draft / Conceptual

Goal: extend Phase F with **non-canonical exploration**, **GPU readiness**, and **empirical validation**,
without weakening the canonical Profile A baseline.

Axis framing:
1) **Non-canonical analysis (Profile B)**
2) **Representation / compute projection (toCuda)**

Principle: **no new semantics** introduced in Phase G.

Scope summary (recommended order, fixed):
- **G1**: Profile B (non-canonical) on CPU only.
- **G2**: `toCuda()` skeleton (state projection + round-trip tests).
- **G3**: empirical study phase (what emerges, governance patterns, failure modes).
- **G4**: CUDA compute for core operations (reweight, stats, drawdown, rebirth-mask).

Non-goals (v0):
- changing Profile A semantics
- adding new learning objectives
- using sediment data causally

---

## G1 – Profile B (non-canonical, CPU)

**Intent:** define an explicit, documented **non-canonical** configuration bundle for analysis and research.

Deliverables:
- Profile B config bundle (clearly flagged non-canonical).
- Tests for Profile B invariants and deltas vs Profile A.
- Documentation of allowed deviations.

Key decisions:
- Which deviations are allowed (scores, stats, stack weighting, freeze behavior).
- Which invariants remain hard (sediment non-causal, no reward/target, no external intervention).

Minimal scope (B0):
- Freeze may keep **weights fixed** while **stats continue**. Freeze semantics in Profile B are explicitly experimental and must be logged per run.
- No topology changes during Freeze.
- No sediment insertion during Freeze.

Acceptance criteria:
- Profile B is runnable and explicitly labeled non-canonical.
- Tests for Profile B pass independently from Profile A.
- Docs clearly state **non-comparability** to Profile A results.

---

## G2 – toCuda() Skeleton (no compute)

**Intent:** define the projection contract and round-trip safety before any GPU logic.

Deliverables:
- `toCuda()` state projection (CPU → GPU tensors).
- No-op kernel (identity / copy) or stub execution.
- Round-trip tests (CPU → GPU → CPU preserves observational state). Observational state = all fields exposed via state() and relevant for next-step dynamics.
- Introduce an explicit `TensorState` container for GPU projection.

Acceptance criteria:
- Round-trip equivalence on state fields.
- Deterministic tensor shapes/types.
- No semantic changes introduced.

---

## G3 – Empirical Phase (learning & governance)

**Intent:** investigate actual dynamics and governance patterns under controlled conditions.

Questions:
- What does the system learn empirically?
- Which governance patterns emerge?
- Where are the failure modes or unintended incentives?

Deliverables:
- Minimal world/curriculum setups for controlled experiments.
- Metric definitions (stability, sediment dynamics, stack churn). Metrics are descriptive only and must not be reused as control signals.
- Structured report of observed behaviors.
- Observation axes: stability, churn, sediment growth, weight diversity (entropy).

Example worlds:
1) Stationary i.i.d. returns
2) Regime shifts
3) Exogenous shocks (state-agnostic)

Acceptance criteria:
- Repeatable experiments with logged seeds and inputs.
- Clear interpretation boundaries (Profile A vs B).
- Findings documented without overclaiming.

Non-goal:
- proving optimality
- intelligence
- convergence.

---

## G4 – CUDA Compute (core ops)

**Intent:** move canonical operations to GPU with **exact CPU semantics**.

Targets:
- Reweighting (EG update).
- Stats (EWMA on Π).
- Drawdown.
- Rebirth-gate / freeze semantics.

Recommended parity sequence:
1) Reweighting
2) Stats
3) Drawdown
4) Rebirth

Acceptance criteria:
- GPU results match CPU within defined tolerances.
- Determinism in test mode.
- Profile A semantics unchanged.

---

## Ordering (fixed)

The order is **prescribed**: **G1 → G2 → G3 → G4**.  
Reason: semantics must be stable before GPU work, and empirical results must be interpretable
within a clearly defined (non-canonical) Profile B frame.

---

## Risks / Open Questions

- Profile B scope creep (must remain explicitly non-canonical).
- GPU semantic drift vs CPU reference.
- Empirical phase interpretations depend on world design quality.

---

## Exit Criteria (Phase G complete)

- Profile B documented, tested, and isolated.
- `toCuda()` contract verified by round-trip tests.
- GPU core ops validated against CPU.
- Empirical report with reproducible runs and clear governance findings.
- No governance claims are made beyond the observed experimental scope.

---

Leitmotiv:
Not: "What can the system achieve?"
But: "What survives — and why?"
