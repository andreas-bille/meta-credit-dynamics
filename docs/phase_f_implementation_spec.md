# Phase F – Implementation Spec (Prod Readiness) v0

Scope: Implement `math-v1.md` **Profile A – Canonical Inhibition Mode** end-to-end,
with deterministic tests and a clean external interface boundary (`interface.md`).

Non-scope (v1): `math-v1.md` Part II (Equity/Debt, credit condition, pool finance, sparsity masks as active mechanism)
=> explicitly **inactive** in Profile A.

---

## 0. Baseline & Invariants (no behavior change)

### Goal
Establish a deterministic test harness and a minimal invariant layer,
without changing runtime behavior.

### Work items
- Add `pytest` as primary test runner (already in requirements).
- Create `tests/` structure:
  - `tests/test_invariants.py`
  - `tests/test_reweighting.py`
  - `tests/test_stats.py`
  - `tests/test_rebirth.py`
  - `tests/test_freeze.py`
  - `tests/test_sediment_filter.py`
- Add a `Config` object for Profile A defaults (single source of truth):
  - `capitalmarket/capitalselector/config.py` (or `profiles.py`)
- Encode Profile A decisions: D1, D3, D5, D7, D8, D11, D13 inactive, D14 not implemented in v1, D15 disabled.

### Acceptance tests
- **T0.1**: `pytest` runs in < 5s and passes (initially only invariant tests).
- **T0.2**: Simplex invariant: `w` is non-negative and sums to 1 after each `stack_step()`.
- **T0.3**: Determinism: given fixed seeds in channels, two runs produce identical traces (within exact equality for CPU reference).

---

## 1. Canonical Net Flow (`Π`) and Cost Handling (Profile A)

### Goal
Make net-flow (`Π`) the canonical scalar signal and align cost semantics with `math-v1.md` §2.2 and Profile A.

### Work items
- Ensure the engine computes:
  - `R_i(t) = Σ_k w_ik r_k`
  - `C_i(t)` as scalar total cost
  - `Π_i(t) = R_i(t) - C_i(t)`
- In current code, `feedback_vector(r_vec, c)` already receives:
  - `r_vec`: **weighted** per-channel gross contributions (because `Channel.step(weight)` returns weighted payout)
  - `c`: scalar total cost
- Implement canonical per-channel net contribution for scoring:
  - `pi_vec_k = r_vec_k - w_k * C_total` (since `Σ_k w_k = 1` in Profile A)
- Add a helper in `CapitalSelector`:
  - `compute_pi(r_vec, c_total) -> (R, C, Pi, pi_vec)`

### Acceptance tests
- **T1.1**: If all channels return zero and costs are zero ⇒ `Π=0`.
- **T1.2**: If costs are positive and returns zero ⇒ `Π<0`.
- **T1.3**: `Σ_k pi_vec_k == Π` (within small float tolerance).

---

## 2. Statistics on Net Flow + Drawdown (Profile A, §6)

### Goal
Implement `μ, σ²` on **Π** and add `DD` (drawdown) exactly per `math-v1.md` Profile A.

### Work items
- Extend `EWMAStats` to track:
  - `mu`, `var` (EWMA of Π)
  - `dd` (drawdown)
  - minimal internal state for DD (canonical option):
    - `cum_pi` and `peak_cum_pi`, then `dd = peak - cum`
- Implement:
  - `update(pi: float)` updates mu/var/dd
- Update callers:
  - `CapitalSelector.feedback_vector(...)` updates stats with **Π**, not gross `r`.

### Acceptance tests
- **T2.1**: EWMA sanity: constant Π input converges `mu→Π` and `var→0`.
- **T2.2**: DD monotonicity: for strictly positive Π sequence, `dd==0` always.
- **T2.3**: DD increases correctly on negative streak after a peak.

Implementation note:
The cumulative-profit / peak formulation of drawdown
is mathematically equivalent to the definition in math-v1.md §6.

---

## 3. Reweighting Score = Net Contribution minus μ (Profile A, §4, D5)

### Goal
Implement Profile A score:
`s_ik(t) = pi_ik(t) - μ_i(t)`
and ensure exponentiated-gradient update stays on simplex.

### Work items
- In `reweight.py` (or where `exp_reweight` is):
  - accept `score_vec` explicitly (already likely supported)
- In `CapitalSelector`:
  - build `score_vec = pi_vec - stats.mu`
  - run EG update to produce `w(t+1)`
- Keep Profile A semantics:
  - no ETF baseline (D4.A)
  - no activity-mix alpha (not implemented in v1)

### Acceptance tests
- **T3.1**: After reweight, `w` is on simplex (sum=1, all ≥0).
- **T3.2**: If all scores equal ⇒ weights unchanged (or only numerical epsilon change).
- **T3.3**: If one channel dominates score repeatedly ⇒ its weight increases (monotonic trend).

---

## 4. Stack Weights (D8) = Equal Weighting (Profile A, §7)

### Goal
Make stack weighting explicitly `α_i = 1/|S|` for stacks in Profile A, matching `math-v1.md`.

### Work items
- Ensure `StackChannel.step(weight)` uses equal internal distribution among members (currently appears so).
- Make it explicit in `StackChannel` / `StackManager` config:
  - `stack_weighting="equal"` as Profile-A default
- Ensure stability predicate uses the canonical stats thresholds (already in StackConfig; align naming with `τ_μ, τ_σ, τ_dd`).

### Acceptance tests
- **T4.1**: For a stack with N identical members, returns equal to average of member returns.
- **T4.2**: Stack stability flips when any configured threshold is violated (μ low, σ high, dd high).

Note:
CVaR-based stability metrics, if present in the codebase,
are non-canonical and must be disabled in Profile A.

---

## 5. Canonical Update Loop Ordering (Architecture §5)

### Goal
Enforce canonical ordering and remove ambiguity:
**Rebirth → Invariants** (in that order).

### Work items
- The effective update semantics must follow the canonical update order:

  1) Observe  
  2) Compute Π  
  3) Update stats  
  4) Evaluate stability  
  5) Topology maintenance + sediment insertion  
  6) Reweight  
  7) Rebirth (if triggered)  
  8) Invariant enforcement

- A single `engine_step` API is optional and not required in v1.
- Add an internal trace/log event list (optional) to test ordering deterministically.

### Acceptance tests
- **T5.1**: If rebirth triggers, invariants are still enforced after the reset.
- **T5.2**: Ordering trace matches the canonical sequence for one step.

---

## 6. Rebirth = BehavioralState Reset (D10.A, Profile A, §9)

### Goal
Implement D10.A precisely:
Reset **only** BehavioralState = (`w`, `μ`, `σ²`, `DD`), not topology/sediment/environment.
Wealth is clamped, not reset.

### Work items
- Introduce a `BehavioralState` container (dataclass) OR define reset methods on `CapitalSelector` and `EWMAStats`.
- Implement `CapitalSelector.rebirth()` to:
  - reset `w` → uniform simplex
  - reset stats → `mu=0, var=seed_var, dd=0` (+ reset dd internals)
  - clamp wealth (and equity if present later)
- Ensure stack/sediment objects are untouched.

### Acceptance tests
- **T6.1**: After rebirth, `w` is uniform.
- **T6.2**: After rebirth, stats are reset to initial values.
- **T6.3**: Sediment node count unchanged by rebirth; stack topology unchanged by rebirth.
- **T6.4**: Wealth after rebirth is `>= τ_rebirth`.

---

## 7. Freeze Mode (Profile A strict, §10 + Architecture §9)

### Goal
Implement Freeze semantics per Profile A:
**no state change** (weights, stats, topology, sediment insertion).

### Work items
- Add `freeze: bool` to runtime config / step call.
- Gate:
  - no reweighting
  - no stats update
  - no stack dissolve/form
  - no sediment insertion
  - no rebirth
- Provide Profile-B hooks later (weights freeze but stats update), but Profile A is strict.

### Acceptance tests
- **T7.1**: With Freeze on, running 100 steps leaves full internal state observationally equal.
- **T7.2**: Random state must be frozen or excluded from comparison.
- **T7.3**: With Freeze on, sediment stays constant and topology stays constant.

---

## 8. Sediment Causality vs Exclusion Rule (Architecture §6)

### Goal
Make the “no causal influence” claim precise:
Sediment does **not** influence weights/stats/wealth,
but it **does** act as a **structural filter** when forming stacks.

### Work items
- Ensure `SedimentDAG.is_forbidden(...)` is only called from stack formation logic.
- Document this explicitly in code docstrings and in `sediment.py`.

### Acceptance tests
- **T8.1**: Changing sediment nodes does not change reweighting outcome in a topology-free run.
- **T8.2**: A forbidden member/phase combination cannot be formed as a stack (stack formation rejects it).

---

## 9. External Boundary: World / Curriculum / Teacher (interface.md)

### Goal
Implement the external control surface:
World provides exogenous signals, curriculum selects worlds, teacher selects profile/mode/config.
No intervention on internal state.

### Work items
- Add interfaces/protocols:
  - `World.step(t) -> { r: array[float], c: float }`
  - `Curriculum.next(t) -> World`
  - `Teacher.configure(run_id, profile, mode, params)`
- Clarify “Not allowed: adaptive intervention” as:
  - **No state intervention** (curriculum may be parameter-adaptive but state-agnostic).
- Ensure the internal engine never passes internal state to curriculum/teacher.

### Acceptance tests
- **T9.1**: World has no access path to weights/stats/topology (by API shape).
- **T9.2**: Curriculum can vary parameters over time but cannot read internal state (no reference passed).
- **T9.3**: Teacher config changes only at run start (or explicit safe boundary), not per-step.

---

## 10. Packaging: Profile A as the Required Canonical Mode

### Goal
Make Profile A a first-class runtime choice with a single entry point:
`run(profile="A", freeze=False, mode=...)`.

### Work items
- Implement `ProfileAConfig` (defaults aligned to `math-v1.md` §18).
- Ensure docs reflect that Profile A is required for Prod-v0; Profile B optional.

### Acceptance tests
- **T10.1**: A run in Profile A uses Π-based stats and net-score reweighting (validated via trace hooks).
- **T10.2**: Switching to Profile B (if implemented) is explicit and labeled non-canonical.

---

## 11. toCuda() Readiness (Phase F.5 hook)

### Goal
Prepare the state layout for a deterministic `toCuda()` projection later:
clear ownership, contiguous arrays, no hidden Python-only behavior.

### Work items
- Ensure all BehavioralState pieces are representable as tensors:
  - `w` (float32/float64)
  - `mu`, `var`, `dd` (scalars)
- Add `state()` methods that return strictly typed numeric containers.

### Acceptance tests
- **T11.1**: `state()` output contains all fields needed to reconstruct BehavioralState.
- **T11.2**: Roundtrip CPU state -> serialization -> reconstruction yields identical behavior for one step.

---

## Appendix: Concrete Code Touchpoints (current repo)

Observed key files:
- `capitalmarket/capitalselector/core.py` (CapitalSelector, step loop, feedback)
- `capitalmarket/capitalselector/stats.py` (EWMAStats)
- `capitalmarket/capitalselector/reweight.py` (EG reweight)
- `capitalmarket/capitalselector/stack.py` (StackChannel, StackManager)
- `capitalmarket/capitalselector/sediment.py` (SedimentDAG)
- `capitalmarket/capitalselector/channels.py` (test channels, RNG seeding)

Primary deltas vs Profile A:
- Stats currently update on gross `r`, not `Π`
- Drawdown missing
- Reweight score ignores costs
- Rebirth resets weights but not stats/DD
- Freeze mode not implemented
- External boundary exists as docs but not as explicit interfaces
