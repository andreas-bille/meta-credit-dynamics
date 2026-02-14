# Phase G – Implementation Spec G1 (Profile Bundles)

Scope: Implement **Profile B** as a non-canonical bundle while keeping the
canonical kernel profile-agnostic. This spec covers Phase G1 only.

Normative sources:
- `docs/math-v1.md` (§18, §19)
- `docs/architecture.md` (§10, Builder boundary)
- `docs/phase_g_concept_g1_v0.md`

Non-scope:
- CUDA / `toCuda()`
- empirical runs
- any new semantics or state variables

---

## 0. Goals

- Provide **closed configuration bundles** for Profile A and Profile B.
- Enforce **profile selection only at instantiation time** (builder boundary).
- Ensure the **canonical kernel is profile-agnostic**.
- Keep Profile A canonical and Profile B explicitly non-canonical.

---

## 1. Work Items

### 1.1 Profile Bundles
- Introduce `ProfileAConfig` and `ProfileBConfig` as closed bundles.
- Each bundle must fully resolve:
  - scoring signal selection
  - statistics accumulation rules
  - stack weighting strategy
  - freeze behavior
  - rebirth activation **triggers only** (semantics invariant)
- Bundles must not add state variables.

### 1.2 Builder Boundary
- Add a `KernelBuilder` (or equivalent constructor) that:
  - accepts exactly one profile
  - resolves all choices into a concrete runtime config
  - returns a fully instantiated kernel without profile identity

### 1.3 Canonical Kernel Isolation
- Ensure no profile-conditionals exist inside:
  - update equations
  - invariants
  - rebirth reset logic
  - drawdown logic

### 1.4 Explicit Non-Canonical Labeling
- Profile B must be explicitly labeled **non-canonical**:
  - in logs
  - in run metadata
  - in any exported reports

---

## 2. Acceptance Tests (Minimum)

### T1 – Profile Equivalence (Identical Parameters)
If Profile A and Profile B resolve to **identical parameter values**, then
two instances must produce **identical behavior** for the same inputs.

### T2 – No Profile Leakage
After instantiation:
- The kernel instance must expose **no profile identity**
- No profile field may be reachable via public attributes or state dumps
- There must be **no profile-conditional branch** in the kernel code path

### T3 – Rebirth Reset Invariance
Rebirth reset semantics must be **identical** across profiles:
- weights reset to uniform simplex
- stats reset to canonical seeds
- sediment / topology unchanged
Only **activation triggers** may differ by profile.

---

## 3. Additional Required Tests (Recommended)

### T4 – Canonical Invariants (Profile A)
Profile A must satisfy all canonical invariants from Phase F:
- simplex weights
- non-negativity
- freeze semantics (no state change)
- sediment non-causal

### T5 – Profile B Non-Canonical Labeling
Profile B runs must be flagged:
- run metadata includes `profile="B"` and `canonical=false`
- reports and logs include non-canonical disclaimer

### T6 – Closed Bundle Validation
Profile bundles must be **closed**:
- no missing parameters
- no optional fallback logic at runtime
- builder fails if unresolved fields remain

### T7 – Freeze Semantics Separation
If Profile B allows “stats continue during freeze”:
- weights remain fixed
- stats change only via Π updates
- no topology or sediment changes

### T8 – Determinism Parity
With fixed seeds and identical world inputs:
- Profile A and Profile B runs are deterministic within themselves
- Any differences are attributable **only** to profile-defined deltas

---

## 4. Implementation Notes

- Profile selection must be resolved **before** kernel execution.
- The kernel must not import profile-specific code paths.
- Profile B is non-canonical by definition and must be **explicitly** treated as such.

---

## 5. Exit Criteria

G1 is complete when:
- Profile A and Profile B are closed bundles
- A builder boundary enforces profile selection at instantiation
- All tests T1–T3 pass
- Recommended tests T4–T8 pass or are explicitly waived with rationale
