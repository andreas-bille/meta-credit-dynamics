# Phase G – Implementation Spec G2 (toCuda Skeleton)

Status: Draft (implementation-facing)

Goal: implement **G2** only: a **representation change** from canonical Python/CPU state to a CUDA-capable tensor state,
with **round-trip correctness tests**. **No new semantics** may be introduced.

---

## Scope

Implement:

- A complete **state inventory** for the canonical kernel state that must be representable on GPU.
- A passive **CudaState** (or equivalent) that holds tensor representations of the canonical state.
- `toCuda()` projection (**CPU/Python → CUDA tensors**) and `fromCuda()` back-projection (**CUDA tensors → CPU/Python**).
- Determinism and round-trip equivalence tests on CPU and GPU devices.

This G2 spec is intentionally **minimal and mechanical**.

---

## Normative sources (contract)

- `docs/math-v1.md` (canonical semantics; no changes allowed)
- `docs/architecture.md` (§7 compute architecture, §7.2 toCuda() contract; and any relevant invariants)
- `docs/interface.md` (external control surface; must not change)
- `docs/phase_g_concept_v0.md` (Phase G order and non-goals)
- `docs/phase_g_concept_g1_v0.md` and `docs/phase_g_implementation_spec_g1.md` (G1 boundary conditions: profile isolation, kernel remains profile-agnostic)

---

## Non-scope (explicit)

- Any semantic changes to the update loop, invariants, reweighting, stability logic, rebirth reset semantics, freeze semantics.
- Any new state variables or new metrics.
- Any CUDA compute kernels for core operations (reserved for G4).
- Any empirical study / governance experiments (reserved for G3).
- Any changes to `docs/architecture.md` or `docs/interface.md`.

---

## Definitions

- **Canonical state**: the current CPU/Python representation used by Profile A (and Profile B) execution.
- **Cuda state**: a tensor representation of the canonical state on a target device (typically `cuda`),
  plus immutable structural metadata required for reconstruction.
- **Representation change**: converting between CPU/Python objects and device tensors **without changing meaning**.

---

## G2 Work Items

### G2.1 State Inventory (canonical)
Create/maintain a single authoritative list of state elements that must be representable on GPU.

For each element, specify:
- name
- owner (Unit / CapitalSelector / Stack / Manager)
- type (Python scalar / NumPy array / torch tensor)
- shape
- dtype
- device (CPU by default)
- invariants (e.g., simplex, non-negativity)

> Constraint: **Do not introduce new state.** The inventory is descriptive only.

**Deliverable:** `docs/phase_g_g2_state_inventory.md` plus code comments near the state definition.

---

### G2.2 CudaState (passive container)
Introduce a passive `CudaState` (or equivalent) that contains tensors plus **immutable structural metadata**
required to reconstruct the canonical state. It must not contain kernel logic.

**Rules**
- No update methods.
- No learning logic.
- No profile identity.
- Must preserve canonical shapes and invariants.
- Non-tensor structural metadata is allowed **only** if required to reconstruct canonical state
  (e.g., topology descriptors, ids, shapes). It must be passive and non-causal.
- Disallowed metadata: any callable, any object with behavior (methods), any RNG state,
  any profile identity, any kernel references.

---

### G2.3 toCuda() projection (CPU/Python → CUDA)
Implement `toCuda(state, device, dtype=None)` (exact placement per codebase conventions) that:
- converts canonical CPU/Python state into `CudaState` on `device`
- uses explicit dtypes
- validates shapes
- preserves invariants (simplex, non-negativity) **by construction**, not by “repair logic”

**Rules**
- No stochasticity.
- No semantic changes.
- No mutation of the input state unless explicitly documented as safe.
- Default dtype policy: **no dtype change**. If `dtype` is omitted, preserve canonical dtype.
- Any dtype casting must be explicit and documented, with tolerances adjusted accordingly.

---

### G2.4 fromCuda() back-projection (CUDA → CPU/Python)
Implement `fromCuda(cuda_state)` that reconstructs the canonical CPU/Python state (or a byte-identical equivalent)
from `CudaState`.

**Rules**
- No semantic changes.
- Deterministic.
- Must reconstruct all inventoried elements.

---

### G2.5 Guardrails (anti-drift)
Add guardrails to prevent accidental semantic creep:
- `toCuda()`/`fromCuda()` must not call update/reweight/stability/rebirth logic.
- `CudaState` must remain a passive data container.
- No profile information may be stored in `CudaState` or returned kernel instances.

---

### G2.6 Minimal Documentation
Update internal developer notes only if strictly necessary (no canonical doc edits in G2).
This spec + state inventory must be sufficient.

---

## Acceptance Tests (Minimum)

### T1 – Round-trip identity (CPU → CUDA → CPU)
For a representative canonical state `S`:
- `S2 = fromCuda(toCuda(S, device="cuda"))`
- `S2` must be observationally equal to `S` within declared numeric tolerances.

Observational equality is defined by a single dump function
`canonical_state_dump()` (or `state()` if it already contains all required fields)
that includes **all fields required for next-step dynamics**.

If `state()` is incomplete, introduce `canonical_state_dump()` specifically for
test equivalence and next-step sufficiency.

Equality must cover:
- weights simplex
- statistics (μ, σ², DD and any canonical seeds)
- topology encodings (if represented)
- sediment representation (structural, non-causal)

Non-tensor structural metadata required for reconstruction is part of the dump and must match exactly.

For non-float fields (ids, shapes, adjacency indices, topology descriptors),
equality is **exact**, not tolerant.

RNG state is **out of scope** for G2 and must not be part of the round-trip dump.

---

### T2 – Shape & dtype preservation
All inventoried elements must preserve:
- shape
- dtype
- value semantics (within tolerance)

---

### T3 – No semantic calls
Instrumented test ensures `toCuda()` and `fromCuda()` do **not** call:
- reweighting
- stability evaluation
- topology maintenance
- rebirth reset
- freeze logic

(Implementation options: monkeypatch hooks or call counters on
`reweight_fn`, `rebirth`, `StackManager.maintain`, `StackChannel.stable`,
or equivalent call sites.)

---

### T4 – Profile agnosticism
`toCuda()`/`fromCuda()` must not branch on profiles and must not store profile identity anywhere.

---

### T5 – Determinism parity (within tolerance)
With fixed seeds and deterministic world inputs:

1) Run CPU for `t → t+1` from state `S` and record outputs/state `S1`.
2) Compute `S_rt = fromCuda(toCuda(S, device="cuda"))`.
3) Run CPU for `t → t+1` from `S_rt` and record outputs/state `S1_rt`.

Then `S1` and `S1_rt` must match within tolerance.

This tests that round-tripping does not alter outcomes, without requiring CUDA compute.

---

### CUDA availability in tests
CUDA-dependent tests must be optional and skip when CUDA is unavailable, e.g.:
`pytest.mark.skipif(not torch.cuda.is_available(), ...)`.

---

## Numeric Tolerances

Define tolerances explicitly (default guidance):
- float32 tensors: `rtol=1e-6`, `atol=1e-7`
- float64 tensors: `rtol=1e-12`, `atol=1e-12`

Tolerance values must be chosen based on current canonical dtypes and should be kept minimal.
Use explicit `rtol/atol` in `torch.testing.assert_close` or `np.testing.assert_allclose` per field.

---

## Implementation Notes (minimal-invasiveness)

- Prefer extending existing state/config objects rather than inventing parallel hierarchies.
- Keep file diffs small; refactors only if strictly required for tests.
- `CudaState` should mirror canonical state layout to keep mapping obvious.
- All conversion code must be explicit about dtype/device to avoid silent casts.

---

## Exit Criteria (G2 complete)

G2 is complete when:

- A complete state inventory exists and matches the actual canonical state.
- `CudaState`, `toCuda()`, and `fromCuda()` exist and are passive/mechanical.
- All acceptance tests (T1–T4) pass on CPU and GPU.
- Determinism parity test (T5) passes or is explicitly waived with rationale (and tracked).

---

## Appendix A – State Inventory (placeholder)

> Fill this section after implementing G2.1 (or link to `docs/phase_g_g2_state_inventory.md`).

| Element | Owner | Type | Shape | DType | Invariants |
|---|---|---|---|---|---|
| weights `w` | Unit/CapitalSelector | tensor/ndarray | (K,) | float32/float64 | simplex, >=0 |
| stats `μ` | EWMAStats | scalar | () | float32/float64 | finite |
| stats `σ²` | EWMAStats | scalar | () | float32/float64 | >=0 |
| drawdown `DD` | EWMAStats | scalar | () | float32/float64 | >=0 |
| ... | ... | ... | ... | ... | ... |
