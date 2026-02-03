# Phase E – Tech Specification (v0)

**Project:** GAP / Economy of Neurons
**Phase:** E – AI in the Sandbox
**Branch:** `phases/phase_e_gap_in_the_sandbox`
**Purpose:** Minimal, binding technical decisions required to implement Phase E **without changing the model**.

This document complements:

* `phase_e_implementation_specification_v1.md` (conceptual / behavioural spec)

---

## 0. Non-Negotiables

1. **World ≠ Sediment**: the World is an adapter (Mazeworld, Binomialworld, Drivingworld, …). Sediment is not the World.
2. **Sediment is passive**: no agent reads the Sediment-DAG. Sediment only constrains formation implicitly via the StackManager.
3. **Teacher is external**: Teacher is not part of Stack/System state.
4. **Rebirth is structural**: Rebirth resets *behavioural* state, not Sediment history.

---

## 1. Sediment-DAG

### 1.1 Purpose

Sediment-DAG records **dead Broker mediation configurations** as compressed history.

It provides a single service to the system:

* `is_forbidden(candidate_config) -> bool` (hard filter, v0)

No other access is permitted in Phase E v0.

---

### 1.2 Node schema (v0)

A Sediment node represents a **failed mediation configuration** at the moment of dissolution.

**Node fields (minimal, binding):**

* `node_id: int`
  Monotone per run (starts at 1).

* `members: list[str]`
  Stable identifiers of the bound units (StackChannels / unit ids). Sorted.

* `mask: dict`
  Minimal masking signature, v0:

  * `masked_members: list[str]` (sorted)
  * `mask_depth: int` (optional; 0 if unknown)

* `world_id: str`
  Teacher-provided label (e.g. `maze_v1`, `binomial_v0`).

* `phase_id: str`
  Teacher-provided label (e.g. `E1`, `E2`).

* `t: int`
  Global step counter within the run (Telemetry step index).

* `run_id: str`
  Correlation id of the run/episode.

**Explicitly excluded (v0):**

* learned weights
* rewards
* world coordinates
* internal broker parameters

Rationale: keep the fingerprint structural, not behavioural.

---

### 1.3 Edge schema (v0)

An edge expresses **temporal ordering of failures**.

**Edge rule (binding):**

* For nodes `a`, `b` within the same `run_id`: add directed edge `a -> b` iff `a.t < b.t`.

**Implementation note (v0 simplification):**

* Maintain `last_node_id_by_run_id` and add a single chain edge per insertion:

  * when adding node `b`, add edge `(last_node_id -> b)` if it exists.

This guarantees a DAG in v0.

---

### 1.4 Persistence format

Persist as **append-only JSONL**:

* path: `logs/sediment.jsonl`

Each line is one of:

* `{"event":"SEDIMENT_NODE_ADDED", "payload": <node>}`
* `{"event":"SEDIMENT_EDGE_ADDED", "payload": {"from": int, "to": int, "run_id": str, "t": int}}`

Monotone ids are guaranteed per process run.

---

## 2. Matching / Filter Rules (v0)

### 2.1 Primary rule (hard forbid)

A candidate configuration is forbidden if:

* `set(candidate.members) == set(node.members)` for any node in the DAG **within the same phase_id**.

(Phase-local forbidding avoids permanently banning structures across curriculum phases in v0.)

---

### 2.2 Optional rule (pair forbids) – feature flag

Optional additional constraint behind feature flag `SEDIMENT_FORBID_PAIRS`:

* For any sediment node with member set `M`, forbid any candidate that contains any 2-subset from `M`.

This is a conservative approximation of "bad cliques".

Default: **OFF**.

---

### 2.3 No soft penalties in v0

Phase E v0 uses **hard rejection**, not cost shaping.

(Soft penalties may be introduced only in v1+, after E1–E3 invariants are validated.)

---

## 3. Lifecycle Events and Hooks

### 3.1 Definition: “Broker death” vs “Stack dissolve”

For Phase E implementation, **Sediment nodes are created on Stack dissolution**, not on individual broker objects.

Reason: dissolution is the observable boundary where a mediation configuration ceases to be operational.

---

### 3.2 Hook point A – Dissolution

**Hook location (binding):**

* In `StackManager` (or the equivalent owner of stack lifecycle): at the point where a stack is dissolved / disbanded.

**Actions (in order):**

1. Compute `candidate_config` fingerprint (members + mask signature).
2. Emit telemetry event `STACK_DISSOLVED` including the fingerprint.
3. Call `SedimentDAG.add_node(candidate_config)`.
4. Call `SedimentDAG.add_edge(last_node, new_node)` via chain rule.

---

### 3.3 Hook point B – Formation attempt

**Hook location (binding):**

* In `StackManager.try_form_stack()` (or equivalent): before finalising a new stack.

**Action:**

* Compute candidate fingerprint.
* If `SedimentDAG.is_forbidden(candidate)` then reject formation (return false / no-op).

No logging beyond `SEDIMENT_FORMATION_REJECTED` telemetry.

---

## 4. Teacher / Curriculum API (Minimal)

### 4.1 Teacher responsibilities (binding)

Teacher provides per run:

* `world_id`
* `phase_id`
* selection pressure configuration
* rebirth trigger

Teacher does **not** provide:

* guidance signals
* rewards
* gradients
* internal state

---

### 4.2 Interface sketch (conceptual)

Teacher is implemented as a thin driver layer (script / notebook / test harness):

* `start_phase(phase_id, world_id, params)`
* `run_episode(n_steps)`
* `should_rebirth(metrics) -> bool`
* `rebirth()`

Teacher can swap worlds between phases.

---

## 5. Rebirth Semantics (Sediment-Aware)

### 5.1 Reset Matrix (binding)

On Rebirth:

**Reset (behavioural state):**

* active stacks
* active brokers
* budgets / capital of living brokers
* selector / routing weights (if present)

**Preserve (structural history):**

* `SedimentDAG` in memory
* `sediment.jsonl` log
* curriculum phase progression state (Teacher-owned)

**World reset:**

* Teacher-defined (may reset or continue world depending on phase)

---

### 5.2 Implementation approach (v0)

Introduce `SedimentAwareRebirthPolicy`:

* delegates to existing `RebirthPolicy` for behavioural reset
* explicitly keeps a reference to `SedimentDAG` untouched

---

## 6. Telemetry (Events)

Add telemetry events (names binding):

* `SEDIMENT_NODE_ADDED`
* `SEDIMENT_EDGE_ADDED`
* `SEDIMENT_FORMATION_REJECTED`
* `REBIRTH`
* `REBIRTH_PHASE_START`
* `REBIRTH_PHASE_END`
* `STACK_DISSOLVED` (with fingerprint)

Telemetry payloads must not include world coordinates unless already present in Phase D telemetry.

---

## 7. Tests / Invariants (Phase E v0)

### E1 – Monotone sediment growth

Given a run where stacks dissolve at least twice:

* sediment node count is monotone increasing
* node_id is monotone increasing
* JSONL lines append only

---

### E2 – Re-formation suppression

Scenario:

1. induce dissolution of stack configuration `C`
2. attempt to form `C` again

Expected:

* without Sediment filter: `C` can re-form (baseline)
* with Sediment filter (v0): `C` formation is rejected

---

### E3 – Canalisation (statistical shift)

Across multiple episodes in the same phase:

* frequency of `SEDIMENT_FORMATION_REJECTED` increases initially
* then stabilises as formation avoids already-failed configurations

No claim of performance improvement is required.

---

## 8. File / Module Plan (v0)

* `capitalmarket/capitalselector/sediment.py`

  * `class SedimentDAG`
  * `add_node(config) -> node_id`
  * `add_edge(a, b) -> None`
  * `is_forbidden(candidate) -> bool`

* integrate into:

  * `stack.py` / `StackManager`
  * `rebirth.py` (SedimentAwareRebirthPolicy)
  * `telemetry.py`
  * Phase E tests (new file or extend Phase D tests)

---

## 9. Deliberate omissions (v0)

Not in v0:

* multi-layer sediment decay
* cross-phase reactivation logic
* soft penalties instead of hard forbid
* higher-order matching beyond identical set / pair forbids
* world-specific fingerprints (coordinates)

These may be introduced only after v0 invariants hold.
