# Phase G – G2 Canonical State Inventory

Scope: canonical CPU/Python state that influences **next-step dynamics** and must be representable on GPU.
This inventory is **descriptive only** (no new state).

Note:
Only fields with dump_role ∈ {tensor, meta}
must be present in canonical_state_dump().

Fields marked as excluded are intentionally not serialized.

Legend:
- Type: Python scalar / NumPy array / torch tensor / other
- Device: CPU (canonical)
- Invariants: constraints that must hold

---

## CapitalSelector (core)

| Element | Owner | Type | Shape | DType | Device | Invariants | dump_role |
|---|---|---|---|---|---|---|---|
| `wealth` | CapitalSelector | Python scalar | () | float | CPU | finite | tensor |
| `rebirth_threshold` | CapitalSelector | Python scalar | () | float | CPU | finite | tensor |
| `stats` | CapitalSelector | object | n/a | n/a | CPU | EWMAStats instance (dump via stats.* fields) | meta |
| `reweight_fn` | CapitalSelector | callable | n/a | n/a | CPU | non-null | excluded |
| `kind` | CapitalSelector | Python scalar | () | str | CPU | non-empty | meta |
| `rebirth_policy` | CapitalSelector | object | n/a | n/a | CPU | optional | excluded |
| `channels` | CapitalSelector | list | (K) | object | CPU | length = K | excluded |
| `K` | CapitalSelector | Python scalar | () | int | CPU | K ≥ 0 | meta |
| `w` | CapitalSelector | NumPy array | (K,) | float | CPU | simplex, ≥0 | tensor |
| `_last_r` | CapitalSelector | Python scalar | () | float | CPU | finite | tensor |
| `_last_c` | CapitalSelector | Python scalar | () | float | CPU | finite | tensor |

---

## EWMAStats

| Element | Owner | Type | Shape | DType | Device | Invariants | dump_role |
|---|---|---|---|---|---|---|---|
| `beta` | EWMAStats | Python scalar | () | float | CPU | 0<β≤1 | tensor |
| `mu` | EWMAStats | Python scalar | () | float | CPU | finite | tensor |
| `var` | EWMAStats | Python scalar | () | float | CPU | ≥0 | tensor |
| `dd` | EWMAStats | Python scalar | () | float | CPU | ≥0 | tensor |
| `cum_pi` | EWMAStats | Python scalar | () | float | CPU | finite | tensor |
| `peak_cum_pi` | EWMAStats | Python scalar | () | float | CPU | finite | tensor |
| `seed_var` | EWMAStats | Python scalar | () | float | CPU | ≥0 | tensor |

---

## StackConfig / StackFormationThresholds

| Element | Owner | Type | Shape | DType | Device | Invariants | dump_role |
|---|---|---|---|---|---|---|---|
| `C_agg` | StackConfig | Python scalar | () | float | CPU | ≥0 | meta |
| `min_size` | StackConfig | Python scalar | () | int | CPU | ≥1 | meta |
| `max_size` | StackConfig | Python scalar | () | int | CPU | ≥min_size | meta |
| `stack_weighting` | StackConfig | Python scalar | () | str | CPU | "equal" | meta |
| `tau_mu` | StackConfig | Python scalar | () | float | CPU | finite | meta |
| `tau_vol` | StackConfig | Python scalar | () | float | CPU | ≥0 | meta |
| `tau_cvar` | StackConfig | Python scalar | () | float | CPU | finite | meta |
| `tau_dd` | StackConfig | Python scalar | () | float | CPU | ≥0 | meta |
| `use_cvar` | StackConfig | Python scalar | () | bool | CPU | bool | meta |
| `tau_mu` | StackFormationThresholds | Python scalar | () | float | CPU | finite | meta |
| `tau_vol` | StackFormationThresholds | Python scalar | () | float | CPU | ≥0 | meta |
| `tau_cvar` | StackFormationThresholds | Python scalar | () | float | CPU | finite | meta |
| `tau_surv` | StackFormationThresholds | Python scalar | () | float | CPU | 0–1 | meta |
| `tau_corr` | StackFormationThresholds | Python scalar | () | float | CPU | 0–1 | meta |
| `min_size` | StackFormationThresholds | Python scalar | () | int | CPU | ≥1 | meta |
| `max_size` | StackFormationThresholds | Python scalar | () | int | CPU | ≥min_size | meta |

---

## StackChannel (topology node)

| Element | Owner | Type | Shape | DType | Device | Invariants | dump_role |
|---|---|---|---|---|---|---|---|
| `members` | StackChannel | dict | (N) | PhaseCChannel | CPU | N≥1 | meta |
| `cfg` | StackChannel | object | n/a | StackConfig | CPU | non-null (see stack_cfg) | excluded |
| `stack_id` | StackChannel | Python scalar | () | str | CPU | non-empty | meta |
| `_w_internal` | StackChannel | dict | (N) | float | CPU | sum=1 | tensor |
| `_alive` | StackChannel | Python scalar | () | bool | CPU | bool | tensor |
| `_dt` | StackChannel | Python scalar | () | float | CPU | >0 | tensor |
| `_mu` | StackChannel | Python scalar | () | float | CPU | finite | tensor |
| `_var` | StackChannel | Python scalar | () | float | CPU | ≥0 | tensor |
| `_beta` | StackChannel | Python scalar | () | float | CPU | 0<β≤1 | tensor |
| `_dd_peak` | StackChannel | Python scalar | () | float | CPU | finite | tensor |
| `_dd_val` | StackChannel | Python scalar | () | float | CPU | finite | tensor |
| `_dd_max` | StackChannel | Python scalar | () | float | CPU | ≥0 | tensor |
| `_cvar` | StackChannel | Python scalar | () | float | CPU | finite | tensor |

---

## StackManager (topology controller)

| Element | Owner | Type | Shape | DType | Device | Invariants | dump_role |
|---|---|---|---|---|---|---|---|
| `stack_cfg` | StackManager | object | n/a | StackConfig | CPU | non-null (dumped via stack_cfg fields) | meta |
| `thresholds` | StackManager | object | n/a | StackFormationThresholds | CPU | non-null (dumped via thresholds fields) | meta |
| `stacks` | StackManager | dict | (M) | StackChannel | CPU | M≥0 | meta |
| `sediment` | StackManager | object | n/a | SedimentDAG | CPU | optional (serialized separately as top-level "sediment") | excluded |
| `telemetry` | StackManager | object | n/a | TelemetryLogger | CPU | optional | excluded |
| `world_id` | StackManager | Python scalar | () | str | CPU | non-empty | meta |
| `phase_id` | StackManager | Python scalar | () | str | CPU | non-empty | meta |
| `run_id` | StackManager | Python scalar | () | str | CPU | non-empty | meta |
| `_counter` | StackManager | Python scalar | () | int | CPU | ≥0 | meta |
| `_t_counter` | StackManager | Python scalar | () | int | CPU | ≥0 | meta |
| `_last_reject_t` | StackManager | Python scalar | () | int | CPU | ≥-1 | meta |

---

## SedimentDAG (structural memory)

| Element | Owner | Type | Shape | DType | Device | Invariants | dump_role |
|---|---|---|---|---|---|---|---|
| `persist_path` | SedimentDAG | object | n/a | Path | CPU | optional | excluded |
| `forbid_pairs` | SedimentDAG | Python scalar | () | bool | CPU | bool | meta |
| `_nodes` | SedimentDAG | dict | (N) | SedimentNode | CPU | N≥0 | meta |
| `_last_node_id_by_run` | SedimentDAG | dict | (R) | int | CPU | ids ≥1 | meta |
| `_next_node_id` | SedimentDAG | Python scalar | () | int | CPU | ≥1 | meta |

---

## SedimentNode (structural record)

| Element | Owner | Type | Shape | DType | Device | Invariants | dump_role |
|---|---|---|---|---|---|---|---|
| `node_id` | SedimentNode | Python scalar | () | int | CPU | ≥1 | meta |
| `members` | SedimentNode | list | (K) | str | CPU | sorted | meta |
| `mask` | SedimentNode | dict | (M) | Any | CPU | passive | meta |
| `world_id` | SedimentNode | Python scalar | () | str | CPU | non-empty | meta |
| `phase_id` | SedimentNode | Python scalar | () | str | CPU | non-empty | meta |
| `t` | SedimentNode | Python scalar | () | int | CPU | ≥0 | meta |
| `run_id` | SedimentNode | Python scalar | () | str | CPU | non-empty | meta |
