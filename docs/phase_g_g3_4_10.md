# Phase G – G3.4.10 Stack Trigger World

This document defines G3.4.10: a deterministic clustered‑signal world
intended to make stack formation likely without kernel changes.

Purpose:
- test whether a persistent “good‑cluster” signal triggers stacks
- verify determinism under fixed conditions
- ensure no kernel semantic drift

Scope:
- Profile A only
- horizon T = 500 (t = 0..499)
- Topology enabled (`CAPM_ENABLE_TOPOLOGY=1`)

Returns (constant per step):
- r_t = [0.02, 0.02, 0.00, -0.01, -0.01]
- c_t = 0.0

---

## Results

### Run summary (aggregates)

| final_wealth | mean_stack_count | total_rebirths | weight_entropy | weight_share_0 | weight_share_1 |
|---:|---:|---:|---:|---:|---:|
| 11.000000 | 1.000 | 0 | 0.871586 | 0.464320 | 0.464320 |

---

## Interpretation (brief)

- Deterministic clustered signal yields stable topological activity (mean_stack_count = 1.0).
- Weights concentrate on the two positive channels (0 and 1), with symmetric shares.
- No rebirth observed under this benign, stationary signal.
