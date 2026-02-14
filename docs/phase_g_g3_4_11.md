# Phase G – G3.4.11 Regime Flip → Sediment

This document defines G3.4.11: a deterministic regime flip designed to
form stacks and then dissolve them, producing sediment.

Purpose:
- trigger stack formation and dissolution
- verify sediment creation under controlled conditions
- ensure determinism and no kernel semantic drift

Scope:
- Profile A only
- horizon T = 500 (t = 0..499)
- flip_time = 250
- Topology enabled (`CAPM_ENABLE_TOPOLOGY=1`)

Returns:
- For t < flip_time: [0.05, 0.05, 0.00, -0.03, -0.03]
- For t ≥ flip_time: [-0.03, -0.03, 0.00, 0.05, 0.05]
- For t in shock_window: [-2.00, -2.00, 0.00, -2.00, -2.00]
- c_t = 0.0

---

## Results

### Run summary (aggregates)

| final_wealth | mean_stack_count | total_rebirths | final_sediment_count |
|---:|---:|---:|---:|
| 10.100000 | 0.918 | 9 | 0 |

---

## Interpretation (brief)

- Shock window triggers **rebirths** (7), but still **no sediment**.
- Stack count drops slightly (mean_stack_count < 1), suggesting destabilization without dissolution.
