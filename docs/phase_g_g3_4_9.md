# Phase G – G3.4.9 Controlled Ruin Run

This document defines G3.4.9: a controlled ruin regime intended to trigger at least one rebirth
under deterministic conditions, without kernel changes.

Purpose:
- verify that rebirth can be triggered under a ruinous regime
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p = 0.01
- seed = 0
- horizon T = 500 (t = 0..499)
- Topology enabled (`CAPM_ENABLE_TOPOLOGY=1`)

---

## Results

### Run summary (aggregates)

| p | seed | final_wealth | mean_stack_count | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | 2.871288 | 0.868 | 13 | 0.717766 | 0.756938 | 0.060782 |

---

## Interpretation (brief)

- Controlled ruin regime triggers **rebirth** as intended (13 total rebirths in this run).
- Topology is active (mean_stack_count < 1, indicating stack turnover under stress).
- Wealth declines substantially relative to earlier benign worlds, consistent with negative expected regime.
