# Phase G – G3.4.5 Channel‑Subset Regimes

This document defines G3.4.5: subset‑based regime advantages
compared against the single‑channel baseline.

Purpose:
- test diversification response to subset regimes
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)
- costs c = 0.0

---

## Results (subset vs. baseline)

### Δ‑Table (subset − baseline)

| p | seed | d_entropy(sub-baseline) | d_share0 | d_share1 | d_share3 | d_share4 | d_final_wealth | d_mean_stack | d_total_rebirths |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0 | 0.409767 | -0.339188 | 0.356181 | -0.026582 | 0.019667 | 10.000000 | 0.346 | 0 |
| 0.01 | 1 | 0.451293 | -0.008516 | -0.009170 | -0.348240 | 0.375645 | 10.000000 | 0.220 | 0 |
| 0.01 | 2 | 0.324315 | -0.259977 | 0.289186 | -0.114428 | 0.101189 | 10.000000 | 0.378 | 0 |
| 0.01 | 3 | 0.345760 | -0.162514 | 0.215071 | -0.167736 | 0.133022 | 10.000000 | 0.510 | 0 |
| 0.01 | 4 | 0.341754 | -0.157384 | 0.137702 | -0.235645 | 0.267657 | 10.000000 | 0.160 | 0 |
| 0.05 | 0 | 0.374237 | -0.298992 | 0.332588 | -0.057852 | 0.035294 | 10.000000 | 0.466 | 0 |
| 0.05 | 1 | 0.364395 | -0.104025 | 0.130161 | -0.239441 | 0.229143 | 10.000000 | 0.242 | 0 |
| 0.05 | 2 | 0.383328 | -0.049938 | 0.053806 | -0.306667 | 0.317163 | 10.000000 | 0.306 | 0 |
| 0.05 | 3 | 0.325376 | -0.288023 | 0.340180 | -0.065948 | 0.031719 | 10.000000 | 0.380 | 0 |
| 0.05 | 4 | 0.324891 | -0.227741 | 0.246091 | -0.144615 | 0.139642 | 10.000000 | 0.454 | 0 |

---

### Δ Summary (mean/std)

| p | mean d_entropy | std | mean d_final_wealth | std |
|---|---|---|---|---|
| 0.01 | 0.374578 | 0.048078 | 10.000000 | 0.000000 |
| 0.05 | 0.354445 | 0.024671 | 10.000000 | 0.000000 |

---

## Interpretation (descriptive)

- Subset regimes materially increase weight entropy relative to the single‑channel baseline.
- Wealth is uniformly higher in the subset regime runs under the same seeds and p values.
- Stack activity is present (mean_stack_count ≈ 1.0), while rebirth remains absent.
