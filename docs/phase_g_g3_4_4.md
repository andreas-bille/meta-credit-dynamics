# Phase G – G3.4.4 Cost‑Shocks (Exogenous)

This document defines G3.4.4: an exogenous cost‑shock variant on
RegimeSwitchBanditWorld to test whether rare, high costs impact observables
without kernel changes.

Purpose:
- stress inhibition with cost spikes while keeping returns unchanged
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05}
- q ∈ {0.0, 0.02}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)

---

## Results

### Run table (aggregates)

| p | q | seed | final_wealth | mean_stack_count | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.00 | 0 | 10.347599 | 1.048 | 0 | 0.649110 | 0.762564 | 0.099742 |
| 0.01 | 0.00 | 1 | 11.337870 | 1.136 | 0 | 0.478572 | 0.036804 | 0.839039 |
| 0.01 | 0.00 | 2 | 10.262962 | 1.106 | 0 | 1.029813 | 0.547851 | 0.279131 |
| 0.01 | 0.00 | 3 | 11.443279 | 1.080 | 0 | 0.907713 | 0.409747 | 0.431183 |
| 0.01 | 0.00 | 4 | 11.390624 | 1.182 | 0 | 0.967022 | 0.324932 | 0.520761 |
| 0.01 | 0.02 | 0 | 9.747599 | 1.048 | 0 | 0.666201 | 0.758181 | 0.101326 |
| 0.01 | 0.02 | 1 | 10.687870 | 1.136 | 0 | 0.497224 | 0.038108 | 0.834125 |
| 0.01 | 0.02 | 2 | 9.962962 | 1.106 | 0 | 1.034179 | 0.545316 | 0.280285 |
| 0.01 | 0.02 | 3 | 10.893279 | 1.080 | 0 | 0.920629 | 0.394125 | 0.441542 |
| 0.01 | 0.02 | 4 | 10.990624 | 1.182 | 0 | 0.972063 | 0.323570 | 0.520939 |
| 0.05 | 0.00 | 0 | 10.347599 | 1.008 | 0 | 0.846742 | 0.702374 | 0.154579 |
| 0.05 | 0.00 | 1 | 11.337870 | 1.084 | 0 | 1.055684 | 0.249370 | 0.573355 |
| 0.05 | 0.00 | 2 | 10.262962 | 1.102 | 0 | 0.844330 | 0.149553 | 0.685593 |
| 0.05 | 0.00 | 3 | 11.443279 | 1.136 | 0 | 0.923155 | 0.669241 | 0.167036 |
| 0.05 | 0.00 | 4 | 11.390624 | 1.010 | 0 | 1.090081 | 0.497321 | 0.324046 |
| 0.05 | 0.02 | 0 | 9.747599 | 1.008 | 0 | 0.881062 | 0.684779 | 0.167909 |
| 0.05 | 0.02 | 1 | 10.687870 | 1.084 | 0 | 1.082829 | 0.265074 | 0.552120 |
| 0.05 | 0.02 | 2 | 9.962962 | 1.102 | 0 | 0.850085 | 0.150078 | 0.683827 |
| 0.05 | 0.02 | 3 | 10.893279 | 1.136 | 0 | 0.959647 | 0.648629 | 0.182094 |
| 0.05 | 0.02 | 4 | 10.990624 | 1.010 | 0 | 1.094201 | 0.491192 | 0.329177 |

### Δ‑table (q=0.02 − q=0.0)

| p | seed | Δ final_wealth | Δ weight_entropy | Δ weight_share_0 | Δ weight_share_3 |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0 | -0.600000 | 0.017091 | -0.004383 | 0.001584 |
| 0.01 | 1 | -0.650000 | 0.018652 | 0.001304 | -0.004915 |
| 0.01 | 2 | -0.300000 | 0.004366 | -0.002535 | 0.001155 |
| 0.01 | 3 | -0.550000 | 0.012916 | -0.015621 | 0.010359 |
| 0.01 | 4 | -0.400000 | 0.005041 | -0.001362 | 0.000177 |
| 0.05 | 0 | -0.600000 | 0.034321 | -0.017595 | 0.013330 |
| 0.05 | 1 | -0.650000 | 0.027145 | 0.015704 | -0.021235 |
| 0.05 | 2 | -0.300000 | 0.005756 | 0.000526 | -0.001767 |
| 0.05 | 3 | -0.550000 | 0.036493 | -0.020612 | 0.015058 |
| 0.05 | 4 | -0.400000 | 0.004121 | -0.006129 | 0.005131 |

---

## Interpretation (brief)

- Cost‑shocks reduce final_wealth consistently across seeds (Δ final_wealth ≈ -0.3 to -0.65) for both p values.
- Entropy shifts are small but mostly positive, with larger deltas at p=0.05.
- Weight shares shift modestly and inconsistently in sign; no clear directional reallocation signal.
- Stack activity is present (mean_stack_count ≈ 1.0), while rebirths remain absent.

Conclusion: exogenous cost shocks primarily depress wealth and mildly perturb weight entropy, without triggering topology dynamics under current settings.
