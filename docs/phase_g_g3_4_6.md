# Phase G – G3.4.6 Non‑stationary Noise Floor

This document defines G3.4.6: a time‑varying volatility schedule σ(t)
with unchanged mean returns and zero costs.

Purpose:
- test whether a drifting noise floor affects weight dynamics
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05}
- volatility_mode ∈ {"stationary", "drift_up"}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)

---

## Results

### Run table (aggregates)

| p | mode | seed | final_wealth | mean_stack_count | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | drift_up | 0 | 10.093823 | 1.056 | 0 | 0.659519 | 0.758264 | 0.098908 |
| 0.01 | drift_up | 1 | 11.779751 | 1.120 | 0 | 0.479829 | 0.038571 | 0.839329 |
| 0.01 | drift_up | 2 | 10.223504 | 1.096 | 0 | 1.036821 | 0.542662 | 0.283832 |
| 0.01 | drift_up | 3 | 11.279288 | 1.080 | 0 | 0.897725 | 0.408068 | 0.435604 |
| 0.01 | drift_up | 4 | 11.579125 | 1.180 | 0 | 0.957938 | 0.347534 | 0.499085 |
| 0.01 | stationary | 0 | 10.347599 | 1.048 | 0 | 0.649110 | 0.762564 | 0.099742 |
| 0.01 | stationary | 1 | 11.337870 | 1.136 | 0 | 0.478572 | 0.036804 | 0.839039 |
| 0.01 | stationary | 2 | 10.262962 | 1.106 | 0 | 1.029813 | 0.547851 | 0.279131 |
| 0.01 | stationary | 3 | 11.443279 | 1.080 | 0 | 0.907713 | 0.409747 | 0.431183 |
| 0.01 | stationary | 4 | 11.390624 | 1.182 | 0 | 0.967022 | 0.324932 | 0.520761 |
| 0.05 | drift_up | 0 | 10.093823 | 1.028 | 0 | 0.857282 | 0.698485 | 0.152865 |
| 0.05 | drift_up | 1 | 11.779751 | 1.106 | 0 | 1.059416 | 0.262977 | 0.563687 |
| 0.05 | drift_up | 2 | 10.223504 | 1.106 | 0 | 0.839927 | 0.148636 | 0.686366 |
| 0.05 | drift_up | 3 | 11.279288 | 1.144 | 0 | 0.923701 | 0.666023 | 0.172865 |
| 0.05 | drift_up | 4 | 11.579125 | 1.116 | 0 | 1.079180 | 0.523946 | 0.299993 |
| 0.05 | stationary | 0 | 10.347599 | 1.008 | 0 | 0.846742 | 0.702374 | 0.154579 |
| 0.05 | stationary | 1 | 11.337870 | 1.084 | 0 | 1.055684 | 0.249370 | 0.573355 |
| 0.05 | stationary | 2 | 10.262962 | 1.102 | 0 | 0.844330 | 0.149553 | 0.685593 |
| 0.05 | stationary | 3 | 11.443279 | 1.136 | 0 | 0.923155 | 0.669241 | 0.167036 |
| 0.05 | stationary | 4 | 11.390624 | 1.010 | 0 | 1.090081 | 0.497321 | 0.324046 |

### Δ‑table (drift_up − stationary)

| p | seed | Δ final_wealth | Δ weight_entropy | Δ weight_share_0 | Δ weight_share_3 |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0 | -0.253777 | 0.010409 | -0.004300 | -0.000834 |
| 0.01 | 1 | 0.441881 | 0.001257 | 0.001768 | 0.000289 |
| 0.01 | 2 | -0.039458 | 0.007008 | -0.005189 | 0.004702 |
| 0.01 | 3 | -0.163990 | -0.009989 | -0.001678 | 0.004421 |
| 0.01 | 4 | 0.188501 | -0.009083 | 0.022602 | -0.021676 |
| 0.05 | 0 | -0.253777 | 0.010540 | -0.003889 | -0.001714 |
| 0.05 | 1 | 0.441881 | 0.003732 | 0.013607 | -0.009668 |
| 0.05 | 2 | -0.039458 | -0.004403 | -0.000917 | 0.000773 |
| 0.05 | 3 | -0.163990 | 0.000546 | -0.003219 | 0.005830 |
| 0.05 | 4 | 0.188501 | -0.010900 | 0.026624 | -0.024053 |

---

## Interpretation (brief)

- Drift_up introduces mixed wealth shifts (both positive and negative deltas across seeds), no uniform direction.
- Entropy deltas are small and inconsistent in sign, suggesting limited sensitivity to the slow σ(t) drift at current scales.
- Weight share changes remain modest and seed‑dependent; no clear reallocation trend.
- Stack activity is present (mean_stack_count ≈ 1.0), while rebirths remain absent.

Conclusion: the non‑stationary noise floor perturbs outcomes mildly but does not create a stable selection signal or topology dynamics under current settings.
