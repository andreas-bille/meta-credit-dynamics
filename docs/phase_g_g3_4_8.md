# Phase G – G3.4.8 Stack‑Trigger Shock

This document defines G3.4.8: deterministic drawdown shocks on the active
regime channel to trigger topology dynamics under extreme but controlled conditions.

Purpose:
- test whether topology/rebirth can be triggered without kernel changes
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05}
- mode ∈ {"baseline", "shock"}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)

---

## Results

### Run table (aggregates)

| p | mode | seed | final_wealth | mean_stack_count | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | baseline | 0 | 10.347599 | 1.048 | 0 | 0.649110 | 0.762564 | 0.099742 |
| 0.01 | baseline | 1 | 11.337870 | 1.136 | 0 | 0.478572 | 0.036804 | 0.839039 |
| 0.01 | baseline | 2 | 10.262962 | 1.106 | 0 | 1.029813 | 0.547851 | 0.279131 |
| 0.01 | baseline | 3 | 11.443279 | 1.080 | 0 | 0.907713 | 0.409747 | 0.431183 |
| 0.01 | baseline | 4 | 11.390624 | 1.182 | 0 | 0.967022 | 0.324932 | 0.520761 |
| 0.01 | shock | 0 | 9.547599 | 0.970 | 0 | 0.688294 | 0.749134 | 0.104833 |
| 0.01 | shock | 1 | 10.537870 | 1.140 | 0 | 0.519944 | 0.039654 | 0.828181 |
| 0.01 | shock | 2 | 9.462962 | 1.150 | 0 | 1.091472 | 0.511301 | 0.301168 |
| 0.01 | shock | 3 | 10.643279 | 1.076 | 0 | 0.973178 | 0.411416 | 0.415628 |
| 0.01 | shock | 4 | 10.590624 | 1.192 | 0 | 1.014489 | 0.314653 | 0.517709 |
| 0.05 | baseline | 0 | 10.347599 | 1.008 | 0 | 0.846742 | 0.702374 | 0.154579 |
| 0.05 | baseline | 1 | 11.337870 | 1.084 | 0 | 1.055684 | 0.249370 | 0.573355 |
| 0.05 | baseline | 2 | 10.262962 | 1.102 | 0 | 0.844330 | 0.149553 | 0.685593 |
| 0.05 | baseline | 3 | 11.443279 | 1.136 | 0 | 0.923155 | 0.669241 | 0.167036 |
| 0.05 | baseline | 4 | 11.390624 | 1.010 | 0 | 1.090081 | 0.497321 | 0.324046 |
| 0.05 | shock | 0 | 9.547599 | 0.980 | 0 | 0.911928 | 0.671077 | 0.174455 |
| 0.05 | shock | 1 | 10.537870 | 1.072 | 0 | 1.016135 | 0.191166 | 0.625367 |
| 0.05 | shock | 2 | 9.462962 | 1.076 | 0 | 0.873349 | 0.145446 | 0.681118 |
| 0.05 | shock | 3 | 10.643279 | 1.198 | 0 | 0.932148 | 0.672408 | 0.152716 |
| 0.05 | shock | 4 | 10.590624 | 1.044 | 0 | 1.133226 | 0.473785 | 0.334048 |

### Δ‑table (shock − baseline)

| p | seed | Δ final_wealth | Δ weight_entropy | Δ weight_share_0 | Δ weight_share_3 | Δ mean_stack_count | Δ total_rebirths |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | -0.800000 | 0.039184 | -0.013430 | 0.005091 | -0.078 | 0 |
| 0.01 | 1 | -0.800000 | 0.041373 | 0.002851 | -0.010858 | 0.004 | 0 |
| 0.01 | 2 | -0.800000 | 0.061660 | -0.036550 | 0.022038 | 0.044 | 0 |
| 0.01 | 3 | -0.800000 | 0.065465 | 0.001669 | -0.015555 | -0.004 | 0 |
| 0.01 | 4 | -0.800000 | 0.047468 | -0.010279 | -0.003052 | 0.010 | 0 |
| 0.05 | 0 | -0.800000 | 0.065186 | -0.031297 | 0.019876 | -0.028 | 0 |
| 0.05 | 1 | -0.800000 | -0.039549 | -0.058204 | 0.052012 | -0.012 | 0 |
| 0.05 | 2 | -0.800000 | 0.029019 | -0.004106 | -0.004475 | -0.026 | 0 |
| 0.05 | 3 | -0.800000 | 0.008993 | 0.003166 | -0.014320 | 0.062 | 0 |
| 0.05 | 4 | -0.800000 | 0.043145 | -0.023536 | 0.010002 | 0.034 | 0 |

---

## Interpretation (brief)

- The deterministic shocks reduce final_wealth by exactly 0.8 across all seeds (four shocks × 0.2).
- Entropy shifts are consistently positive in most runs, indicating mild diversification under shocks.
- Weight share shifts are modest and mixed in sign; no strong reallocation pattern.
- Stack formation is now observed (mean_stack_count ≈ 1.0), but rebirths remain absent.

Conclusion: targeted, deterministic shocks perturb weights and wealth as expected. They now coincide with non-zero stack activity, but still do not trigger rebirth under current settings.
