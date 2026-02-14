# Phase G – G3.4.6.1 Asymmetric Noise Drift

This document defines G3.4.6.1: a channel‑asymmetric noise drift where
neutral channels experience a rising σ(t) while the active regime channel
keeps a constant σ.

Purpose:
- test whether **relative** SNR changes (not global noise) affect weights/entropy
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05}
- mode ∈ {"stationary", "asym_drift"}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)

---

## Results

### Run table (aggregates)

| p | mode | seed | final_wealth | mean_stack_count | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | asym_drift | 0 | 10.129169 | 1.068 | 0 | 0.660031 | 0.757614 | 0.101474 |
| 0.01 | asym_drift | 1 | 11.711487 | 1.126 | 0 | 0.481843 | 0.038891 | 0.838086 |
| 0.01 | asym_drift | 2 | 10.275055 | 1.116 | 0 | 1.018422 | 0.559757 | 0.269105 |
| 0.01 | asym_drift | 3 | 11.204593 | 1.138 | 0 | 0.903061 | 0.413312 | 0.430523 |
| 0.01 | asym_drift | 4 | 11.611385 | 1.146 | 0 | 0.959008 | 0.342159 | 0.505865 |
| 0.01 | stationary | 0 | 10.347599 | 1.048 | 0 | 0.649110 | 0.762564 | 0.099742 |
| 0.01 | stationary | 1 | 11.337870 | 1.136 | 0 | 0.478572 | 0.036804 | 0.839039 |
| 0.01 | stationary | 2 | 10.262962 | 1.106 | 0 | 1.029813 | 0.547851 | 0.279131 |
| 0.01 | stationary | 3 | 11.443279 | 1.080 | 0 | 0.907713 | 0.409747 | 0.431183 |
| 0.01 | stationary | 4 | 11.390624 | 1.182 | 0 | 0.967022 | 0.324932 | 0.520761 |
| 0.05 | asym_drift | 0 | 10.225639 | 1.026 | 0 | 0.848238 | 0.702877 | 0.150819 |
| 0.05 | asym_drift | 1 | 11.603544 | 1.088 | 0 | 1.063254 | 0.262266 | 0.562306 |
| 0.05 | asym_drift | 2 | 10.354565 | 1.072 | 0 | 0.838547 | 0.146989 | 0.687344 |
| 0.05 | asym_drift | 3 | 11.124839 | 1.178 | 0 | 0.920110 | 0.668759 | 0.170383 |
| 0.05 | asym_drift | 4 | 11.571739 | 1.156 | 0 | 1.083849 | 0.510909 | 0.313860 |
| 0.05 | stationary | 0 | 10.347599 | 1.008 | 0 | 0.846742 | 0.702374 | 0.154579 |
| 0.05 | stationary | 1 | 11.337870 | 1.084 | 0 | 1.055684 | 0.249370 | 0.573355 |
| 0.05 | stationary | 2 | 10.262962 | 1.102 | 0 | 0.844330 | 0.149553 | 0.685593 |
| 0.05 | stationary | 3 | 11.443279 | 1.136 | 0 | 0.923155 | 0.669241 | 0.167036 |
| 0.05 | stationary | 4 | 11.390624 | 1.010 | 0 | 1.090081 | 0.497321 | 0.324046 |

### Δ‑table (asym_drift − stationary)

| p | seed | Δ final_wealth | Δ weight_entropy | Δ weight_share_0 | Δ weight_share_3 |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0 | -0.218430 | 0.010921 | -0.004950 | 0.001732 |
| 0.01 | 1 | 0.373617 | 0.003272 | 0.002088 | -0.000954 |
| 0.01 | 2 | 0.012093 | -0.011391 | 0.011906 | -0.010026 |
| 0.01 | 3 | -0.238686 | -0.004652 | 0.003566 | -0.000660 |
| 0.01 | 4 | 0.220761 | -0.008014 | 0.017226 | -0.014896 |
| 0.05 | 0 | -0.121960 | 0.001497 | 0.000503 | -0.003760 |
| 0.05 | 1 | 0.265674 | 0.007570 | 0.012896 | -0.011048 |
| 0.05 | 2 | 0.091602 | -0.005783 | -0.002563 | 0.001751 |
| 0.05 | 3 | -0.318439 | -0.003045 | -0.000482 | 0.003348 |
| 0.05 | 4 | 0.181116 | -0.006232 | 0.013588 | -0.010186 |

---

## Interpretation (brief)

- Asymmetric drift produces mixed wealth shifts across seeds; no uniform direction.
- Entropy deltas remain small and inconsistent in sign; no strong, directional selection signal.
- Weight share changes are modest and seed‑dependent; no clear reallocation trend.
- Stack activity is present (mean_stack_count ≈ 1.1), while rebirths remain absent.

Conclusion: channel‑asymmetric noise drift perturbs outcomes mildly, but does not by itself create stable topology dynamics under current settings.
