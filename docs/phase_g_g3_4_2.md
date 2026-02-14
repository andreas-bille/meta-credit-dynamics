# Phase G – G3.4.2 Stronger Regime Persistence

This document defines G3.4.2: a persistence sweep with p = 0.001
compared to p = 0.01 using RegimeSwitchBanditWorld (no costs).

Purpose:
- test sensitivity to longer regime persistence
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.001, 0.01}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)
- costs c = 0.0

---

## Results (all runs)

| p | seed | final_wealth | mean_stack | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---|---|---|---|---|---|---|---|
| 0.001 | 0 | 10.347599 | 1.042 | 0 | 0.455350 | 0.041892 | 0.846141 |
| 0.001 | 1 | 11.337870 | 1.160 | 0 | 0.724361 | 0.157480 | 0.709750 |
| 0.001 | 2 | 10.262962 | 1.126 | 0 | 0.489865 | 0.834373 | 0.042233 |
| 0.001 | 3 | 11.443279 | 1.096 | 0 | 0.710650 | 0.724842 | 0.144264 |
| 0.001 | 4 | 11.390624 | 1.092 | 0 | 0.462293 | 0.037249 | 0.843084 |
| 0.010 | 0 | 10.347599 | 1.048 | 0 | 0.649110 | 0.762564 | 0.099742 |
| 0.010 | 1 | 11.337870 | 1.136 | 0 | 0.478572 | 0.036804 | 0.839039 |
| 0.010 | 2 | 10.262962 | 1.106 | 0 | 1.029813 | 0.547851 | 0.279131 |
| 0.010 | 3 | 11.443279 | 1.080 | 0 | 0.907713 | 0.409747 | 0.431183 |
| 0.010 | 4 | 11.390624 | 1.182 | 0 | 0.967022 | 0.324932 | 0.520761 |

---

## Δ‑Table (p=0.001 − p=0.01)

| seed | d_entropy(p001-p01) | d_share0 | d_share3 | d_final_wealth | d_mean_stack | d_total_rebirths |
|---|---|---|---|---|---|---|
| 0 | -0.193760 | -0.720672 | 0.746399 | 0.000000 | -0.006 | 0 |
| 1 | 0.245789 | 0.120676 | -0.129289 | 0.000000 | 0.024 | 0 |
| 2 | -0.539947 | 0.286522 | -0.236898 | -0.000000 | 0.020 | 0 |
| 3 | -0.197064 | 0.315095 | -0.286919 | 0.000000 | 0.016 | 0 |
| 4 | -0.504729 | -0.287683 | 0.322323 | 0.000000 | -0.090 | 0 |

---

## Δ Summary (mean/std)

| metric | mean | std |
|---|---|---|
| d_entropy | -0.237942 | 0.282843 |
| d_share0 | -0.057212 | 0.395400 |
| d_share3 | 0.083123 | 0.395377 |
| d_final_wealth | 0.000000 | 0.000000 |

---

## Autocorrelation of weight shares (lag‑1)

| seed | ac_share0(p001) | ac_share0(p01) | d_ac_share0 | ac_share3(p001) | ac_share3(p01) | d_ac_share3 |
|---|---|---|---|---|---|---|
| 0 | 0.999920 | 0.999963 | -0.000043 | 0.999974 | 0.999788 | 0.000186 |
| 1 | 0.999931 | 0.999923 | 0.000007 | 0.999857 | 0.999976 | -0.000119 |
| 2 | 0.999979 | 0.999802 | 0.000177 | 0.999918 | 0.999024 | 0.000894 |
| 3 | 0.999862 | 0.999856 | 0.000006 | 0.999909 | 0.999958 | -0.000050 |
| 4 | 0.999923 | 0.999849 | 0.000074 | 0.999974 | 0.999684 | 0.000290 |

| metric | mean | std |
|---|---|---|
| d_ac_share0 | 0.000044 | 0.000076 |
| d_ac_share3 | 0.000240 | 0.000359 |

---

## Interpretation (descriptive)

- Wealth is identical per seed across p values; persistence does not change wealth aggregates in this setup.
- Weight observables shift: entropy tends to be lower for p=0.001 on average (more concentration), but direction is seed‑dependent.
- Autocorrelation of weight shares is effectively unchanged (Δ ≈ 1e‑4).
- Stacks/Rebirth remain 0 across runs; effects are confined to weight dynamics.
