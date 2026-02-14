# Phase G – G3.4.7 Autocorrelation Control

This document defines G3.4.7: a persistent vs shuffled regime comparison
using RegimeSwitchBanditWorld and ShuffledRegimeBanditWorld.

Purpose:
- isolate temporal structure effects (autocorrelation) from marginal distributions
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)
- noise sequence shared per‑t between World A and B

---

## Results (A/B per (p, seed))

| p | seed | world | final_wealth | mean_stack | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0 | A | 10.630195 | 1.140 | 0 | 0.872351 | 0.682436 | 0.163993 |
| 0.01 | 0 | B | 10.630195 | 1.028 | 0 | 1.029801 | 0.588535 | 0.228795 |
| 0.01 | 1 | A | 9.980916 | 1.058 | 0 | 0.710812 | 0.094124 | 0.752937 |
| 0.01 | 1 | B | 9.980916 | 0.970 | 0 | 0.919179 | 0.171197 | 0.653996 |
| 0.01 | 2 | A | 11.693133 | 1.188 | 0 | 0.853385 | 0.361987 | 0.479597 |
| 0.01 | 2 | B | 11.693133 | 1.078 | 0 | 0.802628 | 0.121184 | 0.713106 |
| 0.01 | 3 | A | 11.065128 | 1.154 | 0 | 0.654064 | 0.086562 | 0.774850 |
| 0.01 | 3 | B | 11.065128 | 1.116 | 0 | 0.911455 | 0.164066 | 0.669693 |
| 0.01 | 4 | A | 11.492163 | 1.252 | 0 | 1.036706 | 0.310921 | 0.530451 |
| 0.01 | 4 | B | 11.492163 | 1.102 | 0 | 1.109810 | 0.337599 | 0.487314 |
| 0.05 | 0 | A | 10.630195 | 1.142 | 0 | 1.002111 | 0.606297 | 0.224948 |
| 0.05 | 0 | B | 10.630195 | 1.058 | 0 | 1.151548 | 0.422957 | 0.388638 |
| 0.05 | 1 | A | 9.980916 | 0.994 | 0 | 0.992496 | 0.614963 | 0.212644 |
| 0.05 | 1 | B | 9.980916 | 0.928 | 0 | 1.077322 | 0.555959 | 0.263108 |
| 0.05 | 2 | A | 11.693133 | 1.124 | 0 | 1.044654 | 0.505823 | 0.315448 |
| 0.05 | 2 | B | 11.693133 | 1.088 | 0 | 1.041612 | 0.577083 | 0.243907 |
| 0.05 | 3 | A | 11.065128 | 1.170 | 0 | 1.096684 | 0.483811 | 0.343944 |
| 0.05 | 3 | B | 11.065128 | 1.132 | 0 | 1.129261 | 0.368342 | 0.452267 |
| 0.05 | 4 | A | 11.492163 | 1.230 | 0 | 1.036684 | 0.265203 | 0.563767 |
| 0.05 | 4 | B | 11.492163 | 1.186 | 0 | 0.974886 | 0.203128 | 0.630055 |

---

## Δ‑Table (B − A)

| p | seed | d_entropy(B-A) | d_share0(B-A) | d_share3(B-A) |
|---|---|---|---|---|
| 0.01 | 0 | 0.157450 | -0.093901 | 0.064802 |
| 0.01 | 1 | 0.208367 | 0.077074 | -0.098940 |
| 0.01 | 2 | -0.050757 | -0.240803 | 0.233509 |
| 0.01 | 3 | 0.257391 | 0.077504 | -0.105157 |
| 0.01 | 4 | 0.073104 | 0.026678 | -0.043138 |
| 0.05 | 0 | 0.149436 | -0.183340 | 0.163690 |
| 0.05 | 1 | 0.084826 | -0.059004 | 0.050464 |
| 0.05 | 2 | -0.003042 | 0.071260 | -0.071540 |
| 0.05 | 3 | 0.032577 | -0.115469 | 0.108324 |
| 0.05 | 4 | -0.061798 | -0.062075 | 0.066288 |

---

## ΔEntropy Summary (B − A)

| p | mean Δentropy | std | t (mean / (std/√n)) | n |
|---|---|---|---|---|
| 0.01 | 0.129111 | 0.121480 | 2.377 | 5 |
| 0.05 | 0.040400 | 0.081018 | 1.115 | 5 |

---

## Interpretation (narrow, descriptive)

- **Wealth is identical between A/B per seed** → the wealth aggregate is driven by marginals/noise, not by autocorrelation in this setup.
- **Weight dynamics differ between A/B** (entropy and weight shares shift) → **time structure influences weights** even under identical marginals and shared noise.
- **Stacks are present (mean ~1), rebirths remain 0** → topology is active but stable; effects are confined to weight dynamics rather than rebirth/reset events.
