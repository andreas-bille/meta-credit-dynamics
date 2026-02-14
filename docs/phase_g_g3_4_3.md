# Phase G – G3.4.3 Volatility Regimes

This document defines G3.4.3: a regime‑dependent volatility variant
with unchanged means.

Purpose:
- test sensitivity to volatility structure
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)
- σ_low = 0.005, σ_high = 0.02
- costs c = 0.0

---

## Results (baseline vs. volatility regime)

### Δ‑Table (volatility − baseline)

| p | seed | d_entropy(vol-baseline) | d_share0 | d_share3 | d_final_wealth | d_mean_stack | d_total_rebirths |
|---|---|---|---|---|---|---|---|
| 0.01 | 0 | 0.021380 | -0.009012 | 0.014937 | -0.191578 | -0.060 | 0 |
| 0.01 | 1 | -0.018550 | -0.004663 | 0.003153 | 0.297647 | -0.046 | 0 |
| 0.01 | 2 | 0.037932 | -0.032200 | 0.017637 | 0.163292 | -0.080 | 0 |
| 0.01 | 3 | -0.014908 | -0.000710 | 0.004004 | -0.275543 | -0.012 | 0 |
| 0.01 | 4 | -0.024284 | -0.028323 | 0.029789 | 0.149246 | -0.070 | 0 |
| 0.05 | 0 | 0.002252 | -0.002759 | 0.000293 | -0.557316 | -0.006 | 0 |
| 0.05 | 1 | -0.026074 | -0.020914 | 0.020031 | -0.205808 | -0.020 | 0 |
| 0.05 | 2 | 0.011252 | 0.009276 | -0.009434 | -0.536405 | -0.100 | 0 |
| 0.05 | 3 | 0.000311 | 0.001088 | -0.002369 | 0.268465 | 0.020 | 0 |
| 0.05 | 4 | -0.002032 | 0.013946 | -0.015016 | -0.183214 | 0.172 | 0 |

---

### Δ Summary (mean/std)

| p | mean d_entropy | std | mean d_final_wealth | std |
|---|---|---|---|---|
| 0.01 | 0.000314 | 0.024704 | 0.028613 | 0.221840 |
| 0.05 | -0.002858 | 0.012449 | -0.242856 | 0.300478 |

---

## Interpretation (descriptive)

- Volatility regimes produce only small shifts in weight entropy and shares; effects are mixed by seed.
- Wealth deltas are small and inconsistent; no clear aggregate shift emerges.
- Stack activity is present (mean_stack_count ≈ 1.0), while rebirths remain absent.
