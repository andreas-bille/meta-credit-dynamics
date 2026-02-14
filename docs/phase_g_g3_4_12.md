# Phase G – G3.4.12 Adversarial Phase‑Shift (Relative SNR Reversal)

This document defines G3.4.12: a two‑phase adversarial SNR reversal world
intended to stress learned weight concentration and topology under fixed means.

Purpose:
- test reversibility of learned structure under relative SNR inversion
- verify determinism under fixed seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p = 0.001
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)
- costs c = 0.0

Means (both phases):
- Regime A: [0.02, 0, 0, 0, 0]
- Regime B: [0, 0, 0, 0.02, 0]

Phase logic:
- Phase 1 (t < T/2): σ_k = 0.005 for all channels
- Phase 2 (t ≥ T/2):
  - active regime channel: σ = 0.02
  - opposing regime channel: σ = 0.005
  - others: σ = 0.01

---

## Results

### Run table (aggregates)

| seed | final_wealth | mean_stack_count | total_rebirths | weight_entropy | weight_share_0 | weight_share_3 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.352959 | 0.998 | 0 | 0.556718 | 0.063301 | 0.818768 |
| 1 | 11.032102 | 1.170 | 0 | 0.460994 | 0.039890 | 0.844264 |
| 2 | 10.233775 | 0.998 | 0 | 0.478428 | 0.838328 | 0.040395 |
| 3 | 11.350259 | 1.168 | 0 | 0.479976 | 0.836323 | 0.040811 |
| 4 | 11.140174 | 1.064 | 0 | 0.481734 | 0.041134 | 0.835282 |

---

## Interpretation (brief)

- Topology is active and stacks form consistently (mean_stack_count ≈ 1.0–1.17).
- No rebirths observed under these conditions.
- Weight mass concentrates strongly on channel 0 or 3, with seed‑dependent dominance.
- Entropy remains low, reflecting high concentration despite the Phase‑2 SNR reversal.

---

## Phase‑wise weight diagnostics

Pre = t < T/2, Post = t ≥ T/2.

| seed | pre_entropy | post_entropy | pre_w0 | post_w0 | pre_w3 | post_w3 | weight_cross_time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.868597 | 0.244838 | 0.075219 | 0.051382 | 0.699766 | 0.937770 | 0 |
| 1 | 0.890530 | 0.031459 | 0.078569 | 0.001212 | 0.692955 | 0.995573 | 0 |
| 2 | 0.911792 | 0.045064 | 0.683312 | 0.993345 | 0.079202 | 0.001589 | — |
| 3 | 0.923289 | 0.036664 | 0.677895 | 0.994751 | 0.080389 | 0.001232 | — |
| 4 | 0.921811 | 0.041656 | 0.080665 | 0.001603 | 0.676672 | 0.993892 | 0 |

Notes:
- `weight_cross_time` is the first t where w3 > w0; “—” means no cross occurred.
- The Phase‑2 SNR shift tends to **amplify the pre‑phase dominant channel** rather than invert it.

---

## Interpretation (consolidated)

Now it gets structurally interesting.

### 1) Stacks are real and stable
mean_stack_count ≈ 1.0–1.17  
→ StackManager engages, topology exists, we are past the “all zeros” regime.

### 2) No rebirths
total_rebirths = 0  
→ wealth does not collapse, inhibition is not ruin‑triggered, the Phase‑2 SNR shift is **not** a system shock.  
This is a stability signal, not a weakness.

### 3) No systematic reversal
Phase 2 inverts **relative SNR**, but weights do **not** systematically flip.  
They stay concentrated, and Phase 2 typically **amplifies the pre‑phase dominant channel**.

Interpretation: the system tracks **expected value**, not variance alone.  
Variance increases uncertainty but does not create a directional disadvantage without a mean shift or costs.

---

## What would actually “break” it?

1) **Mean inversion**  
e.g., Phase 2 swaps μ so the opposing channel becomes truly better.

2) **Costs + SNR combined**  
High‑variance on the dominant channel **plus** explicit costs to create a real disadvantage.

3) **False evidence lock‑in**  
Trigger inhibition under false evidence, then flip means to expose mis‑adaptation.

---

## What G3.4.12 proves

1. Stack mechanics function under controlled stress.  
2. The system is robust to **pure variance shifts**.  
3. Rebirth does not trigger lightly.  
4. Topology is stable, not fragile.

This is Phase‑G‑worthy evidence of a stable, adaptive dynamical system.
