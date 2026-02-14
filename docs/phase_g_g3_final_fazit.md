# Phase G – G3 Final Fazit (v0.6.3)

This document summarizes what was learned across G3.1–G3.4 runs, up to the
current checkpoint (v0.6.3). It is descriptive and non‑canonical.

---

## Scope Covered

- G3.1–G3.3 deterministic and regime‑switch experiments
- G3.4.* world variations (costs, persistence, volatility, subset regimes, shocks)
- Topology activation toggle (G3.5) and reruns
- Controlled ruin test (G3.4.9)
- Stack trigger world (G3.4.10)
- Regime‑flip + shock attempts for sediment (G3.4.11)

---

## Core Findings (Concise)

1) **Topology is active and stable**  
With topology enabled, stacks form and persist (mean_stack_count ~1).  
Topology was previously absent due to a missing call in the loop.

2) **Rebirth is functional but needs economic pressure**  
Rebirth triggers under ruinous regimes (G3.4.9).  
In benign worlds, wealth remains above the threshold → no rebirth.

3) **Weight dynamics respond to time structure**  
Autocorrelation vs. shuffled regimes shows measurable changes in entropy/weight share
even with identical marginals and shared noise.

4) **World perturbations are mostly “soft”**  
Costs, volatility drift, and mild shocks move wealth/entropy but do not
induce strong structural breaks in topology.

5) **Sediment has not been produced yet**  
Even with aggressive flip + shock windows (G3.4.11), stacks destabilize only
marginally and do not dissolve → no sediment.

---

## What Worked

- Deterministic experiment scaffolding across G3.1–G3.4
- Topology activation toggle with reproducible reruns
- Controlled ruin run confirming rebirth path
- Deterministic cluster world demonstrating stack formation

---

## What Remains Open

- **Sediment creation**: no stack dissolution observed yet under exogenous shocks.
- **Stack stability thresholds**: likely conservative; needs diagnostic visibility.
- **Structural break design**: stronger, localized destabilization may be required.

---

## Practical Interpretation

The system **learns via weights** under many regimes, while **topology remains stable**.
