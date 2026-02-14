# Phase G – G3.2 RegimeSwitchBanditWorld

This document describes the stochastic regime-switching baseline and its
marginal-matched control world for G3.2.

World A (RegimeSwitchBanditWorld):
- K=5, two regimes
- Regime A mean: [0.02, 0, 0, 0, 0]
- Regime B mean: [0, 0, 0, 0.02, 0]
- Regime flips with probability p per step
- Gaussian noise with sigma=0.01
- c = 0.0

World B (MarginalMatchedControlWorld):
- No regime state
- Per-step choose channel 0 or 3 as high-mean with equal probability
- Same noise sigma
- Matches marginal mean/variance of World A

Purpose:
- test regime effects vs marginal-matched control
- verify deterministic behavior under fixed seed
- ensure no kernel semantic drift
