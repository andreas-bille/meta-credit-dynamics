# Phase G – G3.3 Parameter Sweep Robustness

This document defines the G3.3 parameter sweep over regime switching probability `p`
and seed replication for RegimeSwitchBanditWorld.

Purpose:
- test robustness of observable dynamics across regime persistence levels
- verify determinism across seeds
- ensure no kernel semantic drift

Scope:
- Profile A only
- p ∈ {0.01, 0.05, 0.1}
- seeds ∈ {0, 1, 2, 3, 4}
- horizon T = 500 (t = 0..499)
