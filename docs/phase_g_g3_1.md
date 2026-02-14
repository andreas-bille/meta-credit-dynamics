# Phase G – G3.1 DeterministicScriptWorld

This document describes the deterministic baseline world used in G3.1.

Purpose:
- validate reproducibility
- verify logging integrity
- ensure no kernel semantic drift

World contract:
- step(t) -> { r: array[float], c: float }
- K=3, r=[0.01, 0.00, -0.01], c=0.0
- deterministic, no stochasticity, no seed dependence
