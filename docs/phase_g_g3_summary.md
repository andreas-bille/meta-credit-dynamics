# Phase G – G3.1–G3.3 Summary (Tonspur)

## G3.1 – DeterministicScriptWorld
- Deterministische Baseline bestätigt.
- Identische Seeds → identische Observables + identische `canonical_state_dump()` an t=0/100/200.
- Logging/Observables verändern den Kernelzustand nicht.

## G3.2 – RegimeSwitchBanditWorld vs. MarginalMatchedControlWorld
- Beide Welten reproduzierbar unter festen Seeds.
- Marginal‑Match (mean/variance per channel) erfüllt.
- No‑semantic‑mutation Test bestanden.

## G3.3 – Parameter‑Sweep (p × seed)
- Sweep p∈{0.01,0.05,0.1} × seeds{0..4} deterministisch.
- Alle Runs liefern finite final_wealth.
- No‑semantic‑mutation Test bestanden.

## Meta‑Resultat
- Experiment‑Pipeline stabil, deterministisch, ohne Semantik‑Drift.
- Grundlage für empirische Auswertung (G3.4) vorhanden.
