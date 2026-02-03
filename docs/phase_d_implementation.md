# Phase D – Minimal Implementation

This bundle adds a minimal, non-optimizing Phase D implementation on top of Phase C.

## What’s included

- `capitalmarket/capitalselector/repair.py`
  - `RepairPolicySet` and policies:
    - `CapsPolicy`
    - `LagPolicy`
    - `SoftBailoutPolicy`
    - `IsolationPolicy`
  - `RepairContext` (explicit deterministic state: prev weights, lag buffers, cooldowns)

- `capitalmarket/capitalselector/telemetry.py`
  - `TelemetryLogger` (optional JSONL events)

- `capitalmarket/capitalselector/tests_phase_d.py`
  - Tests A6, A8, A9, A10 (minimal acceptance tests)

- Notebooks (ON/OFF):
  - `phase_d_caps_demo.ipynb`
  - `phase_d_lag_demo.ipynb`
  - `phase_d_bailout_demo.ipynb`
  - `phase_d_isolation_demo.ipynb`

## Run tests

```bash
python -m capitalmarket.capitalselector.tests_phase_d
```

## Notes

- Policies are deterministic and do not introduce new learning rules.
- Weight-changing policies (caps/bail-out/isolation) always re-normalize to the simplex (sum=1 on active ids).
- Lag injection is implemented as an observation wrapper; signature checks accept both ~L and ~2L shifts depending on the collapse proxy.
