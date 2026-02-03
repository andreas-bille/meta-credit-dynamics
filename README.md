# Meta-Credit Dynamics

This repository contains a minimal research implementation exploring
**repair, stabilization, and meta-credit dynamics** in capital-like systems.

The focus is on **forensic analysis**, not optimization:
repair mechanisms are introduced only to observe how they delay collapse,
shift risk, and create emergent broker paths.

The project is intentionally:
- non-optimizing
- semantically minimal
- scale-invariant
- research-oriented

## Structure

- `capitalmarket/` – core implementation (selectors, repair policies, telemetry)
- `doc/` – conceptual papers and technical specifications
- `notebooks/` – demonstrative ON/OFF experiments for Phase D
- `tests/` – reproducible tests validating Phase-D properties

## Status

Phase D (Repair & Stabilization) is complete as of **v0.3.1**.  
Phase E (Sedimented Mediation Paths) is complete as of **v0.4.0**.  
Phase F (Production Readiness) is planned.

## License

This project is released under a **restricted research-use license**.
See `LICENSE.md` for details.

For commercial use, derivative works, or extended permissions,
please contact the author.
