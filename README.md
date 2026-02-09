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
Phase E (technical completion and feedback coupling) is planned.

Phase F (Production Readiness, Profile A) is complete as of **v0.5.12**.
See `docs/phase_f_final_fazit.md` for the final summary and test status.

### Phase F Scope (v1 / Profile A)
- Canonical spec: `docs/math-v1.md`
- Architecture: `docs/architecture.md`
- External interface: `docs/interface.md`
- Implementation spec: `docs/phase_f_implementation_spec.md`

Non‑scope (v1): Part II (Equity/Debt, credit condition, pool finance), sparsity mechanics, and Profile B.

### Tests
Run tests via Docker:
- `make test-cpu`
- `make test-gpu`

### Diagrams
SVG diagrams are rendered on Windows for correct math glyphs; do not overwrite them from Linux containers.

## License

This project is released under a **restricted research-use license**.
See `LICENSE.md` for details.

For commercial use, derivative works, or extended permissions,
please contact the author.
