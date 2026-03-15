# Meta-Credit Dynamics

> Note: Public snapshot of the meta-credit-dynamics research project. Development happens in a private repository.

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

## Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gap-labs/meta-credit-dynamics/blob/main/notebooks/team_demo.ipynb)

Public demo notebook:
- `notebooks/team_demo.ipynb` (generated from `notebooks/team.ipynb` during publish)

Notes:
- The Colab link opens the notebook directly.
- For security reasons, Colab does not auto-run notebooks via URL.
- In Colab, run `Runtime -> Run all` once.

## Structure

- `capitalmarket/` – core implementation (runtime, selectors, repair/stabilization, telemetry)
- `capitalselector/` – package namespace entrypoint
- `docs/` – public-facing specs, architecture notes, and release notes
- `notebooks/` – public demo notebook (`team_demo.ipynb`, generated during publish)
- `tests/` – representative, reproducible tests for invariants and phase behavior

## Status

Current public snapshot references release **v1.0.0**.

v0.9.x stabilization and readiness work is integrated in this branch, including:

- long-run harness and checkpoint/replay path
- experiment dataset + visualization comparison utilities
- architecture/module-boundary hardening
- readiness checklist and evidence automation updates

Reference docs:

- v1 baseline: `docs/v1/math-v1.md`, `docs/v1/architecture.md`, `docs/v1/interface.md`
- v2 baseline: `docs/v2/math-v2.md`, `docs/v2/architecture.md`, `docs/v2/README.md`
- v3 implementation details: `docs/v3/impl_spec_phase_ii.md`, `docs/v3/architecture_phase_ii.md`, `docs/v3/phase_ii_prompt_seq.md`
- v4 stabilization and readiness artifacts: `docs/v4/issues-0.9.5.md`, `docs/v4/v1-readiness-checklist.md`
- v5 post-v1 planning and review artifacts: `docs/v5/post_v1_structural_mutation_roadmap.md`, `docs/v5/v1_0_0_rc_diff_audit_and_architecture_stress_review.md`

### Tests
Run tests with Makefile/docker workflow:

- CPU suite: `make test-cpu`
- GPU suite: `make test-gpu`

Phase-II protocol tests include deterministic contracts and evaluation CI logic,
including `tests/test_phase_ii_evaluation_protocol.py`.

Direct pytest (optional):

- All tests: `pytest -q`
- CPU-focused run (skip CUDA/GPU tests): `CAPM_SKIP_CUDA_TESTS=1 pytest -q`

Optional public-facing subset:
- `pytest -q tests/test_invariants.py tests/tests_phase_c.py tests/tests_phase_d.py tests/tests_phase_e.py`

## License

This project is released under a **restricted research-use license**.
See `LICENSE.md` for details.

For commercial use, derivative works, or extended permissions,
please contact the author.
