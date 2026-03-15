# v1.0 Readiness Checklist (v0.9.5-9)

This checklist is the readiness gate for the v0.9.5 iteration before v1.0 release stabilization.

| Item | Binary pass/fail criterion | Result | Evidence link |
| --- | --- | --- | --- |
| 1. Deterministic replay (long-run, ESS probe, regime robustness) | PASS if fixed-seed replay hash matches reference hash for each harness under identical (seed, config, backend). FAIL otherwise. | PASS | [tests/test_v095_v1_readiness.py](../../tests/test_v095_v1_readiness.py) `test_v095_v1_readiness_deterministic_replay_reference_hashes_for_three_harnesses` |
| 2. Long-run stability (500 generations) | PASS if 500-generation reference long-run simulation has no NaN/Inf in fitness values and no NaN/Inf in flow-matrix row sums for any generation. FAIL otherwise. | PASS | [tests/test_v095_v1_readiness.py](../../tests/test_v095_v1_readiness.py) `test_v095_v1_readiness_long_run_stability_500_generations_no_nan_inf` |
| 3. Experiment reproducibility (evidence artifacts) | PASS if two independent evidence command runs produce identical SHA-256 hashes for each declared artifact file. FAIL otherwise. | PASS | [tests/test_v095_v1_readiness.py](../../tests/test_v095_v1_readiness.py) `test_v095_v1_readiness_reproducibility_evidence_artifact_hashes_stable_across_two_runs`; `make generate-v095-evidence`; [scripts/generate_v095_evidence.py](../../scripts/generate_v095_evidence.py) |
| 4. Architecture documentation integrity | PASS if all three architecture artifacts exist and pass stale-reference/diagram checks. FAIL otherwise. | PASS | [docs/v4/architecture-v095.md](architecture-v095.md), [docs/v4/generation-loop-sequence.md](generation-loop-sequence.md), [docs/v4/data-flow.md](data-flow.md); [tests/test_v095_architecture_docs.py](../../tests/test_v095_architecture_docs.py) `test_v095_architecture_docs_artifact_existence`, `test_v095_architecture_docs_no_stale_module_references`, `test_v095_architecture_docs_sequence_diagram_plantuml_syntax`, `test_v095_architecture_docs_data_flow_plantuml_syntax` |
| 5. Public API finalized | PASS if all symbols declared in `capitalmarket/capitalselector/__init__.py::__all__` are individually importable and no implicit public symbols exist. FAIL otherwise. | PASS | [capitalmarket/capitalselector/__init__.py](../../capitalmarket/capitalselector/__init__.py); [tests/test_v095_module_boundaries.py](../../tests/test_v095_module_boundaries.py) `test_v095_module_boundaries_public_api_symbols_are_individually_importable`, `test_v095_module_boundaries_no_implicit_public_api_surface` |
| 6. Backward compatibility (0.9.x experiment configs) | PASS if `ESSProbeConfig`, `RegimeRobustnessConfig`, and `PeriodicityExperimentConfig` each complete a minimal single-generation integration run without error. FAIL otherwise. | PASS | [tests/test_v095_v1_readiness.py](../../tests/test_v095_v1_readiness.py) `test_v095_v1_readiness_backward_compat_configs_run_single_generation_without_error` |
| 7. Test gates | PASS if `make test-cpu` is green, and if CUDA is available then `make test-gpu` is green. FAIL otherwise. | PASS | [Makefile](../../Makefile) targets `test-cpu`, `test-gpu`; release evidence: [docs/v4/evidence-0.9.5-8.md](evidence-0.9.5-8.md) |
| 8. No open blockers in v0.9.5 milestone | PASS if `gh issue list --state open --label blocker --milestone v0.9.5 --json number,title,labels,state,url` returns `[]`. FAIL otherwise. | PASS | Command snapshot in this issue implementation: `[]` |

## Waiver Notes

No checklist item is waived in this revision.
