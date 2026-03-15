# Evidence 0.9.5-1 Automation and Regression Guard

Issue: #116  
Parent: #114  
Scope: CO-0.9.5-2 + CO-0.9.5-3

## Production and Test Scope

- `scripts/generate_v095_evidence.py`
  - deterministic evidence artifact generation command
  - baseline update command for performance regression guard
- `tests/_v095_perf_baseline.json`
  - committed baseline artifact (versioned with test code)
- `tests/test_v095_evidence_and_regression.py`
  - required issue scenarios for artifacts, determinism, performance guard, baseline update, and table content
- `Makefile`
  - documented command surface
  - `generate-v095-evidence`
  - `update-perf-baseline`

## Documented Commands

```bash
make generate-v095-evidence
make update-perf-baseline
```

Make target note:

- baseline update process is documented inline in `Makefile` comments above `update-perf-baseline`
- default baseline path is `tests/_v095_perf_baseline.json`

## Evidence 1: Artifact Directory Listing

Containerized run command:

```bash
docker run --rm -v "/home/andreas/prj/dl":/workspace -w /workspace -u 1000:1000 dl:cpu \
  bash -lc "PYTHONPATH=/workspace python scripts/generate_v095_evidence.py generate --output-dir /tmp/v095-evidence-a --seed 2026"
```

Listing:

```text
caching_boundary_trace.json
exact_vs_sampled_comparison.md
semantic_invariant_output.json
```

## Evidence 2: Relative Performance Guard Expression

Containerized baseline update command:

```bash
docker run --rm -v "/home/andreas/prj/dl":/workspace -w /workspace -u 1000:1000 dl:cpu \
  bash -lc "PYTHONPATH=/workspace python scripts/generate_v095_evidence.py update-perf-baseline --baseline-file /tmp/v095-perf-baseline-116.json"
```

Output:

```text
performance baseline updated: baseline_time=0.253064s, threshold=baseline_time*1.6=0.404902s
```

This demonstrates the required relative threshold contract:

- threshold = `baseline_time * 1.6`
- baseline_time is computed via `median_of_3` independent measurements
- no absolute wall-clock constant used as the guard boundary

## Evidence 3: Byte-Identical Determinism Across Independent Runs

Two independent artifact generations were compared via SHA-256:

```text
a49f837454c38cf01160242823f0afa9dc3040ba263e680565a1067e43f7af3a  /tmp/v095-evidence-a/caching_boundary_trace.json
e063659ea94f65835198fe29ca00ef96e4c5c804744a1c6599efd844033c82bd  /tmp/v095-evidence-a/exact_vs_sampled_comparison.md
3502dca92471bff4c3e5422acafb3d4494bbb3e0eb5c14422cfcc8124b29b6e6  /tmp/v095-evidence-a/semantic_invariant_output.json
a49f837454c38cf01160242823f0afa9dc3040ba263e680565a1067e43f7af3a  /tmp/v095-evidence-b/caching_boundary_trace.json
e063659ea94f65835198fe29ca00ef96e4c5c804744a1c6599efd844033c82bd  /tmp/v095-evidence-b/exact_vs_sampled_comparison.md
3502dca92471bff4c3e5422acafb3d4494bbb3e0eb5c14422cfcc8124b29b6e6  /tmp/v095-evidence-b/semantic_invariant_output.json
```

Matching hashes for all three artifact files confirm byte-identical determinism for identical `(seed, config)`.

## CPU Gate

```bash
make test-cpu
```

Result:

- 574 passed
- 87 skipped
- 1 xfailed
- 3 warnings

Runtime:

- ~21.17s
