from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tests" / "_v095_perf_baseline.json"

ARTIFACT_FILENAMES = (
    "semantic_invariant_output.json",
    "exact_vs_sampled_comparison.md",
    "caching_boundary_trace.json",
)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    root = str(REPO_ROOT)
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{root}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = root
    return env


def _run_make(target: str, *vars_: str) -> subprocess.CompletedProcess[str]:
    command = ["make", target, *vars_]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        check=True,
        text=True,
        capture_output=True,
    )


def test_v095_evidence_command_emits_all_declared_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence_artifacts"

    _run_make(
        "generate-v095-evidence",
        "V095_EVIDENCE_RUNNER=local",
        f"V095_EVIDENCE_OUTPUT_DIR={output_dir}",
    )

    for filename in ARTIFACT_FILENAMES:
        artifact_path = output_dir / filename
        assert artifact_path.exists(), f"missing artifact file: {artifact_path}"
        assert artifact_path.is_file(), f"artifact path is not a file: {artifact_path}"


def test_v095_evidence_artifacts_are_byte_identical_across_two_runs(tmp_path: Path) -> None:
    output_dir_a = tmp_path / "run_a"
    output_dir_b = tmp_path / "run_b"

    _run_make(
        "generate-v095-evidence",
        "V095_EVIDENCE_RUNNER=local",
        f"V095_EVIDENCE_OUTPUT_DIR={output_dir_a}",
    )
    _run_make(
        "generate-v095-evidence",
        "V095_EVIDENCE_RUNNER=local",
        f"V095_EVIDENCE_OUTPUT_DIR={output_dir_b}",
    )

    for filename in ARTIFACT_FILENAMES:
        bytes_a = (output_dir_a / filename).read_bytes()
        bytes_b = (output_dir_b / filename).read_bytes()
        assert bytes_a == bytes_b, f"artifact differs between deterministic runs: {filename}"


def test_v095_performance_regression_guard_uses_relative_threshold_expression(tmp_path: Path) -> None:
    baseline_payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    baseline_time = float(baseline_payload["baseline_seconds"])
    factor = float(baseline_payload["regression_multiplier"])

    measured_file = tmp_path / "measured_perf_baseline.json"
    _run_make(
        "update-perf-baseline",
        "V095_EVIDENCE_RUNNER=local",
        f"V095_PERF_BASELINE_FILE={measured_file}",
    )

    measured_payload = json.loads(measured_file.read_text(encoding="utf-8"))
    measured_time = float(measured_payload["baseline_seconds"])

    assert int(measured_payload["scenario"]["population_size"]) == 256
    assert int(measured_payload["scenario"]["runtime_horizon"]) == 16
    assert str(measured_payload["measurement_method"]) == "median_of_3"
    assert int(measured_payload["measurement_statistics"]["sample_count"]) == 3
    assert len(list(measured_payload["measurement_runs"])) == 3

    threshold = baseline_time * factor
    assert measured_time <= threshold, (
        "performance regression guard failed: "
        f"elapsed={measured_time:.6f}s > baseline_time*{factor:.1f}={baseline_time:.6f}*{factor:.1f}={threshold:.6f}s"
    )


def test_v095_update_perf_baseline_command_updates_target_artifact(tmp_path: Path) -> None:
    baseline_path = tmp_path / "perf_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "version": 1,
                "baseline_seconds": 0.0,
                "regression_multiplier": 1.6,
                "scenario": {
                    "population_size": 256,
                    "runtime_horizon": 16,
                },
                "last_measurement": {
                    "elapsed_seconds": 0.0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    _run_make(
        "update-perf-baseline",
        "V095_EVIDENCE_RUNNER=local",
        f"V095_PERF_BASELINE_FILE={baseline_path}",
    )

    updated_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert float(updated_payload["baseline_seconds"]) > 0.0
    assert float(updated_payload["regression_multiplier"]) == 1.6
    assert str(updated_payload["measurement_method"]) == "median_of_3"
    assert int(updated_payload["measurement_statistics"]["sample_count"]) == 3
    assert len(list(updated_payload["measurement_runs"])) == 3
    assert int(updated_payload["scenario"]["population_size"]) == 256
    assert int(updated_payload["scenario"]["runtime_horizon"]) == 16


def test_v095_exact_vs_sampled_artifact_contains_metric_table(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence_artifacts"
    _run_make(
        "generate-v095-evidence",
        "V095_EVIDENCE_RUNNER=local",
        f"V095_EVIDENCE_OUTPUT_DIR={output_dir}",
    )

    comparison_path = output_dir / "exact_vs_sampled_comparison.md"
    content = comparison_path.read_text(encoding="utf-8")

    assert "| mode | pair_count | mean_pairwise_distance | structural_entropy | normalized_structural_entropy |" in content

    exact_row = ""
    sampled_row = ""
    for line in content.splitlines():
        if line.startswith("| exact |"):
            exact_row = line
        if line.startswith("| fallback_sampled |"):
            sampled_row = line

    assert exact_row
    assert sampled_row

    exact_cells = [cell.strip() for cell in exact_row.strip("|").split("|")]
    sampled_cells = [cell.strip() for cell in sampled_row.strip("|").split("|")]

    assert int(exact_cells[1]) > 0
    assert int(sampled_cells[1]) > 0

    float(exact_cells[2])
    float(exact_cells[3])
    float(exact_cells[4])
    float(sampled_cells[2])
    float(sampled_cells[3])
    float(sampled_cells[4])


def test_v095_makefile_declares_documented_evidence_commands() -> None:
    makefile_content = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "generate-v095-evidence" in makefile_content
    assert "update-perf-baseline" in makefile_content
