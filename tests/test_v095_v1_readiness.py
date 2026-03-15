"""Tests for Issue #124: v1.0 readiness checklist (D3)."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from capitalmarket.capitalselector.ess_evaluator import ESSProbeConfig
from capitalmarket.capitalselector.ess_evaluator import run_ess_probe_experiment
from capitalmarket.capitalselector.experiments.long_run_harness import LongRunConfig
from capitalmarket.capitalselector.experiments.long_run_harness import run_long_run_harness
from capitalmarket.capitalselector.genome_serialization import genome_from_dict
from capitalmarket.capitalselector.periodicity_experiment import PeriodicityExperimentConfig
from capitalmarket.capitalselector.periodicity_experiment import run_periodicity_experiment
from capitalmarket.capitalselector.regime_robustness import RegimeRobustnessConfig
from capitalmarket.capitalselector.regime_robustness import RobustnessPerturbationConfig
from capitalmarket.capitalselector.regime_robustness import run_regime_robustness_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_FILENAMES = (
    "semantic_invariant_output.json",
    "exact_vs_sampled_comparison.md",
    "caching_boundary_trace.json",
)


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


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
    return subprocess.run(
        ["make", target, *vars_],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        check=True,
        text=True,
        capture_output=True,
    )


def _resident_payload_2ch() -> dict[str, object]:
    return {
        "flow_matrix": [[0.7, 0.3], [0.3, 0.7]],
        "output_weights": [0.5, 0.5],
        "lambda_risk": 0.2,
        "selector_policy": "term_risk",
        "settlement_params": {
            "lambda_cash_share": 0.6,
            "accept_by_default": 1.0,
            "future_maturity_offset": 1.0,
        },
    }


def _resident_payload_3ch() -> dict[str, object]:
    return {
        "flow_matrix": [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.3, 0.0, 0.7]],
        "output_weights": [0.4, 0.3, 0.3],
        "lambda_risk": 0.2,
        "selector_policy": "term_risk",
        "settlement_params": {
            "lambda_cash_share": 0.5,
            "accept_by_default": 1.0,
            "future_maturity_offset": 1.0,
        },
    }


def _long_run_world_parameters(*, channel_count: int) -> dict[str, object]:
    returns = np.linspace(0.01, 0.03, int(channel_count), dtype=np.float64)
    costs = np.linspace(0.01, 0.013, int(channel_count), dtype=np.float64)
    return {
        "returns": [float(v) for v in returns.tolist()],
        "costs": [float(v) for v in costs.tolist()],
        "jitter_scale": 0.0,
        "initial_wealth": 1.0,
        "rebirth_enabled": False,
    }


def _ess_worlds() -> dict[str, dict[str, object]]:
    return {
        "stable": {
            "returns": [0.03, 0.01],
            "costs": [0.01, 0.012, 0.011, 0.013, 0.01, 0.012],
            "jitter_scale": 0.001,
            "initial_wealth": 1.0,
        },
        "stress": {
            "returns": [0.01, -0.01],
            "costs": [0.012, 0.013, 0.014, 0.013, 0.012, 0.013],
            "jitter_scale": 0.001,
            "initial_wealth": 1.0,
        },
    }


def _regime_worlds() -> dict[str, dict[str, object]]:
    return {
        "periodic_stable": {
            "world_type": "periodic",
            "world_parameters": {
                "channel_productivity": [0.95, 0.75],
                "channel_risk": [0.08, 0.12],
                "liquidity_scale": 1.0,
                "regime_period": 3,
                "stress_probability": 0.05,
                "stress_intensity": 0.05,
            },
        },
        "non_periodic_stressed": {
            "world_type": "non_periodic",
            "world_parameters": {
                "channel_productivity": [0.55, 0.35],
                "channel_risk": [0.22, 0.28],
                "liquidity_scale": 0.9,
                "regime_period": 3,
                "stress_probability": 0.7,
                "stress_intensity": 0.5,
            },
        },
    }


def test_v095_v1_readiness_deterministic_replay_reference_hashes_for_three_harnesses() -> None:
    # Long-run harness deterministic replay (50 generations)
    long_run_cfg = LongRunConfig(
        seed=2026031601,
        population_size=4,
        channel_count=3,
        horizon=2,
        generations=50,
        world_parameters=_long_run_world_parameters(channel_count=3),
        checkpoint_interval=0,
        checkpoint_path=None,
        backend="cpu",
    )
    long_run_ref = run_long_run_harness(config=long_run_cfg)
    long_run_ref_hash = _canonical_sha256(asdict(long_run_ref))
    long_run_replay_hash = _canonical_sha256(asdict(run_long_run_harness(config=long_run_cfg)))
    assert long_run_replay_hash == long_run_ref_hash

    # ESS probe deterministic replay (50 resident generations, 50-tau invasion horizon)
    ess_cfg = ESSProbeConfig(
        seed=2026031602,
        world_parameters_by_regime=_ess_worlds(),
        regimes=["stable", "stress"],
        n_resident_generations=50,
        n_mutant_trials=2,
        runtime_horizon=50,
        resident_population_size=2,
        mutant_population_size=1,
        dt=1.0,
        backend="cpu",
        min_survival_tau=1,
    )
    ess_ref = run_ess_probe_experiment(genome_from_dict(_resident_payload_2ch()), ess_cfg)
    ess_ref_hash = _canonical_sha256(asdict(ess_ref))
    ess_replay_hash = _canonical_sha256(
        asdict(run_ess_probe_experiment(genome_from_dict(_resident_payload_2ch()), ess_cfg))
    )
    assert ess_replay_hash == ess_ref_hash

    # Regime robustness deterministic replay (50-tau horizon)
    robust_cfg = RegimeRobustnessConfig(
        seed=2026031603,
        world_parameters_by_regime=_regime_worlds(),
        regimes=["periodic_stable", "non_periodic_stressed"],
        runtime_horizon=50,
        resident_population_size=3,
        dt=1.0,
        backend="cpu",
        perturbation=RobustnessPerturbationConfig(
            start_tau=4,
            duration=3,
            attenuation=0.2,
            recovery_horizon=20,
        ),
        perturbation_regime="periodic_stable",
    )
    robust_ref = run_regime_robustness_experiment(genome_from_dict(_resident_payload_2ch()), robust_cfg)
    robust_ref_hash = _canonical_sha256(asdict(robust_ref))
    robust_replay_hash = _canonical_sha256(
        asdict(run_regime_robustness_experiment(genome_from_dict(_resident_payload_2ch()), robust_cfg))
    )
    assert robust_replay_hash == robust_ref_hash


def test_v095_v1_readiness_long_run_stability_500_generations_no_nan_inf() -> None:
    config = LongRunConfig(
        seed=2026031604,
        population_size=4,
        channel_count=3,
        horizon=1,
        generations=500,
        world_parameters=_long_run_world_parameters(channel_count=3),
        checkpoint_interval=0,
        checkpoint_path=None,
        backend="cpu",
    )

    result = run_long_run_harness(config=config)

    assert len(result.generation_records) == 500

    for record in result.generation_records:
        assert record.fitness_values_finite
        assert record.flow_row_sums_finite
        assert np.isfinite(record.min_fitness)
        assert np.isfinite(record.max_fitness)
        assert np.isfinite(record.min_flow_row_sum)
        assert np.isfinite(record.max_flow_row_sum)

    for generation_state in result.population_states_by_generation:
        for genome_payload in generation_state:
            flow_matrix = np.asarray(genome_payload["flow_matrix"], dtype=np.float64)
            row_sums = np.sum(flow_matrix, axis=1)
            assert np.all(np.isfinite(row_sums))


def test_v095_v1_readiness_reproducibility_evidence_artifact_hashes_stable_across_two_runs(tmp_path: Path) -> None:
    output_dir_a = tmp_path / "evidence_run_a"
    output_dir_b = tmp_path / "evidence_run_b"

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

    hashes_a = {filename: _sha256_file(output_dir_a / filename) for filename in ARTIFACT_FILENAMES}
    hashes_b = {filename: _sha256_file(output_dir_b / filename) for filename in ARTIFACT_FILENAMES}

    assert hashes_a == hashes_b


def test_v095_v1_readiness_backward_compat_configs_run_single_generation_without_error() -> None:
    resident_2ch = genome_from_dict(_resident_payload_2ch())

    # ESS probe (0.9.x config) single-generation integration run.
    ess_cfg = ESSProbeConfig(
        seed=2026031605,
        world_parameters_by_regime=_ess_worlds(),
        regimes=["stable", "stress"],
        n_resident_generations=1,
        n_mutant_trials=1,
        runtime_horizon=1,
        resident_population_size=2,
        mutant_population_size=1,
        dt=1.0,
        backend="cpu",
        min_survival_tau=1,
    )
    ess_result = run_ess_probe_experiment(resident_2ch, ess_cfg)
    assert len(ess_result.trial_results) == 2

    # Regime robustness (0.9.x config) single-generation integration run.
    robust_cfg = RegimeRobustnessConfig(
        seed=2026031606,
        world_parameters_by_regime=_regime_worlds(),
        regimes=["periodic_stable", "non_periodic_stressed"],
        runtime_horizon=1,
        resident_population_size=2,
        dt=1.0,
        backend="cpu",
        perturbation=RobustnessPerturbationConfig(
            start_tau=0,
            duration=1,
            attenuation=0.2,
            recovery_horizon=1,
        ),
        perturbation_regime="periodic_stable",
    )
    robust_result = run_regime_robustness_experiment(resident_2ch, robust_cfg)
    assert len(robust_result.regime_comparison_metrics) == 2

    # Periodicity experiment (0.9.x config) single-generation integration run.
    genomes = [genome_from_dict(_resident_payload_3ch()), genome_from_dict(_resident_payload_3ch())]
    periodicity_cfg = PeriodicityExperimentConfig(
        seed=2026031607,
        runtime_horizon=1,
        regime_period=3,
        stress_probability=0.4,
        stress_intensity=0.25,
        channel_productivity=(0.9, 0.6, 0.3),
        channel_risk=(0.1, 0.15, 0.2),
        liquidity_scale=1.0,
        initial_wealth=1.0,
        rebirth_enabled=False,
    )
    periodicity_result = run_periodicity_experiment(genomes, config=periodicity_cfg, backend="cpu")
    assert periodicity_result.metrics.step_count == 1
