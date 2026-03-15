from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from capitalmarket.capitalselector.evolution_contracts import PAIRWISE_DISTANCE_MODE_EXACT
from capitalmarket.capitalselector.evolution_contracts import PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED
from capitalmarket.capitalselector.fitness_engine import FitnessEvaluationConfig
from capitalmarket.capitalselector.fitness_engine import simulate_genome_population
from capitalmarket.capitalselector.genome_serialization import genome_from_dict

DEFAULT_ARTIFACT_DIR = Path("docs/v4/evidence-0.9.5-1-artifacts")
DEFAULT_BASELINE_FILE = Path("tests/_v095_perf_baseline.json")

SEMANTIC_INVARIANT_FILE = "semantic_invariant_output.json"
EXACT_VS_SAMPLED_FILE = "exact_vs_sampled_comparison.md"
CACHING_BOUNDARY_TRACE_FILE = "caching_boundary_trace.json"

PERF_REGRESSION_FACTOR = 1.6
PERF_BASELINE_SAMPLE_COUNT = 3
PERF_SCENARIO_POPULATION = 256
PERF_SCENARIO_HORIZON = 16
PERF_SCENARIO_CHANNELS = 3
PERF_SCENARIO_SAMPLE_PAIRS = 1024
PERF_SCENARIO_SEED = 9501


def _world_parameters(*, channel_count: int, jitter_scale: float = 0.0) -> dict[str, Any]:
    returns = np.linspace(0.01, 0.03, int(channel_count), dtype=np.float64)
    costs = np.linspace(0.01, 0.013, int(channel_count), dtype=np.float64)
    return {
        "returns": [float(v) for v in returns.tolist()],
        "costs": [float(v) for v in costs.tolist()],
        "jitter_scale": float(jitter_scale),
        "initial_wealth": 1.0,
    }


def _genome_payload(flow_matrix: np.ndarray, *, lambda_risk: float = 0.2) -> dict[str, Any]:
    matrix = np.asarray(flow_matrix, dtype=np.float64)
    n = int(matrix.shape[0])
    return {
        "flow_matrix": matrix.tolist(),
        "output_weights": (np.ones(n, dtype=np.float64) / float(n)).tolist(),
        "lambda_risk": float(lambda_risk),
        "selector_policy": "term_risk",
        "settlement_params": {
            "lambda_cash_share": 0.6,
            "accept_by_default": 1.0,
            "future_maturity_offset": 1.0,
        },
    }


def _random_row_stochastic_matrices(*, population: int, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(np.random.PCG64(np.uint64(seed)))
    matrices: list[np.ndarray] = []
    for _ in range(int(population)):
        matrix = np.zeros((n, n), dtype=np.float64)
        for row in range(n):
            matrix[row, :] = rng.dirichlet(np.ones(n, dtype=np.float64))
        matrices.append(matrix)
    return matrices


def _population_genomes(*, population: int, n: int, seed: int) -> list[Any]:
    matrices = _random_row_stochastic_matrices(population=population, n=n, seed=seed)
    return [genome_from_dict(_genome_payload(matrix)) for matrix in matrices]


def _config(
    *,
    seed: int,
    population_size: int,
    runtime_horizon: int,
    world_parameters: dict[str, Any],
    backend: str = "cpu",
    pairwise_distance_mode: str = PAIRWISE_DISTANCE_MODE_EXACT,
    pairwise_distance_max_exact_population: int = 1024,
    pairwise_distance_sampled_pair_count: int = 10_000,
    pairwise_distance_sample_seed: int = 0,
    generation_event_cache_enabled: bool | None = None,
) -> FitnessEvaluationConfig:
    return FitnessEvaluationConfig(
        seed=int(seed),
        population_size=int(population_size),
        runtime_horizon=int(runtime_horizon),
        world_parameters=world_parameters,
        dt=1.0,
        backend=str(backend),
        generation_event_cache_enabled=generation_event_cache_enabled,
        pairwise_distance_mode=str(pairwise_distance_mode),
        pairwise_distance_max_exact_population=int(pairwise_distance_max_exact_population),
        pairwise_distance_sampled_pair_count=int(pairwise_distance_sampled_pair_count),
        pairwise_distance_sample_seed=int(pairwise_distance_sample_seed),
    )


def _summary_dict(report: Any) -> dict[str, Any]:
    summary = report.generation_summary.structural_diversity
    return {
        "pair_count": int(summary.pair_count),
        "mean_pairwise_distance": float(summary.mean_pairwise_distance),
        "structural_entropy": float(summary.structural_entropy),
        "normalized_structural_entropy": float(summary.normalized_structural_entropy),
        "pairwise_matrix_distances": [float(v) for v in summary.pairwise_matrix_distances],
        "distance_histogram_probabilities": [float(v) for v in summary.distance_histogram_probabilities],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_semantic_invariant_payload(*, seed: int) -> dict[str, Any]:
    world = _world_parameters(channel_count=3, jitter_scale=0.0)

    genomes_cache_on = _population_genomes(population=12, n=3, seed=seed)
    genomes_cache_off = _population_genomes(population=12, n=3, seed=seed)

    cfg_cache_on = _config(
        seed=seed + 1,
        population_size=12,
        runtime_horizon=3,
        world_parameters=world,
        pairwise_distance_mode=PAIRWISE_DISTANCE_MODE_EXACT,
        generation_event_cache_enabled=True,
    )
    cfg_cache_off = _config(
        seed=seed + 1,
        population_size=12,
        runtime_horizon=3,
        world_parameters=world,
        pairwise_distance_mode=PAIRWISE_DISTANCE_MODE_EXACT,
        generation_event_cache_enabled=False,
    )

    report_cache_on = simulate_genome_population(genomes_cache_on, cfg_cache_on)
    report_cache_off = simulate_genome_population(genomes_cache_off, cfg_cache_off)

    summary_on = _summary_dict(report_cache_on)
    summary_off = _summary_dict(report_cache_off)

    checks = {
        "pairwise_matrix_distances_equal": summary_on["pairwise_matrix_distances"] == summary_off["pairwise_matrix_distances"],
        "distance_histogram_probabilities_equal": summary_on["distance_histogram_probabilities"]
        == summary_off["distance_histogram_probabilities"],
        "pair_count_equal": summary_on["pair_count"] == summary_off["pair_count"],
        "mean_pairwise_distance_equal": summary_on["mean_pairwise_distance"] == summary_off["mean_pairwise_distance"],
        "structural_entropy_equal": summary_on["structural_entropy"] == summary_off["structural_entropy"],
        "normalized_structural_entropy_equal": summary_on["normalized_structural_entropy"]
        == summary_off["normalized_structural_entropy"],
    }

    return {
        "artifact": "semantic_invariant_output",
        "scenario": {
            "population_size": 12,
            "runtime_horizon": 3,
            "pairwise_distance_mode": PAIRWISE_DISTANCE_MODE_EXACT,
            "world_jitter_scale": 0.0,
            "seed": seed,
        },
        "cache_on": summary_on,
        "cache_off": summary_off,
        "checks": checks,
    }


def _build_exact_vs_sampled_markdown(*, seed: int) -> str:
    world = _world_parameters(channel_count=3, jitter_scale=0.0)
    population = 48
    sampled_pair_count = 320

    genomes_exact = _population_genomes(population=population, n=3, seed=seed + 10)
    genomes_sampled = _population_genomes(population=population, n=3, seed=seed + 10)

    cfg_exact = _config(
        seed=seed + 11,
        population_size=population,
        runtime_horizon=6,
        world_parameters=world,
        pairwise_distance_mode=PAIRWISE_DISTANCE_MODE_EXACT,
    )
    cfg_sampled = _config(
        seed=seed + 11,
        population_size=population,
        runtime_horizon=6,
        world_parameters=world,
        pairwise_distance_mode=PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED,
        pairwise_distance_sampled_pair_count=sampled_pair_count,
        pairwise_distance_sample_seed=12345,
    )

    report_exact = simulate_genome_population(genomes_exact, cfg_exact)
    report_sampled = simulate_genome_population(genomes_sampled, cfg_sampled)

    exact_summary = _summary_dict(report_exact)
    sampled_summary = _summary_dict(report_sampled)

    lines = [
        "# Exact vs Sampled Pairwise Diagnostics",
        "",
        "Fixed scenario:",
        f"- seed: {seed}",
        f"- population_size: {population}",
        "- runtime_horizon: 6",
        "- channels: 3",
        f"- sampled_pair_count: {sampled_pair_count}",
        "",
        "| mode | pair_count | mean_pairwise_distance | structural_entropy | normalized_structural_entropy |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            "| exact | "
            f"{exact_summary['pair_count']} | "
            f"{exact_summary['mean_pairwise_distance']:.10f} | "
            f"{exact_summary['structural_entropy']:.10f} | "
            f"{exact_summary['normalized_structural_entropy']:.10f} |"
        ),
        (
            "| fallback_sampled | "
            f"{sampled_summary['pair_count']} | "
            f"{sampled_summary['mean_pairwise_distance']:.10f} | "
            f"{sampled_summary['structural_entropy']:.10f} | "
            f"{sampled_summary['normalized_structural_entropy']:.10f} |"
        ),
        "",
    ]
    return "\n".join(lines)


def _build_caching_boundary_trace_payload(*, seed: int) -> dict[str, Any]:
    world = _world_parameters(channel_count=3, jitter_scale=0.0)
    cfg = _config(
        seed=seed + 21,
        population_size=10,
        runtime_horizon=6,
        world_parameters=world,
        pairwise_distance_mode=PAIRWISE_DISTANCE_MODE_EXACT,
        generation_event_cache_enabled=True,
    )

    genomes_warmup = _population_genomes(population=10, n=3, seed=seed + 22)
    _ = simulate_genome_population(genomes_warmup, cfg)

    genomes_same_process = _population_genomes(population=10, n=3, seed=seed + 22)
    same_process_report = simulate_genome_population(genomes_same_process, cfg)

    genomes_fresh = _population_genomes(population=10, n=3, seed=seed + 22)
    fresh_report = simulate_genome_population(genomes_fresh, cfg)

    same_summary = _summary_dict(same_process_report)
    fresh_summary = _summary_dict(fresh_report)

    return {
        "artifact": "caching_boundary_trace",
        "scenario": {
            "population_size": 10,
            "runtime_horizon": 6,
            "pairwise_distance_mode": PAIRWISE_DISTANCE_MODE_EXACT,
            "generation_event_cache_enabled": True,
            "world_jitter_scale": 0.0,
            "seed": seed,
        },
        "same_process": same_summary,
        "fresh_invocation": fresh_summary,
        "checks": {
            "reports_equal": same_process_report == fresh_report,
            "pairwise_matrix_distances_equal": same_summary["pairwise_matrix_distances"] == fresh_summary["pairwise_matrix_distances"],
        },
    }


def measure_performance_guard_scenario() -> dict[str, Any]:
    world = _world_parameters(channel_count=PERF_SCENARIO_CHANNELS, jitter_scale=0.0)
    genomes = _population_genomes(
        population=PERF_SCENARIO_POPULATION,
        n=PERF_SCENARIO_CHANNELS,
        seed=PERF_SCENARIO_SEED,
    )
    cfg = _config(
        seed=PERF_SCENARIO_SEED + 1,
        population_size=PERF_SCENARIO_POPULATION,
        runtime_horizon=PERF_SCENARIO_HORIZON,
        world_parameters=world,
        pairwise_distance_mode=PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED,
        pairwise_distance_sampled_pair_count=PERF_SCENARIO_SAMPLE_PAIRS,
        pairwise_distance_sample_seed=777,
        generation_event_cache_enabled=True,
    )

    start = time.perf_counter()
    report = simulate_genome_population(genomes, cfg)
    elapsed = time.perf_counter() - start

    return {
        "elapsed_seconds": float(elapsed),
        "population_size": PERF_SCENARIO_POPULATION,
        "runtime_horizon": PERF_SCENARIO_HORIZON,
        "pairwise_distance_mode": PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED,
        "pairwise_distance_sampled_pair_count": PERF_SCENARIO_SAMPLE_PAIRS,
        "mean_fitness": float(report.mean_fitness),
        "mean_time_to_death": float(report.mean_time_to_death),
        "pair_count": int(report.generation_summary.structural_diversity.pair_count),
    }


def generate_evidence_artifacts(*, output_dir: Path, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_path = output_dir / SEMANTIC_INVARIANT_FILE
    exact_vs_sampled_path = output_dir / EXACT_VS_SAMPLED_FILE
    caching_boundary_path = output_dir / CACHING_BOUNDARY_TRACE_FILE

    _write_json(semantic_path, _build_semantic_invariant_payload(seed=seed))
    _write_markdown(exact_vs_sampled_path, _build_exact_vs_sampled_markdown(seed=seed))
    _write_json(caching_boundary_path, _build_caching_boundary_trace_payload(seed=seed))

    return [semantic_path, exact_vs_sampled_path, caching_boundary_path]


def update_perf_baseline(*, baseline_file: Path) -> dict[str, Any]:
    measurements = [measure_performance_guard_scenario() for _ in range(PERF_BASELINE_SAMPLE_COUNT)]
    elapsed_values = np.asarray([float(item["elapsed_seconds"]) for item in measurements], dtype=np.float64)

    median_elapsed = float(np.median(elapsed_values))
    median_index = int(np.argmin(np.abs(elapsed_values - median_elapsed)))
    representative_measurement = dict(measurements[median_index])
    representative_measurement["elapsed_seconds"] = float(median_elapsed)

    payload = {
        "version": 1,
        "baseline_seconds": float(median_elapsed),
        "regression_multiplier": PERF_REGRESSION_FACTOR,
        "measurement_method": f"median_of_{PERF_BASELINE_SAMPLE_COUNT}",
        "measurement_statistics": {
            "sample_count": int(PERF_BASELINE_SAMPLE_COUNT),
            "min_elapsed_seconds": float(np.min(elapsed_values)),
            "max_elapsed_seconds": float(np.max(elapsed_values)),
            "median_elapsed_seconds": float(median_elapsed),
        },
        "scenario": {
            "population_size": PERF_SCENARIO_POPULATION,
            "runtime_horizon": PERF_SCENARIO_HORIZON,
            "channel_count": PERF_SCENARIO_CHANNELS,
            "pairwise_distance_mode": PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED,
            "pairwise_distance_sampled_pair_count": PERF_SCENARIO_SAMPLE_PAIRS,
            "seed": PERF_SCENARIO_SEED,
        },
        "measurement_runs": measurements,
        "last_measurement": representative_measurement,
    }
    _write_json(baseline_file, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic evidence artifacts for v0.9.5-1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate all required evidence artifacts")
    generate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Artifact output directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    generate_parser.add_argument("--seed", type=int, default=2026, help="Deterministic evidence seed")

    baseline_parser = subparsers.add_parser(
        "update-perf-baseline",
        help="Update committed performance baseline artifact",
    )
    baseline_parser.add_argument(
        "--baseline-file",
        type=Path,
        default=DEFAULT_BASELINE_FILE,
        help=f"Baseline file path (default: {DEFAULT_BASELINE_FILE})",
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.command == "generate":
        written = generate_evidence_artifacts(output_dir=args.output_dir, seed=int(args.seed))
        for path in written:
            print(str(path))
        return 0

    if args.command == "update-perf-baseline":
        payload = update_perf_baseline(baseline_file=args.baseline_file)
        baseline = float(payload["baseline_seconds"])
        threshold = baseline * float(payload["regression_multiplier"])
        print(
            "performance baseline updated: "
            f"baseline_time={baseline:.6f}s, threshold=baseline_time*{float(payload['regression_multiplier']):.1f}={threshold:.6f}s"
        )
        return 0

    raise RuntimeError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
