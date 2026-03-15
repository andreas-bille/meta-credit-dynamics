from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .builder import build_selector_from_genome
from .evolution_contracts import classify_flow_structure
from .evolution_contracts import PAIRWISE_DISTANCE_DEFAULT_SAMPLE_SEED
from .evolution_contracts import PAIRWISE_DISTANCE_DEFAULT_SAMPLED_PAIR_COUNT
from .evolution_contracts import PAIRWISE_DISTANCE_MAX_EXACT_POPULATION
from .evolution_contracts import PAIRWISE_DISTANCE_MODE_EXACT
from .evolution_contracts import PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED
from .evolution_contracts import pairwise_structural_distances_l1
from .evolution_contracts import structural_entropy_from_distances
from .evolution_contracts import validate_flow_matrix_invariants
from .genome import SelectorGenome
from .parent_selection import ParentSelectionPolicy
from .population_manager import PopulationManager, RebirthConfig
from .structured_world_adapter import StructuredWorldAdapter
from .mutation_scaling import MutationScalingConfig


@dataclass(frozen=True)
class FitnessEvaluationConfig:
    seed: int | None
    population_size: int | None
    runtime_horizon: int | None
    world_parameters: Mapping[str, Any] | None
    dt: float = 1.0
    backend: str = "cpu"
    evaluation_mode: str = "genome_pipeline"
    rebirth_enabled: bool = False
    rebirth_base_liquidity: float = 0.0
    rebirth_eta: float = 0.0
    rebirth_kappa: float = 1.0
    mutation_noise_scale: float = 0.01
    mutation_redistribution_share: float = 0.05
    mutation_lambda_risk_scale: float = 0.05
    rebirth_parent_selection_policy: ParentSelectionPolicy | None = None
    mutation_scaling_config: MutationScalingConfig | None = None
    # Optional explicit toggle for generation-scoped process-event caching.
    # None preserves the legacy default behavior (enabled iff jitter_scale == 0.0).
    generation_event_cache_enabled: bool | None = None
    pairwise_distance_mode: str = PAIRWISE_DISTANCE_MODE_EXACT
    pairwise_distance_max_exact_population: int = PAIRWISE_DISTANCE_MAX_EXACT_POPULATION
    pairwise_distance_sampled_pair_count: int = PAIRWISE_DISTANCE_DEFAULT_SAMPLED_PAIR_COUNT
    pairwise_distance_sample_seed: int = PAIRWISE_DISTANCE_DEFAULT_SAMPLE_SEED


@dataclass(frozen=True)
class GenomeFitnessResult:
    process_id: int
    cohort: str
    fitness_wealth: float
    fitness_norm: float
    survived: bool
    time_to_death: float
    death_tau: int | None
    wealth_trajectory: tuple[float, ...]


@dataclass(frozen=True)
class GenerationMutationRecord:
    newborn_process_id: int
    child_generation_id: int
    parent_process_id: int
    parent_generation_id: int
    mutation_magnitude: float
    modified_edge_count: int
    structural_distance: float
    flow_matrix: tuple[tuple[float, ...], ...]
    output_weights: tuple[float, ...]
    lambda_risk: float


@dataclass(frozen=True)
class GenerationProcessSnapshot:
    process_id: int
    generation_id: int
    parent_process_id: int | None
    parent_generation_id: int | None
    cohort: str
    wealth: float
    dead: bool
    flow_structure_classification: str


@dataclass(frozen=True)
class GenerationStepRecord:
    tau: int
    active_process_ids: tuple[int, ...] = ()
    dead_process_ids: tuple[int, ...] = ()
    newborn_process_ids: tuple[int, ...] = ()
    mutation_events: tuple[GenerationMutationRecord, ...] = ()
    population_snapshot: tuple[GenerationProcessSnapshot, ...] = ()


@dataclass(frozen=True)
class GenerationLoopTrajectory:
    steps: tuple[GenerationStepRecord, ...] = ()


@dataclass(frozen=True)
class MutationStepStatistics:
    """Aggregated per-step mutation statistics (from 0.9.2-2 diagnostics)."""

    event_count: int = 0
    mean_mutation_magnitude: float = 0.0
    mean_modified_edge_count: float = 0.0
    mean_structural_distance: float = 0.0


@dataclass(frozen=True)
class PopulationStepStatistics:
    """Per-step population statistics."""

    active_count: int = 0
    dead_count: int = 0
    newborn_count: int = 0
    mean_wealth: float = 0.0


@dataclass(frozen=True)
class GenerationStepLogRecord:
    """Structured log record for a single generation step."""

    tau: int
    mutation_stats: MutationStepStatistics = field(default_factory=MutationStepStatistics)
    population_stats: PopulationStepStatistics = field(default_factory=PopulationStepStatistics)


@dataclass(frozen=True)
class StructuralDiversitySummary:
    pairwise_matrix_distances: tuple[float, ...] = ()
    pair_count: int = 0
    mean_pairwise_distance: float = 0.0
    structural_entropy: float = 0.0
    normalized_structural_entropy: float = 0.0
    distance_histogram_probabilities: tuple[float, ...] = ()


@dataclass(frozen=True)
class FlowStructureFrequencySummary:
    zero_flow_count: int = 0
    single_edge_dominated_count: int = 0
    sparse_complex_count: int = 0
    dense_complex_count: int = 0
    total_observations: int = 0
    zero_flow_frequency: float = 0.0
    single_edge_dominated_frequency: float = 0.0
    sparse_complex_frequency: float = 0.0
    dense_complex_frequency: float = 0.0


@dataclass(frozen=True)
class GenerationSummary:
    structural_diversity: StructuralDiversitySummary = field(default_factory=StructuralDiversitySummary)
    flow_structure_frequency: FlowStructureFrequencySummary = field(default_factory=FlowStructureFrequencySummary)


@dataclass(frozen=True)
class GenerationLoopLogRecord:
    """Structured log record for a complete generation loop run.

    Includes generation summary (structural diversity from 0.9.2-3) and per-step
    mutation and population statistics (from 0.9.2-2 and 0.9.2-3).
    """

    total_steps: int = 0
    step_records: tuple[GenerationStepLogRecord, ...] = ()
    generation_summary: GenerationSummary = field(default_factory=GenerationSummary)


@dataclass(frozen=True)
class PopulationFitnessReport:
    per_process: tuple[GenomeFitnessResult, ...]
    mean_fitness: float
    median_fitness: float
    survival_rate: float
    population_size: int
    mean_time_to_death: float
    mean_normalized_fitness: float
    cpu_fallback_used: int
    backend: str
    generation_summary: GenerationSummary = field(default_factory=GenerationSummary)
    generation_trajectory: GenerationLoopTrajectory = field(default_factory=GenerationLoopTrajectory)


def _splitmix64(value: int) -> int:
    x = int(value) & 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return int(z & 0xFFFFFFFFFFFFFFFF)


def _require_genome(genome: SelectorGenome | None) -> SelectorGenome:
    if genome is None:
        raise ValueError("fitness evaluation requires genome")
    return genome


def _validate_fitness_config(config: FitnessEvaluationConfig) -> None:
    if float(config.dt) <= 0.0:
        raise ValueError("invalid fitness configuration: dt must be > 0")

    if config.evaluation_mode != "genome_pipeline":
        raise ValueError("fitness evaluation bypassed genome pipeline")

    missing = (
        config.seed is None
        or config.population_size is None
        or config.runtime_horizon is None
        or config.world_parameters is None
    )
    if missing:
        raise ValueError("invalid fitness evaluation configuration")

    if int(config.population_size) <= 0 or int(config.runtime_horizon) <= 0:
        raise ValueError("invalid fitness evaluation configuration")

    if not isinstance(config.world_parameters, Mapping):
        raise ValueError("invalid fitness evaluation configuration")

    numeric_config = (
        config.rebirth_base_liquidity,
        config.rebirth_eta,
        config.rebirth_kappa,
        config.mutation_noise_scale,
        config.mutation_redistribution_share,
        config.mutation_lambda_risk_scale,
    )
    if any(not np.isfinite(float(value)) for value in numeric_config):
        raise ValueError("invalid fitness evaluation configuration")

    if (
        float(config.rebirth_base_liquidity) < 0.0
        or float(config.rebirth_eta) < 0.0
        or float(config.rebirth_kappa) < 0.0
        or float(config.mutation_noise_scale) < 0.0
        or float(config.mutation_redistribution_share) < 0.0
        or float(config.mutation_lambda_risk_scale) < 0.0
    ):
        raise ValueError("invalid fitness evaluation configuration")

    mode_norm = str(config.pairwise_distance_mode).strip().lower()
    if mode_norm not in {
        PAIRWISE_DISTANCE_MODE_EXACT,
        PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED,
    }:
        raise ValueError("invalid fitness evaluation configuration")

    if int(config.pairwise_distance_max_exact_population) < 2:
        raise ValueError("invalid fitness evaluation configuration")

    if int(config.pairwise_distance_sampled_pair_count) <= 0:
        raise ValueError("invalid fitness evaluation configuration")

    if config.pairwise_distance_sample_seed is None:
        raise ValueError("invalid fitness evaluation configuration")

    if config.generation_event_cache_enabled is not None and not isinstance(config.generation_event_cache_enabled, bool):
        raise ValueError("invalid fitness evaluation configuration")


def fitness_from_wealth_trajectory(
    wealth_trajectory: Sequence[float],
    *,
    dt: float,
    death_index: int | None,
) -> tuple[float, float, float]:
    if float(dt) <= 0.0:
        raise ValueError("invalid fitness configuration: dt must be > 0")

    values = np.asarray(list(wealth_trajectory), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("invalid normalized fitness state")

    if death_index is None:
        stop = int(values.shape[0])
    else:
        stop = min(int(values.shape[0]), int(death_index) + 1)

    if stop <= 0:
        raise ValueError("invalid normalized fitness state")

    if np.any(~np.isfinite(values[:stop])):
        raise ValueError("invalid normalized fitness state")

    fitness_wealth = float(np.sum(values[:stop]) * float(dt))
    t_eff = float(stop * float(dt))
    if not np.isfinite(fitness_wealth):
        raise ValueError("invalid normalized fitness state")

    if not np.isfinite(t_eff) or t_eff <= 0.0:
        raise ValueError("invalid normalized fitness state")

    fitness_norm = float(fitness_wealth / max(1e-12, t_eff))
    if not np.isfinite(fitness_norm):
        raise ValueError("invalid normalized fitness state")

    return fitness_wealth, fitness_norm, t_eff


def aggregate_population_metrics(results: Sequence[GenomeFitnessResult]) -> tuple[float, float, float, int, float, float]:
    if not results:
        raise ValueError("invalid fitness evaluation configuration")

    fitness_values = np.asarray([float(item.fitness_wealth) for item in results], dtype=np.float64)
    normalized_values = np.asarray([float(item.fitness_norm) for item in results], dtype=np.float64)
    survival_values = np.asarray([1.0 if bool(item.survived) else 0.0 for item in results], dtype=np.float64)
    ttd_values = np.asarray([float(item.time_to_death) for item in results], dtype=np.float64)

    return (
        float(np.mean(fitness_values)),
        float(np.median(fitness_values)),
        float(np.mean(survival_values)),
        int(len(results)),
        float(np.mean(ttd_values)),
        float(np.mean(normalized_values)),
    )


def _serialize_float_matrix(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(values, dtype=np.float64)
    return tuple(tuple(float(item) for item in row) for row in matrix.tolist())


def _serialize_float_vector(values: np.ndarray) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float64)
    return tuple(float(item) for item in vector.tolist())


def _build_process_snapshot(process_id: int, selector: Any) -> GenerationProcessSnapshot:
    flow_matrix = np.asarray(getattr(selector, "flow_matrix", np.zeros((0, 0))), dtype=np.float64)
    return GenerationProcessSnapshot(
        process_id=int(process_id),
        generation_id=int(getattr(selector, "generation_id", 0)),
        parent_process_id=(None if getattr(selector, "parent_process_id", None) is None else int(selector.parent_process_id)),
        parent_generation_id=(None if getattr(selector, "parent_generation_id", None) is None else int(selector.parent_generation_id)),
        cohort=str(getattr(selector, "_fitness_cohort", "resident")),
        wealth=float(getattr(selector, "wealth", 0.0)),
        dead=bool(getattr(selector, "dead", False)),
        flow_structure_classification=str(classify_flow_structure(flow_matrix)),
    )


def _build_generation_step_record(
    *,
    tau: int,
    step_result: Mapping[str, Any],
    manager: PopulationManager,
) -> GenerationStepRecord:
    dead_process_ids = tuple(int(process_id) for process_id in step_result.get("dead_ids", []))
    newborn_process_ids = tuple(int(process_id) for process_id in step_result.get("newborn_ids", []))

    mutation_events: list[GenerationMutationRecord] = []
    for event in step_result.get("mutation_diagnostics", []):
        newborn_process_id = int(event["newborn_process_id"])
        selector = manager.processes[newborn_process_id]
        mutation_events.append(
            GenerationMutationRecord(
                newborn_process_id=newborn_process_id,
                child_generation_id=int(getattr(selector, "generation_id", 0)),
                parent_process_id=int(event["parent_process_id"]),
                parent_generation_id=int(event["parent_generation_id"]),
                mutation_magnitude=float(event["mutation_magnitude"]),
                modified_edge_count=int(event["modified_edge_count"]),
                structural_distance=float(event["structural_distance"]),
                flow_matrix=_serialize_float_matrix(np.asarray(getattr(selector, "flow_matrix", []), dtype=np.float64)),
                output_weights=_serialize_float_vector(np.asarray(getattr(selector, "output_weights", []), dtype=np.float64)),
                lambda_risk=float(getattr(selector, "lambda_risk", 0.0)),
            )
        )

    population_snapshot = tuple(
        _build_process_snapshot(int(process_id), selector)
        for process_id, selector in sorted(manager.processes.items())
    )
    return GenerationStepRecord(
        tau=int(tau),
        active_process_ids=tuple(int(process_id) for process_id, _ in sorted(manager.processes.items())),
        dead_process_ids=dead_process_ids,
        newborn_process_ids=newborn_process_ids,
        mutation_events=tuple(mutation_events),
        population_snapshot=population_snapshot,
    )


def _summarize_population_structural_diversity(
    flow_matrices: Sequence[np.ndarray],
    *,
    pairwise_distance_mode: str,
    pairwise_distance_max_exact_population: int,
    pairwise_distance_sampled_pair_count: int,
    pairwise_distance_sample_seed: int,
) -> StructuralDiversitySummary:
    try:
        distances = pairwise_structural_distances_l1(
            flow_matrices,
            mode=str(pairwise_distance_mode),
            max_exact_population=int(pairwise_distance_max_exact_population),
            sampled_pair_count=int(pairwise_distance_sampled_pair_count),
            sample_seed=int(pairwise_distance_sample_seed),
        )
    except TypeError as exc:
        # Backward-compatible path for monkeypatched helper spies that still
        # expose the legacy one-argument signature in tests.
        if "unexpected keyword argument" not in str(exc):
            raise
        distances = pairwise_structural_distances_l1(flow_matrices)

    entropy, normalized_entropy, histogram_probabilities = structural_entropy_from_distances(distances)
    entropy = max(0.0, float(entropy))
    normalized_entropy = max(0.0, float(normalized_entropy))
    mean_pairwise_distance = 0.0
    if distances.size > 0:
        mean_pairwise_distance = float(np.mean(distances))

    return StructuralDiversitySummary(
        pairwise_matrix_distances=tuple(float(value) for value in distances.tolist()),
        pair_count=int(distances.size),
        mean_pairwise_distance=float(mean_pairwise_distance),
        structural_entropy=float(entropy),
        normalized_structural_entropy=float(normalized_entropy),
        distance_histogram_probabilities=tuple(float(value) for value in histogram_probabilities.tolist()),
    )


def _summarize_flow_structure_frequencies(
    generation_steps: Sequence[GenerationStepRecord],
) -> FlowStructureFrequencySummary:
    counts = {
        "ZERO_FLOW": 0,
        "SINGLE_EDGE_DOMINATED": 0,
        "SPARSE_COMPLEX": 0,
        "DENSE_COMPLEX": 0,
    }

    counted_dead_process_ids: set[int] = set()

    for step in generation_steps:
        for snapshot in step.population_snapshot:
            process_id = int(snapshot.process_id)
            is_dead = bool(snapshot.dead)
            if is_dead and process_id in counted_dead_process_ids:
                continue

            label = str(snapshot.flow_structure_classification)
            if label not in counts:
                raise ValueError("invalid flow structure classification in generation snapshot")
            counts[label] += 1

            if is_dead:
                counted_dead_process_ids.add(process_id)

    total = int(sum(counts.values()))
    if total <= 0:
        return FlowStructureFrequencySummary()

    return FlowStructureFrequencySummary(
        zero_flow_count=int(counts["ZERO_FLOW"]),
        single_edge_dominated_count=int(counts["SINGLE_EDGE_DOMINATED"]),
        sparse_complex_count=int(counts["SPARSE_COMPLEX"]),
        dense_complex_count=int(counts["DENSE_COMPLEX"]),
        total_observations=total,
        zero_flow_frequency=float(counts["ZERO_FLOW"] / total),
        single_edge_dominated_frequency=float(counts["SINGLE_EDGE_DOMINATED"] / total),
        sparse_complex_frequency=float(counts["SPARSE_COMPLEX"] / total),
        dense_complex_frequency=float(counts["DENSE_COMPLEX"] / total),
    )


def _build_generation_summary(
    flow_matrices: Sequence[np.ndarray],
    generation_steps: Sequence[GenerationStepRecord],
    *,
    pairwise_distance_mode: str,
    pairwise_distance_max_exact_population: int,
    pairwise_distance_sampled_pair_count: int,
    pairwise_distance_sample_seed: int,
) -> GenerationSummary:
    return GenerationSummary(
        structural_diversity=_summarize_population_structural_diversity(
            flow_matrices,
            pairwise_distance_mode=str(pairwise_distance_mode),
            pairwise_distance_max_exact_population=int(pairwise_distance_max_exact_population),
            pairwise_distance_sampled_pair_count=int(pairwise_distance_sampled_pair_count),
            pairwise_distance_sample_seed=int(pairwise_distance_sample_seed),
        ),
        flow_structure_frequency=_summarize_flow_structure_frequencies(generation_steps),
    )


def _build_mutation_step_statistics(
    mutation_events: tuple[GenerationMutationRecord, ...],
) -> MutationStepStatistics:
    if not mutation_events:
        return MutationStepStatistics()
    n = len(mutation_events)
    return MutationStepStatistics(
        event_count=n,
        mean_mutation_magnitude=float(np.mean([e.mutation_magnitude for e in mutation_events])),
        mean_modified_edge_count=float(np.mean([e.modified_edge_count for e in mutation_events])),
        mean_structural_distance=float(np.mean([e.structural_distance for e in mutation_events])),
    )


def _build_population_step_statistics(step: GenerationStepRecord) -> PopulationStepStatistics:
    active_snapshots = [s for s in step.population_snapshot if not s.dead]
    mean_wealth = float(np.mean([s.wealth for s in active_snapshots])) if active_snapshots else 0.0
    return PopulationStepStatistics(
        active_count=len(step.active_process_ids),
        dead_count=len(step.dead_process_ids),
        newborn_count=len(step.newborn_process_ids),
        mean_wealth=mean_wealth,
    )


def build_generation_step_log_record(step: GenerationStepRecord) -> GenerationStepLogRecord:
    """Build a :class:`GenerationStepLogRecord` from a trajectory step."""
    return GenerationStepLogRecord(
        tau=step.tau,
        mutation_stats=_build_mutation_step_statistics(step.mutation_events),
        population_stats=_build_population_step_statistics(step),
    )


def build_generation_loop_log_record(report: PopulationFitnessReport) -> GenerationLoopLogRecord:
    """Build a :class:`GenerationLoopLogRecord` from a completed fitness report."""
    step_records = tuple(build_generation_step_log_record(s) for s in report.generation_trajectory.steps)
    return GenerationLoopLogRecord(
        total_steps=len(step_records),
        step_records=step_records,
        generation_summary=report.generation_summary,
    )


def _extract_world_base_returns(world_parameters: Mapping[str, Any]) -> np.ndarray:
    if "returns" not in world_parameters:
        raise ValueError("invalid fitness evaluation configuration")
    returns = np.asarray(world_parameters["returns"], dtype=np.float64)
    if returns.ndim != 1 or returns.shape[0] <= 0:
        raise ValueError("invalid fitness evaluation configuration")
    if np.any(~np.isfinite(returns)):
        raise ValueError("invalid fitness evaluation configuration")
    return returns


def _extract_cost_for_tau(world_parameters: Mapping[str, Any], tau: int) -> float:
    if "costs" not in world_parameters:
        raise ValueError("invalid fitness evaluation configuration")

    costs = world_parameters["costs"]
    if isinstance(costs, (float, int)):
        value = float(costs)
    else:
        arr = np.asarray(costs, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] == 0:
            raise ValueError("invalid fitness evaluation configuration")
        idx = min(int(tau), int(arr.shape[0]) - 1)
        value = float(arr[idx])

    if not np.isfinite(value):
        raise ValueError("invalid fitness evaluation configuration")
    return value


def _extract_jackpot_for_tau(world_parameters: Mapping[str, Any], tau: int) -> float:
    jackpot = world_parameters.get("jackpot_by_tau", 0.0)
    if isinstance(jackpot, (float, int)):
        value = float(jackpot)
    else:
        arr = np.asarray(jackpot, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] == 0:
            raise ValueError("invalid fitness evaluation configuration")
        idx = min(int(tau), int(arr.shape[0]) - 1)
        value = float(arr[idx])

    if not np.isfinite(value):
        raise ValueError("invalid fitness evaluation configuration")
    return value


def _build_process_event(
    *,
    config: FitnessEvaluationConfig,
    tau: int,
    process_id: int,
    channel_count: int,
) -> dict[str, Any]:
    assert config.world_parameters is not None
    world_parameters = config.world_parameters

    base_returns = _extract_world_base_returns(world_parameters)
    if int(base_returns.shape[0]) != int(channel_count):
        raise ValueError("invalid fitness evaluation configuration")

    jitter_scale = float(world_parameters.get("jitter_scale", 0.0))
    seed_value = _splitmix64(int(config.seed) ^ (int(tau) << 32) ^ int(process_id))
    rng = np.random.default_rng(np.random.PCG64(np.uint64(seed_value)))

    if jitter_scale > 0.0:
        jitter = rng.normal(loc=0.0, scale=jitter_scale, size=base_returns.shape)
    else:
        jitter = np.zeros_like(base_returns)

    r_vec = np.asarray(base_returns + jitter, dtype=np.float64)
    c_total = _extract_cost_for_tau(world_parameters, tau)
    return {"r_vec": r_vec, "c_total": float(c_total), "freeze": False}


def _initial_wealth(world_parameters: Mapping[str, Any]) -> float:
    value = float(world_parameters.get("initial_wealth", 1.0))
    if not np.isfinite(value):
        raise ValueError("invalid fitness evaluation configuration")
    return value


def _structured_world_adapter_from_world_parameters(
    world_parameters: Mapping[str, Any],
) -> StructuredWorldAdapter | None:
    structured_world = world_parameters.get("structured_world", None)
    if structured_world is None:
        return None
    return StructuredWorldAdapter(world=structured_world)


def simulate_genome_population(
    resident_genomes: Sequence[SelectorGenome | None],
    config: FitnessEvaluationConfig,
    *,
    injected_genomes_by_tau: Mapping[int, Sequence[SelectorGenome | None]] | None = None,
) -> PopulationFitnessReport:
    _validate_fitness_config(config)
    assert config.population_size is not None
    assert config.runtime_horizon is not None
    assert config.world_parameters is not None

    resident_genomes = list(resident_genomes)
    if len(resident_genomes) != int(config.population_size):
        raise ValueError("invalid fitness evaluation configuration")

    backend = str(config.backend).strip().lower()
    if backend not in {"cpu", "cuda"}:
        raise ValueError("invalid fitness evaluation configuration")

    processes: dict[int, Any] = {}
    for process_id, genome in enumerate(resident_genomes):
        validated = _require_genome(genome)
        selector = build_selector_from_genome(
            validated,
            process_id=int(process_id),
            generation_id=0,
            initial_wealth=_initial_wealth(config.world_parameters),
            rebirth_threshold=-1.0,
        )
        selector._fitness_cohort = "resident"
        processes[int(process_id)] = selector

    manager = PopulationManager(
        processes=processes,
        rebirth_config=RebirthConfig(
            enabled=bool(config.rebirth_enabled),
            base_liquidity=float(config.rebirth_base_liquidity),
            eta=float(config.rebirth_eta),
            kappa=float(config.rebirth_kappa),
            deterministic_seed=int(config.seed),
            mutation_noise_scale=float(config.mutation_noise_scale),
            mutation_redistribution_share=float(config.mutation_redistribution_share),
            mutation_lambda_risk_scale=float(config.mutation_lambda_risk_scale),
            parent_selection_policy=config.rebirth_parent_selection_policy,
            mutation_scaling_config=config.mutation_scaling_config,
        ),
        backend=backend,
    )

    structured_world_adapter = _structured_world_adapter_from_world_parameters(config.world_parameters)

    wealth_trajectories: dict[int, list[float]] = {int(pid): [] for pid in manager.processes.keys()}
    death_tau: dict[int, int | None] = {int(pid): None for pid in manager.processes.keys()}
    generation_steps: list[GenerationStepRecord] = []

    inject_map = {int(t): list(items) for t, items in dict(injected_genomes_by_tau or {}).items()}
    jitter_scale = float(config.world_parameters.get("jitter_scale", 0.0))
    if config.generation_event_cache_enabled is None:
        use_generation_event_cache = np.isclose(jitter_scale, 0.0, rtol=0.0, atol=0.0)
    else:
        use_generation_event_cache = bool(config.generation_event_cache_enabled)

    for tau in range(int(config.runtime_horizon)):
        if tau in inject_map:
            for genome in inject_map[tau]:
                validated = _require_genome(genome)
                new_id = manager._allocate_process_id()  # noqa: SLF001
                selector = build_selector_from_genome(
                    validated,
                    process_id=int(new_id),
                    generation_id=0,
                    initial_wealth=_initial_wealth(config.world_parameters),
                    rebirth_threshold=-1.0,
                )
                selector._fitness_cohort = "mutant"
                manager.processes[int(new_id)] = selector
                manager._cores[int(new_id)] = manager._build_core(start_tau=int(tau))  # noqa: SLF001
                wealth_trajectories[int(new_id)] = []
                death_tau[int(new_id)] = None

        process_events: dict[int, dict[str, Any]] = {}
        if structured_world_adapter is not None:
            shared_event = structured_world_adapter.build_process_event(tau=int(tau))
            expected_channels = int(np.asarray(shared_event["r_vec"], dtype=np.float64).shape[0])

            for process_id in sorted(manager.processes.keys()):
                selector = manager.processes[int(process_id)]
                flow_matrix = np.asarray(getattr(selector, "flow_matrix", np.zeros((0, 0))), dtype=np.float64)
                channel_count = int(flow_matrix.shape[0])
                if channel_count <= 0:
                    raise ValueError("invalid fitness evaluation configuration")
                if channel_count != expected_channels:
                    raise ValueError("invalid fitness evaluation configuration")
                process_events[int(process_id)] = {
                    "r_vec": np.asarray(shared_event["r_vec"], dtype=np.float64),
                    "c_total": float(shared_event["c_total"]),
                    "freeze": bool(shared_event.get("freeze", False)),
                }
        else:
            # Generation-scoped cache with explicit eviction at generation end.
            # Key is channel_count, valid only when jitter_scale == 0.0 where
            # process_id has no effect on generated events.
            event_cache_by_channel: dict[int, dict[str, Any]] = {}
            for process_id in sorted(manager.processes.keys()):
                selector = manager.processes[int(process_id)]
                flow_matrix = np.asarray(getattr(selector, "flow_matrix", np.zeros((0, 0))), dtype=np.float64)
                channel_count = int(flow_matrix.shape[0])
                if channel_count <= 0:
                    raise ValueError("invalid fitness evaluation configuration")
                if use_generation_event_cache:
                    cached = event_cache_by_channel.get(channel_count)
                    if cached is None:
                        cached = _build_process_event(
                            config=config,
                            tau=int(tau),
                            process_id=0,
                            channel_count=channel_count,
                        )
                        event_cache_by_channel[channel_count] = {
                            "r_vec": np.asarray(cached["r_vec"], dtype=np.float64),
                            "c_total": float(cached["c_total"]),
                            "freeze": bool(cached.get("freeze", False)),
                        }
                    process_events[int(process_id)] = {
                        "r_vec": np.asarray(event_cache_by_channel[channel_count]["r_vec"], dtype=np.float64),
                        "c_total": float(event_cache_by_channel[channel_count]["c_total"]),
                        "freeze": bool(event_cache_by_channel[channel_count]["freeze"]),
                    }
                else:
                    process_events[int(process_id)] = _build_process_event(
                        config=config,
                        tau=int(tau),
                        process_id=int(process_id),
                        channel_count=channel_count,
                    )

        step_result = manager.step_tau(
            tau=int(tau),
            process_events=process_events,
            jackpot=_extract_jackpot_for_tau(config.world_parameters, tau),
        )

        for newborn_process_id in step_result.get("newborn_ids", []):
            process_id = int(newborn_process_id)
            wealth_trajectories.setdefault(process_id, [])
            death_tau.setdefault(process_id, None)

        generation_steps.append(
            _build_generation_step_record(
                tau=int(tau),
                step_result=step_result,
                manager=manager,
            )
        )

        for process_id, selector in sorted(manager.processes.items()):
            wealth_trajectories[int(process_id)].append(float(selector.wealth))
            if death_tau[int(process_id)] is None and bool(getattr(selector, "dead", False)):
                death_tau[int(process_id)] = int(tau)

    per_process: list[GenomeFitnessResult] = []
    for process_id in sorted(wealth_trajectories.keys()):
        selector = manager.processes[int(process_id)]
        trajectory = wealth_trajectories[int(process_id)]
        dead_idx = death_tau[int(process_id)]

        fitness_wealth, fitness_norm, t_eff = fitness_from_wealth_trajectory(
            trajectory,
            dt=float(config.dt),
            death_index=dead_idx,
        )

        per_process.append(
            GenomeFitnessResult(
                process_id=int(process_id),
                cohort=str(getattr(selector, "_fitness_cohort", "resident")),
                fitness_wealth=float(fitness_wealth),
                fitness_norm=float(fitness_norm),
                survived=dead_idx is None,
                time_to_death=float(t_eff),
                death_tau=dead_idx,
                wealth_trajectory=tuple(float(value) for value in trajectory),
            )
        )

    (
        mean_fitness,
        median_fitness,
        survival_rate,
        population_size,
        mean_time_to_death,
        mean_normalized_fitness,
    ) = aggregate_population_metrics(per_process)

    cpu_fallback_used = 0
    if backend == "cuda":
        for core in manager._cores.values():  # noqa: SLF001
            if hasattr(core, "metrics_snapshot"):
                metrics = core.metrics_snapshot()
                cpu_fallback_used += int(metrics.get("cpu_fallback_used", 0))

    generation_summary = _build_generation_summary(
        [
            np.asarray(manager.processes[int(process_id)].flow_matrix, dtype=np.float64)
            for process_id in sorted(wealth_trajectories.keys())
        ],
        generation_steps=generation_steps,
        pairwise_distance_mode=str(config.pairwise_distance_mode),
        pairwise_distance_max_exact_population=int(config.pairwise_distance_max_exact_population),
        pairwise_distance_sampled_pair_count=int(config.pairwise_distance_sampled_pair_count),
        pairwise_distance_sample_seed=int(config.pairwise_distance_sample_seed),
    )

    return PopulationFitnessReport(
        per_process=tuple(per_process),
        mean_fitness=float(mean_fitness),
        median_fitness=float(median_fitness),
        survival_rate=float(survival_rate),
        population_size=int(population_size),
        mean_time_to_death=float(mean_time_to_death),
        mean_normalized_fitness=float(mean_normalized_fitness),
        cpu_fallback_used=int(cpu_fallback_used),
        backend=backend,
        generation_summary=generation_summary,
        generation_trajectory=GenerationLoopTrajectory(steps=tuple(generation_steps)),
    )


def evaluate_population_fitness(
    genomes: Sequence[SelectorGenome | None],
    config: FitnessEvaluationConfig,
) -> PopulationFitnessReport:
    return simulate_genome_population(genomes, config)


@dataclass
class FlowVisualizationDataset:
    """Deterministic dataset for flow-structure heatmap and distance visualization.

    Fields follow the spec contract (Issue 0.9.2-4):
      flow_matrices_per_generation  — one flow matrix per genome (list[ndarray[n,m]])
      pairwise_distances_per_generation — L1 distances for all unique pairs (i<j),
                                          length K*(K-1)/2 (list[float])
    """
    flow_matrices_per_generation: list[np.ndarray]
    pairwise_distances_per_generation: list[float]


def build_flow_visualization_dataset(
    genomes: Sequence[SelectorGenome | None],
    *,
    seed: int,
    runtime_horizon: int,
    world_parameters: Mapping[str, Any],
    dt: float = 1.0,
    backend: str = "cpu",
) -> FlowVisualizationDataset:
    """Build a deterministic visualization dataset for a population of genomes.

    Flow matrices are validated against the Issue 0.9.2-1 invariants (finite,
    non-negative, row-stochastic) and stored as a list of ndarrays. Pairwise
    L1 distances are computed directly from the validated matrices via the shared
    helper from evolution_contracts, guaranteeing internal dataset coherence.

    The seed, runtime_horizon, world_parameters, dt, and backend parameters are
    accepted for API symmetry with the generation-loop family but do not affect
    the dataset content (which is a deterministic function of the input genomes).
    """
    genome_list = [_require_genome(g) for g in genomes]

    # Validate and extract flow matrices (Issue 0.9.2-1 invariants).
    flow_matrices: list[np.ndarray] = []
    for genome in genome_list:
        matrix = validate_flow_matrix_invariants(
            np.asarray(genome.flow_matrix, dtype=np.float64),
            expected_shape=None,
            shape_error_message="flow visualization dataset: invalid matrix shape",
            finite_error_message="flow visualization dataset: non-finite flow matrix",
            negative_error_message="flow visualization dataset: negative flow matrix",
            require_row_stochastic=True,
            row_stochastic_error_message="flow visualization dataset: non-row-stochastic flow matrix",
        )
        flow_matrices.append(matrix)

    # Compute pairwise distances directly from the validated matrices via the
    # shared helper — guarantees that distances[k] == L1(matrices[i], matrices[j])
    # for the k-th (i<j) pair, eliminating any coupling to the simulation path.
    distances_array = pairwise_structural_distances_l1(flow_matrices)
    distances = [float(v) for v in distances_array.tolist()]

    return FlowVisualizationDataset(
        flow_matrices_per_generation=flow_matrices,
        pairwise_distances_per_generation=distances,
    )
