from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..fitness_engine import FitnessEvaluationConfig
from ..fitness_engine import simulate_genome_population
from ..genome import SelectorGenome
from ..genome_mutation import MutationConfig
from ..genome_mutation import mutate_selector_genome
from ..genome_serialization import genome_from_dict
from ..genome_serialization import genome_to_dict


LONG_RUN_CHECKPOINT_SCHEMA_VERSION = "v0.9.5-2"
LONG_RUN_SELECTION_FITNESS_PROPORTIONAL = "fitness_proportional"
LONG_RUN_SELECTION_UNIFORM = "uniform"
LONG_RUN_CHECKPOINT_FILE_STEM_TEMPLATE = "generation_{generation_index:06d}"


@dataclass(frozen=True)
class LongRunConfig:
    """Long-run harness configuration.

    Shared base contract fields across consolidated configs:
    ``seed``, ``world_parameters``, ``backend``.
    """

    seed: int | None = None
    population_size: int | None = None
    channel_count: int | None = None
    horizon: int | None = None
    generations: int | None = None
    world_parameters: Mapping[str, Any] | None = None
    mutation_config: MutationConfig = MutationConfig()
    selection_policy: str = LONG_RUN_SELECTION_FITNESS_PROPORTIONAL
    checkpoint_interval: int = 0
    checkpoint_path: str | None = None
    dt: float = 1.0
    backend: str = "cpu"


@dataclass(frozen=True)
class LongRunGenerationRecord:
    generation_index: int
    mean_fitness: float
    median_fitness: float
    survival_rate: float
    min_fitness: float
    max_fitness: float
    min_flow_row_sum: float
    max_flow_row_sum: float
    fitness_values_finite: bool
    flow_row_sums_finite: bool
    per_process_fitness: tuple[float, ...]
    per_process_survived: tuple[bool, ...]


@dataclass(frozen=True)
class LongRunResult:
    config: LongRunConfig
    backend: str
    start_generation: int
    completed_generations: int
    generation_records: tuple[LongRunGenerationRecord, ...]
    population_states_by_generation: tuple[tuple[dict[str, Any], ...], ...]
    final_population_state: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LongRunEmergenceEvaluation:
    metric_points: tuple[Any, ...]
    pathology_warning_messages: tuple[str, ...]


def _to_host_json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_host_json_compatible(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}

    if isinstance(value, (list, tuple)):
        return [_to_host_json_compatible(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (bool, int, float, str)) or value is None:
        return value

    try:
        import torch  # local import to avoid hard dependency for non-torch contexts

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
    except Exception:
        pass

    raise ValueError("long-run checkpoint serialization requires JSON-compatible host values")


def _coerce_mutation_config(raw: Any) -> MutationConfig:
    if isinstance(raw, MutationConfig):
        return raw
    if raw is None:
        return MutationConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("invalid long-run configuration")
    return MutationConfig(
        noise_scale=float(raw.get("noise_scale", 0.01)),
        redistribution_share=float(raw.get("redistribution_share", 0.05)),
        lambda_risk_scale=float(raw.get("lambda_risk_scale", 0.05)),
    )


def _coerce_config(raw: LongRunConfig | Mapping[str, Any]) -> LongRunConfig:
    if isinstance(raw, LongRunConfig):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("invalid long-run configuration")

    return LongRunConfig(
        seed=None if raw.get("seed") is None else int(raw.get("seed")),
        population_size=None if raw.get("population_size") is None else int(raw.get("population_size")),
        channel_count=None if raw.get("channel_count") is None else int(raw.get("channel_count")),
        horizon=(
            None
            if raw.get("horizon", raw.get("runtime_horizon")) is None
            else int(raw.get("horizon", raw.get("runtime_horizon")))
        ),
        generations=(None if raw.get("generations") is None else int(raw.get("generations"))),
        world_parameters=raw.get("world_parameters"),
        mutation_config=_coerce_mutation_config(raw.get("mutation_config", None)),
        selection_policy=str(raw.get("selection_policy", LONG_RUN_SELECTION_FITNESS_PROPORTIONAL)),
        checkpoint_interval=int(raw.get("checkpoint_interval", 0)),
        checkpoint_path=None if raw.get("checkpoint_path") is None else str(raw.get("checkpoint_path")),
        dt=float(raw.get("dt", 1.0)),
        backend=str(raw.get("backend", "cpu")),
    )


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.PCG64(np.uint64(int(seed))))


def _extract_channel_count_from_world_parameters(world_parameters: Mapping[str, Any]) -> int | None:
    if "returns" in world_parameters:
        returns = np.asarray(world_parameters["returns"], dtype=np.float64)
        if returns.ndim == 1 and returns.shape[0] > 0:
            return int(returns.shape[0])
    if "events" in world_parameters:
        events = list(world_parameters["events"])
        if events:
            first = events[0]
            if isinstance(first, Mapping) and "r_vec" in first:
                r_vec = np.asarray(first["r_vec"], dtype=np.float64)
                if r_vec.ndim == 1 and r_vec.shape[0] > 0:
                    return int(r_vec.shape[0])
    return None


def _normalize_world_parameters(world_parameters: Mapping[str, Any], *, horizon: int) -> dict[str, Any]:
    if not isinstance(world_parameters, Mapping):
        raise ValueError("invalid long-run configuration")

    out = dict(world_parameters)

    # Backward-compatible support for legacy event-sequence payloads.
    if "events" in out and "returns" not in out:
        events = list(out["events"])
        if len(events) == 0:
            raise ValueError("invalid long-run configuration")
        first_event = events[0]
        if not isinstance(first_event, Mapping) or "r_vec" not in first_event:
            raise ValueError("invalid long-run configuration")
        returns = np.asarray(first_event["r_vec"], dtype=np.float64)
        if returns.ndim != 1 or returns.shape[0] <= 0 or np.any(~np.isfinite(returns)):
            raise ValueError("invalid long-run configuration")

        costs: list[float] = []
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("invalid long-run configuration")
            c_total = float(event.get("c_total", 0.0))
            if not np.isfinite(c_total):
                raise ValueError("invalid long-run configuration")
            costs.append(float(c_total))

        out["returns"] = [float(value) for value in returns.tolist()]
        out["costs"] = costs

    if "returns" not in out:
        raise ValueError("invalid long-run configuration")

    if "costs" not in out:
        out["costs"] = [0.0 for _ in range(max(1, int(horizon)))]

    if "initial_wealth" not in out:
        out["initial_wealth"] = 1.0

    if "jitter_scale" not in out:
        out["jitter_scale"] = 0.0

    return out


def _validate_config(config: LongRunConfig, *, resident_genomes: Sequence[SelectorGenome | None] | None) -> LongRunConfig:
    if config.seed is None or config.horizon is None or config.world_parameters is None:
        raise ValueError("invalid long-run configuration")

    if config.generations is None:
        generations = 1
    else:
        generations = int(config.generations)

    if int(generations) <= 0:
        raise ValueError("invalid long-run configuration")

    if int(config.horizon) <= 0:
        raise ValueError("invalid long-run configuration")

    if float(config.dt) <= 0.0:
        raise ValueError("invalid long-run configuration")

    backend = str(config.backend).strip().lower()
    if backend not in {"cpu", "cuda"}:
        raise ValueError("invalid long-run configuration")

    policy = str(config.selection_policy).strip().lower()
    if policy not in {
        LONG_RUN_SELECTION_FITNESS_PROPORTIONAL,
        LONG_RUN_SELECTION_UNIFORM,
    }:
        raise ValueError("invalid long-run configuration")

    if int(config.checkpoint_interval) < 0:
        raise ValueError("invalid long-run configuration")

    if resident_genomes is None and config.population_size is None:
        raise ValueError("invalid long-run configuration")

    if resident_genomes is not None and len(list(resident_genomes)) == 0:
        raise ValueError("invalid long-run configuration")

    return LongRunConfig(
        seed=int(config.seed),
        population_size=None if config.population_size is None else int(config.population_size),
        channel_count=None if config.channel_count is None else int(config.channel_count),
        horizon=int(config.horizon),
        generations=int(generations),
        world_parameters=dict(config.world_parameters),
        mutation_config=_coerce_mutation_config(config.mutation_config),
        selection_policy=policy,
        checkpoint_interval=int(config.checkpoint_interval),
        checkpoint_path=config.checkpoint_path,
        dt=float(config.dt),
        backend=backend,
    )


def _random_row_stochastic_matrices(*, population: int, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for _ in range(int(population)):
        matrix = np.zeros((n, n), dtype=np.float64)
        for row in range(n):
            matrix[row, :] = rng.dirichlet(np.ones(n, dtype=np.float64))
        matrices.append(matrix)
    return matrices


def _default_settlement_params() -> dict[str, float]:
    return {
        "lambda_cash_share": 0.6,
        "accept_by_default": 1.0,
        "future_maturity_offset": 1.0,
    }


def _initial_population_from_seed(
    *,
    seed: int,
    population_size: int,
    channel_count: int,
) -> list[SelectorGenome]:
    rng = _seeded_rng(seed)
    matrices = _random_row_stochastic_matrices(population=population_size, n=channel_count, rng=rng)

    genomes: list[SelectorGenome] = []
    for matrix in matrices:
        output_weights = np.asarray(rng.dirichlet(np.ones(channel_count, dtype=np.float64)), dtype=np.float64)
        lambda_risk = float(rng.uniform(0.05, 0.5))
        genomes.append(
            SelectorGenome(
                flow_matrix=np.asarray(matrix, dtype=np.float64),
                output_weights=output_weights,
                lambda_risk=lambda_risk,
                selector_policy="term_risk",
                settlement_params=_default_settlement_params(),
            )
        )
    return genomes


def _fitness_vector_by_resident_id(*, report: Any, population_size: int) -> np.ndarray:
    values = np.zeros(int(population_size), dtype=np.float64)
    by_id = {int(row.process_id): float(row.fitness_wealth) for row in report.per_process}
    for process_id in range(int(population_size)):
        values[process_id] = float(by_id.get(process_id, 0.0))
    return values


def _selection_probabilities(fitness_values: np.ndarray, *, policy: str) -> np.ndarray:
    n = int(fitness_values.shape[0])
    if n <= 0:
        raise ValueError("invalid long-run configuration")

    policy_norm = str(policy).strip().lower()
    if policy_norm == LONG_RUN_SELECTION_UNIFORM:
        return np.full(n, 1.0 / float(n), dtype=np.float64)

    values = np.asarray(fitness_values, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("invalid long-run configuration")

    min_value = float(values.min())
    shifted = values - min_value if min_value < 0.0 else values.copy()
    shifted = np.maximum(shifted, 0.0)

    total = float(shifted.sum())
    if total <= 0.0:
        return np.full(n, 1.0 / float(n), dtype=np.float64)

    return shifted / total


def _build_next_generation(
    *,
    current_genomes: Sequence[SelectorGenome],
    report: Any,
    rng: np.random.Generator,
    mutation_config: MutationConfig,
    selection_policy: str,
) -> list[SelectorGenome]:
    population_size = int(len(current_genomes))
    fitness_values = _fitness_vector_by_resident_id(report=report, population_size=population_size)
    probabilities = _selection_probabilities(fitness_values, policy=selection_policy)

    next_genomes: list[SelectorGenome] = []
    for _ in range(population_size):
        parent_index = int(rng.choice(population_size, p=probabilities))
        parent_genome = current_genomes[parent_index]
        child: SelectorGenome | None = None

        # Deterministic retry path: invalid mutation draws consume additional
        # deterministic sub-seeds; final fallback preserves valid semantics.
        for _attempt in range(16):
            mutation_seed = int(rng.integers(low=0, high=np.iinfo(np.int64).max, dtype=np.int64))
            try:
                child = mutate_selector_genome(
                    parent_genome,
                    mutation_seed=mutation_seed,
                    config=mutation_config,
                    execution_backend="cpu",
                )
                break
            except ValueError:
                child = None

        if child is None:
            child = parent_genome

        next_genomes.append(child)

    return next_genomes


def _flow_row_sum_stats(genomes: Sequence[SelectorGenome]) -> tuple[float, float, bool]:
    row_sums: list[float] = []
    for genome in genomes:
        matrix = np.asarray(genome.flow_matrix, dtype=np.float64)
        row_sums.extend(float(value) for value in np.asarray(matrix.sum(axis=1), dtype=np.float64).tolist())
    if len(row_sums) == 0:
        return 0.0, 0.0, True
    arr = np.asarray(row_sums, dtype=np.float64)
    return float(arr.min()), float(arr.max()), bool(np.all(np.isfinite(arr)))


def _record_from_generation(
    *,
    generation_index: int,
    report: Any,
    genomes: Sequence[SelectorGenome],
) -> LongRunGenerationRecord:
    per_process_fitness = tuple(float(row.fitness_wealth) for row in report.per_process)
    per_process_survived = tuple(bool(row.survived) for row in report.per_process)
    min_flow_row_sum, max_flow_row_sum, flow_row_sums_finite = _flow_row_sum_stats(genomes)
    fitness_values_finite = bool(np.all(np.isfinite(np.asarray(per_process_fitness, dtype=np.float64))))
    return LongRunGenerationRecord(
        generation_index=int(generation_index),
        mean_fitness=float(report.mean_fitness),
        median_fitness=float(report.median_fitness),
        survival_rate=float(report.survival_rate),
        min_fitness=float(min(per_process_fitness) if per_process_fitness else 0.0),
        max_fitness=float(max(per_process_fitness) if per_process_fitness else 0.0),
        min_flow_row_sum=float(min_flow_row_sum),
        max_flow_row_sum=float(max_flow_row_sum),
        fitness_values_finite=bool(fitness_values_finite),
        flow_row_sums_finite=bool(flow_row_sums_finite),
        per_process_fitness=per_process_fitness,
        per_process_survived=per_process_survived,
    )


def _population_payload(genomes: Sequence[SelectorGenome]) -> tuple[dict[str, Any], ...]:
    return tuple(genome_to_dict(genome) for genome in genomes)


def _build_fitness_config(
    *,
    generation_seed: int,
    population_size: int,
    config: LongRunConfig,
    world_parameters: Mapping[str, Any],
) -> FitnessEvaluationConfig:
    return FitnessEvaluationConfig(
        seed=int(generation_seed),
        population_size=int(population_size),
        runtime_horizon=int(config.horizon),
        world_parameters=world_parameters,
        dt=float(config.dt),
        backend=str(config.backend),
    )


def _checkpoint_payload(
    *,
    config: LongRunConfig,
    next_generation_index: int,
    rng_state: Mapping[str, Any],
    genomes: Sequence[SelectorGenome],
) -> dict[str, Any]:
    return {
        "schema_version": LONG_RUN_CHECKPOINT_SCHEMA_VERSION,
        "config": {
            "seed": int(config.seed),
            "population_size": int(config.population_size) if config.population_size is not None else None,
            "channel_count": int(config.channel_count) if config.channel_count is not None else None,
            "horizon": int(config.horizon),
            "generations": int(config.generations),
            "world_parameters": _to_host_json_compatible(config.world_parameters),
            "mutation_config": {
                "noise_scale": float(config.mutation_config.noise_scale),
                "redistribution_share": float(config.mutation_config.redistribution_share),
                "lambda_risk_scale": float(config.mutation_config.lambda_risk_scale),
            },
            "selection_policy": str(config.selection_policy),
            "checkpoint_interval": int(config.checkpoint_interval),
            "checkpoint_path": config.checkpoint_path,
            "dt": float(config.dt),
            "backend": str(config.backend),
        },
        "next_generation_index": int(next_generation_index),
        "rng_state": _to_host_json_compatible(rng_state),
        "genomes": [_to_host_json_compatible(genome_to_dict(genome)) for genome in genomes],
    }


def resolve_long_run_checkpoint_file(
    *,
    checkpoint_path: str | Path,
    generation_index: int,
) -> Path:
    generation_i = int(generation_index)
    if generation_i <= 0:
        raise ValueError("invalid long-run checkpoint generation boundary")

    base = Path(checkpoint_path)
    stem = LONG_RUN_CHECKPOINT_FILE_STEM_TEMPLATE.format(generation_index=generation_i)
    if base.suffix:
        return base.with_name(f"{base.stem}.{stem}{base.suffix}")
    return base / f"{stem}.json"


def save_long_run_checkpoint(
    *,
    path: str | Path,
    config: LongRunConfig,
    next_generation_index: int,
    rng_state: Mapping[str, Any],
    genomes: Sequence[SelectorGenome],
) -> None:
    target = resolve_long_run_checkpoint_file(
        checkpoint_path=path,
        generation_index=int(next_generation_index),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError("long-run checkpoint already exists for generation boundary")
    payload = _checkpoint_payload(
        config=config,
        next_generation_index=next_generation_index,
        rng_state=rng_state,
        genomes=genomes,
    )
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_long_run_checkpoint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("invalid long-run checkpoint")

    if str(raw.get("schema_version")) != LONG_RUN_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("invalid long-run checkpoint")

    if "rng_state" not in raw:
        raise ValueError("invalid long-run checkpoint")

    if "genomes" not in raw:
        raise ValueError("invalid long-run checkpoint")

    return dict(raw)


def _checkpoint_config(raw_config: Mapping[str, Any]) -> LongRunConfig:
    return _coerce_config(raw_config)


def _merge_resume_config(checkpoint_config: LongRunConfig, override: LongRunConfig | None) -> LongRunConfig:
    if override is None:
        return checkpoint_config

    # Resume may extend total generations and adjust checkpoint controls,
    # but semantic core fields must remain identical.
    core_pairs = (
        ("seed", checkpoint_config.seed, override.seed),
        ("population_size", checkpoint_config.population_size, override.population_size),
        ("channel_count", checkpoint_config.channel_count, override.channel_count),
        ("horizon", checkpoint_config.horizon, override.horizon),
        ("dt", checkpoint_config.dt, override.dt),
        ("backend", checkpoint_config.backend, override.backend),
        ("selection_policy", checkpoint_config.selection_policy, override.selection_policy),
    )
    for _, from_checkpoint, from_override in core_pairs:
        if from_override is not None and from_checkpoint != from_override:
            raise ValueError("invalid long-run configuration")

    if override.world_parameters is not None and dict(checkpoint_config.world_parameters or {}) != dict(override.world_parameters):
        raise ValueError("invalid long-run configuration")

    if override.mutation_config is not None and _coerce_mutation_config(checkpoint_config.mutation_config) != _coerce_mutation_config(
        override.mutation_config
    ):
        raise ValueError("invalid long-run configuration")

    generations = checkpoint_config.generations if override.generations is None else int(override.generations)
    checkpoint_interval = (
        int(checkpoint_config.checkpoint_interval)
        if override.checkpoint_interval is None
        else int(override.checkpoint_interval)
    )
    checkpoint_path = checkpoint_config.checkpoint_path if override.checkpoint_path is None else override.checkpoint_path

    return LongRunConfig(
        seed=checkpoint_config.seed,
        population_size=checkpoint_config.population_size,
        channel_count=checkpoint_config.channel_count,
        horizon=checkpoint_config.horizon,
        generations=generations,
        world_parameters=checkpoint_config.world_parameters,
        mutation_config=_coerce_mutation_config(checkpoint_config.mutation_config),
        selection_policy=checkpoint_config.selection_policy,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=checkpoint_path,
        dt=checkpoint_config.dt,
        backend=checkpoint_config.backend,
    )


def run_long_run_harness(
    *,
    config: LongRunConfig | Mapping[str, Any],
    resident_genomes: Sequence[SelectorGenome | None] | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> LongRunResult:
    user_config = _coerce_config(config)
    validated_user_config = _validate_config(user_config, resident_genomes=resident_genomes)

    world_parameters = _normalize_world_parameters(
        validated_user_config.world_parameters or {},
        horizon=int(validated_user_config.horizon),
    )

    channel_count_from_world = _extract_channel_count_from_world_parameters(world_parameters)

    if resume_from_checkpoint is not None:
        checkpoint = load_long_run_checkpoint(resume_from_checkpoint)
        checkpoint_config = _validate_config(
            _checkpoint_config(dict(checkpoint["config"])),
            resident_genomes=None,
        )
        merged_config = _validate_config(
            _merge_resume_config(checkpoint_config, validated_user_config),
            resident_genomes=None,
        )
        config_eff = merged_config

        start_generation = int(checkpoint["next_generation_index"])
        current_genomes = [genome_from_dict(payload) for payload in list(checkpoint["genomes"])]
        rng = _seeded_rng(int(config_eff.seed))
        rng.bit_generator.state = dict(checkpoint["rng_state"])
    else:
        config_eff = validated_user_config
        start_generation = 0
        rng = _seeded_rng(int(config_eff.seed))

        if resident_genomes is None:
            population_size = int(config_eff.population_size or 0)
            if population_size <= 0:
                raise ValueError("invalid long-run configuration")

            if config_eff.channel_count is not None:
                channel_count = int(config_eff.channel_count)
            elif channel_count_from_world is not None:
                channel_count = int(channel_count_from_world)
            else:
                raise ValueError("invalid long-run configuration")

            if channel_count <= 0:
                raise ValueError("invalid long-run configuration")

            current_genomes = _initial_population_from_seed(
                seed=int(config_eff.seed),
                population_size=population_size,
                channel_count=channel_count,
            )
        else:
            current_genomes = [genome for genome in resident_genomes if genome is not None]
            if len(current_genomes) == 0:
                raise ValueError("invalid long-run configuration")

    if config_eff.generations is None or int(config_eff.generations) <= 0:
        raise ValueError("invalid long-run configuration")

    if int(config_eff.generations) < int(start_generation):
        raise ValueError("invalid long-run configuration")

    generation_records: list[LongRunGenerationRecord] = []
    population_states_by_generation: list[tuple[dict[str, Any], ...]] = []

    for generation_index in range(int(start_generation), int(config_eff.generations)):
        generation_seed = int(rng.integers(low=0, high=np.iinfo(np.int64).max, dtype=np.int64))
        fitness_config = _build_fitness_config(
            generation_seed=generation_seed,
            population_size=len(current_genomes),
            config=config_eff,
            world_parameters=world_parameters,
        )

        report = simulate_genome_population(current_genomes, fitness_config)
        population_states_by_generation.append(_population_payload(current_genomes))
        generation_records.append(
            _record_from_generation(
                generation_index=generation_index,
                report=report,
                genomes=current_genomes,
            )
        )

        current_genomes = _build_next_generation(
            current_genomes=current_genomes,
            report=report,
            rng=rng,
            mutation_config=config_eff.mutation_config,
            selection_policy=config_eff.selection_policy,
        )

        checkpoint_due = (
            config_eff.checkpoint_path is not None
            and int(config_eff.checkpoint_interval) > 0
            and ((generation_index + 1) % int(config_eff.checkpoint_interval) == 0)
        )
        if checkpoint_due:
            save_long_run_checkpoint(
                path=str(config_eff.checkpoint_path),
                config=config_eff,
                next_generation_index=int(generation_index + 1),
                rng_state=dict(rng.bit_generator.state),
                genomes=current_genomes,
            )

    return LongRunResult(
        config=config_eff,
        backend=str(config_eff.backend),
        start_generation=int(start_generation),
        completed_generations=int(config_eff.generations),
        generation_records=tuple(generation_records),
        population_states_by_generation=tuple(population_states_by_generation),
        final_population_state=_population_payload(current_genomes),
    )


def evaluate_long_run_emergence_metrics(
    result: LongRunResult,
    *,
    metrics_config: Any | None = None,
    pathology_config: Any | None = None,
    mutation_scale_factors: Sequence[float] | None = None,
) -> LongRunEmergenceEvaluation:
    from ..metrics.emergence_metrics import EmergenceMetricsConfig
    from ..metrics.emergence_metrics import EmergenceMetricsPoint
    from ..metrics.emergence_metrics import channel_utilization_sparsity
    from ..metrics.emergence_metrics import fitness_variance
    from ..metrics.emergence_metrics import strategy_lifetime_distribution
    from ..metrics.emergence_metrics import strategy_signature
    from ..metrics.emergence_metrics import structural_entropy
    from ..metrics.emergence_metrics import update_strategy_first_seen
    from ..metrics.pathology_detection import PathologyDetectionConfig
    from ..metrics.pathology_detection import PathologyDetectionState
    from ..metrics.pathology_detection import detect_pathologies_for_generation

    if metrics_config is None:
        metrics_cfg = EmergenceMetricsConfig()
    elif isinstance(metrics_config, EmergenceMetricsConfig):
        metrics_cfg = metrics_config
    else:
        raise ValueError("invalid long-run emergence metrics configuration")

    if pathology_config is None:
        pathology_cfg = PathologyDetectionConfig()
    elif isinstance(pathology_config, PathologyDetectionConfig):
        pathology_cfg = pathology_config
    else:
        raise ValueError("invalid long-run emergence metrics configuration")

    generation_records = tuple(result.generation_records)
    population_states = tuple(result.population_states_by_generation)

    if len(generation_records) != len(population_states):
        raise ValueError("invalid long-run emergence metrics state")

    if mutation_scale_factors is None:
        mutation_scales = tuple(1.0 for _ in generation_records)
    else:
        mutation_scales = tuple(float(value) for value in mutation_scale_factors)
        if len(mutation_scales) != len(generation_records):
            raise ValueError("invalid long-run emergence metrics state")
        if any(not np.isfinite(value) for value in mutation_scales):
            raise ValueError("invalid long-run emergence metrics state")

    first_seen_generation_by_strategy: dict[str, int] = {}
    pathology_state: PathologyDetectionState | None = None
    metric_points: list[EmergenceMetricsPoint] = []
    warning_messages: list[str] = []

    for generation_record, population_state, mutation_scale in zip(generation_records, population_states, mutation_scales):
        generation_index = int(generation_record.generation_index)
        survivors_mask = tuple(bool(value) for value in generation_record.per_process_survived)
        if len(survivors_mask) != len(population_state):
            raise ValueError("invalid long-run emergence metrics state")

        surviving_population = [
            dict(payload)
            for payload, survived in zip(population_state, survivors_mask)
            if bool(survived)
        ]
        if len(surviving_population) <= 0:
            raise ValueError("invalid long-run emergence metrics state")

        surviving_strategy_ids = tuple(strategy_signature(payload) for payload in surviving_population)
        first_seen_generation_by_strategy = update_strategy_first_seen(
            first_seen_generation_by_strategy,
            surviving_strategy_ids,
            generation_index=generation_index,
        )

        metric_points.append(
            EmergenceMetricsPoint(
                generation_index=generation_index,
                fitness_variance=fitness_variance(generation_record.per_process_fitness),
                structural_entropy=structural_entropy(surviving_population),
                channel_utilization_sparsity=channel_utilization_sparsity(
                    surviving_population,
                    threshold=float(metrics_cfg.channel_utilization_threshold),
                ),
                strategy_lifetime_distribution=strategy_lifetime_distribution(
                    surviving_strategy_ids,
                    first_seen_generation_by_strategy=first_seen_generation_by_strategy,
                    generation_index=generation_index,
                ),
            )
        )

        pathology_state, generation_messages = detect_pathologies_for_generation(
            generation_index=generation_index,
            population=population_state,
            surviving_population_size=sum(1 for item in survivors_mask if item),
            mutation_scale_factor=float(mutation_scale),
            config=pathology_cfg,
            state=pathology_state,
        )
        warning_messages.extend(generation_messages)

    return LongRunEmergenceEvaluation(
        metric_points=tuple(metric_points),
        pathology_warning_messages=tuple(str(message) for message in warning_messages),
    )
