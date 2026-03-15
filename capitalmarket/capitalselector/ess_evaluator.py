"""Legacy evaluation utility.

This module predates the v0.9.x canonical evolution runner.
It intentionally operates below run_generation_loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .fitness_engine import FitnessEvaluationConfig
from .fitness_engine import GenomeFitnessResult
from .fitness_engine import PopulationFitnessReport
from .fitness_engine import simulate_genome_population
from .genome import SelectorGenome
from .genome_mutation import MutationConfig
from .genome_mutation import mutate_selector_genome
from .genome_serialization import genome_from_dict
from .genome_serialization import genome_to_dict
from .mutation_scaling import apply_mutation_scaling
from .mutation_scaling import compute_mutation_scale_factor
from .mutation_scaling import MutationScalingConfig


@dataclass(frozen=True)
class ESSEvaluationConfig:
    seeds: Sequence[int] | None
    regimes: Sequence[str] | None
    resident_population_size: int | None
    mutant_population_size: int | None
    mutant_insertion_tau: int | None
    runtime_horizon: int | None
    dt: float
    world_parameters: Mapping[str, Mapping[str, Any]] | None
    backend: str = "cpu"
    epsilon: float = 1e-9


@dataclass(frozen=True)
class ESSScenarioReport:
    seed: int
    regime: str
    resident_baseline: PopulationFitnessReport
    invasion_report: PopulationFitnessReport
    mutant_survival_rate: float
    mutant_mean_fitness: float
    resident_mean_fitness: float
    resident_fitness_shift: float
    is_mutant_dominant: bool


@dataclass(frozen=True)
class ESSEvaluationReport:
    scenario_reports: tuple[ESSScenarioReport, ...]
    mutant_survival_rate: float
    mutant_mean_fitness: float
    resident_mean_fitness: float
    resident_fitness_shift: float
    is_mutant_dominant: bool
    cpu_fallback_used: int
    backend: str


def _validate_ess_config(config: ESSEvaluationConfig) -> None:
    missing = (
        config.seeds is None
        or config.regimes is None
        or config.resident_population_size is None
        or config.mutant_population_size is None
        or config.mutant_insertion_tau is None
        or config.runtime_horizon is None
        or config.world_parameters is None
    )
    if missing:
        raise ValueError("invalid ESS evaluation configuration")

    if len(list(config.seeds)) == 0 or len(list(config.regimes)) == 0:
        raise ValueError("invalid ESS evaluation configuration")

    if int(config.resident_population_size) <= 0 or int(config.mutant_population_size) <= 0:
        raise ValueError("invalid ESS evaluation configuration")

    if int(config.runtime_horizon) <= 0 or float(config.dt) <= 0.0:
        raise ValueError("invalid ESS evaluation configuration")

    if int(config.mutant_insertion_tau) < 0 or int(config.mutant_insertion_tau) >= int(config.runtime_horizon):
        raise ValueError("invalid ESS evaluation configuration")

    if not isinstance(config.world_parameters, Mapping):
        raise ValueError("invalid ESS evaluation configuration")

    for regime in list(config.regimes):
        if str(regime) not in config.world_parameters:
            raise ValueError("invalid ESS evaluation configuration")


def _mean_fitness(results: Sequence[GenomeFitnessResult]) -> float:
    if not results:
        raise ValueError("invalid ESS comparison state")
    values = np.asarray([float(item.fitness_wealth) for item in results], dtype=np.float64)
    if values.ndim != 1 or values.shape[0] == 0 or np.any(~np.isfinite(values)):
        raise ValueError("invalid ESS comparison state")
    return float(np.mean(values))


def _survival_rate(results: Sequence[GenomeFitnessResult]) -> float:
    if not results:
        raise ValueError("invalid ESS comparison state")
    values = np.asarray([1.0 if bool(item.survived) else 0.0 for item in results], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("invalid ESS comparison state")
    return float(np.mean(values))


def _dominance_decision(*, mutant_mean: float, resident_mean: float, epsilon: float) -> bool:
    if not np.isfinite(float(mutant_mean)) or not np.isfinite(float(resident_mean)):
        raise ValueError("invalid ESS comparison state")
    if not np.isfinite(float(epsilon)):
        raise ValueError("invalid ESS comparison state")
    return bool(float(mutant_mean) > float(resident_mean) + float(epsilon))


def _enforce_cuda_fallback_guard(*, backend: str, cpu_fallback_used: int) -> None:
    if str(backend).strip().lower() == "cuda" and int(cpu_fallback_used) > 0:
        raise ValueError("ESS evaluation detected cpu fallback on cuda path")


def evaluate_ess_stability(
    resident_genome: SelectorGenome | None,
    mutant_genome: SelectorGenome | None,
    config: ESSEvaluationConfig,
) -> ESSEvaluationReport:
    if resident_genome is None or mutant_genome is None:
        raise ValueError("fitness evaluation requires genome")

    _validate_ess_config(config)
    assert config.seeds is not None
    assert config.regimes is not None
    assert config.world_parameters is not None
    assert config.resident_population_size is not None
    assert config.mutant_population_size is not None
    assert config.mutant_insertion_tau is not None
    assert config.runtime_horizon is not None

    scenario_reports: list[ESSScenarioReport] = []
    cpu_fallback_used = 0

    for seed in [int(item) for item in config.seeds]:
        for regime in [str(item) for item in config.regimes]:
            world = dict(config.world_parameters[regime])

            resident_cfg = FitnessEvaluationConfig(
                seed=int(seed),
                population_size=int(config.resident_population_size),
                runtime_horizon=int(config.runtime_horizon),
                world_parameters=world,
                dt=float(config.dt),
                backend=str(config.backend),
                evaluation_mode="genome_pipeline",
            )

            resident_report = simulate_genome_population(
                [resident_genome for _ in range(int(config.resident_population_size))],
                resident_cfg,
            )

            invasion_report = simulate_genome_population(
                [resident_genome for _ in range(int(config.resident_population_size))],
                resident_cfg,
                injected_genomes_by_tau={
                    int(config.mutant_insertion_tau): [mutant_genome for _ in range(int(config.mutant_population_size))]
                },
            )

            mutant_results = [item for item in invasion_report.per_process if str(item.cohort) == "mutant"]
            resident_results = [item for item in invasion_report.per_process if str(item.cohort) == "resident"]

            mutant_mean_fitness = _mean_fitness(mutant_results)
            resident_mean_fitness = _mean_fitness(resident_results)
            mutant_survival_rate = _survival_rate(mutant_results)
            resident_fitness_shift = float(resident_mean_fitness - float(resident_report.mean_fitness))
            dominant = _dominance_decision(
                mutant_mean=mutant_mean_fitness,
                resident_mean=resident_mean_fitness,
                epsilon=float(config.epsilon),
            )

            cpu_fallback_used += int(resident_report.cpu_fallback_used)
            cpu_fallback_used += int(invasion_report.cpu_fallback_used)

            scenario_reports.append(
                ESSScenarioReport(
                    seed=int(seed),
                    regime=str(regime),
                    resident_baseline=resident_report,
                    invasion_report=invasion_report,
                    mutant_survival_rate=float(mutant_survival_rate),
                    mutant_mean_fitness=float(mutant_mean_fitness),
                    resident_mean_fitness=float(resident_mean_fitness),
                    resident_fitness_shift=float(resident_fitness_shift),
                    is_mutant_dominant=bool(dominant),
                )
            )

    if not scenario_reports:
        raise ValueError("invalid ESS comparison state")

    mutant_survival_rate = float(np.mean([item.mutant_survival_rate for item in scenario_reports]))
    mutant_mean_fitness = float(np.mean([item.mutant_mean_fitness for item in scenario_reports]))
    resident_mean_fitness = float(np.mean([item.resident_mean_fitness for item in scenario_reports]))
    resident_fitness_shift = float(np.mean([item.resident_fitness_shift for item in scenario_reports]))

    is_mutant_dominant = _dominance_decision(
        mutant_mean=mutant_mean_fitness,
        resident_mean=resident_mean_fitness,
        epsilon=float(config.epsilon),
    )

    _enforce_cuda_fallback_guard(backend=str(config.backend), cpu_fallback_used=int(cpu_fallback_used))

    return ESSEvaluationReport(
        scenario_reports=tuple(scenario_reports),
        mutant_survival_rate=float(mutant_survival_rate),
        mutant_mean_fitness=float(mutant_mean_fitness),
        resident_mean_fitness=float(resident_mean_fitness),
        resident_fitness_shift=float(resident_fitness_shift),
        is_mutant_dominant=bool(is_mutant_dominant),
        cpu_fallback_used=int(cpu_fallback_used),
        backend=str(config.backend).strip().lower(),
    )


ESS_MUTATION_SCALING_ENABLED = "enabled"
ESS_MUTATION_SCALING_FROZEN = "frozen"

ESS_MUTANT_PATH_CANONICAL = "canonical_mutation_path"
ESS_MUTANT_PATH_SCALED_OVERRIDE = "scaled_mutation_path"


@dataclass(frozen=True)
class ESSProbeConfig:
    """ESS probe configuration.

    Shared base contract fields across consolidated configs:
    ``seed``, ``world_parameters_by_regime`` (world parameters), ``backend``.
    """

    seed: int | None
    world_parameters_by_regime: Mapping[str, Mapping[str, Any]] | None
    regimes: Sequence[str] | None
    n_resident_generations: int | None
    n_mutant_trials: int | None
    runtime_horizon: int | None
    resident_population_size: int = 3
    mutant_population_size: int = 1
    dt: float = 1.0
    backend: str = "cpu"
    min_survival_tau: int = 1
    mutation_scaling_mode: str | None = None
    mutation_scaling_decay_rate: float = 0.01
    mutant_generation_path: str = ESS_MUTANT_PATH_CANONICAL
    mutation_noise_scale: float = 0.01
    mutation_redistribution_share: float = 0.05
    mutation_lambda_risk_scale: float = 0.05


@dataclass(frozen=True)
class ESSProbeTrialResult:
    trial_index: int
    regime: str
    invasion_success: bool
    survival_time: int
    mutant_mean_fitness: float
    resident_mean_fitness: float
    resident_fitness_shift: float
    mutant_hash: str


@dataclass(frozen=True)
class ESSProbeResult:
    seed: int
    backend: str
    mutation_scaling_mode: str
    mutant_generation_path: str
    min_survival_tau: int
    n_resident_generations: int
    n_mutant_trials: int
    trial_results: tuple[ESSProbeTrialResult, ...]
    invasion_successes: int
    mutant_fixation_probability: float
    mean_survival_time: float
    mean_mutant_mean_fitness: float
    mean_resident_mean_fitness: float
    mean_resident_fitness_shift: float
    resident_hash_before: str
    resident_hash_after: str
    cpu_fallback_used: int


def resolve_probe_mutation_scaling_mode(mode: str | None) -> str:
    normalized = "" if mode is None else str(mode).strip().lower()
    if normalized == "":
        return ESS_MUTATION_SCALING_FROZEN
    if normalized in {ESS_MUTATION_SCALING_ENABLED, ESS_MUTATION_SCALING_FROZEN}:
        return normalized
    raise ValueError("invalid ESS probe configuration")


def _splitmix64(value: int) -> int:
    x = int(value) & 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return int(z & 0xFFFFFFFFFFFFFFFF)


def _stable_text_seed(text: str) -> int:
    acc = 0
    for value in text.encode("utf-8"):
        acc = _splitmix64(acc ^ int(value))
    return int(acc)


def _genome_hash(genome: SelectorGenome) -> str:
    payload = genome_to_dict(genome)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _validate_probe_config(config: ESSProbeConfig) -> None:
    missing = (
        config.seed is None
        or config.world_parameters_by_regime is None
        or config.regimes is None
        or config.n_resident_generations is None
        or config.n_mutant_trials is None
        or config.runtime_horizon is None
    )
    if missing:
        raise ValueError("invalid ESS probe configuration")

    if int(config.n_resident_generations) <= 0:
        raise ValueError("invalid ESS probe configuration")
    if int(config.n_mutant_trials) <= 0:
        raise ValueError("invalid ESS probe configuration")
    if int(config.runtime_horizon) <= 0:
        raise ValueError("invalid ESS probe configuration")
    if int(config.resident_population_size) <= 0:
        raise ValueError("invalid ESS probe configuration")
    if int(config.mutant_population_size) <= 0:
        raise ValueError("invalid ESS probe configuration")
    if int(config.min_survival_tau) <= 0:
        raise ValueError("invalid ESS probe configuration")
    if float(config.dt) <= 0.0:
        raise ValueError("invalid ESS probe configuration")

    if not isinstance(config.world_parameters_by_regime, Mapping):
        raise ValueError("invalid ESS probe configuration")

    regimes = [str(item) for item in config.regimes]
    if len(regimes) == 0:
        raise ValueError("invalid ESS probe configuration")
    for regime in regimes:
        if regime not in config.world_parameters_by_regime:
            raise ValueError("invalid ESS probe configuration")

    if str(config.mutant_generation_path).strip() not in {
        ESS_MUTANT_PATH_CANONICAL,
        ESS_MUTANT_PATH_SCALED_OVERRIDE,
    }:
        raise ValueError("invalid ESS probe configuration")

    resolve_probe_mutation_scaling_mode(config.mutation_scaling_mode)

    numeric = (
        config.mutation_scaling_decay_rate,
        config.mutation_noise_scale,
        config.mutation_redistribution_share,
        config.mutation_lambda_risk_scale,
    )
    if any(not np.isfinite(float(item)) for item in numeric):
        raise ValueError("invalid ESS probe configuration")
    if any(float(item) < 0.0 for item in numeric):
        raise ValueError("invalid ESS probe configuration")


def _base_mutation_config(config: ESSProbeConfig) -> MutationConfig:
    return MutationConfig(
        noise_scale=float(config.mutation_noise_scale),
        redistribution_share=float(config.mutation_redistribution_share),
        lambda_risk_scale=float(config.mutation_lambda_risk_scale),
    )


def _effective_mutation_config(
    *,
    base_config: MutationConfig,
    scaling_mode: str,
    scaling_config: MutationScalingConfig,
    n_resident_generations: int,
    trial_index: int,
) -> MutationConfig:
    if scaling_mode == ESS_MUTATION_SCALING_FROZEN:
        # Fixed evaluation scaling during probe unless explicitly overridden.
        factor = compute_mutation_scale_factor(
            generation_index=int(n_resident_generations),
            config=scaling_config,
        )
    else:
        factor = compute_mutation_scale_factor(
            generation_index=int(n_resident_generations + trial_index),
            config=scaling_config,
        )
    return apply_mutation_scaling(base_config, scale_factor=float(factor))


def _build_mutant_genome(
    *,
    resident_genome: SelectorGenome,
    mutation_seed: int,
    mutation_config: MutationConfig,
    mutant_generation_path: str,
) -> SelectorGenome:
    path = str(mutant_generation_path).strip()
    if path == ESS_MUTANT_PATH_CANONICAL:
        return mutate_selector_genome(
            resident_genome,
            mutation_seed=int(mutation_seed),
            config=mutation_config,
            execution_backend="cpu",
        )

    if path == ESS_MUTANT_PATH_SCALED_OVERRIDE:
        override_config = apply_mutation_scaling(mutation_config, scale_factor=0.5)
        return mutate_selector_genome(
            resident_genome,
            mutation_seed=int(mutation_seed),
            config=override_config,
            execution_backend="cpu",
        )

    raise ValueError("invalid ESS probe configuration")


def _survival_time_tau(*, rows: Sequence[GenomeFitnessResult], dt: float) -> int:
    if not rows:
        return 0
    values = [int(round(float(row.time_to_death) / float(dt))) for row in rows]
    return max(0, max(values))


def run_ess_probe_experiment(
    resident_genome: SelectorGenome | None,
    config: ESSProbeConfig,
) -> ESSProbeResult:
    if resident_genome is None:
        raise ValueError("fitness evaluation requires genome")

    _validate_probe_config(config)
    assert config.seed is not None
    assert config.world_parameters_by_regime is not None
    assert config.regimes is not None
    assert config.n_resident_generations is not None
    assert config.n_mutant_trials is not None
    assert config.runtime_horizon is not None

    seed = int(config.seed)
    backend = str(config.backend).strip().lower()
    regimes = [str(item) for item in config.regimes]
    n_resident_generations = int(config.n_resident_generations)
    n_mutant_trials = int(config.n_mutant_trials)
    runtime_horizon = int(config.runtime_horizon)
    min_survival_tau = int(config.min_survival_tau)

    scaling_mode = resolve_probe_mutation_scaling_mode(config.mutation_scaling_mode)
    scaling_config = MutationScalingConfig(
        mode="generation_dependent",
        decay_rate=float(config.mutation_scaling_decay_rate),
    )
    base_mutation = _base_mutation_config(config)

    # Structural freeze guard: use canonical clone and verify hash unchanged.
    resident_frozen = genome_from_dict(genome_to_dict(resident_genome))
    resident_hash_before = _genome_hash(resident_frozen)

    trial_results: list[ESSProbeTrialResult] = []
    cpu_fallback_used = 0

    for regime in regimes:
        world = dict(config.world_parameters_by_regime[regime])
        regime_seed = _stable_text_seed(regime)

        training_seed = _splitmix64(seed ^ regime_seed ^ 0x1111111111111111)
        resident_cfg = FitnessEvaluationConfig(
            seed=int(training_seed),
            population_size=int(config.resident_population_size),
            runtime_horizon=int(n_resident_generations),
            world_parameters=world,
            dt=float(config.dt),
            backend=backend,
            evaluation_mode="genome_pipeline",
        )
        resident_baseline = simulate_genome_population(
            [resident_frozen for _ in range(int(config.resident_population_size))],
            resident_cfg,
        )
        cpu_fallback_used += int(resident_baseline.cpu_fallback_used)

        for trial_index in range(n_mutant_trials):
            mutation_config = _effective_mutation_config(
                base_config=base_mutation,
                scaling_mode=scaling_mode,
                scaling_config=scaling_config,
                n_resident_generations=n_resident_generations,
                trial_index=int(trial_index),
            )

            mutation_seed = _splitmix64(
                seed
                ^ regime_seed
                ^ (int(trial_index + 1) << 32)
                ^ 0x3333333333333333
            )
            mutant = _build_mutant_genome(
                resident_genome=resident_frozen,
                mutation_seed=int(mutation_seed),
                mutation_config=mutation_config,
                mutant_generation_path=str(config.mutant_generation_path),
            )

            invasion_seed = _splitmix64(
                seed
                ^ regime_seed
                ^ (int(trial_index + 1) << 24)
                ^ 0x2222222222222222
            )
            invasion_cfg = FitnessEvaluationConfig(
                seed=int(invasion_seed),
                population_size=int(config.resident_population_size),
                runtime_horizon=int(runtime_horizon),
                world_parameters=world,
                dt=float(config.dt),
                backend=backend,
                evaluation_mode="genome_pipeline",
            )

            invasion_report = simulate_genome_population(
                [resident_frozen for _ in range(int(config.resident_population_size))],
                invasion_cfg,
                injected_genomes_by_tau={
                    0: [mutant for _ in range(int(config.mutant_population_size))],
                },
            )
            cpu_fallback_used += int(invasion_report.cpu_fallback_used)

            mutant_rows = [item for item in invasion_report.per_process if str(item.cohort) == "mutant"]
            resident_rows = [item for item in invasion_report.per_process if str(item.cohort) == "resident"]
            mutant_mean = _mean_fitness(mutant_rows)
            resident_mean = _mean_fitness(resident_rows)

            survival_time = _survival_time_tau(rows=mutant_rows, dt=float(config.dt))
            invasion_success = bool(survival_time >= int(min_survival_tau))

            trial_results.append(
                ESSProbeTrialResult(
                    trial_index=int(trial_index),
                    regime=str(regime),
                    invasion_success=bool(invasion_success),
                    survival_time=int(survival_time),
                    mutant_mean_fitness=float(mutant_mean),
                    resident_mean_fitness=float(resident_mean),
                    resident_fitness_shift=float(resident_mean - float(resident_baseline.mean_fitness)),
                    mutant_hash=_genome_hash(mutant),
                )
            )

    resident_hash_after = _genome_hash(resident_frozen)
    if resident_hash_after != resident_hash_before:
        raise ValueError("ESS probe modified frozen resident strategy")

    if not trial_results:
        raise ValueError("invalid ESS probe configuration")

    # Trial-level fixation metric: when multiple regimes are configured, a
    # mutant trial is counted as successful only if it invades in every regime.
    outcomes_by_trial: dict[int, list[bool]] = {}
    for item in trial_results:
        outcomes_by_trial.setdefault(int(item.trial_index), []).append(bool(item.invasion_success))

    expected_trial_ids = set(range(n_mutant_trials))
    if set(outcomes_by_trial.keys()) != expected_trial_ids:
        raise ValueError("invalid ESS probe metrics")

    invasion_successes = int(
        sum(
            1
            for trial_index in range(n_mutant_trials)
            if all(outcomes_by_trial[int(trial_index)])
        )
    )
    mutant_fixation_probability = float(invasion_successes / float(n_mutant_trials))
    if mutant_fixation_probability < 0.0 or mutant_fixation_probability > 1.0:
        raise ValueError("invalid ESS probe metrics")

    return ESSProbeResult(
        seed=int(seed),
        backend=backend,
        mutation_scaling_mode=str(scaling_mode),
        mutant_generation_path=str(config.mutant_generation_path),
        min_survival_tau=int(min_survival_tau),
        n_resident_generations=int(n_resident_generations),
        n_mutant_trials=int(n_mutant_trials),
        trial_results=tuple(trial_results),
        invasion_successes=int(invasion_successes),
        mutant_fixation_probability=float(mutant_fixation_probability),
        mean_survival_time=float(np.mean([item.survival_time for item in trial_results])),
        mean_mutant_mean_fitness=float(np.mean([item.mutant_mean_fitness for item in trial_results])),
        mean_resident_mean_fitness=float(np.mean([item.resident_mean_fitness for item in trial_results])),
        mean_resident_fitness_shift=float(np.mean([item.resident_fitness_shift for item in trial_results])),
        resident_hash_before=str(resident_hash_before),
        resident_hash_after=str(resident_hash_after),
        cpu_fallback_used=int(cpu_fallback_used),
    )
