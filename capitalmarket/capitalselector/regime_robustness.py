from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .evolution_contracts import PARITY_ATOL
from .evolution_contracts import PARITY_RTOL
from .fitness_engine import FitnessEvaluationConfig
from .fitness_engine import PopulationFitnessReport
from .fitness_engine import simulate_genome_population
from .genome import SelectorGenome
from .world_interface import CanonicalWorldObservation
from .world_parameters import WorldParametersSchema
from .world_parameters import world_parameters_from_dict
from .worlds.periodic_world import PeriodicWorld


ROBUSTNESS_PARITY_RTOL: float = PARITY_RTOL
ROBUSTNESS_PARITY_ATOL: float = PARITY_ATOL

ROBUSTNESS_PERTURBATION_TEMPORARY_RESOURCE_SHOCK = "temporary_resource_shock"
ROBUSTNESS_WORLD_TYPE_PERIODIC = "periodic"
ROBUSTNESS_WORLD_TYPE_NON_PERIODIC = "non_periodic"


@dataclass(frozen=True)
class _RegimeWorldSpec:
    world_type: str
    world_parameters: WorldParametersSchema


@dataclass(frozen=True)
class RobustnessPerturbationConfig:
    perturbation_type: str = ROBUSTNESS_PERTURBATION_TEMPORARY_RESOURCE_SHOCK
    start_tau: int = 1
    duration: int = 2
    attenuation: float = 0.2
    recovery_horizon: int = 8


@dataclass(frozen=True)
class RegimeRobustnessConfig:
    """Regime robustness configuration.

    Shared base contract fields across consolidated configs:
    ``seed``, ``world_parameters_by_regime`` (world parameters), ``backend``.
    """

    seed: int | None
    world_parameters_by_regime: Mapping[str, Mapping[str, Any]] | None
    regimes: Sequence[str] | None
    runtime_horizon: int | None
    resident_population_size: int = 4
    dt: float = 1.0
    backend: str = "cpu"
    perturbation: RobustnessPerturbationConfig = RobustnessPerturbationConfig()
    perturbation_regime: str | None = None


@dataclass(frozen=True)
class RegimeComparisonMetrics:
    regime: str
    viability_score: float
    mean_fitness: float
    survival_rate: float


@dataclass(frozen=True)
class PerturbationRecoveryMetrics:
    perturbation_type: str
    regime: str
    perturbation_start_tau: int
    perturbation_duration: int
    recovery_horizon: int
    recovery_generation_index: int
    extinction_flag: bool
    pre_shock_mean_wealth: float
    min_shock_window_mean_wealth: float
    end_of_horizon_mean_wealth: float


@dataclass(frozen=True)
class RobustnessResult:
    seed: int
    backend: str
    runtime_horizon: int
    resident_population_size: int
    regime_comparison_metrics: tuple[RegimeComparisonMetrics, ...]
    perturbation_recovery_metrics: tuple[PerturbationRecoveryMetrics, ...]
    cpu_fallback_used: int


class _TemporaryResourceShockWorld:
    def __init__(
        self,
        *,
        base_parameters: WorldParametersSchema,
        start_tau: int,
        duration: int,
        attenuation: float,
    ) -> None:
        self._params = base_parameters
        self._tau = 0
        self._start_tau = int(start_tau)
        self._duration = int(duration)
        self._attenuation = float(attenuation)

    def step(self, tau: int) -> None:
        tau_i = int(tau)
        if tau_i < 0:
            raise ValueError("temporary resource shock world requires tau >= 0")
        self._tau = tau_i

    def observe(self) -> CanonicalWorldObservation:
        in_window = self._start_tau <= int(self._tau) < self._start_tau + self._duration
        multiplier = 1.0 - self._attenuation if in_window else 1.0
        productivity = tuple(
            float(value * multiplier) for value in self._params.channel_productivity
        )
        return CanonicalWorldObservation(
            channel_productivity=productivity,
            channel_risk=tuple(float(v) for v in self._params.channel_risk),
            liquidity_scale=float(self._params.liquidity_scale),
            regime_id=int(self._tau // max(1, int(self._params.regime_period))),
            stress_active=bool(in_window),
        )

    def parameters(self) -> WorldParametersSchema:
        return self._params


class _NonPeriodicWorld:
    def __init__(self, *, world_parameters: WorldParametersSchema) -> None:
        self._params = world_parameters
        self._tau = 0

    def step(self, tau: int) -> None:
        tau_i = int(tau)
        if tau_i < 0:
            raise ValueError("non-periodic world requires tau >= 0")
        self._tau = tau_i

    def observe(self) -> CanonicalWorldObservation:
        stress_active = bool(
            float(self._params.stress_probability) > 0.0 and float(self._params.stress_intensity) > 0.0
        )
        stress_scale = float(1.0 - float(self._params.stress_intensity)) if stress_active else 1.0
        productivity = tuple(float(value * stress_scale) for value in self._params.channel_productivity)
        return CanonicalWorldObservation(
            channel_productivity=productivity,
            channel_risk=tuple(float(v) for v in self._params.channel_risk),
            liquidity_scale=float(self._params.liquidity_scale),
            regime_id=0,
            stress_active=bool(stress_active),
        )

    def parameters(self) -> WorldParametersSchema:
        return self._params


def _validate_non_negative_finite(value: float, *, field_name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"invalid robustness configuration: {field_name} must be finite")
    if out < 0.0:
        raise ValueError(f"invalid robustness configuration: {field_name} must be >= 0.0")
    return out


def _coerce_regime_world_spec(raw: Mapping[str, Any]) -> _RegimeWorldSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("invalid regime robustness configuration")

    if "world_type" in raw or "world_parameters" in raw:
        if "world_type" not in raw or "world_parameters" not in raw:
            raise ValueError("invalid regime robustness configuration")
        world_type = str(raw["world_type"]).strip().lower()
        world_payload_raw = raw["world_parameters"]
        if not isinstance(world_payload_raw, Mapping):
            raise ValueError("invalid regime robustness configuration")
        world_payload = world_payload_raw
    else:
        # Backward-compatible default: plain world-parameter payload implies periodic world.
        world_type = ROBUSTNESS_WORLD_TYPE_PERIODIC
        world_payload = raw

    if world_type not in {
        ROBUSTNESS_WORLD_TYPE_PERIODIC,
        ROBUSTNESS_WORLD_TYPE_NON_PERIODIC,
    }:
        raise ValueError("invalid regime robustness configuration")

    return _RegimeWorldSpec(
        world_type=world_type,
        world_parameters=world_parameters_from_dict(world_payload),
    )


def _regime_world_signature(spec: _RegimeWorldSpec) -> str:
    payload = spec.world_parameters.to_dict(include_schema_version=False)
    return (
        f"{spec.world_type}:"
        f"{json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)}"
    )


def _validate_config(config: RegimeRobustnessConfig) -> tuple[tuple[str, ...], dict[str, _RegimeWorldSpec]]:
    missing = (
        config.seed is None
        or config.world_parameters_by_regime is None
        or config.regimes is None
        or config.runtime_horizon is None
    )
    if missing:
        raise ValueError("invalid regime robustness configuration")

    if int(config.runtime_horizon) <= 0:
        raise ValueError("invalid regime robustness configuration")
    if int(config.resident_population_size) <= 0:
        raise ValueError("invalid regime robustness configuration")
    if float(config.dt) <= 0.0:
        raise ValueError("invalid regime robustness configuration")

    backend = str(config.backend).strip().lower()
    if backend not in {"cpu", "cuda"}:
        raise ValueError("invalid regime robustness configuration")

    if not isinstance(config.world_parameters_by_regime, Mapping):
        raise ValueError("invalid regime robustness configuration")

    regimes = tuple(str(item) for item in config.regimes)
    if len(regimes) < 2:
        raise ValueError("invalid regime robustness configuration")
    if len(set(regimes)) != len(regimes):
        raise ValueError("invalid regime robustness configuration")

    world_by_regime: dict[str, _RegimeWorldSpec] = {}
    for regime in regimes:
        if regime not in config.world_parameters_by_regime:
            raise ValueError("invalid regime robustness configuration")
        world_by_regime[regime] = _coerce_regime_world_spec(config.world_parameters_by_regime[regime])

    if len({_regime_world_signature(world_by_regime[regime]) for regime in regimes}) < 2:
        raise ValueError("invalid regime robustness configuration")

    perturbation = config.perturbation
    if str(perturbation.perturbation_type) != ROBUSTNESS_PERTURBATION_TEMPORARY_RESOURCE_SHOCK:
        raise ValueError("invalid regime robustness configuration")

    if int(perturbation.start_tau) < 0:
        raise ValueError("invalid regime robustness configuration")
    if int(perturbation.duration) <= 0:
        raise ValueError("invalid regime robustness configuration")
    if int(perturbation.start_tau) + int(perturbation.duration) > int(config.runtime_horizon):
        raise ValueError("invalid regime robustness configuration")
    if int(perturbation.recovery_horizon) <= 0:
        raise ValueError("invalid regime robustness configuration")

    attenuation = _validate_non_negative_finite(float(perturbation.attenuation), field_name="attenuation")
    if attenuation > 1.0:
        raise ValueError("invalid regime robustness configuration")

    perturbation_regime = regimes[0] if config.perturbation_regime is None else str(config.perturbation_regime)
    if perturbation_regime not in world_by_regime:
        raise ValueError("invalid regime robustness configuration")

    _validate_shock_magnitude_against_schema(
        world_by_regime[perturbation_regime].world_parameters,
        attenuation=float(attenuation),
    )

    return regimes, world_by_regime


def _build_world_for_regime(
    *,
    spec: _RegimeWorldSpec,
    seed: int,
    runtime_horizon: int,
) -> Any:
    if spec.world_type == ROBUSTNESS_WORLD_TYPE_PERIODIC:
        return PeriodicWorld(
            world_parameters=spec.world_parameters,
            seed=int(seed),
            stress_precompute_horizon=max(64, int(runtime_horizon) + 1),
        )
    if spec.world_type == ROBUSTNESS_WORLD_TYPE_NON_PERIODIC:
        return _NonPeriodicWorld(world_parameters=spec.world_parameters)
    raise ValueError("invalid regime robustness configuration")


def _validate_shock_magnitude_against_schema(
    world_parameters: WorldParametersSchema,
    *,
    attenuation: float,
) -> None:
    attenuation_f = float(attenuation)
    if attenuation_f < 0.0 or attenuation_f > 1.0:
        raise ValueError("invalid regime robustness configuration")

    scaled_productivity = tuple(
        float(value * (1.0 - attenuation_f)) for value in world_parameters.channel_productivity
    )

    # Validate candidate perturbation against schema bounds before application.
    _ = WorldParametersSchema(
        channel_productivity=scaled_productivity,
        channel_risk=world_parameters.channel_risk,
        liquidity_scale=world_parameters.liquidity_scale,
        regime_period=world_parameters.regime_period,
        stress_probability=world_parameters.stress_probability,
        stress_intensity=world_parameters.stress_intensity,
    )

    for base, perturbed in zip(world_parameters.channel_productivity, scaled_productivity):
        if perturbed > base + 1e-12:
            raise ValueError("invalid regime robustness configuration")


def _build_fitness_config(
    *,
    seed: int,
    runtime_horizon: int,
    resident_population_size: int,
    dt: float,
    backend: str,
    world: Any,
) -> FitnessEvaluationConfig:
    return FitnessEvaluationConfig(
        seed=int(seed),
        population_size=int(resident_population_size),
        runtime_horizon=int(runtime_horizon),
        world_parameters={
            "structured_world": world,
            "initial_wealth": 1.0,
            "rebirth_enabled": False,
        },
        dt=float(dt),
        backend=str(backend),
        evaluation_mode="genome_pipeline",
    )


def _mean_active_wealth_by_tau(report: PopulationFitnessReport) -> tuple[float, ...]:
    values: list[float] = []
    for step in report.generation_trajectory.steps:
        active_wealth = [float(snapshot.wealth) for snapshot in step.population_snapshot if not snapshot.dead]
        if active_wealth:
            values.append(float(np.mean(active_wealth)))
        else:
            values.append(0.0)
    return tuple(values)


def _build_perturbation_recovery_metrics(
    *,
    regime: str,
    report: PopulationFitnessReport,
    perturbation: RobustnessPerturbationConfig,
) -> PerturbationRecoveryMetrics:
    trajectory = _mean_active_wealth_by_tau(report)
    if not trajectory:
        raise ValueError("invalid robustness state")

    start_tau = int(perturbation.start_tau)
    duration = int(perturbation.duration)
    recovery_horizon = int(perturbation.recovery_horizon)

    pre_shock_idx = max(0, start_tau - 1)
    pre_shock_mean_wealth = float(trajectory[pre_shock_idx])

    shock_end_exclusive = min(len(trajectory), start_tau + duration)
    shock_window = trajectory[start_tau:shock_end_exclusive]
    if not shock_window:
        raise ValueError("invalid robustness state")

    min_shock_window_mean_wealth = float(min(shock_window))

    recovery_search_start = shock_end_exclusive
    recovery_search_end = min(len(trajectory) - 1, start_tau + recovery_horizon)
    recovery_generation_index = recovery_horizon + 1
    extinction_flag = True
    for tau in range(recovery_search_start, recovery_search_end + 1):
        if float(trajectory[tau]) >= float(pre_shock_mean_wealth):
            recovery_generation_index = int(tau - start_tau)
            extinction_flag = False
            break

    end_idx = min(len(trajectory) - 1, start_tau + recovery_horizon)
    end_of_horizon_mean_wealth = float(trajectory[end_idx])

    return PerturbationRecoveryMetrics(
        perturbation_type=str(perturbation.perturbation_type),
        regime=str(regime),
        perturbation_start_tau=start_tau,
        perturbation_duration=duration,
        recovery_horizon=recovery_horizon,
        recovery_generation_index=int(recovery_generation_index),
        extinction_flag=bool(extinction_flag),
        pre_shock_mean_wealth=float(pre_shock_mean_wealth),
        min_shock_window_mean_wealth=float(min_shock_window_mean_wealth),
        end_of_horizon_mean_wealth=float(end_of_horizon_mean_wealth),
    )


def run_regime_robustness_experiment(
    resident_genome: SelectorGenome | None,
    config: RegimeRobustnessConfig,
) -> RobustnessResult:
    if resident_genome is None:
        raise ValueError("fitness evaluation requires genome")

    regimes, world_by_regime = _validate_config(config)
    assert config.seed is not None
    assert config.runtime_horizon is not None

    seed = int(config.seed)
    runtime_horizon = int(config.runtime_horizon)
    resident_population_size = int(config.resident_population_size)
    backend = str(config.backend).strip().lower()

    regime_metrics: list[RegimeComparisonMetrics] = []
    cpu_fallback_used = 0

    for regime in regimes:
        world = _build_world_for_regime(
            spec=world_by_regime[regime],
            seed=seed,
            runtime_horizon=runtime_horizon,
        )
        report = simulate_genome_population(
            [resident_genome for _ in range(resident_population_size)],
            _build_fitness_config(
                seed=seed,
                runtime_horizon=runtime_horizon,
                resident_population_size=resident_population_size,
                dt=float(config.dt),
                backend=backend,
                world=world,
            ),
        )
        cpu_fallback_used += int(report.cpu_fallback_used)

        viability_score = float(report.mean_normalized_fitness) * float(report.survival_rate)
        if not np.isfinite(viability_score):
            raise ValueError("invalid robustness state")

        regime_metrics.append(
            RegimeComparisonMetrics(
                regime=str(regime),
                viability_score=float(viability_score),
                mean_fitness=float(report.mean_fitness),
                survival_rate=float(report.survival_rate),
            )
        )

    perturbation_regime = regimes[0] if config.perturbation_regime is None else str(config.perturbation_regime)
    perturbation = config.perturbation

    shock_world = _TemporaryResourceShockWorld(
        base_parameters=world_by_regime[perturbation_regime].world_parameters,
        start_tau=int(perturbation.start_tau),
        duration=int(perturbation.duration),
        attenuation=float(perturbation.attenuation),
    )
    perturbation_report = simulate_genome_population(
        [resident_genome for _ in range(resident_population_size)],
        _build_fitness_config(
            seed=seed,
            runtime_horizon=runtime_horizon,
            resident_population_size=resident_population_size,
            dt=float(config.dt),
            backend=backend,
            world=shock_world,
        ),
    )
    cpu_fallback_used += int(perturbation_report.cpu_fallback_used)

    perturbation_metrics = _build_perturbation_recovery_metrics(
        regime=perturbation_regime,
        report=perturbation_report,
        perturbation=perturbation,
    )

    if backend == "cuda" and int(cpu_fallback_used) > 0:
        raise ValueError("regime robustness experiment detected cpu fallback on cuda path")

    return RobustnessResult(
        seed=seed,
        backend=backend,
        runtime_horizon=runtime_horizon,
        resident_population_size=resident_population_size,
        regime_comparison_metrics=tuple(regime_metrics),
        perturbation_recovery_metrics=(perturbation_metrics,),
        cpu_fallback_used=int(cpu_fallback_used),
    )
