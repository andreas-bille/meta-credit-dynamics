"""Periodicity experiment harness — deterministic multi-generation runs over PeriodicWorld.

CPU-first determinism is the primary contract.  GPU parity applies to runtime
outputs within tolerances rtol=1e-6, atol=1e-8: per-process wealth
trajectories and fitness values, plus summary metrics mean_fitness,
median_fitness, and survival_rate. Internal helper artefacts need not be
bitwise identical across CPU and GPU.

Structured world inputs are population-shared per tau: all processes receive
the same world-derived runtime event for a given tau unless an explicitly
different model is introduced.

``stress_probability`` parameterizes deterministic stress schedule generation
under seed; runtime stepping consumes precomputed schedule entries and does not
resample stress stochastically per process/tau.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .genome import SelectorGenome
from .population_manager import run_generation_loop
from .world_parameters import WorldParametersSchema
from .world_regime_dataset import (
    WorldRegimeTimelineDataset,
    build_world_regime_timeline_dataset,
)
from .worlds.periodic_world import PeriodicWorld

# Documented parity tolerances for CPU/GPU runtime output comparison.
PARITY_RTOL: float = 1e-6
PARITY_ATOL: float = 1e-8


@dataclass(frozen=True)
class PeriodicityExperimentConfig:
    """Full configuration for a periodicity experiment run.

    Carries all knobs needed to reproduce the experiment from scratch:
    * seed — master seed for world and generation loop
    * runtime_horizon — number of tau steps per run
    * regime_period — world period in tau steps
    * stress_probability / stress_intensity — deterministic stress schedule knobs
    * channel_productivity / channel_risk / liquidity_scale — channel config
        * initial_wealth / rebirth_enabled — runtime controls forwarded to the
            canonical generation loop; part of the reproducibility contract
    """

    seed: int
    runtime_horizon: int
    regime_period: int
    stress_probability: float
    stress_intensity: float
    channel_productivity: tuple[float, ...]
    channel_risk: tuple[float, ...]
    liquidity_scale: float
    initial_wealth: float = 1.0
    rebirth_enabled: bool = False

    def world_parameters(self) -> WorldParametersSchema:
        """Build WorldParametersSchema from this config."""
        return WorldParametersSchema(
            channel_productivity=self.channel_productivity,
            channel_risk=self.channel_risk,
            liquidity_scale=self.liquidity_scale,
            regime_period=self.regime_period,
            stress_probability=self.stress_probability,
            stress_intensity=self.stress_intensity,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "runtime_horizon": int(self.runtime_horizon),
            "regime_period": int(self.regime_period),
            "stress_probability": float(self.stress_probability),
            "stress_intensity": float(self.stress_intensity),
            "channel_productivity": list(float(v) for v in self.channel_productivity),
            "channel_risk": list(float(v) for v in self.channel_risk),
            "liquidity_scale": float(self.liquidity_scale),
            "initial_wealth": float(self.initial_wealth),
            "rebirth_enabled": bool(self.rebirth_enabled),
        }


@dataclass(frozen=True)
class PeriodicityExperimentProcessRecord:
    """Per-process generation metrics record."""

    process_id: int
    fitness_wealth: float
    fitness_norm: float
    survived: bool
    death_tau: int | None
    wealth_trajectory: tuple[float, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "process_id": int(self.process_id),
            "fitness_wealth": float(self.fitness_wealth),
            "fitness_norm": float(self.fitness_norm),
            "survived": bool(self.survived),
            "death_tau": None if self.death_tau is None else int(self.death_tau),
            "wealth_trajectory": [float(v) for v in self.wealth_trajectory],
        }


@dataclass(frozen=True)
class PeriodicityExperimentMetrics:
    """Aggregated and per-process generation metrics from a single experiment run.

    These fields are sufficient for downstream comparison pipelines:
    mean/median fitness, survival rate, and full per-process records (including
    wealth trajectories) are the primary CPU/GPU parity targets.
    """

    mean_fitness: float
    median_fitness: float
    survival_rate: float
    per_process: tuple[PeriodicityExperimentProcessRecord, ...]
    step_count: int
    backend: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "mean_fitness": float(self.mean_fitness),
            "median_fitness": float(self.median_fitness),
            "survival_rate": float(self.survival_rate),
            "per_process": [rec.to_payload() for rec in self.per_process],
            "step_count": int(self.step_count),
            "backend": str(self.backend),
        }


@dataclass(frozen=True)
class PeriodicityExperimentResult:
    """Complete output of a periodicity experiment run.

    Combines the experiment configuration, generation metrics, and world regime
    timeline into a single deterministic artefact for downstream analysis and
    reproducibility checks.

        Determinism contract:
        * Identical (config, genomes, backend="cpu") → identical result.
        * CPU vs GPU: per-process wealth_trajectory, fitness_wealth, fitness_norm,
            and summary metrics mean_fitness, median_fitness, survival_rate must
            satisfy rtol=1e-6, atol=1e-8 (PARITY_RTOL / PARITY_ATOL).
    """

    config: PeriodicityExperimentConfig
    metrics: PeriodicityExperimentMetrics
    regime_timeline: WorldRegimeTimelineDataset

    def to_payload(self) -> dict[str, Any]:
        return {
            "config": self.config.to_payload(),
            "metrics": self.metrics.to_payload(),
            "regime_timeline": self.regime_timeline.to_payload(),
        }



def run_periodicity_experiment(
    genomes: Sequence[SelectorGenome | None],
    *,
    config: PeriodicityExperimentConfig,
    backend: str = "cpu",
) -> PeriodicityExperimentResult:
    """Run a periodicity experiment over a PeriodicWorld and return a deterministic result.

    Drives ``run_generation_loop`` with a ``_PeriodicRuntimeWorld`` wrapper so
    the canonical generation loop receives structured runtime inputs from the
    world.  Regime timeline is built independently from ``config`` using
    ``build_world_regime_timeline_dataset`` — identical seed and params guarantee
    alignment with the world driving the loop.

    Args:
        genomes: Population of selector genomes (must not be empty).
        config: Full experiment configuration (seed, horizon, period, stress knobs).
        backend: Runtime backend — "cpu" or "cuda".

    Returns:
        ``PeriodicityExperimentResult`` with config, metrics, and regime timeline.
    """
    if not isinstance(config, PeriodicityExperimentConfig):
        raise ValueError("run_periodicity_experiment requires a PeriodicityExperimentConfig")

    world_params = config.world_parameters()

    world = PeriodicWorld(
        world_parameters=world_params,
        seed=int(config.seed),
        stress_precompute_horizon=max(64, int(config.runtime_horizon) + 1),
    )
    world_parameters_dict: dict[str, Any] = {
        "structured_world": world,
        "initial_wealth": float(config.initial_wealth),
        "rebirth_enabled": bool(config.rebirth_enabled),
    }

    report = run_generation_loop(
        list(genomes),
        seed=int(config.seed),
        runtime_horizon=int(config.runtime_horizon),
        world_parameters=world_parameters_dict,
        dt=1.0,
        backend=str(backend),
    )

    per_process = tuple(
        PeriodicityExperimentProcessRecord(
            process_id=int(row.process_id),
            fitness_wealth=float(row.fitness_wealth),
            fitness_norm=float(row.fitness_norm),
            survived=bool(row.survived),
            death_tau=None if row.death_tau is None else int(row.death_tau),
            wealth_trajectory=tuple(float(v) for v in row.wealth_trajectory),
        )
        for row in report.per_process
    )

    metrics = PeriodicityExperimentMetrics(
        mean_fitness=float(report.mean_fitness),
        median_fitness=float(report.median_fitness),
        survival_rate=float(report.survival_rate),
        per_process=per_process,
        step_count=int(len(report.generation_trajectory.steps)),
        backend=str(report.backend),
    )

    regime_timeline = build_world_regime_timeline_dataset(
        horizon=int(config.runtime_horizon),
        world_parameters=world_params,
        seed=int(config.seed),
    )

    return PeriodicityExperimentResult(
        config=config,
        metrics=metrics,
        regime_timeline=regime_timeline,
    )
