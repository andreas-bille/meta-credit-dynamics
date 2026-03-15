"""Evolution demo: dataset extraction and visualization for the generation loop."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from capitalmarket.capitalselector.fitness_engine import (
    GenomeFitnessResult,
    PopulationFitnessReport,
)
from capitalmarket.capitalselector.evolution_contracts import validate_flow_matrix_invariants
from capitalmarket.capitalselector.genome import SelectorGenome
from capitalmarket.capitalselector.population_manager import run_generation_loop


@dataclass(frozen=True)
class EvolutionDemoDataset:
    """Numeric snapshot of a single generation-loop run.

    All fields contain only plain Python/numpy primitives so the dataset is
    fully serialisable and unambiguously comparable between runs.
    """

    # Per-process wealth trajectory: outer index = process, inner = time step.
    # Includes all processes — founding and reborn. Reborn entries start at step 0
    # of their own trajectory, not the global simulation step. Use survival_curve
    # for population-level alive/dead analysis (founders only).
    wealth_trajectories: tuple[tuple[float, ...], ...]

    # Fraction of founding processes still alive at each simulation step.
    # survival_curve[t] = alive_founders(t) / initial_population_size
    # Only founding processes (IDs 0..N-1) are tracked; reborn processes are excluded.
    survival_curve: tuple[float, ...]

    # F[n,m] snapshot: row-major nested tuple from the first genome's flow matrix.
    flow_matrix_snapshot: tuple[tuple[float, ...], ...]

    # Aggregate metrics.
    mean_fitness: float
    survival_rate: float


def _survival_curve(
    per_process: Sequence[GenomeFitnessResult],
    runtime_horizon: int,
    initial_population_size: int,
) -> tuple[float, ...]:
    if initial_population_size == 0:
        return tuple()
    # per_process is sorted by process_id (fitness_engine iterates
    # sorted(wealth_trajectories.keys())). Founding processes receive IDs 0..N-1
    # assigned via enumerate; reborn processes receive higher IDs via
    # _next_process_id. The slice below therefore reliably selects only founders.
    if len(per_process) < initial_population_size:
        raise ValueError(
            f"per_process has {len(per_process)} entries, "
            f"expected at least {initial_population_size} founding processes"
        )
    founders = per_process[:initial_population_size]
    result: list[float] = []
    for t in range(runtime_horizon):
        alive = sum(1 for r in founders if len(r.wealth_trajectory) > t)
        result.append(float(alive) / float(initial_population_size))
    return tuple(result)


def _flow_matrix_as_nested_tuple(genome: SelectorGenome) -> tuple[tuple[float, ...], ...]:
    raw = np.asarray(genome.flow_matrix, dtype=float)
    fm = validate_flow_matrix_invariants(
        raw,
        expected_shape=(int(raw.shape[0]), int(raw.shape[1])) if raw.ndim == 2 else None,
        shape_error_message="invalid evolution demo flow_matrix shape",
        finite_error_message="invalid evolution demo flow_matrix values",
        negative_error_message="invalid evolution demo flow_matrix values",
    )
    return tuple(tuple(float(v) for v in row) for row in fm)


def _build_dataset(
    report: PopulationFitnessReport,
    genomes: Sequence[SelectorGenome],
    runtime_horizon: int,
) -> EvolutionDemoDataset:
    wealth_trajectories = tuple(r.wealth_trajectory for r in report.per_process)
    survival_curve = _survival_curve(
        report.per_process, runtime_horizon, initial_population_size=len(genomes)
    )
    flow_matrix_snapshot = _flow_matrix_as_nested_tuple(genomes[0])
    return EvolutionDemoDataset(
        wealth_trajectories=wealth_trajectories,
        survival_curve=survival_curve,
        flow_matrix_snapshot=flow_matrix_snapshot,
        mean_fitness=float(report.mean_fitness),
        survival_rate=float(report.survival_rate),
    )


def _save_visualizations(
    dataset: EvolutionDemoDataset,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Wealth trajectories
    fig, ax = plt.subplots()
    for i, traj in enumerate(dataset.wealth_trajectories):
        ax.plot(list(traj), label=f"process {i}")
    ax.set_xlabel("step")
    ax.set_ylabel("wealth")
    ax.set_title("Wealth Trajectories")
    ax.legend(loc="upper left", fontsize="small")
    fig.savefig(output_dir / "wealth_trajectories.png", dpi=80)
    plt.close(fig)

    # 2. Population survival curve
    fig, ax = plt.subplots()
    ax.plot(list(dataset.survival_curve))
    ax.set_xlabel("step")
    ax.set_ylabel("survival fraction")
    ax.set_title("Population Survival Curve")
    ax.set_ylim(0.0, 1.05)
    fig.savefig(output_dir / "survival_curve.png", dpi=80)
    plt.close(fig)

    # 3. F[n,m] heatmap
    fm = np.array([[v for v in row] for row in dataset.flow_matrix_snapshot])
    fig, ax = plt.subplots()
    im = ax.imshow(fm, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("output channel m")
    ax.set_ylabel("input channel n")
    ax.set_title("F[n,m] Flow Matrix Heatmap")
    fig.savefig(output_dir / "flow_matrix_heatmap.png", dpi=80)
    plt.close(fig)


def run_evolution_demo(
    genomes: Sequence[SelectorGenome],
    world_parameters: dict,
    seed: int,
    runtime_horizon: int = 8,
    dt: float = 1.0,
    backend: str = "cpu",
    output_dir: str | os.PathLike | None = None,
) -> tuple[EvolutionDemoDataset, str | None]:
    """Run one generation loop and return a reproducible dataset.

    Args:
        genomes: Initial selector genome population.
        world_parameters: World configuration passed to the generation loop.
        seed: Deterministic seed (required; the generation loop enforces this).
        runtime_horizon: Simulation step count.
        dt: Time-step size.
        backend: ``"cpu"`` or ``"cuda"``.
        output_dir: If given, wealth-trajectory, survival-curve, and F[n,m]
            heatmap plots are saved there.  The directory is created if needed.

    Returns:
        ``(dataset, artifact_dir)`` where *artifact_dir* is the resolved string
        path passed as *output_dir* (or ``None`` when no output_dir was given).
    """
    report = run_generation_loop(
        list(genomes),
        seed=seed,
        runtime_horizon=runtime_horizon,
        world_parameters=world_parameters,
        dt=dt,
        backend=backend,
    )
    dataset = _build_dataset(report, genomes, runtime_horizon)

    artifact_dir: str | None = None
    if output_dir is not None:
        out_path = Path(output_dir)
        _save_visualizations(dataset, out_path)
        artifact_dir = str(out_path.resolve())

    return dataset, artifact_dir
