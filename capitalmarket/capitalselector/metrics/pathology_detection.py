from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from ..genome import SelectorGenome
from ..genome_serialization import genome_to_dict


class PathologyWarning(UserWarning):
    """Warning category for emergence/pathology detections."""


@dataclass(frozen=True)
class PathologyDetectionConfig:
    min_survival_fraction: float = 0.25
    max_mutation_scale: float = 2.0
    n_runaway_generations: int = 3
    row_sum_tolerance: float = 1e-12


@dataclass(frozen=True)
class PathologyDetectionState:
    initial_population_size: int | None = None
    mutation_runaway_streak: int = 0
    population_collapse_emitted: bool = False
    mutation_runaway_emitted: bool = False


def _validate_config(config: PathologyDetectionConfig) -> None:
    if not np.isfinite(float(config.min_survival_fraction)):
        raise ValueError("invalid pathology detection configuration")
    if float(config.min_survival_fraction) < 0.0:
        raise ValueError("invalid pathology detection configuration")

    if not np.isfinite(float(config.max_mutation_scale)):
        raise ValueError("invalid pathology detection configuration")
    if float(config.max_mutation_scale) < 0.0:
        raise ValueError("invalid pathology detection configuration")

    if int(config.n_runaway_generations) < 0:
        raise ValueError("invalid pathology detection configuration")

    if not np.isfinite(float(config.row_sum_tolerance)):
        raise ValueError("invalid pathology detection configuration")
    if float(config.row_sum_tolerance) < 0.0:
        raise ValueError("invalid pathology detection configuration")


def _population_payload(population: Sequence[SelectorGenome | Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for entry in population:
        if isinstance(entry, SelectorGenome):
            payloads.append(dict(genome_to_dict(entry)))
        elif isinstance(entry, Mapping):
            payloads.append(dict(entry))
        else:
            raise ValueError("invalid pathology detection population")
    return tuple(payloads)


def detect_pathologies_for_generation(
    *,
    generation_index: int,
    population: Sequence[SelectorGenome | Mapping[str, Any]],
    surviving_population_size: int,
    mutation_scale_factor: float,
    config: PathologyDetectionConfig,
    state: PathologyDetectionState | None = None,
) -> tuple[PathologyDetectionState, tuple[str, ...]]:
    _validate_config(config)

    payloads = _population_payload(population)
    total_population = int(len(payloads))
    if total_population <= 0:
        raise ValueError("invalid pathology detection population")

    survivors = int(surviving_population_size)
    if survivors < 0 or survivors > total_population:
        raise ValueError("invalid pathology detection population")

    scale = float(mutation_scale_factor)
    if not np.isfinite(scale):
        raise ValueError("invalid pathology detection mutation scale factor")

    previous = state if state is not None else PathologyDetectionState(initial_population_size=total_population)
    initial_population = int(previous.initial_population_size or total_population)
    if initial_population <= 0:
        raise ValueError("invalid pathology detection population")

    population_collapse_emitted = bool(previous.population_collapse_emitted)
    mutation_runaway_emitted = bool(previous.mutation_runaway_emitted)
    mutation_runaway_streak = int(previous.mutation_runaway_streak)

    messages: list[str] = []

    collapse_threshold = float(config.min_survival_fraction) * float(initial_population)
    if float(survivors) < collapse_threshold and not population_collapse_emitted:
        message = (
            "population collapse detected: "
            f"generation={int(generation_index)}, survivors={survivors}, "
            f"threshold={collapse_threshold:.6f}"
        )
        warnings.warn(message, PathologyWarning, stacklevel=2)
        messages.append(message)
        population_collapse_emitted = True

    if scale > float(config.max_mutation_scale):
        mutation_runaway_streak = int(mutation_runaway_streak) + 1
    else:
        mutation_runaway_streak = 0

    if (
        int(mutation_runaway_streak) > int(config.n_runaway_generations)
        and not mutation_runaway_emitted
    ):
        message = (
            "mutation runaway detected: "
            f"generation={int(generation_index)}, mutation_scale_factor={scale:.6f}, "
            f"max_mutation_scale={float(config.max_mutation_scale):.6f}, "
            f"consecutive_generations={int(mutation_runaway_streak)}"
        )
        warnings.warn(message, PathologyWarning, stacklevel=2)
        messages.append(message)
        mutation_runaway_emitted = True

    row_tolerance = float(config.row_sum_tolerance)
    degenerate_found = False
    for payload in payloads:
        if "flow_matrix" not in payload:
            raise ValueError("invalid pathology detection population")
        matrix = np.asarray(payload["flow_matrix"], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
            raise ValueError("invalid pathology detection population")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("invalid pathology detection population")
        row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64)
        if np.any(~np.isfinite(row_sums)):
            raise ValueError("invalid pathology detection population")
        if bool(np.any(row_sums < row_tolerance)):
            degenerate_found = True
            break

    if degenerate_found:
        message = (
            "degenerate flow matrix detected: "
            f"generation={int(generation_index)}, row_sum_tolerance={row_tolerance:.6f}"
        )
        warnings.warn(message, PathologyWarning, stacklevel=2)
        messages.append(message)

    next_state = PathologyDetectionState(
        initial_population_size=int(initial_population),
        mutation_runaway_streak=int(mutation_runaway_streak),
        population_collapse_emitted=bool(population_collapse_emitted),
        mutation_runaway_emitted=bool(mutation_runaway_emitted),
    )
    return next_state, tuple(messages)
