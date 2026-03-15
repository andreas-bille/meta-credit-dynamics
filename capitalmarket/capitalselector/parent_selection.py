from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .evolution_contracts import EDGE_EPSILON


ParentIndices = tuple[int, ...]
Population = Sequence[Any]
FitnessVector = Sequence[float] | np.ndarray

PARENT_SELECTION_FITNESS_PROPORTIONAL = "fitness_proportional"
PARENT_SELECTION_TOURNAMENT = "tournament"


class ParentSelectionPolicy(Protocol):
    def select_parents(
        self,
        population: Population,
        fitness: FitnessVector,
        n_parents: int,
        rng: np.random.Generator,
    ) -> ParentIndices:
        """Select parent indices using deterministic RNG substreams."""


def _normalize_selection_inputs(
    *,
    population: Population,
    fitness: FitnessVector,
    n_parents: int,
) -> tuple[int, np.ndarray, int]:
    population_size = int(len(population))
    if population_size <= 0:
        raise ValueError("parent selection requires non-empty population")

    fitness_arr = np.asarray(fitness, dtype=np.float64)
    if fitness_arr.ndim != 1 or int(fitness_arr.shape[0]) != population_size:
        raise ValueError("parent selection fitness shape mismatch")
    if np.any(~np.isfinite(fitness_arr)):
        raise ValueError("parent selection fitness must be finite")

    try:
        requested = int(n_parents)
    except (TypeError, ValueError) as exc:
        raise ValueError("parent selection requires integer n_parents") from exc

    effective = max(1, min(requested, population_size))
    return population_size, fitness_arr, effective


def _validate_parent_indices(
    indices: Sequence[int],
    *,
    population_size: int,
    expected_count: int,
) -> ParentIndices:
    normalized = tuple(int(idx) for idx in indices)
    if len(normalized) != int(expected_count):
        raise ValueError("parent selection produced invalid parent count")
    if len(normalized) == 0:
        raise ValueError("parent selection produced empty index set")
    if len(set(normalized)) != len(normalized):
        raise ValueError("parent selection produced duplicate indices")

    for idx in normalized:
        if idx < 0 or idx >= int(population_size):
            raise ValueError("parent selection produced out-of-bounds index")
    return normalized


@dataclass(frozen=True)
class FitnessProportionalSelectionPolicy:
    epsilon: float = EDGE_EPSILON

    def select_parents(
        self,
        population: Population,
        fitness: FitnessVector,
        n_parents: int,
        rng: np.random.Generator,
    ) -> ParentIndices:
        population_size, fitness_arr, effective_count = _normalize_selection_inputs(
            population=population,
            fitness=fitness,
            n_parents=n_parents,
        )

        selected: list[int] = []
        available = list(range(population_size))
        epsilon = float(self.epsilon)
        for _ in range(effective_count):
            available_fitness = np.maximum(fitness_arr[np.asarray(available, dtype=np.int64)], 0.0)
            fitness_sum = float(np.sum(available_fitness))
            if fitness_sum > epsilon:
                probabilities = available_fitness / fitness_sum
                pick_pos = int(rng.choice(len(available), p=probabilities))
            else:
                pick_pos = int(rng.integers(low=0, high=len(available)))
            selected.append(int(available.pop(pick_pos)))

        return _validate_parent_indices(
            selected,
            population_size=population_size,
            expected_count=effective_count,
        )


@dataclass(frozen=True)
class TournamentSelectionPolicy:
    tournament_size: int = 3

    def select_parents(
        self,
        population: Population,
        fitness: FitnessVector,
        n_parents: int,
        rng: np.random.Generator,
    ) -> ParentIndices:
        population_size, fitness_arr, effective_count = _normalize_selection_inputs(
            population=population,
            fitness=fitness,
            n_parents=n_parents,
        )

        bracket_size = max(1, min(int(self.tournament_size), population_size))

        selected: list[int] = []
        available = list(range(population_size))
        for _ in range(effective_count):
            if bracket_size >= len(available):
                bracket = list(available)
            else:
                bracket = [
                    int(idx)
                    for idx in np.asarray(
                        rng.choice(np.asarray(available, dtype=np.int64), size=bracket_size, replace=False),
                        dtype=np.int64,
                    ).tolist()
                ]

            best_fitness = float(np.max(fitness_arr[np.asarray(bracket, dtype=np.int64)]))
            winners = [idx for idx in bracket if float(fitness_arr[int(idx)]) == best_fitness]
            winner = int(min(winners))
            selected.append(winner)
            available.remove(winner)

        return _validate_parent_indices(
            selected,
            population_size=population_size,
            expected_count=effective_count,
        )


def build_parent_selection_policy(policy: str, *, tournament_size: int = 3) -> ParentSelectionPolicy:
    normalized = str(policy).strip().lower()
    if normalized == PARENT_SELECTION_FITNESS_PROPORTIONAL:
        return FitnessProportionalSelectionPolicy()
    if normalized == PARENT_SELECTION_TOURNAMENT:
        return TournamentSelectionPolicy(tournament_size=int(tournament_size))
    allowed = ", ".join(
        sorted(
            (
                PARENT_SELECTION_FITNESS_PROPORTIONAL,
                PARENT_SELECTION_TOURNAMENT,
            )
        )
    )
    raise ValueError(f"unknown parent selection policy '{policy}', expected one of: {allowed}")
