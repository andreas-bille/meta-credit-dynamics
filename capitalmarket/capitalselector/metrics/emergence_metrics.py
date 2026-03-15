from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np

from ..genome import SelectorGenome
from ..genome_serialization import genome_to_dict


@dataclass(frozen=True)
class EmergenceMetricsConfig:
    channel_utilization_threshold: float = 0.25


@dataclass(frozen=True)
class EmergenceMetricsPoint:
    generation_index: int
    fitness_variance: float
    structural_entropy: float
    channel_utilization_sparsity: float
    strategy_lifetime_distribution: tuple[int, ...]


def _population_payload(population: Sequence[SelectorGenome | Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for entry in population:
        if isinstance(entry, SelectorGenome):
            payloads.append(dict(genome_to_dict(entry)))
        elif isinstance(entry, Mapping):
            payloads.append(dict(entry))
        else:
            raise ValueError("invalid emergence metrics payload")
    return tuple(payloads)


def _require_finite_1d(values: Sequence[float], *, error_message: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] <= 0:
        raise ValueError(error_message)
    if np.any(~np.isfinite(arr)):
        raise ValueError(error_message)
    return arr


def strategy_signature(strategy: SelectorGenome | Mapping[str, Any]) -> str:
    payloads = _population_payload([strategy])
    canonical = json.dumps(payloads[0], sort_keys=True, separators=(",", ":"))
    return str(canonical)


def update_strategy_first_seen(
    first_seen_generation_by_strategy: Mapping[str, int],
    strategy_ids: Sequence[str],
    *,
    generation_index: int,
) -> dict[str, int]:
    out = {str(key): int(value) for key, value in first_seen_generation_by_strategy.items()}
    for strategy_id in strategy_ids:
        sid = str(strategy_id)
        if sid not in out:
            out[sid] = int(generation_index)
    return out


def fitness_variance(fitness_scores: Sequence[float]) -> float:
    values = _require_finite_1d(fitness_scores, error_message="invalid emergence metrics fitness scores")
    variance = float(np.var(values))
    if not np.isfinite(variance):
        raise ValueError("invalid emergence metrics fitness scores")
    return variance


def strategy_lifetime_distribution(
    strategy_ids: Sequence[str],
    *,
    first_seen_generation_by_strategy: Mapping[str, int],
    generation_index: int,
) -> tuple[int, ...]:
    normalized_ids = tuple(str(value) for value in strategy_ids)
    seen = {str(key): int(value) for key, value in first_seen_generation_by_strategy.items()}

    out: list[int] = []
    for strategy_id in sorted(normalized_ids):
        if strategy_id not in seen:
            raise ValueError("invalid emergence strategy lifetime state")
        lifetime = int(generation_index) - int(seen[strategy_id]) + 1
        if lifetime <= 0:
            raise ValueError("invalid emergence strategy lifetime state")
        out.append(int(lifetime))
    return tuple(out)


def structural_entropy(population: Sequence[SelectorGenome | Mapping[str, Any]]) -> float:
    payloads = _population_payload(population)
    if len(payloads) <= 0:
        raise ValueError("invalid emergence metrics population")

    normalized_vectors: list[np.ndarray] = []
    for payload in payloads:
        if "flow_matrix" not in payload:
            raise ValueError("invalid emergence metrics population")
        matrix = np.asarray(payload["flow_matrix"], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
            raise ValueError("invalid emergence metrics population")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("invalid emergence metrics population")

        total = float(matrix.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("invalid emergence metrics population")

        normalized = np.asarray(matrix.reshape(-1) / total, dtype=np.float64)
        normalized_vectors.append(normalized)

    first_shape = normalized_vectors[0].shape
    if any(vector.shape != first_shape for vector in normalized_vectors):
        raise ValueError("invalid emergence metrics population")

    mean_distribution = np.mean(np.stack(normalized_vectors, axis=0), axis=0)
    if np.any(~np.isfinite(mean_distribution)):
        raise ValueError("invalid emergence metrics population")

    mass = float(mean_distribution.sum())
    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError("invalid emergence metrics population")

    probabilities = mean_distribution / mass
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    if not np.isfinite(entropy):
        raise ValueError("invalid emergence metrics population")
    return entropy


def channel_utilization_sparsity(
    population: Sequence[SelectorGenome | Mapping[str, Any]],
    *,
    threshold: float,
) -> float:
    payloads = _population_payload(population)
    if len(payloads) <= 0:
        raise ValueError("invalid emergence metrics population")

    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError("invalid channel utilization threshold")

    utilization_accumulator: np.ndarray | None = None
    for payload in payloads:
        if "output_weights" not in payload:
            raise ValueError("invalid emergence metrics population")
        output_weights = _require_finite_1d(payload["output_weights"], error_message="invalid emergence metrics population")
        if np.any(output_weights < 0.0):
            raise ValueError("invalid emergence metrics population")
        if utilization_accumulator is None:
            utilization_accumulator = np.zeros_like(output_weights)
        if utilization_accumulator.shape != output_weights.shape:
            raise ValueError("invalid emergence metrics population")
        utilization_accumulator = utilization_accumulator + output_weights

    assert utilization_accumulator is not None
    mean_utilization = np.asarray(utilization_accumulator / float(len(payloads)), dtype=np.float64)
    if np.any(~np.isfinite(mean_utilization)):
        raise ValueError("invalid emergence metrics population")

    sparse_flags = mean_utilization < threshold_value
    fraction_sparse = float(np.mean(sparse_flags.astype(np.float64)))
    if not np.isfinite(fraction_sparse):
        raise ValueError("invalid emergence metrics population")
    return fraction_sparse
