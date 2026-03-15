from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# Global thresholds/constants for v0.9.2 architecture contracts.
EDGE_EPSILON = 1e-12
ZERO_FLOW_EPSILON = 1e-12
DOMINANCE_THRESHOLD = 0.9
SPARSE_DENSITY_THRESHOLD = 0.5
ENTROPY_HIST_BINS = 20
ENTROPY_LOG_EPSILON = 1e-12
PARITY_RTOL = 1e-6
PARITY_ATOL = 1e-8

PAIRWISE_DISTANCE_MODE_EXACT = "exact"
PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED = "fallback_sampled"
PAIRWISE_DISTANCE_MODE_AUTO = "auto"
PAIRWISE_DISTANCE_MAX_EXACT_POPULATION = 64
PAIRWISE_DISTANCE_DEFAULT_SAMPLED_PAIR_COUNT = 10_000
PAIRWISE_DISTANCE_DEFAULT_SAMPLE_SEED = 0


@dataclass(frozen=True)
class DeterminismTolerances:
    rtol: float = PARITY_RTOL
    atol: float = PARITY_ATOL


DEFAULT_DETERMINISM_TOLERANCES = DeterminismTolerances()


def _as_float_array(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def validate_array_shape(
    values: np.ndarray | Sequence[float],
    *,
    expected_shape: tuple[int, ...],
    shape_error_message: str,
) -> np.ndarray:
    arr = _as_float_array(values)
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(shape_error_message)
    return arr


def validate_finite_non_negative_array(
    values: np.ndarray | Sequence[float],
    *,
    finite_error_message: str,
    negative_error_message: str,
) -> np.ndarray:
    arr = _as_float_array(values)
    if np.any(~np.isfinite(arr)):
        raise ValueError(finite_error_message)
    if np.any(arr < 0.0):
        raise ValueError(negative_error_message)
    return arr


def validate_finite_non_negative_scalar(
    value: float,
    *,
    finite_error_message: str,
    negative_error_message: str,
) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(finite_error_message)
    if out < 0.0:
        raise ValueError(negative_error_message)
    return out


def validate_flow_matrix_invariants(
    flow_matrix: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None = None,
    shape_error_message: str,
    finite_error_message: str,
    negative_error_message: str,
    require_row_stochastic: bool = False,
    row_stochastic_error_message: str = "flow_matrix row sums must be 1.0",
    row_sum_atol: float = PARITY_ATOL,
) -> np.ndarray:
    flow = _as_float_array(flow_matrix)
    if flow.ndim != 2:
        raise ValueError(shape_error_message)

    if expected_shape is not None:
        flow = validate_array_shape(
            flow,
            expected_shape=expected_shape,
            shape_error_message=shape_error_message,
        )

    flow = validate_finite_non_negative_array(
        flow,
        finite_error_message=finite_error_message,
        negative_error_message=negative_error_message,
    )

    if require_row_stochastic:
        row_sums = np.sum(flow, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=float(row_sum_atol)):
            raise ValueError(row_stochastic_error_message)

    return flow


def structural_distance_l1(flow_a: np.ndarray, flow_b: np.ndarray) -> float:
    left = _as_float_array(flow_a)
    right = _as_float_array(flow_b)
    if left.shape != right.shape:
        raise ValueError("structural distance requires equal matrix shape")
    return float(np.sum(np.abs(left - right)))


def _pair_from_linear_index(pair_index: int, population_size: int) -> tuple[int, int]:
    if pair_index < 0:
        raise ValueError("pair index must be >= 0")

    remaining = int(pair_index)
    for i in range(int(population_size) - 1):
        row_pairs = int(population_size) - i - 1
        if remaining < row_pairs:
            return i, i + 1 + remaining
        remaining -= row_pairs
    raise ValueError("pair index out of range")


def pairwise_structural_distances_l1(
    flow_matrices: Sequence[np.ndarray],
    *,
    mode: str = PAIRWISE_DISTANCE_MODE_EXACT,
    max_exact_population: int = PAIRWISE_DISTANCE_MAX_EXACT_POPULATION,
    sampled_pair_count: int = PAIRWISE_DISTANCE_DEFAULT_SAMPLED_PAIR_COUNT,
    sample_seed: int = PAIRWISE_DISTANCE_DEFAULT_SAMPLE_SEED,
) -> np.ndarray:
    if len(flow_matrices) <= 1:
        return np.zeros(0, dtype=np.float64)

    matrices = [_as_float_array(matrix) for matrix in flow_matrices]
    population_size = int(len(matrices))

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {
        PAIRWISE_DISTANCE_MODE_EXACT,
        PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED,
        PAIRWISE_DISTANCE_MODE_AUTO,
    }:
        raise ValueError("invalid pairwise distance mode")

    if int(max_exact_population) < 2:
        raise ValueError("max_exact_population must be >= 2")
    if int(sampled_pair_count) <= 0:
        raise ValueError("sampled_pair_count must be > 0")

    reference_shape = matrices[0].shape
    for matrix in matrices:
        if matrix.shape != reference_shape:
            raise ValueError("pairwise distances require equal matrix shape")

    effective_mode = mode_norm
    if mode_norm == PAIRWISE_DISTANCE_MODE_AUTO:
        if population_size <= int(max_exact_population):
            effective_mode = PAIRWISE_DISTANCE_MODE_EXACT
        else:
            effective_mode = PAIRWISE_DISTANCE_MODE_FALLBACK_SAMPLED

    if effective_mode == PAIRWISE_DISTANCE_MODE_EXACT:
        distances: list[float] = []
        for i in range(population_size):
            for j in range(i + 1, population_size):
                distances.append(structural_distance_l1(matrices[i], matrices[j]))
        return np.asarray(distances, dtype=np.float64)

    if sample_seed is None:
        raise ValueError("sample_seed must be provided for fallback_sampled mode")

    total_pairs = population_size * (population_size - 1) // 2
    sample_size = min(int(sampled_pair_count), int(total_pairs))
    if sample_size == int(total_pairs):
        # Fallback degenerates to exact mode when all pairs are sampled.
        distances: list[float] = []
        for i in range(population_size):
            for j in range(i + 1, population_size):
                distances.append(structural_distance_l1(matrices[i], matrices[j]))
        return np.asarray(distances, dtype=np.float64)

    rng = np.random.default_rng(np.random.PCG64(np.uint64(int(sample_seed))))
    sampled_pair_indices = np.asarray(
        rng.choice(int(total_pairs), size=int(sample_size), replace=False),
        dtype=np.int64,
    )
    sampled_pair_indices.sort()

    distances: list[float] = []
    for pair_idx in sampled_pair_indices.tolist():
        i, j = _pair_from_linear_index(int(pair_idx), population_size)
        distances.append(structural_distance_l1(matrices[i], matrices[j]))
    return np.asarray(distances, dtype=np.float64)


def distance_histogram_probabilities(
    distances: Iterable[float],
    *,
    bins: int = ENTROPY_HIST_BINS,
) -> np.ndarray:
    values = np.asarray(list(distances), dtype=np.float64)
    if bins <= 0:
        raise ValueError("bins must be > 0")

    if values.size == 0:
        return np.zeros(int(bins), dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("distance distribution must be finite")

    d_min = float(np.min(values))
    d_max = float(np.max(values))
    if np.isclose(d_min, d_max, rtol=0.0, atol=0.0):
        probs = np.zeros(int(bins), dtype=np.float64)
        probs[0] = 1.0
        return probs

    hist, _ = np.histogram(values, bins=int(bins), range=(d_min, d_max), density=False)
    total = float(np.sum(hist))
    if total <= 0.0:
        return np.zeros(int(bins), dtype=np.float64)
    return np.asarray(hist, dtype=np.float64) / total


def shannon_entropy(probabilities: np.ndarray, *, log_epsilon: float = ENTROPY_LOG_EPSILON) -> float:
    probs = _as_float_array(probabilities).reshape(-1)
    if probs.size == 0:
        return 0.0
    if np.any(~np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")

    prob_sum = float(np.sum(probs))
    if prob_sum <= 0.0:
        return 0.0

    normalized = probs / prob_sum
    return float(-np.sum(normalized * np.log(normalized + float(log_epsilon))))


def normalized_shannon_entropy(probabilities: np.ndarray, *, log_epsilon: float = ENTROPY_LOG_EPSILON) -> float:
    probs = _as_float_array(probabilities).reshape(-1)
    if probs.size <= 1:
        return 0.0

    entropy = shannon_entropy(probs, log_epsilon=log_epsilon)
    denom = float(np.log(float(probs.size)))
    if denom <= 0.0:
        return 0.0
    return float(entropy / denom)


def structural_entropy_from_distances(
    distances: Iterable[float],
    *,
    bins: int = ENTROPY_HIST_BINS,
    log_epsilon: float = ENTROPY_LOG_EPSILON,
) -> tuple[float, float, np.ndarray]:
    probs = distance_histogram_probabilities(distances, bins=bins)
    entropy = shannon_entropy(probs, log_epsilon=log_epsilon)
    normalized = normalized_shannon_entropy(probs, log_epsilon=log_epsilon)
    return entropy, normalized, probs


def parity_allclose(
    left: np.ndarray | Sequence[float],
    right: np.ndarray | Sequence[float],
    *,
    tolerances: DeterminismTolerances = DEFAULT_DETERMINISM_TOLERANCES,
) -> bool:
    arr_left = _as_float_array(left)
    arr_right = _as_float_array(right)
    if arr_left.shape != arr_right.shape:
        raise ValueError("parity comparison requires equal shape")
    return bool(np.allclose(arr_left, arr_right, rtol=float(tolerances.rtol), atol=float(tolerances.atol)))


def classify_flow_structure(
    flow_matrix: np.ndarray,
    *,
    zero_epsilon: float = ZERO_FLOW_EPSILON,
    dominance_threshold: float = DOMINANCE_THRESHOLD,
    sparse_density_threshold: float = SPARSE_DENSITY_THRESHOLD,
) -> str:
    flow = _as_float_array(flow_matrix)
    if flow.ndim != 2:
        raise ValueError("flow structure classification requires 2D matrix")
    if np.any(~np.isfinite(flow)):
        raise ValueError("flow structure classification requires finite matrix")
    if np.any(flow < 0.0):
        raise ValueError("flow structure classification requires non-negative matrix")

    total = float(np.sum(flow))
    if total <= float(zero_epsilon):
        return "ZERO_FLOW"

    max_share = float(np.max(flow) / total)
    if max_share >= float(dominance_threshold):
        return "SINGLE_EDGE_DOMINATED"

    active = int(np.count_nonzero(flow > float(zero_epsilon)))
    density = float(active) / float(flow.size)
    if density <= float(sparse_density_threshold):
        return "SPARSE_COMPLEX"
    return "DENSE_COMPLEX"
