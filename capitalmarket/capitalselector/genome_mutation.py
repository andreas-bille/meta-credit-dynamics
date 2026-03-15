from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .evolution_contracts import EDGE_EPSILON
from .evolution_contracts import structural_distance_l1
from .evolution_contracts import validate_finite_non_negative_array
from .evolution_contracts import validate_finite_non_negative_scalar
from .evolution_contracts import validate_flow_matrix_invariants
from .genome import SelectorGenome
from .genome_serialization import genome_from_dict, genome_to_dict
from .genome_validation import validate_selector_genome
from .inhabitants import InhabitantEntry


@dataclass(frozen=True)
class MutationConfig:
    noise_scale: float = 0.01
    redistribution_share: float = 0.05
    lambda_risk_scale: float = 0.05


@dataclass(frozen=True)
class MutationDiagnostics:
    mutation_magnitude: float
    modified_edge_count: int
    structural_distance: float


@dataclass(frozen=True)
class MutationResult:
    genome: SelectorGenome
    diagnostics: MutationDiagnostics


def compute_mutation_diagnostics(
    *,
    parent_flow_matrix: np.ndarray,
    child_flow_matrix: np.ndarray,
    edge_epsilon: float = EDGE_EPSILON,
) -> MutationDiagnostics:
    parent = np.asarray(parent_flow_matrix, dtype=np.float64)
    child = np.asarray(child_flow_matrix, dtype=np.float64)
    if parent.shape != child.shape:
        raise ValueError("mutation diagnostics require equal matrix shape")

    validate_finite_non_negative_scalar(
        edge_epsilon,
        finite_error_message="edge epsilon must be finite",
        negative_error_message="edge epsilon must be non-negative",
    )

    structural_distance = structural_distance_l1(parent, child)
    delta = np.abs(child - parent)
    modified_edge_count = int(np.count_nonzero(delta > float(edge_epsilon)))
    return MutationDiagnostics(
        mutation_magnitude=float(structural_distance),
        modified_edge_count=modified_edge_count,
        structural_distance=float(structural_distance),
    )


def _splitmix64(value: int) -> int:
    x = int(value) & 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return int(z & 0xFFFFFFFFFFFFFFFF)


def _as_contract_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-deterministic entropy source in rebirth") from exc


def derive_rebirth_subseeds(
    *,
    deterministic_seed: int | None,
    tau: int,
    dead_process_id: int,
    dead_generation_id: int,
) -> tuple[int, int]:
    if deterministic_seed is None:
        raise ValueError("parent selection requires deterministic seed")

    seed = _as_contract_int(deterministic_seed)
    tau_i = _as_contract_int(tau)
    process_i = _as_contract_int(dead_process_id)
    generation_i = _as_contract_int(dead_generation_id)

    packed = (
        (seed & 0xFFFFFFFFFFFFFFFF)
        ^ ((tau_i & 0xFFFFFFFF) << 32)
        ^ ((process_i & 0xFFFF) << 16)
        ^ (generation_i & 0xFFFF)
    )

    selection_seed = _splitmix64(packed ^ 0xA5A5A5A5A5A5A5A5)
    mutation_seed = _splitmix64(packed ^ 0x5A5A5A5A5A5A5A5A)
    return selection_seed, mutation_seed


def _rng_from_seed(seed: int | None, *, missing_error: str):
    if seed is None:
        raise ValueError(missing_error)
    seed_i = _as_contract_int(seed)
    return np.random.default_rng(np.random.PCG64(np.uint64(seed_i)))


def select_parent_entry_deterministic(
    entries: Sequence[InhabitantEntry],
    *,
    selection_seed: int | None,
    epsilon: float = EDGE_EPSILON,
) -> InhabitantEntry:
    rng = _rng_from_seed(selection_seed, missing_error="parent selection requires deterministic seed")

    ordered = sorted(entries, key=lambda entry: (int(entry.process_id), int(entry.generation_id)))
    if not ordered:
        raise ValueError("cannot select parent from empty inhabitants book")

    positives = [entry for entry in ordered if max(float(entry.fitness), 0.0) > float(epsilon)]
    if positives:
        weights = np.asarray([max(float(entry.fitness), 0.0) for entry in positives], dtype=np.float64)
        weight_sum = float(weights.sum())
        if weight_sum > float(epsilon):
            target = float(rng.random())
            cumulative = 0.0
            for entry, weight in zip(positives, weights):
                cumulative += float(weight / weight_sum)
                # Keep earliest entry when boundary ties happen within epsilon.
                if target <= cumulative + float(epsilon):
                    return entry
            return positives[-1]

    index = int(rng.integers(low=0, high=len(ordered)))
    return ordered[index]


def parent_genome_from_entry(entry: InhabitantEntry) -> SelectorGenome:
    metadata = dict(getattr(entry, "metadata", {}) or {})
    payload = metadata.get("genome")
    if payload is None:
        raise ValueError("rebirth requires parent genome")

    try:
        genome = genome_from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("rebirth requires parent genome") from exc

    return genome


def selector_genome_from_selector(selector: Any) -> SelectorGenome:
    flow_matrix = getattr(selector, "flow_matrix", None)
    output_weights = getattr(selector, "output_weights", None)
    if flow_matrix is None or output_weights is None:
        raise ValueError("rebirth requires parent genome")

    settlement_config = dict(getattr(selector, "settlement_config", {}) or {})
    settlement_params: dict[str, float] = {}
    for key, raw in settlement_config.items():
        if isinstance(raw, bool):
            settlement_params[str(key)] = 1.0 if raw else 0.0
        else:
            settlement_params[str(key)] = float(raw)

    try:
        genome = SelectorGenome(
            flow_matrix=np.asarray(flow_matrix, dtype=np.float64),
            output_weights=np.asarray(output_weights, dtype=np.float64),
            lambda_risk=float(getattr(selector, "lambda_risk", 0.0)),
            selector_policy=str(getattr(selector, "selector_policy", "myopic")),
            settlement_params=settlement_params,
        )
        validate_selector_genome(genome)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("rebirth requires parent genome") from exc

    return genome


def validate_mutation_result(
    *,
    flow_matrix: np.ndarray,
    expected_flow_shape: tuple[int, int],
    output_weights: np.ndarray,
    expected_output_shape: tuple[int],
    lambda_risk: float,
) -> None:
    validate_flow_matrix_invariants(
        flow_matrix,
        expected_shape=expected_flow_shape,
        shape_error_message="mutation changed flow_matrix shape",
        finite_error_message="mutation produced non-finite flow_matrix",
        negative_error_message="mutation produced negative flow_matrix",
        require_row_stochastic=True,
        row_stochastic_error_message="mutation produced non-row-stochastic flow_matrix",
    )

    output = validate_finite_non_negative_array(
        output_weights,
        finite_error_message="mutation produced non-finite output_weights",
        negative_error_message="mutation produced negative output_weights",
    )
    if tuple(output.shape) != tuple(expected_output_shape):
        raise ValueError("mutation changed output_weights shape")

    validate_finite_non_negative_scalar(
        lambda_risk,
        finite_error_message="mutation produced non-finite lambda_risk",
        negative_error_message="mutation produced negative lambda_risk",
    )


def _mutate_flow_matrix(
    flow_matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    config: MutationConfig,
) -> np.ndarray:
    flow = np.asarray(flow_matrix, dtype=np.float64).copy()
    n_inputs, m_outputs = flow.shape

    if float(config.noise_scale) > 0.0:
        flow += rng.normal(loc=0.0, scale=float(config.noise_scale), size=flow.shape)

    share = float(config.redistribution_share)
    if share > 0.0 and n_inputs > 0 and m_outputs > 0:
        share = min(max(share, 0.0), 1.0)
        for row in range(n_inputs):
            row_values = flow[row, :]
            row_sum = float(row_values.sum())
            if row_sum <= 0.0:
                continue
            target = rng.dirichlet(np.ones(m_outputs, dtype=np.float64))
            flow[row, :] = (1.0 - share) * row_values + share * row_sum * target

        for col in range(m_outputs):
            col_values = flow[:, col]
            col_sum = float(col_values.sum())
            if col_sum <= 0.0:
                continue
            target = rng.dirichlet(np.ones(n_inputs, dtype=np.float64))
            flow[:, col] = (1.0 - share) * col_values + share * col_sum * target

    np.maximum(flow, 0.0, out=flow)
    # Project each row back to the probability simplex to enforce row-stochastic mutation output.
    for row in range(n_inputs):
        row_sum = float(np.sum(flow[row, :]))
        if row_sum <= EDGE_EPSILON:
            flow[row, :] = 1.0 / float(m_outputs)
        else:
            flow[row, :] = flow[row, :] / row_sum

    return flow


def _mutate_output_weights(
    output_weights: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    out = np.asarray(output_weights, dtype=np.float64).copy()
    if float(noise_scale) > 0.0:
        out += rng.normal(loc=0.0, scale=float(noise_scale), size=out.shape)
    return out


def mutate_selector_genome_result(
    parent_genome: SelectorGenome | None,
    *,
    mutation_seed: int | None,
    config: MutationConfig | None = None,
    execution_backend: str = "cpu",
    mutation_already_computed: bool = False,
) -> MutationResult:
    if parent_genome is None:
        raise ValueError("rebirth requires parent genome")

    backend = str(execution_backend).strip().lower()
    if backend != "cpu" or bool(mutation_already_computed):
        raise ValueError("rebirth mutation must be computed once in backend-neutral CPU path")

    rng = _rng_from_seed(mutation_seed, missing_error="mutation requires deterministic seed")
    cfg = config or MutationConfig()

    validate_selector_genome(parent_genome)

    parent_flow = np.asarray(parent_genome.flow_matrix, dtype=np.float64)
    parent_output = np.asarray(parent_genome.output_weights, dtype=np.float64)
    parent_lambda = float(parent_genome.lambda_risk)

    flow_mutation_enabled = float(cfg.noise_scale) > 0.0 or float(cfg.redistribution_share) > 0.0
    if flow_mutation_enabled:
        mutated_flow = _mutate_flow_matrix(parent_flow, rng=rng, config=cfg)
    else:
        mutated_flow = parent_flow.copy()

    mutated_output = _mutate_output_weights(parent_output, rng=rng, noise_scale=float(cfg.noise_scale))
    lambda_scale = float(cfg.lambda_risk_scale)
    if lambda_scale <= 0.0:
        lambda_delta = 0.0
    else:
        lambda_delta = float(rng.uniform(-lambda_scale, lambda_scale))
    mutated_lambda = parent_lambda + lambda_delta

    invariant_messages = {
        "mutation changed flow_matrix shape",
        "mutation produced non-finite flow_matrix",
        "mutation produced negative flow_matrix",
        "mutation produced non-row-stochastic flow_matrix",
        "mutation changed output_weights shape",
        "mutation produced non-finite output_weights",
        "mutation produced negative output_weights",
        "mutation produced non-finite lambda_risk",
        "mutation produced negative lambda_risk",
    }

    validate_mutation_result(
        flow_matrix=mutated_flow,
        expected_flow_shape=tuple(parent_flow.shape),
        output_weights=mutated_output,
        expected_output_shape=tuple(parent_output.shape),
        lambda_risk=mutated_lambda,
    )

    newborn = SelectorGenome(
        flow_matrix=mutated_flow,
        output_weights=mutated_output,
        lambda_risk=mutated_lambda,
        selector_policy=str(parent_genome.selector_policy),
        settlement_params=dict(parent_genome.settlement_params),
    )

    try:
        validate_selector_genome(newborn)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ValueError) and str(exc) in invariant_messages:
            raise
        raise ValueError("mutation produced invalid genome") from exc

    diagnostics = compute_mutation_diagnostics(
        parent_flow_matrix=parent_flow,
        child_flow_matrix=mutated_flow,
        edge_epsilon=EDGE_EPSILON,
    )
    return MutationResult(genome=newborn, diagnostics=diagnostics)


def mutate_selector_genome(
    parent_genome: SelectorGenome | None,
    *,
    mutation_seed: int | None,
    config: MutationConfig | None = None,
    execution_backend: str = "cpu",
    mutation_already_computed: bool = False,
) -> SelectorGenome:
    return mutate_selector_genome_result(
        parent_genome,
        mutation_seed=mutation_seed,
        config=config,
        execution_backend=execution_backend,
        mutation_already_computed=mutation_already_computed,
    ).genome


def serialize_genome_for_metadata(genome: SelectorGenome) -> dict[str, Any]:
    return genome_to_dict(genome)
