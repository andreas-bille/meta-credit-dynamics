from __future__ import annotations

from typing import Any

import numpy as np

from .flow_contracts import FlowDimensions
from .flow_contracts import validate_flow_dimensions
from .genome import SelectorGenome
from .selector_policy import validate_selector_policy


ALLOWED_SETTLEMENT_PARAM_KEYS = frozenset(
    {
        "accept_by_default",
        "future_maturity_offset",
        "lambda_cash_share",
    }
)


def _validate_settlement_params(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("invalid genome: settlement_params must be dict[str,float]")

    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("invalid genome: invalid settlement param key")
        if key not in ALLOWED_SETTLEMENT_PARAM_KEYS:
            raise ValueError("invalid genome: unknown settlement param key")

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid genome: settlement_params must be dict[str,float]") from exc
        if not np.isfinite(numeric):
            raise ValueError("invalid genome: non-finite settlement param value")
        if numeric < 0.0:
            raise ValueError("invalid genome: negative settlement param value")


def _validate_d2_projection_guardrail(*, flow_matrix: np.ndarray, output_weights: np.ndarray) -> None:
    projected = np.asarray(flow_matrix @ output_weights, dtype=float)
    if projected.ndim != 1:
        raise ValueError("invalid action shape")
    if projected.shape[0] != int(flow_matrix.shape[0]):
        raise ValueError("invalid action dimensions")
    if np.any(~np.isfinite(projected)) or np.any(projected < 0.0):
        raise ValueError("invalid action values")
    # Preserve legacy adapter contract semantics: non-short normalized exposure
    # requires a strictly positive projection sum.
    if float(np.sum(projected)) <= 0.0:
        raise ValueError("invalid action values")


def validate_selector_genome(genome: SelectorGenome) -> None:
    flow_matrix = np.asarray(genome.flow_matrix, dtype=float)
    output_weights = np.asarray(genome.output_weights, dtype=float)

    if flow_matrix.ndim != 2:
        raise ValueError("invalid genome: flow_matrix must be 2D")
    if output_weights.ndim != 1:
        raise ValueError("invalid genome: output_weights must be 1D")
    if flow_matrix.shape[1] != output_weights.shape[0]:
        raise ValueError("invalid genome: output dimension mismatch")
    if flow_matrix.shape[0] <= 0 or flow_matrix.shape[1] <= 0:
        raise ValueError("invalid genome: dimensions must be > 0")

    if np.any(~np.isfinite(flow_matrix)):
        raise ValueError("invalid genome: non-finite flow_matrix")
    if np.any(~np.isfinite(output_weights)):
        raise ValueError("invalid genome: non-finite output_weights")

    lambda_risk = float(genome.lambda_risk)
    if not np.isfinite(lambda_risk):
        raise ValueError("invalid genome: non-finite lambda_risk")

    if np.any(flow_matrix < 0.0):
        raise ValueError("invalid genome: negative flow_matrix")
    if np.any(output_weights < 0.0):
        raise ValueError("invalid genome: negative output_weights")
    if lambda_risk < 0.0:
        raise ValueError("invalid genome: negative lambda_risk")

    _validate_settlement_params(genome.settlement_params)

    try:
        validate_selector_policy(str(genome.selector_policy))
        _validate_d2_projection_guardrail(flow_matrix=flow_matrix, output_weights=output_weights)

        dims = FlowDimensions(
            n_inputs=int(flow_matrix.shape[0]),
            m_outputs=int(flow_matrix.shape[1]),
        )
        validate_flow_dimensions(dims)
    except Exception as exc:
        raise ValueError("invalid genome: not action-contract compatible") from exc
