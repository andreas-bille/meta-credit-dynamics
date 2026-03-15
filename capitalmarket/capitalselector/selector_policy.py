from __future__ import annotations

from typing import Literal, cast
import numpy as np

from .interfaces import WorldAction, validate_and_normalize_world_action


SelectorPolicy = Literal["myopic", "term_aware", "term_risk"]

DEFAULT_SELECTOR_POLICY: SelectorPolicy = "myopic"

_ALLOWED_SELECTOR_POLICIES = {"myopic", "term_aware", "term_risk"}


def validate_selector_policy(policy: str) -> SelectorPolicy:
    normalized = str(policy).strip().lower()
    if normalized not in _ALLOWED_SELECTOR_POLICIES:
        allowed = ", ".join(sorted(_ALLOWED_SELECTOR_POLICIES))
        raise ValueError(f"unknown selector policy '{policy}', expected one of: {allowed}")
    return cast(SelectorPolicy, normalized)


def build_world_action(
    *,
    weights: np.ndarray | None = None,
    flow_matrix: np.ndarray | None = None,
    output_weights: np.ndarray | None = None,
    gross_exposure: float = 1.0,
    leverage_limit: float = 1.0,
    allow_short: bool = False,
    expected_channels: int | None = None,
) -> WorldAction:
    """Build a validated and normalized WorldAction from vector or matrix policy outputs.

    Canonical matrix mode uses a non-negative flow transform F[n,m] and optional
    output allocation over m outputs. Legacy callers can continue passing only
    `weights`.
    """
    if weights is None:
        if flow_matrix is None:
            raise ValueError("either weights or flow_matrix must be provided")
        fm = np.asarray(flow_matrix, dtype=float)
        if fm.ndim != 2:
            raise ValueError("flow_matrix must be a 2D array")
        if output_weights is None:
            out_w = np.ones(fm.shape[1], dtype=float)
        else:
            out_w = np.asarray(output_weights, dtype=float)
        weights = np.asarray(fm @ out_w, dtype=float)

    action = WorldAction(
        weights=np.asarray(weights, dtype=float),
        flow_matrix=None if flow_matrix is None else np.asarray(flow_matrix, dtype=float),
        output_weights=None if output_weights is None else np.asarray(output_weights, dtype=float),
        gross_exposure=float(gross_exposure),
        leverage_limit=float(leverage_limit),
        allow_short=bool(allow_short),
    )
    return validate_and_normalize_world_action(action, expected_channels=expected_channels)
