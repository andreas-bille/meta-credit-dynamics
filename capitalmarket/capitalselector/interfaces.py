from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any
import numpy as np


@dataclass
class WorldAction:
    """Policy action handed to the world in Phase II."""

    weights: np.ndarray
    flow_matrix: np.ndarray | None = None
    output_weights: np.ndarray | None = None
    gross_exposure: float = 1.0
    leverage_limit: float = 1.0
    allow_short: bool = False


def validate_and_normalize_world_action(
    action: WorldAction,
    *,
    expected_channels: int | None = None,
) -> WorldAction:
    """Validate and normalize a WorldAction in deterministic rule order.

    Validation/normalization order is binding:
    1) finite check
    2) sign constraints
    3) leverage constraint
    4) exposure normalization
    """
    if not isinstance(action, WorldAction):
        raise ValueError("invalid WorldAction: expected WorldAction instance")

    # 1) finite check
    weights = np.asarray(action.weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("invalid WorldAction: weights must be a 1D array")

    flow_matrix = None if action.flow_matrix is None else np.asarray(action.flow_matrix, dtype=float)
    output_weights = None if action.output_weights is None else np.asarray(action.output_weights, dtype=float)

    if flow_matrix is not None:
        if flow_matrix.ndim != 2:
            raise ValueError("invalid WorldAction: flow_matrix must be a 2D array")
        if expected_channels is not None and flow_matrix.shape[0] != int(expected_channels):
            raise ValueError(
                "invalid WorldAction: flow_matrix row count must match expected channels"
            )
        if np.any(flow_matrix < 0.0):
            raise ValueError("invalid WorldAction: flow_matrix must be non-negative")
        if not np.all(np.isfinite(flow_matrix)):
            raise ValueError("invalid WorldAction: flow_matrix must contain only finite values")

        if output_weights is None:
            output_weights = np.ones(flow_matrix.shape[1], dtype=float)
        if output_weights.ndim != 1:
            raise ValueError("invalid WorldAction: output_weights must be a 1D array")
        if output_weights.shape[0] != flow_matrix.shape[1]:
            raise ValueError(
                "invalid WorldAction: output_weights length must match flow_matrix output dimension"
            )
        if np.any(output_weights < 0.0):
            raise ValueError("invalid WorldAction: output_weights must be non-negative")
        if not np.all(np.isfinite(output_weights)):
            raise ValueError("invalid WorldAction: output_weights must contain only finite values")

        # Canonical projection: input-channel exposure induced by F[n,m] and output allocation m.
        weights = np.asarray(flow_matrix @ output_weights, dtype=float)

    if expected_channels is not None and weights.shape[0] != int(expected_channels):
        raise ValueError(
            f"invalid WorldAction: expected {int(expected_channels)} weights, got {weights.shape[0]}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("invalid WorldAction: weights must contain only finite values")

    gross_exposure = float(action.gross_exposure)
    leverage_limit = float(action.leverage_limit)
    if not np.isfinite(gross_exposure):
        raise ValueError("invalid WorldAction: gross_exposure must be finite")
    if not np.isfinite(leverage_limit):
        raise ValueError("invalid WorldAction: leverage_limit must be finite")

    # 2) sign constraints
    allow_short = bool(action.allow_short)
    if not allow_short and np.any(weights < 0.0):
        raise ValueError("invalid WorldAction: negative weights are not allowed when allow_short=False")

    # 3) leverage constraint
    if gross_exposure < 0.0:
        raise ValueError("invalid WorldAction: gross_exposure must be >= 0")
    if leverage_limit < 0.0:
        raise ValueError("invalid WorldAction: leverage_limit must be >= 0")
    if gross_exposure > leverage_limit:
        raise ValueError("invalid WorldAction: gross_exposure exceeds leverage_limit")

    # 4) exposure normalization
    norm = float(np.abs(weights).sum()) if allow_short else float(weights.sum())
    if norm <= 0.0:
        if gross_exposure == 0.0:
            normalized = np.zeros_like(weights, dtype=float)
        else:
            basis = "sum(abs(weights))" if allow_short else "sum(weights)"
            raise ValueError(f"invalid WorldAction: cannot normalize exposure because {basis} <= 0")
    else:
        normalized = weights * (gross_exposure / norm)

    normalized_output_weights = None if output_weights is None else np.asarray(output_weights, dtype=float)

    return WorldAction(
        weights=np.asarray(normalized, dtype=float),
        flow_matrix=None if flow_matrix is None else np.asarray(flow_matrix, dtype=float),
        output_weights=normalized_output_weights,
        gross_exposure=gross_exposure,
        leverage_limit=leverage_limit,
        allow_short=allow_short,
    )


@dataclass
class WorldStepResult:
    """Realized world outcome consumed by the runtime/kernel."""

    realized_return: float
    costs: float
    channel_returns: np.ndarray
    cost_by_channel: np.ndarray
    freeze: bool = False


class World(Protocol):
    """World provides action-conditioned outcomes (Phase II contract)."""

    def step(self, t: int, action: WorldAction) -> WorldStepResult:
        ...


class Curriculum(Protocol):
    """Curriculum provides a sequence of worlds (state-agnostic)."""

    def next(self, t: int) -> World:
        ...


class Teacher(Protocol):
    """Teacher provides configuration authority, not state intervention."""

    def configure(self, run_id: str, profile: str, mode: str, params: Dict[str, Any]) -> None:
        ...


def validate_world_output(out: Any) -> tuple[np.ndarray, float]:
    """Validate world output and provide legacy `(r_vec, c_total)` view.

    Supports both the new WorldStepResult contract and legacy dict payloads.
    """
    step_result = validate_world_step_result(out)
    r = np.asarray(step_result.channel_returns, dtype=float)
    c = float(step_result.costs)
    if r.ndim != 1:
        raise ValueError("World r must be a 1D array")
    return r, c


def make_world_step_result(
    *,
    r_vec: np.ndarray,
    c_total: float,
    action: WorldAction | None = None,
    freeze: bool = False,
) -> WorldStepResult:
    """Create a WorldStepResult from legacy `(r, c)` payload components."""
    channel_returns = np.asarray(r_vec, dtype=float)
    if channel_returns.ndim != 1:
        raise ValueError("World r must be a 1D array")

    if action is None:
        realized_return = 0.0
    else:
        normalized_action = validate_and_normalize_world_action(
            action,
            expected_channels=int(channel_returns.shape[0]),
        )
        weights = np.asarray(normalized_action.weights, dtype=float)
        realized_return = float(np.dot(weights, channel_returns))

    return WorldStepResult(
        realized_return=realized_return,
        costs=float(c_total),
        channel_returns=channel_returns,
        cost_by_channel=np.zeros_like(channel_returns, dtype=float),
        freeze=bool(freeze),
    )


def validate_world_step_result(out: Any) -> WorldStepResult:
    """Validate and normalize world output to WorldStepResult."""
    if isinstance(out, WorldStepResult):
        channel_returns = np.asarray(out.channel_returns, dtype=float)
        cost_by_channel = np.asarray(out.cost_by_channel, dtype=float)
        if channel_returns.ndim != 1:
            raise ValueError("WorldStepResult.channel_returns must be a 1D array")
        if cost_by_channel.ndim != 1:
            raise ValueError("WorldStepResult.cost_by_channel must be a 1D array")
        if channel_returns.shape != cost_by_channel.shape:
            raise ValueError("WorldStepResult channel vectors must have identical shape")
        return WorldStepResult(
            realized_return=float(out.realized_return),
            costs=float(out.costs),
            channel_returns=channel_returns,
            cost_by_channel=cost_by_channel,
            freeze=bool(out.freeze),
        )

    if isinstance(out, dict):
        if "r" not in out or "c" not in out:
            raise ValueError("World.step must return WorldStepResult or dict with keys: r, c")
        return make_world_step_result(
            r_vec=np.asarray(out["r"], dtype=float),
            c_total=float(out["c"]),
            action=None,
            freeze=bool(out.get("freeze", False)),
        )

    raise ValueError("World.step must return WorldStepResult or dict payload")
