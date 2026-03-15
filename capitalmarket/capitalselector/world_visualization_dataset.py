"""World visualization dataset — deterministic, renderer-independent."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .world_interface import CanonicalWorld, CanonicalWorldObservation, validate_world_observation_payload
from .world_parameters import WorldParametersSchema


WORLD_VISUALIZATION_EXTENSION_STRESS_INTENSITY_FIELD = "ext_stress_intensity"


@dataclass(frozen=True)
class WorldVisualizationTimelineEntry:
    tau: int
    channel_productivity: tuple[float, ...]
    channel_risk: tuple[float, ...]
    liquidity_scale: float
    regime_id: int
    stress_active: bool
    stress_intensity: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "tau": int(self.tau),
            "channel_productivity": [float(v) for v in self.channel_productivity],
            "channel_risk": [float(v) for v in self.channel_risk],
            "liquidity_scale": float(self.liquidity_scale),
            "regime_id": int(self.regime_id),
            "stress_active": bool(self.stress_active),
            "stress_intensity": float(self.stress_intensity),
        }


@dataclass(frozen=True)
class WorldVisualizationDataset:
    """Deterministic visualization dataset for a world run over a tau horizon.

    Fields:
    * timeline   — per-tau world observations (ordered by tau, ascending)
    * productivity_heatmap — shape (horizon, num_channels): channel productivity
                             values across time, suitable for 2-D heatmap plots
    """

    horizon: int
    seed: int
    num_channels: int
    timeline: tuple[WorldVisualizationTimelineEntry, ...]
    # productivity_heatmap[tau][channel_index] == channel_productivity for that tau
    productivity_heatmap: tuple[tuple[float, ...], ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "seed": int(self.seed),
            "num_channels": int(self.num_channels),
            "timeline": [entry.to_payload() for entry in self.timeline],
            "productivity_heatmap": [
                [float(v) for v in row] for row in self.productivity_heatmap
            ],
        }


def _as_horizon(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("world visualization dataset horizon must be an integer")
    try:
        horizon = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("world visualization dataset horizon must be an integer") from exc
    if horizon < 0:
        raise ValueError("world visualization dataset horizon must be >= 0")
    return horizon


def _as_seed(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("world visualization dataset seed must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("world visualization dataset seed must be an integer") from exc


def _as_stress_intensity(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("world visualization extension field 'ext_stress_intensity' must be a float") from exc
    if not math.isfinite(out):
        raise ValueError("world visualization extension field 'ext_stress_intensity' must be finite")
    if out < 0.0 or out > 1.0:
        raise ValueError("world visualization extension field 'ext_stress_intensity' must be in [0.0, 1.0]")
    return float(out)


def _stress_intensity_for_entry(
    *,
    raw_observation: Mapping[str, Any] | CanonicalWorldObservation,
    observation: CanonicalWorldObservation,
    world_parameters: WorldParametersSchema,
) -> float:
    if isinstance(raw_observation, Mapping):
        if WORLD_VISUALIZATION_EXTENSION_STRESS_INTENSITY_FIELD in raw_observation:
            return _as_stress_intensity(raw_observation[WORLD_VISUALIZATION_EXTENSION_STRESS_INTENSITY_FIELD])

    if bool(observation.stress_active):
        return float(world_parameters.stress_intensity)
    return 0.0


def build_world_visualization_dataset_from_world(
    *,
    horizon: int,
    world: CanonicalWorld,
    seed: int,
) -> WorldVisualizationDataset:
    """Build a deterministic visualization dataset from a CanonicalWorld.

    This is the reusable interface-level production path. It only depends on
    the canonical world contract and does not require PeriodicWorld-specific
    imports.
    """
    horizon_i = _as_horizon(horizon)
    seed_i = _as_seed(seed)

    params = world.parameters()
    if not isinstance(params, WorldParametersSchema):
        raise ValueError("world visualization dataset requires world.parameters() -> WorldParametersSchema")
    expected_channels = int(len(params.channel_productivity))

    timeline_entries: list[WorldVisualizationTimelineEntry] = []
    heatmap_rows: list[tuple[float, ...]] = []

    for tau in range(horizon_i):
        world.step(tau)
        raw_observation = world.observe()
        observation = validate_world_observation_payload(
            raw_observation,
            expected_channels=expected_channels,
        )
        stress_intensity = _stress_intensity_for_entry(
            raw_observation=raw_observation,
            observation=observation,
            world_parameters=params,
        )

        entry = WorldVisualizationTimelineEntry(
            tau=tau,
            channel_productivity=tuple(float(v) for v in observation.channel_productivity),
            channel_risk=tuple(float(v) for v in observation.channel_risk),
            liquidity_scale=float(observation.liquidity_scale),
            regime_id=int(observation.regime_id),
            stress_active=bool(observation.stress_active),
            stress_intensity=float(stress_intensity),
        )
        timeline_entries.append(entry)
        heatmap_rows.append(entry.channel_productivity)

    return WorldVisualizationDataset(
        horizon=horizon_i,
        seed=seed_i,
        num_channels=expected_channels,
        timeline=tuple(timeline_entries),
        productivity_heatmap=tuple(heatmap_rows),
    )


def build_world_visualization_dataset(
    *,
    horizon: int,
    world_parameters: WorldParametersSchema,
    seed: int,
) -> WorldVisualizationDataset:
    """Build a deterministic visualization dataset from default PeriodicWorld wiring.

    Compatibility wrapper that preserves the existing signature while delegating
    core dataset construction to ``build_world_visualization_dataset_from_world``.
    """
    if not isinstance(world_parameters, WorldParametersSchema):
        raise ValueError("world visualization dataset requires WorldParametersSchema")

    horizon_i = _as_horizon(horizon)
    seed_i = _as_seed(seed)

    # Local import keeps PeriodicWorld dependency at compatibility edge.
    from .worlds.periodic_world import PeriodicWorld

    world = PeriodicWorld(world_parameters=world_parameters, seed=seed_i)
    return build_world_visualization_dataset_from_world(
        horizon=horizon_i,
        world=world,
        seed=seed_i,
    )
