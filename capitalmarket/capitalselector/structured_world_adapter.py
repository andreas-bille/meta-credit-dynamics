from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .evolution_contracts import PARITY_ATOL as CONTRACT_PARITY_ATOL
from .evolution_contracts import PARITY_RTOL as CONTRACT_PARITY_RTOL
from .world_interface import (
    CanonicalWorld,
    CanonicalWorldObservation,
    validate_canonical_world_interface,
    validate_world_observation_payload,
)
from .world_parameters import WorldParametersSchema


PARITY_RTOL: float = CONTRACT_PARITY_RTOL
PARITY_ATOL: float = CONTRACT_PARITY_ATOL

STRUCTURED_WORLD_ADAPTER_R_VEC_FORMULA = (
    "r_vec[n] = (channel_productivity[n] - channel_risk[n] - stress_penalty) * liquidity_scale"
)
STRUCTURED_WORLD_ADAPTER_C_TOTAL_FORMULA = (
    "c_total = liquidity_scale * (mean(channel_risk) + stress_penalty)"
)
STRUCTURED_WORLD_ADAPTER_FREEZE_FORMULA = "freeze = False"


@dataclass(frozen=True)
class StructuredWorldAdapterSnapshot:
    tau: int
    regime_id: int
    stress_active: bool
    liquidity_scale: float
    r_vec: tuple[float, ...]
    c_total: float
    freeze: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "tau": int(self.tau),
            "regime_id": int(self.regime_id),
            "stress_active": bool(self.stress_active),
            "liquidity_scale": float(self.liquidity_scale),
            "r_vec": [float(value) for value in self.r_vec],
            "c_total": float(self.c_total),
            "freeze": bool(self.freeze),
        }


def project_canonical_observation_to_adapter_snapshot(
    *,
    tau: int,
    observation: CanonicalWorldObservation,
    world_parameters: WorldParametersSchema,
) -> StructuredWorldAdapterSnapshot:
    """Project canonical world observation into runtime event vectors.

    Authoritative mapping contract for v0.9.4-1:
    * ``r_vec[n] = (channel_productivity[n] - channel_risk[n] - stress_penalty) * liquidity_scale``
    * ``c_total = liquidity_scale * (mean(channel_risk) + stress_penalty)``
    * ``freeze = False``

    ``stress_penalty`` is derived from world parameters, not from selector-core
    state: ``world_parameters.stress_intensity`` when ``stress_active`` is true,
    else ``0.0``.
    """
    tau_i = int(tau)
    if tau_i < 0:
        raise ValueError("structured world adapter requires tau >= 0")
    if not isinstance(world_parameters, WorldParametersSchema):
        raise ValueError("structured world adapter requires WorldParametersSchema")

    stress_penalty = float(world_parameters.stress_intensity) if observation.stress_active else 0.0
    liquidity_scale = float(observation.liquidity_scale)
    mean_channel_risk = float(sum(observation.channel_risk) / len(observation.channel_risk))

    r_vec = tuple(
        float((channel_productivity - channel_risk - stress_penalty) * liquidity_scale)
        for channel_productivity, channel_risk in zip(
            observation.channel_productivity,
            observation.channel_risk,
        )
    )
    c_total = float(liquidity_scale * (mean_channel_risk + stress_penalty))

    return StructuredWorldAdapterSnapshot(
        tau=tau_i,
        regime_id=int(observation.regime_id),
        stress_active=bool(observation.stress_active),
        liquidity_scale=liquidity_scale,
        r_vec=r_vec,
        c_total=c_total,
        freeze=False,
    )

class StructuredWorldAdapter:
    """Deterministic adapter from CanonicalWorld observations to runtime event vectors.

    The projection formulas are authoritative and documented by
    ``STRUCTURED_WORLD_ADAPTER_R_VEC_FORMULA``,
    ``STRUCTURED_WORLD_ADAPTER_C_TOTAL_FORMULA``, and
    ``STRUCTURED_WORLD_ADAPTER_FREEZE_FORMULA``.
    """

    def __init__(self, *, world: CanonicalWorld) -> None:
        self._world = world
        self._parameters = world.parameters()
        self._expected_channels = int(len(self._parameters.channel_productivity))

        # Validate contract once during wiring so runtime failures are explicit.
        validate_canonical_world_interface(world, tau=0)

    def _snapshot_to_runtime_vectors(self, *, tau: int) -> StructuredWorldAdapterSnapshot:
        tau_i = int(tau)

        self._world.step(tau_i)
        observation = validate_world_observation_payload(
            self._world.observe(),
            expected_channels=self._expected_channels,
        )
        return project_canonical_observation_to_adapter_snapshot(
            tau=tau_i,
            observation=observation,
            world_parameters=self._parameters,
        )

    def snapshot_for_tau(self, *, tau: int) -> StructuredWorldAdapterSnapshot:
        return self._snapshot_to_runtime_vectors(tau=tau)

    def build_process_event(self, *, tau: int) -> dict[str, Any]:
        snapshot = self.snapshot_for_tau(tau=tau)
        return {
            "r_vec": np.asarray(snapshot.r_vec, dtype=np.float64),
            "c_total": float(snapshot.c_total),
            "freeze": bool(snapshot.freeze),
        }
