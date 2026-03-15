from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .world_parameters import WorldParametersSchema


@dataclass(frozen=True)
class RegimeTimelineEntry:
    tau: int
    regime_id: int
    stress_active: bool
    stress_intensity: float


@dataclass(frozen=True)
class WorldRegimeTimelineDataset:
    horizon: int
    seed: int
    regime_period: int
    entries: tuple[RegimeTimelineEntry, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "seed": int(self.seed),
            "regime_period": int(self.regime_period),
            "entries": [
                {
                    "tau": int(entry.tau),
                    "regime_id": int(entry.regime_id),
                    "stress_active": bool(entry.stress_active),
                    "stress_intensity": float(entry.stress_intensity),
                }
                for entry in self.entries
            ],
        }


def _as_seed(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("regime timeline seed must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("regime timeline seed must be an integer") from exc


def _as_horizon(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("regime timeline horizon must be an integer")
    try:
        horizon = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("regime timeline horizon must be an integer") from exc
    if horizon < 0:
        raise ValueError("regime timeline horizon must be >= 0")
    return horizon


def regime_id_for_tau(*, tau: int, regime_period: int) -> int:
    tau_i = int(tau)
    period_i = int(regime_period)
    if tau_i < 0:
        raise ValueError("regime timeline tau must be >= 0")
    if period_i < 1:
        raise ValueError("regime period must be >= 1")
    return int(tau_i // period_i)


def _stress_active_for_tau(*, seed: int, tau: int, stress_probability: float) -> bool:
    if stress_probability <= 0.0:
        return False
    if stress_probability >= 1.0:
        return True

    # Derive an independent deterministic stream per tau so results are order-independent.
    mixed_seed = (int(seed) & 0xFFFFFFFFFFFFFFFF) ^ ((int(tau) + 1) * 0x9E3779B97F4A7C15)
    rng = np.random.default_rng(np.random.PCG64(np.uint64(mixed_seed & 0xFFFFFFFFFFFFFFFF)))
    return bool(float(rng.random()) < float(stress_probability))


def build_regime_timeline_entry(
    *,
    tau: int,
    world_parameters: WorldParametersSchema,
    seed: int,
) -> RegimeTimelineEntry:
    if not isinstance(world_parameters, WorldParametersSchema):
        raise ValueError("regime timeline requires WorldParametersSchema")

    tau_i = int(tau)
    if tau_i < 0:
        raise ValueError("regime timeline tau must be >= 0")

    seed_i = _as_seed(seed)
    regime_id = regime_id_for_tau(tau=tau_i, regime_period=int(world_parameters.regime_period))
    stress_active = _stress_active_for_tau(
        seed=seed_i,
        tau=tau_i,
        stress_probability=float(world_parameters.stress_probability),
    )
    stress_intensity = float(world_parameters.stress_intensity) if stress_active else 0.0

    return RegimeTimelineEntry(
        tau=tau_i,
        regime_id=int(regime_id),
        stress_active=bool(stress_active),
        stress_intensity=float(stress_intensity),
    )


def build_world_regime_timeline_dataset(
    *,
    horizon: int,
    world_parameters: WorldParametersSchema,
    seed: int,
) -> WorldRegimeTimelineDataset:
    if not isinstance(world_parameters, WorldParametersSchema):
        raise ValueError("regime timeline requires WorldParametersSchema")

    horizon_i = _as_horizon(horizon)
    seed_i = _as_seed(seed)

    injector = StressRegimeInjector(
        world_parameters=world_parameters,
        seed=seed_i,
        horizon=horizon_i,
    )
    entries = tuple(
        RegimeTimelineEntry(
            tau=int(snapshot.tau),
            regime_id=int(snapshot.regime_id),
            stress_active=bool(snapshot.stress_active),
            stress_intensity=float(snapshot.stress_intensity),
        )
        for snapshot in injector.schedule()
    )

    return WorldRegimeTimelineDataset(
        horizon=int(horizon_i),
        seed=int(seed_i),
        regime_period=int(world_parameters.regime_period),
        entries=entries,
    )


@dataclass(frozen=True)
class StressRegimeInjectorSnapshot:
    # Duration semantics:
    # - stress_duration is the forward run length (in tau steps) of the current
    #   stress_active state including the current tau.
    # - Example: if state sequence from tau=5 is [True, True, True, False], then
    #   snapshot(tau=5).stress_duration == 3.
    tau: int
    regime_id: int
    stress_active: bool
    stress_intensity: float
    stress_duration: int  # how many taus is this stress active

    def to_payload(self) -> dict[str, Any]:
        return {
            "tau": int(self.tau),
            "regime_id": int(self.regime_id),
            "stress_active": bool(self.stress_active),
            "stress_intensity": float(self.stress_intensity),
            "stress_duration": int(self.stress_duration),
        }


class StressRegimeInjector:
    """
    Deterministic stress regime injector with explicit bounds enforcement.

    Generates deterministic stress schedules from seed and world parameters.
    Ensures:
    * stress_probability parameterizes schedule generation (not runtime sampling)
    * stress_intensity remains bounded in [0.0, 1.0]
    * identical (seed, world_parameters) yields identical stress schedule
    """

    STRESS_INTENSITY_MIN = 0.0
    STRESS_INTENSITY_MAX = 1.0

    def __init__(
        self,
        *,
        world_parameters: WorldParametersSchema,
        seed: int,
        horizon: int,
    ) -> None:
        """
        Initialize stress regime injector.

        Args:
            world_parameters: Config with stress_probability and stress_intensity
            seed: Deterministic seed for reproducible stress schedules
            horizon: Number of tau steps to generate stress schedule for
        """
        self._params = world_parameters
        self._seed = int(seed)
        self._horizon = _as_horizon(horizon)

        # Validate bounds once at construction
        if not (self.STRESS_INTENSITY_MIN <= float(self._params.stress_intensity) <= self.STRESS_INTENSITY_MAX):
            raise ValueError(
                f"stress_intensity must be in [{self.STRESS_INTENSITY_MIN}, {self.STRESS_INTENSITY_MAX}], "
                f"got {self._params.stress_intensity}"
            )
        if not (0.0 <= float(self._params.stress_probability) <= 1.0):
            raise ValueError(
                f"stress_probability must be in [0.0, 1.0], got {self._params.stress_probability}"
            )

        self._schedule: list[StressRegimeInjectorSnapshot] = []
        self.extend_to_horizon(horizon=self._horizon)

    def _build_snapshot_for_tau(self, *, tau: int) -> StressRegimeInjectorSnapshot:
        tau_i = int(tau)
        regime_id = regime_id_for_tau(tau=tau_i, regime_period=int(self._params.regime_period))
        stress_active = _stress_active_for_tau(
            seed=self._seed,
            tau=tau_i,
            stress_probability=float(self._params.stress_probability),
        )
        stress_intensity = float(self._params.stress_intensity) if stress_active else 0.0

        # Verify bounds again (defensive)
        if not (self.STRESS_INTENSITY_MIN <= stress_intensity <= self.STRESS_INTENSITY_MAX):
            raise ValueError(
                f"stress_intensity out of bounds at tau={tau_i}: {stress_intensity}"
            )

        return StressRegimeInjectorSnapshot(
            tau=tau_i,
            regime_id=regime_id,
            stress_active=bool(stress_active),
            stress_intensity=float(stress_intensity),
            stress_duration=1,
        )

    def _recompute_durations_from_index(self, *, start_index: int) -> None:
        """Recompute forward run-length durations for an affected schedule suffix."""
        if not self._schedule:
            return

        start_i = int(start_index)
        if start_i < 0:
            start_i = 0
        if start_i >= len(self._schedule):
            return

        run_length = 0
        next_state: bool | None = None
        for index in range(len(self._schedule) - 1, start_i - 1, -1):
            snapshot = self._schedule[index]
            if next_state is None or bool(snapshot.stress_active) != bool(next_state):
                run_length = 1
            else:
                run_length += 1
            self._schedule[index] = StressRegimeInjectorSnapshot(
                tau=int(snapshot.tau),
                regime_id=int(snapshot.regime_id),
                stress_active=bool(snapshot.stress_active),
                stress_intensity=float(snapshot.stress_intensity),
                stress_duration=int(run_length),
            )
            next_state = bool(snapshot.stress_active)

    def extend_to_horizon(self, *, horizon: int) -> None:
        """Extend deterministic schedule up to ``horizon`` entries."""
        target_horizon = _as_horizon(horizon)
        current_horizon = len(self._schedule)
        if target_horizon <= current_horizon:
            return

        for tau in range(current_horizon, target_horizon):
            self._schedule.append(self._build_snapshot_for_tau(tau=tau))

        self._horizon = int(target_horizon)

        # Only recompute durations for appended suffix plus any bridged tail-run
        # from the previous horizon that has the same stress_active state.
        recompute_start = current_horizon
        if current_horizon == 0:
            recompute_start = 0
        elif current_horizon < len(self._schedule):
            bridge_state = bool(self._schedule[current_horizon].stress_active)
            left = current_horizon - 1
            while left >= 0 and bool(self._schedule[left].stress_active) == bridge_state:
                recompute_start = left
                left -= 1

        self._recompute_durations_from_index(start_index=recompute_start)

    def stress_active_for_tau(self, *, tau: int) -> bool:
        """Get stress_active state for specific tau."""
        tau_i = int(tau)
        if not (0 <= tau_i < len(self._schedule)):
            raise ValueError(f"tau {tau_i} out of range [0, {len(self._schedule)})")
        return bool(self._schedule[tau_i].stress_active)

    def stress_intensity_for_tau(self, *, tau: int) -> float:
        """Get stress_intensity value for specific tau (always bounded)."""
        tau_i = int(tau)
        if not (0 <= tau_i < len(self._schedule)):
            raise ValueError(f"tau {tau_i} out of range [0, {len(self._schedule)})")
        return float(self._schedule[tau_i].stress_intensity)

    def snapshot_for_tau(self, *, tau: int) -> StressRegimeInjectorSnapshot:
        """Get complete stress snapshot for specific tau."""
        tau_i = int(tau)
        if not (0 <= tau_i < len(self._schedule)):
            raise ValueError(f"tau {tau_i} out of range [0, {len(self._schedule)})")
        return self._schedule[tau_i]

    def schedule(self) -> list[StressRegimeInjectorSnapshot]:
        """Get entire deterministic stress schedule."""
        return list(self._schedule)

    def to_trace(self) -> list[dict[str, Any]]:
        """Export stress schedule as trace for evidence documentation."""
        return [snapshot.to_payload() for snapshot in self._schedule]
