"""Periodic world with piecewise-constant schedule over configurable period."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np

from ..world_interface import CanonicalWorldObservation
from ..world_parameters import WorldParametersSchema
from ..world_regime_dataset import (
    RegimeTimelineEntry,
    StressRegimeInjector,
    regime_id_for_tau,
)


class PeriodicWorld:
    """
    World with piecewise-constant periodic schedule.

    Invariants:
    * periodic schedule is piecewise-constant over tau with period P
    * periodic schedule follows configured period exactly
    * phase transitions are deterministic under seed
    * channel productivity stays within configured bounds
    * v0.9.3 baseline periodic contract applies to channel productivity
    """

    _PHASE_FLOOR_MULTIPLIER: float = 0.35
    DEFAULT_STRESS_PRECOMPUTE_HORIZON: int = 4096
    PHASE_PROFILE_MODE_LINEAR_PERMUTATION: str = "linear_permutation"
    PHASE_PROFILE_MODE_RISK_ADJUSTED_ROTATION: str = "risk_adjusted_rotation"
    PHASE_PROFILE_MODES: tuple[str, ...] = (
        PHASE_PROFILE_MODE_LINEAR_PERMUTATION,
        PHASE_PROFILE_MODE_RISK_ADJUSTED_ROTATION,
    )

    def __init__(
        self,
        *,
        world_parameters: WorldParametersSchema,
        seed: int,
        phase_scales: Sequence[float | Sequence[float]] | None = None,
        phase_profile_mode: str = PHASE_PROFILE_MODE_LINEAR_PERMUTATION,
        stress_precompute_horizon: int = DEFAULT_STRESS_PRECOMPUTE_HORIZON,
    ) -> None:
        """
        Initialize periodic world.

        Args:
            world_parameters: Configuration schema with regime_period, stress settings
            seed: Deterministic seed for reproducible phase behavior
            phase_scales: Optional list of phase definitions per phase.
                         Scalar entries act as global multipliers on the
                                                 configured productivity vector (compatibility mode).
                         Sequence entries act as explicit per-channel
                         productivity profiles.
                        phase_profile_mode: Deterministic profile mode used when
                                                 ``phase_scales`` is not provided.
                                                 * ``linear_permutation``: v0.9.3 baseline deterministic
                                                     random permutation profile.
                                                 * ``risk_adjusted_rotation``: deterministic,
                                                     non-synthetic profile derived from channel
                                                     productivity/risk structure.
            stress_precompute_horizon: Number of tau entries to precompute in
                         deterministic stress schedule. ``stress_probability``
                         parameterizes schedule generation under seed and is not
                         sampled inside ``step``/``observe``.
        """
        self._params = world_parameters
        self._seed = int(seed)
        self._tau = 0
        self._regime_period = int(self._params.regime_period)
        self._num_channels = int(len(self._params.channel_productivity))
        self._phase_profile_mode = self._coerce_phase_profile_mode(phase_profile_mode)
        self._phase_scale_scalar_warning_emitted = False
        self._stress_precompute_horizon = int(stress_precompute_horizon)
        if self._stress_precompute_horizon < 1:
            raise ValueError("stress_precompute_horizon must be >= 1")

        # Stress is injected from a deterministic precomputed schedule, not sampled in observe().
        self._stress_injector = StressRegimeInjector(
            world_parameters=self._params,
            seed=self._seed,
            horizon=self._stress_precompute_horizon,
        )

        # Generate deterministic channel-wise phase profiles if not provided.
        if phase_scales is None:
            self._phase_vectors = [
                self._build_phase_vector(phase_id)
                for phase_id in range(100)
            ]
        else:
            self._phase_vectors = [self._coerce_phase_vector(scale) for scale in phase_scales]

    def _coerce_phase_profile_mode(self, mode: str) -> str:
        mode_s = str(mode).strip().lower()
        if mode_s not in self.PHASE_PROFILE_MODES:
            allowed = ", ".join(self.PHASE_PROFILE_MODES)
            raise ValueError(f"phase_profile_mode must be one of: {allowed}")
        return mode_s

    def _phase_seed(self, phase_id: int) -> np.uint64:
        mixed_seed = (self._seed & 0xFFFFFFFFFFFFFFFF) ^ ((int(phase_id) + 1) * 0x9E3779B97F4A7C15)
        return np.uint64(mixed_seed & 0xFFFFFFFFFFFFFFFF)

    def _coerce_phase_vector(self, scale: float | Sequence[float]) -> tuple[float, ...]:
        try:
            scalar = float(scale)
        except (TypeError, ValueError):
            if isinstance(scale, (str, bytes)):
                raise ValueError("phase scale must be a float or sequence of floats")
            try:
                raw_values = list(scale)
            except TypeError as exc:
                raise ValueError("phase scale must be a float or sequence of floats") from exc
            if len(raw_values) != self._num_channels:
                raise ValueError("phase vector must match channel count")
            vector = tuple(float(value) for value in raw_values)
        else:
            if not bool(self._phase_scale_scalar_warning_emitted):
                warnings.warn(
                    "PeriodicWorld scalar phase_scales are compatibility mode; prefer explicit per-channel vectors.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                self._phase_scale_scalar_warning_emitted = True
            vector = tuple(
                float(base_value * scalar)
                for base_value in self._params.channel_productivity
            )

        for n, value in enumerate(vector):
            if not np.isfinite(value) or not (
                0.0 <= value <= float(self._params.channel_productivity[n])
            ):
                raise ValueError("phase scale must be in [0.0, per-channel configured max]")
        return vector

    def _build_phase_vector_linear_permutation(self, phase_id: int) -> tuple[float, ...]:
        """Build baseline v0.9.3 deterministic permutation profile for one phase.

        Each channel n gets:
            phase_value[n] = channel_productivity[n] * multiplier[n]
        where multiplier[n] is drawn from a seed-deterministic random permutation
        of N evenly-spaced values in [_PHASE_FLOOR_MULTIPLIER, 1.0].

        This preserves channel_productivity[n] as the strict per-channel ceiling
        and enables niche rotation when channel productivities are comparable.
        """
        rng = np.random.default_rng(np.random.PCG64(self._phase_seed(phase_id)))
        n = self._num_channels
        multipliers = np.linspace(1.0, self._PHASE_FLOOR_MULTIPLIER, n, dtype=np.float64)
        perm = rng.permutation(n)
        channel_multipliers = np.empty(n, dtype=np.float64)
        for rank, channel_idx in enumerate(perm):
            channel_multipliers[int(channel_idx)] = multipliers[rank]
        return tuple(
            float(self._params.channel_productivity[i]) * float(channel_multipliers[i])
            for i in range(n)
        )

    def _build_phase_vector_risk_adjusted_rotation(self, phase_id: int) -> tuple[float, ...]:
        """Build deterministic non-synthetic profile from productivity/risk structure.

        Channels are ranked by risk-adjusted productivity and then rotated by a
        deterministic phase/seed offset to retain niche rotation without runtime
        entropy.
        """
        n = self._num_channels
        base_productivity = np.asarray(self._params.channel_productivity, dtype=np.float64)
        base_risk = np.asarray(self._params.channel_risk, dtype=np.float64)
        risk_adjusted_score = base_productivity / np.maximum(base_risk, 1e-12)

        ranked_channels = sorted(
            range(n),
            key=lambda idx: (-float(risk_adjusted_score[idx]), int(idx)),
        )
        seed_offset = int((self._seed & 0xFFFFFFFFFFFFFFFF) % max(1, n))
        phase_offset = int(seed_offset + int(phase_id)) % max(1, n)
        rotated_channels = ranked_channels[phase_offset:] + ranked_channels[:phase_offset]

        multipliers = np.linspace(1.0, self._PHASE_FLOOR_MULTIPLIER, n, dtype=np.float64)
        channel_multipliers = np.empty(n, dtype=np.float64)
        for rank, channel_idx in enumerate(rotated_channels):
            channel_multipliers[int(channel_idx)] = multipliers[rank]

        return tuple(
            float(base_productivity[i]) * float(channel_multipliers[i])
            for i in range(n)
        )

    def _build_phase_vector(self, phase_id: int) -> tuple[float, ...]:
        if self._phase_profile_mode == self.PHASE_PROFILE_MODE_LINEAR_PERMUTATION:
            return self._build_phase_vector_linear_permutation(phase_id)
        if self._phase_profile_mode == self.PHASE_PROFILE_MODE_RISK_ADJUSTED_ROTATION:
            return self._build_phase_vector_risk_adjusted_rotation(phase_id)
        raise ValueError(f"unsupported phase_profile_mode: {self._phase_profile_mode}")

    def _get_phase_id(self, tau: int) -> int:
        """Get phase ID for given tau (piecewise-constant with period)."""
        tau_i = int(tau)
        if tau_i < 0:
            raise ValueError("periodic world tau must be >= 0")
        return regime_id_for_tau(tau=tau_i, regime_period=self._regime_period)

    def _get_phase_vector(self, tau: int) -> tuple[float, ...]:
        """Get productivity profile for current phase, ensuring piecewise-constant behavior."""
        phase_id = self._get_phase_id(tau)
        while len(self._phase_vectors) <= phase_id:
            self._phase_vectors.append(self._build_phase_vector(len(self._phase_vectors)))
        return self._phase_vectors[phase_id]

    def _get_regime_entry(self, tau: int) -> RegimeTimelineEntry:
        """Get injected regime timeline entry from deterministic stress schedule."""
        tau_i = int(tau)
        if tau_i < 0:
            raise ValueError("periodic world tau must be >= 0")

        if tau_i >= self._stress_precompute_horizon:
            self._stress_injector.extend_to_horizon(horizon=tau_i + 1)
            self._stress_precompute_horizon = int(tau_i + 1)

        stress_snapshot = self._stress_injector.snapshot_for_tau(tau=tau_i)
        return RegimeTimelineEntry(
            tau=int(stress_snapshot.tau),
            regime_id=int(stress_snapshot.regime_id),
            stress_active=bool(stress_snapshot.stress_active),
            stress_intensity=float(stress_snapshot.stress_intensity),
        )

    def step(self, tau: int) -> None:
        """Step world to given tau (deterministic)."""
        tau_i = int(tau)
        if tau_i < 0:
            raise ValueError("periodic world tau must be >= 0")
        self._tau = tau_i

    def observe(self) -> CanonicalWorldObservation:
        """Observe world state at current tau (piecewise-constant within phase).
        
        Risk modulation is NOT enabled: channel_risk values remain constant per configured parameters.
        Stress effects are injected deterministically and only affect world inputs.
        Period-based productivity scaling is applied via deterministic channel-wise phase profiles.
        """
        # Get deterministic regime state for this tau
        regime_entry = self._get_regime_entry(self._tau)

        # Get phase profile (deterministic, piecewise-constant per period)
        phase_vector = self._get_phase_vector(self._tau)

        # Apply deterministic channel-wise phase profile to channel productivity.
        # Piecewise-constant: all tau in same phase window get the same profile.
        stress_scale = max(0.0, 1.0 - float(regime_entry.stress_intensity))
        scaled_productivity = tuple(
            float(phase_value * stress_scale)
            for phase_value in phase_vector
        )

        # Risk stays constant (no risk modulation in periodic world)
        risk = tuple(self._params.channel_risk)

        return CanonicalWorldObservation(
            channel_productivity=scaled_productivity,
            channel_risk=risk,
            liquidity_scale=float(self._params.liquidity_scale),
            regime_id=int(regime_entry.regime_id),
            stress_active=bool(regime_entry.stress_active),
        )

    def parameters(self) -> WorldParametersSchema:
        """Get world configuration parameters."""
        return self._params
