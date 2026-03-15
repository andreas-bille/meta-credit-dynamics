from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genome_mutation import MutationConfig

MUTATION_SCALING_GENERATION_DEPENDENT = "generation_dependent"
MUTATION_SCALING_NONE = "none"


@dataclass(frozen=True)
class MutationScalingConfig:
    """Configuration for mutation-operator scale parameter control.

    mode:
        "generation_dependent" — scale = 1 / (1 + generation_index * decay_rate),
            i.e. mutation intensity decreases monotonically with generation index.
        "none" — scale factor is always 1.0 (identity, pre-0.9.4 equivalent).

    decay_rate:
        Non-negative rate for generation_dependent mode. Larger values cause
        faster decay. decay_rate=0.0 is equivalent to mode="none".
    """

    mode: str = MUTATION_SCALING_GENERATION_DEPENDENT
    decay_rate: float = 0.01


def compute_mutation_scale_factor(
    *,
    generation_index: int,
    config: MutationScalingConfig,
) -> float:
    """Return the mutation-operator scale factor for the given generation index.

    The returned value is always finite and non-negative (>= 0). For
    ``mode="generation_dependent"`` the returned value is in (0, 1] and
    strictly decreasing with generation_index for decay_rate > 0.
    """
    mode = str(config.mode).strip()

    if mode == MUTATION_SCALING_NONE:
        return 1.0

    if mode == MUTATION_SCALING_GENERATION_DEPENDENT:
        decay = float(config.decay_rate)
        if not np.isfinite(decay) or decay < 0.0:
            raise ValueError("mutation scaling decay_rate must be finite and non-negative")
        g = max(0, int(generation_index))
        return 1.0 / (1.0 + float(g) * decay)

    raise ValueError(f"unknown mutation scaling mode: {config.mode!r}")


def apply_mutation_scaling(
    base_config: MutationConfig,
    *,
    scale_factor: float,
) -> MutationConfig:
    """Return a :class:`MutationConfig` with all scale parameters multiplied by
    *scale_factor*.

    When *scale_factor* is exactly 1.0 the original *base_config* object is
    returned unchanged — this is the regression-guard identity path: any call
    sequence that passes scale_factor=1.0 is guaranteed to produce
    bit-identical mutation output compared to a direct call with *base_config*.
    """
    f = float(scale_factor)
    if not np.isfinite(f):
        raise ValueError("mutation scale factor must be finite")
    if f < 0.0:
        raise ValueError("mutation scale factor must be non-negative")
    if f == 1.0:
        # Identity path: return the exact same config object.
        return base_config
    return MutationConfig(
        noise_scale=float(base_config.noise_scale) * f,
        redistribution_share=float(base_config.redistribution_share) * f,
        lambda_risk_scale=float(base_config.lambda_risk_scale) * f,
    )
