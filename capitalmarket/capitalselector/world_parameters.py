from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


WORLD_PARAMETERS_SCHEMA_VERSION = "v0.9.3"
WORLD_PARAMETERS_REQUIRED_FIELDS = (
    "channel_productivity",
    "channel_risk",
    "liquidity_scale",
    "regime_period",
    "stress_probability",
    "stress_intensity",
)

WORLD_PARAMETER_MUTATION_MUTABLE_FIELDS = (
    "channel_productivity",
    "regime_period",
    "stress_intensity",
)

WORLD_PARAMETER_MUTATION_IMMUTABLE_FIELDS = (
    "channel_risk",
    "liquidity_scale",
    "stress_probability",
)


@dataclass(frozen=True)
class WorldParameterMutationConfig:
    productivity_delta_max: float = 0.05
    regime_period_delta_max: int = 1
    stress_intensity_delta_max: float = 0.1

    def __post_init__(self) -> None:
        productivity_delta_max = _coerce_finite_float(
            self.productivity_delta_max,
            field_name="productivity_delta_max",
        )
        if productivity_delta_max < 0.0:
            raise ValueError("world parameter mutation 'productivity_delta_max' must be >= 0.0")

        regime_period_delta_max = _coerce_int(
            self.regime_period_delta_max,
            field_name="regime_period_delta_max",
        )
        if regime_period_delta_max < 0:
            raise ValueError("world parameter mutation 'regime_period_delta_max' must be >= 0")

        stress_intensity_delta_max = _coerce_finite_float(
            self.stress_intensity_delta_max,
            field_name="stress_intensity_delta_max",
        )
        if stress_intensity_delta_max < 0.0:
            raise ValueError("world parameter mutation 'stress_intensity_delta_max' must be >= 0.0")

        object.__setattr__(self, "productivity_delta_max", float(productivity_delta_max))
        object.__setattr__(self, "regime_period_delta_max", int(regime_period_delta_max))
        object.__setattr__(self, "stress_intensity_delta_max", float(stress_intensity_delta_max))


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"world parameters field '{field_name}' must be a float") from exc
    if not math.isfinite(out):
        raise ValueError(f"world parameters field '{field_name}' must be finite")
    return out


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"world parameters field '{field_name}' must be an integer")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)

    raise ValueError(f"world parameters field '{field_name}' must be an integer")


def _coerce_non_negative_vector(values: Sequence[float] | np.ndarray, *, field_name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"world parameters field '{field_name}' must be a sequence")

    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError(f"world parameters field '{field_name}' must be a sequence") from exc

    coerced: list[float] = []
    for index, raw in enumerate(raw_values):
        value = _coerce_finite_float(raw, field_name=f"{field_name}[{index}]")
        if value < 0.0:
            raise ValueError(f"world parameters field '{field_name}' must be non-negative")
        coerced.append(value)
    return tuple(coerced)


def _coerce_seed(seed: Any) -> int:
    if isinstance(seed, bool):
        raise ValueError("world parameter seed must be an integer")
    try:
        return int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("world parameter seed must be an integer") from exc


@dataclass(frozen=True)
class WorldParametersSchema:
    channel_productivity: tuple[float, ...]
    channel_risk: tuple[float, ...]
    liquidity_scale: float = 1.0
    regime_period: int = 1
    stress_probability: float = 0.0
    stress_intensity: float = 0.0

    def __post_init__(self) -> None:
        productivity = _coerce_non_negative_vector(
            self.channel_productivity,
            field_name="channel_productivity",
        )
        if len(productivity) == 0:
            raise ValueError("world parameters field 'channel_productivity' must have at least one entry")

        risk = _coerce_non_negative_vector(
            self.channel_risk,
            field_name="channel_risk",
        )
        if len(risk) != len(productivity):
            raise ValueError("world parameters field 'channel_risk' must match channel_productivity length")

        liquidity_scale = _coerce_finite_float(self.liquidity_scale, field_name="liquidity_scale")
        if liquidity_scale <= 0.0:
            raise ValueError("world parameters field 'liquidity_scale' must be > 0.0")

        regime_period = _coerce_int(self.regime_period, field_name="regime_period")
        if regime_period < 1:
            raise ValueError("world parameters field 'regime_period' must be >= 1")

        stress_probability = _coerce_finite_float(self.stress_probability, field_name="stress_probability")
        if stress_probability < 0.0 or stress_probability > 1.0:
            raise ValueError("world parameters field 'stress_probability' must be in [0.0, 1.0]")

        stress_intensity = _coerce_finite_float(self.stress_intensity, field_name="stress_intensity")
        if stress_intensity < 0.0 or stress_intensity > 1.0:
            raise ValueError("world parameters field 'stress_intensity' must be in [0.0, 1.0]")

        object.__setattr__(self, "channel_productivity", productivity)
        object.__setattr__(self, "channel_risk", risk)
        object.__setattr__(self, "liquidity_scale", float(liquidity_scale))
        object.__setattr__(self, "regime_period", int(regime_period))
        object.__setattr__(self, "stress_probability", float(stress_probability))
        object.__setattr__(self, "stress_intensity", float(stress_intensity))

    def to_dict(self, *, include_schema_version: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "channel_productivity": [float(value) for value in self.channel_productivity],
            "channel_risk": [float(value) for value in self.channel_risk],
            "liquidity_scale": float(self.liquidity_scale),
            "regime_period": int(self.regime_period),
            "stress_probability": float(self.stress_probability),
            "stress_intensity": float(self.stress_intensity),
        }
        if include_schema_version:
            payload["schema_version"] = WORLD_PARAMETERS_SCHEMA_VERSION
        return payload


def world_parameters_from_dict(payload: Mapping[str, Any]) -> WorldParametersSchema:
    if not isinstance(payload, Mapping):
        raise ValueError("world parameters payload must be a mapping")

    unknown_fields = sorted(
        str(key)
        for key in payload.keys()
        if str(key) not in set(WORLD_PARAMETERS_REQUIRED_FIELDS) and str(key) != "schema_version"
    )
    if unknown_fields:
        raise ValueError(f"world parameters payload contains unknown fields: {', '.join(unknown_fields)}")

    missing_required = [field for field in WORLD_PARAMETERS_REQUIRED_FIELDS if field not in payload]
    if missing_required:
        if len(missing_required) == 1:
            raise ValueError(f"world parameters missing required field: {missing_required[0]}")
        raise ValueError(f"world parameters missing required fields: {', '.join(missing_required)}")

    schema_version = payload.get("schema_version", WORLD_PARAMETERS_SCHEMA_VERSION)
    if str(schema_version) != WORLD_PARAMETERS_SCHEMA_VERSION:
        raise ValueError(
            "world parameters schema_version mismatch: "
            f"expected {WORLD_PARAMETERS_SCHEMA_VERSION}, got {schema_version}"
        )

    return WorldParametersSchema(
        channel_productivity=payload["channel_productivity"],
        channel_risk=payload["channel_risk"],
        liquidity_scale=payload["liquidity_scale"],
        regime_period=payload["regime_period"],
        stress_probability=payload["stress_probability"],
        stress_intensity=payload["stress_intensity"],
    )


def world_parameters_from_seed(
    *,
    seed: int,
    channel_count: int,
    liquidity_scale: float = 1.0,
    regime_period: int = 1,
    stress_probability: float = 0.0,
    stress_intensity: float = 0.0,
) -> WorldParametersSchema:
    seed_i = _coerce_seed(seed)
    channel_count_i = _coerce_int(channel_count, field_name="channel_count")
    if channel_count_i < 1:
        raise ValueError("world parameter seed generation requires channel_count >= 1")

    rng = np.random.default_rng(np.random.PCG64(np.uint64(seed_i)))
    channel_productivity = tuple(float(value) for value in rng.uniform(0.1, 1.0, size=channel_count_i).tolist())
    channel_risk = tuple(float(value) for value in rng.uniform(0.0, 1.0, size=channel_count_i).tolist())

    return WorldParametersSchema(
        channel_productivity=channel_productivity,
        channel_risk=channel_risk,
        liquidity_scale=liquidity_scale,
        regime_period=regime_period,
        stress_probability=stress_probability,
        stress_intensity=stress_intensity,
    )


def world_parameters_payload_from_seed(
    *,
    seed: int,
    channel_count: int,
    liquidity_scale: float = 1.0,
    regime_period: int = 1,
    stress_probability: float = 0.0,
    stress_intensity: float = 0.0,
) -> dict[str, Any]:
    params = world_parameters_from_seed(
        seed=seed,
        channel_count=channel_count,
        liquidity_scale=liquidity_scale,
        regime_period=regime_period,
        stress_probability=stress_probability,
        stress_intensity=stress_intensity,
    )
    return params.to_dict(include_schema_version=True)


def canonical_world_parameters_digest_input(
    world_parameters: WorldParametersSchema,
    *,
    seed: int,
) -> str:
    if not isinstance(world_parameters, WorldParametersSchema):
        raise ValueError("canonical digest input requires WorldParametersSchema")

    seed_i = _coerce_seed(seed)
    payload = {
        "schema_version": WORLD_PARAMETERS_SCHEMA_VERSION,
        "seed": int(seed_i),
        "world_parameters": world_parameters.to_dict(include_schema_version=False),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def world_parameter_mutation_scope() -> dict[str, Any]:
    """Return explicit mutation-scope boundaries for world parameter search.

    This is the authoritative positive list for mutable fields under
    ``mutate_world_parameters``. Fields not listed under ``mutable`` are
    intentionally out of scope in the current mutation contract.
    """
    return {
        "mutable": {
            "channel_productivity": {
                "lower_bound": 0.0,
                "upper_bound": None,
                "operator": "additive_bounded_uniform_delta_per_channel",
            },
            "regime_period": {
                "lower_bound": 1,
                "upper_bound": None,
                "operator": "bounded_integer_delta",
            },
            "stress_intensity": {
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "operator": "additive_bounded_uniform_delta",
            },
        },
        "immutable": {
            "channel_risk": "out_of_scope_preserved_as_configured",
            "liquidity_scale": "out_of_scope_preserved_as_configured",
            "stress_probability": "out_of_scope_preserved_as_configured",
        },
    }


def _bounded_uniform_delta(
    *,
    rng: np.random.Generator,
    value: float,
    delta_max: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    lower = max(-float(delta_max), float(lower_bound) - float(value))
    upper = min(float(delta_max), float(upper_bound) - float(value))
    if lower > upper:
        raise ValueError("world parameter mutation produced invalid bounded delta interval")
    if lower == upper:
        return float(lower)
    return float(rng.uniform(lower, upper))


def _bounded_integer_delta(
    *,
    rng: np.random.Generator,
    value: int,
    delta_max: int,
    lower_bound: int,
) -> int:
    lower = max(-int(delta_max), int(lower_bound) - int(value))
    upper = int(delta_max)
    if lower > upper:
        raise ValueError("world parameter mutation produced invalid bounded integer delta interval")
    if lower == upper:
        return int(lower)
    return int(rng.integers(low=int(lower), high=int(upper) + 1))


def mutate_world_parameters(
    world_parameters: WorldParametersSchema,
    *,
    mutation_seed: int,
    config: WorldParameterMutationConfig | None = None,
) -> WorldParametersSchema:
    """Mutate only the documented positive-list world parameters.

    Mutable fields: channel_productivity, regime_period, stress_intensity.
    Out-of-scope fields are preserved exactly: channel_risk, liquidity_scale,
    stress_probability.
    """
    if not isinstance(world_parameters, WorldParametersSchema):
        raise ValueError("world parameter mutation requires WorldParametersSchema")

    seed_i = _coerce_seed(mutation_seed)
    cfg = config or WorldParameterMutationConfig()
    rng = np.random.default_rng(np.random.PCG64(np.uint64(seed_i)))

    mutated_productivity: list[float] = []
    for value in world_parameters.channel_productivity:
        delta = _bounded_uniform_delta(
            rng=rng,
            value=float(value),
            delta_max=float(cfg.productivity_delta_max),
            lower_bound=0.0,
            upper_bound=float("inf"),
        )
        candidate = float(value) + float(delta)
        if not math.isfinite(candidate) or candidate < 0.0:
            raise ValueError("world parameter mutation produced out-of-bounds channel_productivity")
        mutated_productivity.append(float(candidate))

    regime_delta = _bounded_integer_delta(
        rng=rng,
        value=int(world_parameters.regime_period),
        delta_max=int(cfg.regime_period_delta_max),
        lower_bound=1,
    )
    mutated_regime_period = int(world_parameters.regime_period) + int(regime_delta)
    if mutated_regime_period < 1:
        raise ValueError("world parameter mutation produced out-of-bounds regime_period")

    intensity_delta = _bounded_uniform_delta(
        rng=rng,
        value=float(world_parameters.stress_intensity),
        delta_max=float(cfg.stress_intensity_delta_max),
        lower_bound=0.0,
        upper_bound=1.0,
    )
    mutated_stress_intensity = float(world_parameters.stress_intensity) + float(intensity_delta)
    if mutated_stress_intensity < 0.0 or mutated_stress_intensity > 1.0:
        raise ValueError("world parameter mutation produced out-of-bounds stress_intensity")

    return WorldParametersSchema(
        channel_productivity=tuple(mutated_productivity),
        channel_risk=tuple(world_parameters.channel_risk),
        liquidity_scale=float(world_parameters.liquidity_scale),
        regime_period=int(mutated_regime_period),
        stress_probability=float(world_parameters.stress_probability),
        stress_intensity=float(mutated_stress_intensity),
    )


def mutate_world_parameters_payload(
    payload: Mapping[str, Any],
    *,
    mutation_seed: int,
    config: WorldParameterMutationConfig | None = None,
) -> dict[str, Any]:
    source = world_parameters_from_dict(payload)
    mutated = mutate_world_parameters(source, mutation_seed=mutation_seed, config=config)
    return mutated.to_dict(include_schema_version=True)
