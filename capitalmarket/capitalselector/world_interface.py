from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
import math
from typing import Any, Mapping, Protocol, Sequence, TypedDict

from .world_parameters import WorldParametersSchema


logger = logging.getLogger(__name__)


WORLD_OBSERVATION_REQUIRED_FIELDS = (
    "channel_productivity",
    "channel_risk",
    "liquidity_scale",
    "regime_id",
    "stress_active",
)

WORLD_OBSERVATION_ADAPTER_DERIVED_FIELDS = (
    "runtime_r_vec",
    "runtime_c_total",
    "runtime_freeze",
)

WORLD_OBSERVATION_EXTENSION_FIELD_PREFIX = "ext_"


@dataclass(frozen=True)
class CanonicalWorldObservation:
    channel_productivity: tuple[float, ...]
    channel_risk: tuple[float, ...]
    liquidity_scale: float
    regime_id: int
    stress_active: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "channel_productivity": tuple(float(value) for value in self.channel_productivity),
            "channel_risk": tuple(float(value) for value in self.channel_risk),
            "liquidity_scale": float(self.liquidity_scale),
            "regime_id": int(self.regime_id),
            "stress_active": bool(self.stress_active),
        }


class WorldObservationPayload(TypedDict):
    channel_productivity: Sequence[float]
    channel_risk: Sequence[float]
    liquidity_scale: float
    regime_id: int
    stress_active: bool


@dataclass(frozen=True)
class WorldObservationFieldClassification:
    canonical_fields: tuple[str, ...]
    adapter_derived_fields: tuple[str, ...]
    extension_fields: tuple[str, ...]
    unknown_fields: tuple[str, ...]


class CanonicalWorld(Protocol):
    def step(self, tau: int) -> None:
        ...

    def observe(self) -> WorldObservationPayload | CanonicalWorldObservation:
        ...

    def parameters(self) -> WorldParametersSchema:
        ...


def _as_non_negative_finite_vector(value: Any, *, field_name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"world observe() field '{field_name}' must be a sequence")
    try:
        raw = list(value)
    except TypeError as exc:
        raise ValueError(f"world observe() field '{field_name}' must be a sequence") from exc

    if len(raw) == 0:
        raise ValueError(f"world observe() field '{field_name}' must contain at least one entry")

    out: list[float] = []
    for index, item in enumerate(raw):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"world observe() field '{field_name}[{index}]' must be a float") from exc
        if not math.isfinite(number):
            raise ValueError(f"world observe() field '{field_name}[{index}]' must be finite")
        if number < 0.0:
            raise ValueError(f"world observe() field '{field_name}' must be non-negative")
        out.append(float(number))
    return tuple(out)


def classify_world_observation_fields(
    payload: Mapping[str, Any] | CanonicalWorldObservation,
) -> WorldObservationFieldClassification:
    if isinstance(payload, CanonicalWorldObservation):
        payload = payload.to_payload()

    if not isinstance(payload, Mapping):
        raise ValueError("world observe() must return a mapping payload")

    canonical: list[str] = []
    adapter_derived: list[str] = []
    extension: list[str] = []
    unknown: list[str] = []

    required = set(WORLD_OBSERVATION_REQUIRED_FIELDS)
    adapter = set(WORLD_OBSERVATION_ADAPTER_DERIVED_FIELDS)

    for raw_key in payload.keys():
        key = str(raw_key)
        if key in required:
            canonical.append(key)
        elif key in adapter:
            adapter_derived.append(key)
        elif key.startswith(WORLD_OBSERVATION_EXTENSION_FIELD_PREFIX):
            extension.append(key)
        else:
            unknown.append(key)

    return WorldObservationFieldClassification(
        canonical_fields=tuple(canonical),
        adapter_derived_fields=tuple(adapter_derived),
        extension_fields=tuple(extension),
        unknown_fields=tuple(unknown),
    )


def validate_world_observation_payload(
    payload: Mapping[str, Any] | CanonicalWorldObservation,
    *,
    expected_channels: int | None = None,
) -> CanonicalWorldObservation:
    if isinstance(payload, CanonicalWorldObservation):
        payload = payload.to_payload()

    if not isinstance(payload, Mapping):
        raise ValueError("world observe() must return a mapping payload")

    classification = classify_world_observation_fields(payload)
    if classification.unknown_fields:
        unknown = ", ".join(classification.unknown_fields)
        raise ValueError(
            "world observe() contains unlabeled non-canonical fields: "
            f"{unknown}; extension fields must start with "
            f"'{WORLD_OBSERVATION_EXTENSION_FIELD_PREFIX}'"
        )

    missing = [field for field in WORLD_OBSERVATION_REQUIRED_FIELDS if field not in payload]
    if missing:
        if len(missing) == 1:
            raise ValueError(f"world observe() missing required field: {missing[0]}")
        raise ValueError(f"world observe() missing required fields: {', '.join(missing)}")

    productivity = _as_non_negative_finite_vector(payload["channel_productivity"], field_name="channel_productivity")
    risk = _as_non_negative_finite_vector(payload["channel_risk"], field_name="channel_risk")
    if len(risk) != len(productivity):
        raise ValueError("world observe() field 'channel_risk' must match channel_productivity length")

    if expected_channels is not None and len(productivity) != int(expected_channels):
        raise ValueError(
            "world observe() channel vectors must match expected channel count"
        )

    try:
        liquidity_scale = float(payload["liquidity_scale"])
    except (TypeError, ValueError) as exc:
        raise ValueError("world observe() field 'liquidity_scale' must be a float") from exc
    if not math.isfinite(liquidity_scale):
        raise ValueError("world observe() field 'liquidity_scale' must be finite")
    if liquidity_scale <= 0.0:
        raise ValueError("world observe() field 'liquidity_scale' must be > 0.0")

    regime_raw = payload["regime_id"]
    if isinstance(regime_raw, bool):
        raise ValueError("world observe() field 'regime_id' must be an integer")
    try:
        regime_id = int(regime_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("world observe() field 'regime_id' must be an integer") from exc

    stress_raw = payload["stress_active"]
    if not isinstance(stress_raw, bool):
        raise ValueError("world observe() field 'stress_active' must be a bool")

    return CanonicalWorldObservation(
        channel_productivity=productivity,
        channel_risk=risk,
        liquidity_scale=float(liquidity_scale),
        regime_id=int(regime_id),
        stress_active=bool(stress_raw),
    )


def _validate_step_signature(step_fn: Any) -> None:
    try:
        signature = inspect.signature(step_fn)
    except (TypeError, ValueError) as exc:
        raise ValueError("world interface step(tau) must have an inspectable signature") from exc

    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if len(positional) < 1 or len(required) > 1:
        raise ValueError("world interface step(tau) must accept exactly one required tau argument")


def _validate_zero_arg_signature(fn: Any, *, fn_name: str) -> None:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"world interface {fn_name}() must have an inspectable signature") from exc

    required_positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    if required_positional:
        raise ValueError(f"world interface {fn_name}() must be callable without arguments")


def validate_canonical_world_interface(
    world: Any,
    *,
    tau: int = 0,
    probe_tau_consistency: bool = False,
) -> CanonicalWorldObservation:
    step_fn = getattr(world, "step", None)
    if not callable(step_fn):
        raise ValueError("world interface requires callable step(tau)")
    _validate_step_signature(step_fn)

    observe_fn = getattr(world, "observe", None)
    if not callable(observe_fn):
        raise ValueError("world interface requires callable observe()")
    _validate_zero_arg_signature(observe_fn, fn_name="observe")

    parameters_fn = getattr(world, "parameters", None)
    if not callable(parameters_fn):
        raise ValueError("world interface requires callable parameters()")
    _validate_zero_arg_signature(parameters_fn, fn_name="parameters")

    params = parameters_fn()
    if not isinstance(params, WorldParametersSchema):
        raise ValueError("world interface parameters() must return WorldParametersSchema")

    observation = validate_world_observation_payload(
        observe_fn(),
        expected_channels=len(params.channel_productivity),
    )

    if probe_tau_consistency:
        step_fn(int(tau))
        obs_tau = validate_world_observation_payload(
            observe_fn(),
            expected_channels=len(params.channel_productivity),
        )
        step_fn(int(tau) + 1)
        obs_tau_next = validate_world_observation_payload(
            observe_fn(),
            expected_channels=len(params.channel_productivity),
        )
        if obs_tau_next == obs_tau:
            logger.warning(
                "world interface probe: observe() unchanged between tau=%s and tau=%s",
                int(tau),
                int(tau) + 1,
            )

    return observation
