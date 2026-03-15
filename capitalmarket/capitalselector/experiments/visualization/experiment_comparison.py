"""Experiment comparison utilities (B3).

Operates exclusively on ``ExperimentDataset`` inputs; no direct simulation
access.  All outputs are explicitly typed dataclasses, not plain dicts.
Comparison outputs are deterministic for identical input datasets.

Comparison output schema
------------------------
MetricDelta
    metric_name: str
    delta: float | np.ndarray
    relative_delta: float | None

StabilityComparison
    metric_deltas: tuple[MetricDelta, ...]
    is_structurally_equivalent: bool

Invariants
----------
* Two identical datasets produce a zero delta for all metric fields.
* A missing required metric field raises ``ValueError`` with an explicit
  message (not ``KeyError`` or ``None``).
* Comparison iterates a declared field list, not ad-hoc key introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from ..experiment_dataset import ExperimentDataset

# ---------------------------------------------------------------------------
# Declared metric field list — iteration order is fixed and explicit.
# ---------------------------------------------------------------------------

_GENERATION_METRIC_FIELDS: tuple[str, ...] = (
    "invasion_rate",
    "mean_survival_time",
)

_POPULATION_METRIC_FIELDS: tuple[str, ...] = (
    "total_trials",
    "total_invasions",
    "fixation_probability",
    "mean_survival_time",
    "n_resident_generations",
    "n_mutant_trials",
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDelta:
    """Delta between the same metric across two experiments.

    Fields
    ------
    metric_name
        Dotted name of the compared metric (e.g. ``"generation.invasion_rate"``).
    delta
        Scalar or array difference (experiment_b value minus experiment_a value).
    relative_delta
        ``delta / |experiment_a|`` when ``experiment_a`` is non-zero; ``None``
        when the denominator is zero.
    """

    metric_name: str
    delta: Union[float, np.ndarray]
    relative_delta: Union[float, None]


@dataclass(frozen=True)
class StabilityComparison:
    """Result of comparing two experiments.

    Fields
    ------
    metric_deltas
        One ``MetricDelta`` per declared metric field.
    is_structurally_equivalent
        ``True`` when every ``delta`` is zero (scalar zero or all-zero array).
    """

    metric_deltas: tuple[MetricDelta, ...]
    is_structurally_equivalent: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_delta(name: str, a: float, b: float) -> MetricDelta:
    delta = b - a
    relative_delta: Union[float, None] = (delta / abs(a)) if a != 0.0 else None
    return MetricDelta(metric_name=name, delta=delta, relative_delta=relative_delta)


def _array_delta(name: str, a: np.ndarray, b: np.ndarray) -> MetricDelta:
    delta = b - a
    denom = np.abs(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom != 0, delta / denom, np.nan)
    relative_delta: Union[float, None] = float(rel.mean()) if not np.all(np.isnan(rel)) else None
    return MetricDelta(metric_name=name, delta=delta, relative_delta=relative_delta)


def _is_zero(value: Union[float, np.ndarray]) -> bool:
    if isinstance(value, np.ndarray):
        return bool(np.all(value == 0))
    return value == 0.0


def _require_field(obj: object, field: str, context: str) -> float:
    """Return ``getattr(obj, field)`` or raise ``ValueError``."""
    if not hasattr(obj, field):
        raise ValueError(
            f"Required metric field '{field}' is absent from {context}. "
            f"Cannot compute delta without both values present."
        )
    return getattr(obj, field)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_experiments(
    experiment_a: ExperimentDataset,
    experiment_b: ExperimentDataset,
) -> StabilityComparison:
    """Compare two experiments and return a ``StabilityComparison``.

    Iterates the declared field lists ``_GENERATION_METRIC_FIELDS`` and
    ``_POPULATION_METRIC_FIELDS``; never performs ad-hoc key introspection.
    Raises ``ValueError`` (not ``KeyError``) when a required field is absent.

    Parameters
    ----------
    experiment_a, experiment_b:
        Two ``ExperimentDataset`` instances to compare.

    Returns
    -------
    StabilityComparison
        Contains one ``MetricDelta`` per declared metric and
        ``is_structurally_equivalent = True`` iff all deltas are zero.
    """
    deltas: list[MetricDelta] = []

    # --- Per-generation scalar metrics (averaged across generations) ---
    for field in _GENERATION_METRIC_FIELDS:
        vals_a = np.array(
            [_require_field(gm, field, f"experiment_a generation_metrics[{i}]")
             for i, gm in enumerate(experiment_a.generation_metrics)],
            dtype=float,
        )
        vals_b = np.array(
            [_require_field(gm, field, f"experiment_b generation_metrics[{i}]")
             for i, gm in enumerate(experiment_b.generation_metrics)],
            dtype=float,
        )
        deltas.append(_array_delta(f"generation.{field}", vals_a, vals_b))

    # --- Population-level scalar metrics ---
    for field in _POPULATION_METRIC_FIELDS:
        a_val = float(_require_field(
            experiment_a.population_statistics, field, "experiment_a population_statistics"
        ))
        b_val = float(_require_field(
            experiment_b.population_statistics, field, "experiment_b population_statistics"
        ))
        deltas.append(_scalar_delta(f"population.{field}", a_val, b_val))

    is_equiv = all(_is_zero(d.delta) for d in deltas)
    return StabilityComparison(metric_deltas=tuple(deltas), is_structurally_equivalent=is_equiv)
