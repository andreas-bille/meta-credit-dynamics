"""Experiment Visualization Dataset (B1).

Role separation
---------------
ESSDataset
    ESS-probe specific dataset.  Source of truth for strategy hash values
    and invasion trial records.  Remains unchanged by this module.

ExperimentDataset
    Cross-experiment visualization and comparison dataset.  Built from
    ``ESSDataset`` via the documented adapter function
    ``ess_dataset_to_experiment_dataset``.  Contains aggregated
    per-generation, per-strategy, per-regime, and population-level metrics
    suitable for rendering tools.  Does not embed renderer-specific fields.

Canonical conversion path
-------------------------
    ess_dataset_to_experiment_dataset(ESSDataset) -> ExperimentDataset

Invariants
----------
* ``schema_version`` is the declared string constant
  ``EXPERIMENT_DATASET_SCHEMA_VERSION``; it is not constructed at runtime.
* ``config_hash`` is the first 16 hex characters of the SHA-256 digest of
    the canonical JSON (``sort_keys=True``) of the full ESS probe
    configuration stored in ``ESSDataset.config_payload`` when available.
    Legacy v0.9.4 serialized ``ESSDataset`` artifacts do not carry that field;
    for those, the adapter derives a deterministic legacy projection from the
    recoverable artifact fields.
* Strategy hash values in ``ExperimentDataset`` are propagated directly from
  the source ``ESSDataset`` records; no second hash algorithm is defined in
  this module.
* The in-memory production object stores immutable tuples for
    ``generation_metrics``, ``strategy_metrics``, and
    ``world_regime_timeline``.
* The serialized schema emitted by ``experiment_dataset_to_dict`` stores
    these collections as JSON arrays / Python lists.
* ``generation_metrics`` is ordered by ``generation_index``.
* ``strategy_metrics`` is ordered by ``(generation_index, strategy_hash,
    role)``.
* ``world_regime_timeline`` is ordered by ``(generation_index, regime)``.
* No visualization or plotting library is imported in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ..ess_dataset import ESSDataset
from ..ess_dataset import strategy_hash_from_serialized_payload

# ---------------------------------------------------------------------------
# Schema version (declared constant — never constructed at runtime)
# ---------------------------------------------------------------------------

EXPERIMENT_DATASET_SCHEMA_VERSION: str = "v0.9.5-4"

# ---------------------------------------------------------------------------
# Sub-record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationMetrics:
    """Aggregated per-generation invasion metrics.

    Fields
    ------
    generation_index
        Resident training generation index corresponding to these metrics.
    invasion_rate
        Fraction of trials (across all regimes) where ``invasion_outcome``
        is ``True``.
    mean_survival_time
        Mean ``survival_time`` across all trials at this generation.
    resident_strategy_hash
        Hash of the resident strategy at this generation (invariant: constant
        across all records of the same generation).
    """

    generation_index: int
    invasion_rate: float
    mean_survival_time: float
    resident_strategy_hash: str


@dataclass(frozen=True)
class StrategyMetrics:
    """Per-(generation, strategy hash, role) aggregated metrics.

    Fields
    ------
    generation_index
        Generation index at which this strategy was active.
    strategy_hash
        Strategy hash propagated from the source ``ESSDataset`` record.
        No independent hash algorithm is used.
    role
        Either ``"resident"`` or ``"mutant"``.
    invasion_count
        Number of trials in which this strategy achieved ``invasion_outcome
        == True`` (always 0 for the resident role).
    mean_survival_time
        Mean ``survival_time`` across all trials for this strategy in this
        role (0.0 for the resident role).
    """

    generation_index: int
    strategy_hash: str
    role: str
    invasion_count: int
    mean_survival_time: float


@dataclass(frozen=True)
class WorldRegimeEntry:
    """Invasion statistics for a single (generation, regime) pair.

    Fields
    ------
    generation_index
        Generation index of the resident strategy used in this regime.
    regime
        Regime identifier string from the source ``ESSDataset`` records.
    invasion_count
        Number of trials in which ``invasion_outcome`` is ``True`` for this
        regime.
    invasion_rate
        Fraction of trials with successful invasion for this regime.
    """

    generation_index: int
    regime: str
    invasion_count: int
    invasion_rate: float


@dataclass(frozen=True)
class PopulationStatistics:
    """Aggregate statistics over all trials in the experiment.

    Fields
    ------
    total_trials
        Total number of ``ESSDatasetRecord`` entries in the source dataset.
    total_invasions
        Total number of records where ``invasion_outcome`` is ``True``.
    fixation_probability
        ``total_invasions / total_trials``.
    mean_survival_time
        Mean ``survival_time`` across all records.
    n_resident_generations
        Resident training generation count from the source ``ESSDataset``.
    n_mutant_trials
        Number of mutant trials per regime from the source ``ESSDataset``.
    """

    total_trials: int
    total_invasions: int
    fixation_probability: float
    mean_survival_time: float
    n_resident_generations: int
    n_mutant_trials: int


# ---------------------------------------------------------------------------
# Top-level dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentDataset:
    """Cross-experiment visualization and comparison dataset.

    See module docstring for role separation, invariants, and the canonical
    conversion path from ``ESSDataset``.

    Fields
    ------
    schema_version
        Always equal to ``EXPERIMENT_DATASET_SCHEMA_VERSION``.
    seed
        Experiment seed propagated from ``ESSDataset.seed``.
    config_hash
        SHA-256[:16] of the canonical JSON of the full ESS probe
        configuration stored in ``ESSDataset.config_payload`` when present.
        For legacy v0.9.4 artifacts without that field, the adapter uses a
        deterministic projection of the recoverable legacy artifact fields.
    generation_metrics
        One entry per unique ``generation_index``; ordered by
        ``generation_index``. Stored as an immutable tuple in the
        production object and serialized as a JSON array / Python list.
    strategy_metrics
        One entry per unique ``(generation_index, strategy_hash, role)``;
        ordered by ``(generation_index, strategy_hash, role)``. Stored as
        an immutable tuple in the production object and serialized as a
        JSON array / Python list.
    world_regime_timeline
        One entry per unique ``(generation_index, regime)`` pair; ordered
        by ``(generation_index, regime)``. Stored as an immutable tuple in
        the production object and serialized as a JSON array / Python list.
    population_statistics
        Aggregate over all records in the source ``ESSDataset``.
    """

    schema_version: str
    seed: int
    config_hash: str
    generation_metrics: tuple[GenerationMetrics, ...]
    strategy_metrics: tuple[StrategyMetrics, ...]
    world_regime_timeline: tuple[WorldRegimeEntry, ...]
    population_statistics: PopulationStatistics


# ---------------------------------------------------------------------------
# Adapter: ESSDataset -> ExperimentDataset
# ---------------------------------------------------------------------------


def _config_hash_from_ess_dataset(ess_dataset: ESSDataset) -> str:
    """SHA-256[:16] of canonical JSON of the ESS probe configuration.

    For v0.9.5+ exports, this uses the canonical JSON stored in
    ``ESSDataset.config_payload``. Legacy v0.9.4 artifacts do not carry that
    field, so the adapter falls back to a deterministic projection built from
    the recoverable top-level artifact fields.
    """
    if ess_dataset.config_payload is not None:
        canonical = json.dumps(
            json.loads(str(ess_dataset.config_payload)),
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        config_dict: dict[str, Any] = {
            "seed": int(ess_dataset.seed),
            "n_resident_generations": int(ess_dataset.n_resident_generations),
            "n_mutant_trials": int(ess_dataset.n_mutant_trials),
            "regimes": sorted({str(record.regime) for record in ess_dataset.records}),
            "resident_strategy_payloads": sorted(
                {str(record.resident_strategy) for record in ess_dataset.records}
            ),
        }
        canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def ess_dataset_to_experiment_dataset(ess_dataset: ESSDataset) -> ExperimentDataset:
    """Convert an ``ESSDataset`` to an ``ExperimentDataset``.

    This is the canonical adapter function.  All ESS -> ExperimentDataset
    conversions must use this path; no ``ExperimentDataset`` may be
    constructed from an ESS probe run that bypasses this function.

    Field mapping contract
    ----------------------
    seed
        Propagated from ``ESSDataset.seed``.
    config_hash
        SHA-256[:16] of canonical JSON of
        ``{n_mutant_trials, n_resident_generations, seed}``.
    generation_metrics
        One entry per unique ``generation_index`` in records, ordered by
        ``generation_index``.  Per generation:

        * ``invasion_rate`` — fraction of trials where
          ``invasion_outcome`` is ``True``
        * ``mean_survival_time`` — mean of ``survival_time`` across all
          trials
        * ``resident_strategy_hash`` — from ``resident_strategy_hash`` of
          the first record at that generation (invariant: constant per
          generation)
    strategy_metrics
        One entry per unique ``(generation_index, strategy_hash, role)``
        derived from records, ordered by
        ``(generation_index, strategy_hash, role)``.  Strategy hashes are
        propagated directly from ESSDataset records; no independent hash
        algorithm is used.  For the ``"resident"`` role,
        ``invasion_count`` and ``mean_survival_time`` are 0 / 0.0.
    world_regime_timeline
        One entry per unique ``(generation_index, regime)`` pair, ordered
        by ``(generation_index, regime)``.
    population_statistics
        Aggregate statistics over all records.

    Parameters
    ----------
    ess_dataset
        A completed ``ESSDataset`` produced by ``export_ess_dataset``.

    Returns
    -------
    ExperimentDataset
        Deterministic for identical input.
    """
    records = ess_dataset.records

    # --- generation_metrics -------------------------------------------------
    gen_groups: dict[int, list[Any]] = {}
    for rec in records:
        gen_groups.setdefault(int(rec.generation_index), []).append(rec)

    gen_metrics_list: list[GenerationMetrics] = []
    for gen_idx in sorted(gen_groups.keys()):
        group = gen_groups[gen_idx]
        invasion_rate = float(
            sum(1 for r in group if r.invasion_outcome) / max(1, len(group))
        )
        mean_survival = float(
            sum(r.survival_time for r in group) / max(1, len(group))
        )
        resident_hash = str(group[0].resident_strategy_hash)
        gen_metrics_list.append(
            GenerationMetrics(
                generation_index=int(gen_idx),
                invasion_rate=invasion_rate,
                mean_survival_time=mean_survival,
                resident_strategy_hash=resident_hash,
            )
        )

    # --- strategy_metrics ---------------------------------------------------
    # Key: (generation_index, strategy_hash, role)
    strat_groups: dict[tuple[int, str, str], list[Any]] = {}
    for rec in records:
        r_key = (
            int(rec.generation_index),
            str(rec.resident_strategy_hash),
            "resident",
        )
        strat_groups.setdefault(r_key, []).append(rec)
        mutant_hash = str(getattr(rec, "strategy_hash", "")).strip()
        if mutant_hash == "":
            mutant_hash = str(getattr(rec, "mutant_strategy_hash", "")).strip()
        if mutant_hash == "" and str(getattr(rec, "mutant_strategy", "")).strip() != "":
            mutant_hash = strategy_hash_from_serialized_payload(str(rec.mutant_strategy))
        m_key = (
            int(rec.generation_index),
            str(mutant_hash),
            "mutant",
        )
        strat_groups.setdefault(m_key, []).append(rec)

    strat_metrics_list: list[StrategyMetrics] = []
    for key in sorted(strat_groups.keys()):
        gen_idx, strat_hash, role = key
        group = strat_groups[key]
        if role == "mutant":
            invasion_count = int(sum(1 for r in group if r.invasion_outcome))
            mean_survival = float(
                sum(r.survival_time for r in group) / max(1, len(group))
            )
        else:
            invasion_count = 0
            mean_survival = 0.0
        strat_metrics_list.append(
            StrategyMetrics(
                generation_index=int(gen_idx),
                strategy_hash=str(strat_hash),
                role=str(role),
                invasion_count=invasion_count,
                mean_survival_time=mean_survival,
            )
        )

    # --- world_regime_timeline ----------------------------------------------
    regime_groups: dict[tuple[int, str], list[Any]] = {}
    for rec in records:
        r_key = (int(rec.generation_index), str(rec.regime))
        regime_groups.setdefault(r_key, []).append(rec)

    regime_list: list[WorldRegimeEntry] = []
    for key in sorted(regime_groups.keys()):
        gen_idx, regime = key
        group = regime_groups[key]
        invasion_count = int(sum(1 for r in group if r.invasion_outcome))
        invasion_rate = float(invasion_count / max(1, len(group)))
        regime_list.append(
            WorldRegimeEntry(
                generation_index=int(gen_idx),
                regime=str(regime),
                invasion_count=invasion_count,
                invasion_rate=invasion_rate,
            )
        )

    # --- population_statistics ----------------------------------------------
    total_trials = len(records)
    total_invasions = int(sum(1 for r in records if r.invasion_outcome))
    fixation_prob = float(total_invasions / max(1, total_trials))
    mean_survival = float(
        sum(r.survival_time for r in records) / max(1, total_trials)
    )
    pop_stats = PopulationStatistics(
        total_trials=int(total_trials),
        total_invasions=int(total_invasions),
        fixation_probability=fixation_prob,
        mean_survival_time=mean_survival,
        n_resident_generations=int(ess_dataset.n_resident_generations),
        n_mutant_trials=int(ess_dataset.n_mutant_trials),
    )

    return ExperimentDataset(
        schema_version=str(EXPERIMENT_DATASET_SCHEMA_VERSION),
        seed=int(ess_dataset.seed),
        config_hash=str(_config_hash_from_ess_dataset(ess_dataset)),
        generation_metrics=tuple(gen_metrics_list),
        strategy_metrics=tuple(strat_metrics_list),
        world_regime_timeline=tuple(regime_list),
        population_statistics=pop_stats,
    )


# ---------------------------------------------------------------------------
# Serialisation / deserialisation
# ---------------------------------------------------------------------------


def experiment_dataset_to_dict(dataset: ExperimentDataset) -> dict[str, Any]:
    """Serialise an ``ExperimentDataset`` to a plain Python dict.

    All values are JSON-safe Python scalars or nested structures.  Use
    ``experiment_dataset_from_dict`` as the separate deserialisation path
    required by the round-trip invariant.
    """
    return {
        "schema_version": str(dataset.schema_version),
        "seed": int(dataset.seed),
        "config_hash": str(dataset.config_hash),
        "generation_metrics": [
            {
                "generation_index": int(gm.generation_index),
                "invasion_rate": float(gm.invasion_rate),
                "mean_survival_time": float(gm.mean_survival_time),
                "resident_strategy_hash": str(gm.resident_strategy_hash),
            }
            for gm in dataset.generation_metrics
        ],
        "strategy_metrics": [
            {
                "generation_index": int(sm.generation_index),
                "strategy_hash": str(sm.strategy_hash),
                "role": str(sm.role),
                "invasion_count": int(sm.invasion_count),
                "mean_survival_time": float(sm.mean_survival_time),
            }
            for sm in dataset.strategy_metrics
        ],
        "world_regime_timeline": [
            {
                "generation_index": int(we.generation_index),
                "regime": str(we.regime),
                "invasion_count": int(we.invasion_count),
                "invasion_rate": float(we.invasion_rate),
            }
            for we in dataset.world_regime_timeline
        ],
        "population_statistics": {
            "total_trials": int(dataset.population_statistics.total_trials),
            "total_invasions": int(dataset.population_statistics.total_invasions),
            "fixation_probability": float(
                dataset.population_statistics.fixation_probability
            ),
            "mean_survival_time": float(
                dataset.population_statistics.mean_survival_time
            ),
            "n_resident_generations": int(
                dataset.population_statistics.n_resident_generations
            ),
            "n_mutant_trials": int(dataset.population_statistics.n_mutant_trials),
        },
    }


def experiment_dataset_from_dict(data: dict[str, Any]) -> ExperimentDataset:
    """Reconstruct an ``ExperimentDataset`` from a plain Python dict.

    This is the separate deserialisation path required by the round-trip
    invariant.  It does not call the adapter; it restores stored values
    directly.
    """
    gen_metrics = tuple(
        GenerationMetrics(
            generation_index=int(gm["generation_index"]),
            invasion_rate=float(gm["invasion_rate"]),
            mean_survival_time=float(gm["mean_survival_time"]),
            resident_strategy_hash=str(gm["resident_strategy_hash"]),
        )
        for gm in data["generation_metrics"]
    )
    strat_metrics = tuple(
        StrategyMetrics(
            generation_index=int(sm["generation_index"]),
            strategy_hash=str(sm["strategy_hash"]),
            role=str(sm["role"]),
            invasion_count=int(sm["invasion_count"]),
            mean_survival_time=float(sm["mean_survival_time"]),
        )
        for sm in data["strategy_metrics"]
    )
    regime_timeline = tuple(
        WorldRegimeEntry(
            generation_index=int(we["generation_index"]),
            regime=str(we["regime"]),
            invasion_count=int(we["invasion_count"]),
            invasion_rate=float(we["invasion_rate"]),
        )
        for we in data["world_regime_timeline"]
    )
    pop_data = data["population_statistics"]
    pop_stats = PopulationStatistics(
        total_trials=int(pop_data["total_trials"]),
        total_invasions=int(pop_data["total_invasions"]),
        fixation_probability=float(pop_data["fixation_probability"]),
        mean_survival_time=float(pop_data["mean_survival_time"]),
        n_resident_generations=int(pop_data["n_resident_generations"]),
        n_mutant_trials=int(pop_data["n_mutant_trials"]),
    )
    return ExperimentDataset(
        schema_version=str(data["schema_version"]),
        seed=int(data["seed"]),
        config_hash=str(data["config_hash"]),
        generation_metrics=gen_metrics,
        strategy_metrics=strat_metrics,
        world_regime_timeline=regime_timeline,
        population_statistics=pop_stats,
    )
