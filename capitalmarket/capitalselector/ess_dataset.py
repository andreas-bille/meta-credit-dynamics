"""ESS Dataset Export (B2).

Builds a structured, deterministic ESSDataset from a completed ESS probe
experiment.  Mutant genomes are re-derived from the same seeds used during
the original probe so that two independent exports of an identical
ESSProbeResult produce identical datasets.

No renderer or visualisation library is imported here.

Configuration consolidation contract
----------------------------------
This module documents and uses the shared configuration base contract across
ESS probe, regime robustness, and long-run harness paths:

* ``seed``
* ``world_parameters`` (or ``world_parameters_by_regime``)
* ``backend``

Canonical strategy serialization and strategy-hash helpers are single-source in
this module and are reused by adapter consumers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .ess_evaluator import (
    ESS_MUTANT_PATH_CANONICAL,
    ESS_MUTANT_PATH_SCALED_OVERRIDE,
    ESSProbeConfig,
    ESSProbeResult,
    _base_mutation_config,
    _build_mutant_genome,
    _effective_mutation_config,
    _splitmix64,
    _stable_text_seed,
    resolve_probe_mutation_scaling_mode,
)
from .genome import SelectorGenome
from .genome_serialization import genome_from_dict
from .genome_serialization import genome_to_dict
from .mutation_scaling import MutationScalingConfig

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

ESS_DATASET_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "generation_index",
        "trial_index",
        "regime",
        "invasion_outcome",
        "survival_time",
        "resident_strategy",
        "mutant_strategy",
        "strategy_hash",
        "resident_strategy_hash",
        "mutant_strategy_hash",
    }
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strategy_payload(genome: SelectorGenome) -> dict[str, Any]:
    return dict(genome_to_dict(genome))


def _strategy_json(payload: dict[str, Any]) -> str:
    """Canonical JSON serialisation used for hash computation and storage."""
    return canonical_strategy_serialization(payload)


def _strategy_hash(payload: dict[str, Any]) -> str:
    """SHA-256 hash (first 16 hex chars) of the canonical JSON serialisation.

    Uses the same algorithm as ``_genome_hash`` in ``ess_evaluator`` so that
    ``mutant_strategy_hash`` in ``ESSDatasetRecord`` matches
    ``ESSProbeTrialResult.mutant_hash`` for the same genome.
    """
    return canonical_strategy_hash(payload)


def canonical_strategy_serialization(payload: Mapping[str, Any]) -> str:
    """Canonical JSON serialisation for strategy payloads.

    Uses ``sort_keys=True`` to make output independent of key insertion order.
    """
    return json.dumps(dict(payload), sort_keys=True)


def canonical_strategy_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 hash (first 16 hex chars) over canonical strategy serialisation."""
    canonical = canonical_strategy_serialization(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def strategy_hash_from_serialized_payload(strategy_payload: str) -> str:
    """Compute canonical strategy hash from serialized strategy payload JSON."""
    parsed = json.loads(str(strategy_payload))
    if not isinstance(parsed, Mapping):
        raise ValueError("invalid strategy payload for hashing")
    return canonical_strategy_hash(parsed)


def _canonical_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_json_value(item) for item in value]
    raise TypeError("invalid ESS probe configuration payload")


def _ess_probe_config_payload(config: ESSProbeConfig) -> str:
    assert config.seed is not None
    assert config.world_parameters_by_regime is not None
    assert config.regimes is not None
    assert config.n_resident_generations is not None
    assert config.n_mutant_trials is not None
    assert config.runtime_horizon is not None

    payload = {
        "seed": int(config.seed),
        "world_parameters_by_regime": _canonical_json_value(config.world_parameters_by_regime),
        "regimes": [str(item) for item in config.regimes],
        "n_resident_generations": int(config.n_resident_generations),
        "n_mutant_trials": int(config.n_mutant_trials),
        "runtime_horizon": int(config.runtime_horizon),
        "resident_population_size": int(config.resident_population_size),
        "mutant_population_size": int(config.mutant_population_size),
        "dt": float(config.dt),
        "backend": str(config.backend).strip().lower(),
        "min_survival_tau": int(config.min_survival_tau),
        "mutation_scaling_mode": str(resolve_probe_mutation_scaling_mode(config.mutation_scaling_mode)),
        "mutation_scaling_decay_rate": float(config.mutation_scaling_decay_rate),
        "mutant_generation_path": str(config.mutant_generation_path),
        "mutation_noise_scale": float(config.mutation_noise_scale),
        "mutation_redistribution_share": float(config.mutation_redistribution_share),
        "mutation_lambda_risk_scale": float(config.mutation_lambda_risk_scale),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ESSDatasetRecord:
    """One row in an ESSDataset — one (regime, trial) pair.

    Fields
    ------
    generation_index
        Resident training generation at which the probe was launched
        (equals ``ESSProbeResult.n_resident_generations``).
    trial_index
        Zero-based index of the mutant trial within the regime.
    regime
        Name of the world-parameter regime for this trial.
    invasion_outcome
        True iff the mutant lineage survived for >= ``min_survival_tau`` steps.
    survival_time
        Number of generations the mutant lineage persisted before extinction.
    resident_strategy
        Canonical JSON serialisation of the resident genome payload.
    mutant_strategy
        Canonical JSON serialisation of the mutant genome payload.
    strategy_hash
        Canonical strategy hash for the trial strategy payload. This field is
        always retained even when full payload export is disabled.
        For current ESS probe exports this equals ``mutant_strategy_hash``.
    resident_strategy_hash
        SHA-256 (16 hex chars) of ``resident_strategy``.
    mutant_strategy_hash
        SHA-256 (16 hex chars) of ``mutant_strategy``; matches
        ``ESSProbeTrialResult.mutant_hash`` for the same genome.
    """

    generation_index: int
    trial_index: int
    regime: str
    invasion_outcome: bool
    survival_time: int
    resident_strategy: str
    mutant_strategy: str
    resident_strategy_hash: str
    mutant_strategy_hash: str
    strategy_hash: str = ""


@dataclass(frozen=True)
class ESSDatasetConfig:
    """Export-time configuration for ESSDataset materialization.

    full_strategy_payload
        When ``True`` (default), ``ess_dataset_to_records`` includes
        ``mutant_strategy`` payload in serialized records. When ``False``, the
        payload key is omitted but ``strategy_hash`` is always emitted.
    """

    full_strategy_payload: bool = True


@dataclass(frozen=True)
class ESSDataset:
    """Structured, deterministic dataset produced by an ESS probe experiment.

    Records are ordered by ``(generation_index, trial_index, regime)``.

    ``config_payload`` stores the canonical JSON of the full ESS probe
    configuration for v0.9.5+ exports. Legacy v0.9.4 serialized records do
    not carry this field and therefore deserialize with ``config_payload=None``.
    """

    seed: int
    n_resident_generations: int
    n_mutant_trials: int
    records: tuple[ESSDatasetRecord, ...]
    config_payload: str | None = None
    full_strategy_payload: bool = True


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_ess_dataset(
    resident_genome: SelectorGenome,
    config: ESSProbeConfig,
    probe_result: ESSProbeResult,
    dataset_config: ESSDatasetConfig | None = None,
) -> ESSDataset:
    """Build an ESSDataset from a completed probe experiment.

    Mutant genomes are re-derived deterministically using the same seed logic
    as ``run_ess_probe_experiment``.  The result is therefore identical for
    any two independent calls that receive the same (resident_genome, config,
    probe_result) triple.

    Parameters
    ----------
    resident_genome:
        The resident strategy that was used in the original probe.
    config:
        The probe configuration (must be the same as for *probe_result*).
    probe_result:
        The completed probe result providing invasion outcomes and metrics.
    dataset_config:
        Optional export behavior configuration. Defaults preserve legacy
        behavior (full strategy payload included).
    """
    assert config.seed is not None
    assert config.regimes is not None
    assert config.n_mutant_trials is not None
    assert config.n_resident_generations is not None

    seed = int(config.seed)
    regimes = [str(r) for r in config.regimes]
    n_resident_generations = int(probe_result.n_resident_generations)
    n_mutant_trials = int(probe_result.n_mutant_trials)
    effective_dataset_config = ESSDatasetConfig() if dataset_config is None else dataset_config

    scaling_mode = resolve_probe_mutation_scaling_mode(config.mutation_scaling_mode)
    scaling_config = MutationScalingConfig(
        mode="generation_dependent",
        decay_rate=float(config.mutation_scaling_decay_rate),
    )
    base_mutation = _base_mutation_config(config)

    # Structural freeze of resident (same as in run_ess_probe_experiment).
    resident_frozen = genome_from_dict(genome_to_dict(resident_genome))
    resident_payload = _strategy_payload(resident_frozen)
    resident_json = _strategy_json(resident_payload)
    resident_hash = _strategy_hash(resident_payload)

    # Build lookup from (regime, trial_index) -> ESSProbeTrialResult.
    trial_lookup = {
        (str(item.regime), int(item.trial_index)): item
        for item in probe_result.trial_results
    }

    records: list[ESSDatasetRecord] = []

    for regime in regimes:
        regime_seed = _stable_text_seed(regime)

        for trial_index in range(n_mutant_trials):
            mutation_config = _effective_mutation_config(
                base_config=base_mutation,
                scaling_mode=scaling_mode,
                scaling_config=scaling_config,
                n_resident_generations=n_resident_generations,
                trial_index=int(trial_index),
            )
            mutation_seed = _splitmix64(
                seed
                ^ regime_seed
                ^ (int(trial_index + 1) << 32)
                ^ 0x3333333333333333
            )
            mutant = _build_mutant_genome(
                resident_genome=resident_frozen,
                mutation_seed=int(mutation_seed),
                mutation_config=mutation_config,
                mutant_generation_path=str(config.mutant_generation_path),
            )
            mutant_payload = _strategy_payload(mutant)
            mutant_json = _strategy_json(mutant_payload)
            mutant_hash = _strategy_hash(mutant_payload)

            trial_item = trial_lookup.get((regime, trial_index))
            if trial_item is None:
                raise ValueError(
                    f"missing trial result for regime={regime!r}, trial_index={trial_index}"
                )

            records.append(
                ESSDatasetRecord(
                    generation_index=int(n_resident_generations),
                    trial_index=int(trial_index),
                    regime=str(regime),
                    invasion_outcome=bool(trial_item.invasion_success),
                    survival_time=int(trial_item.survival_time),
                    resident_strategy=str(resident_json),
                    mutant_strategy=(
                        str(mutant_json)
                        if bool(effective_dataset_config.full_strategy_payload)
                        else ""
                    ),
                    strategy_hash=str(mutant_hash),
                    resident_strategy_hash=str(resident_hash),
                    mutant_strategy_hash=str(mutant_hash),
                )
            )

    # Explicit stable sort by (generation_index, trial_index, regime).
    sorted_records = tuple(
        sorted(records, key=lambda r: (r.generation_index, r.trial_index, r.regime))
    )

    return ESSDataset(
        seed=int(seed),
        n_resident_generations=int(n_resident_generations),
        n_mutant_trials=int(n_mutant_trials),
        records=sorted_records,
        config_payload=_ess_probe_config_payload(config),
        full_strategy_payload=bool(effective_dataset_config.full_strategy_payload),
    )


# ---------------------------------------------------------------------------
# Serialisation / deserialisation
# ---------------------------------------------------------------------------


def ess_dataset_to_records(dataset: ESSDataset) -> list[dict[str, Any]]:
    """Serialise an ESSDataset to a list of flat record dicts.

    Each dict contains all dataset-level fields plus all record fields so
    that individual rows are self-contained for downstream consumers.
    """
    rows: list[dict[str, Any]] = []
    for rec in dataset.records:
        row: dict[str, Any] = {
            "seed": int(dataset.seed),
            "n_resident_generations": int(dataset.n_resident_generations),
            "n_mutant_trials": int(dataset.n_mutant_trials),
            "config_payload": None if dataset.config_payload is None else str(dataset.config_payload),
            "generation_index": int(rec.generation_index),
            "trial_index": int(rec.trial_index),
            "regime": str(rec.regime),
            "invasion_outcome": bool(rec.invasion_outcome),
            "survival_time": int(rec.survival_time),
            "resident_strategy": str(rec.resident_strategy),
            "strategy_hash": str(rec.strategy_hash) if str(rec.strategy_hash) else str(rec.mutant_strategy_hash),
            "resident_strategy_hash": str(rec.resident_strategy_hash),
            "mutant_strategy_hash": str(rec.mutant_strategy_hash),
        }
        if bool(dataset.full_strategy_payload):
            row["mutant_strategy"] = str(rec.mutant_strategy)
        rows.append(row)
    return rows


def ess_dataset_from_records(records: list[dict[str, Any]]) -> ESSDataset:
    """Reconstruct an ESSDataset from a list of flat record dicts.

    This is the separate deserialisation path required by the round-trip
    invariant.  It does *not* re-derive genomes from seeds; it restores the
    stored string representations directly.
    """
    if not records:
        raise ValueError("cannot deserialise empty records list")

    first = records[0]
    seed = int(first["seed"])
    n_resident_generations = int(first["n_resident_generations"])
    n_mutant_trials = int(first["n_mutant_trials"])
    config_payload = None if first.get("config_payload") is None else str(first["config_payload"])
    full_strategy_payload = "mutant_strategy" in first

    reconstructed: list[ESSDatasetRecord] = []
    for row in records:
        row_config_payload = None if row.get("config_payload") is None else str(row["config_payload"])
        if row_config_payload != config_payload:
            raise ValueError("invalid ESS dataset records")

        row_has_mutant_payload = "mutant_strategy" in row
        if row_has_mutant_payload != full_strategy_payload:
            raise ValueError("invalid ESS dataset records")

        resident_strategy = str(row["resident_strategy"])
        mutant_strategy = str(row.get("mutant_strategy", ""))

        strategy_hash_raw = row.get("strategy_hash", None)
        if strategy_hash_raw is None or str(strategy_hash_raw) == "":
            strategy_hash_raw = row.get("mutant_strategy_hash", None)
        if (strategy_hash_raw is None or str(strategy_hash_raw) == "") and mutant_strategy != "":
            strategy_hash_raw = strategy_hash_from_serialized_payload(mutant_strategy)
        if strategy_hash_raw is None or str(strategy_hash_raw) == "":
            raise ValueError("invalid ESS dataset records")

        reconstructed.append(
            ESSDatasetRecord(
                generation_index=int(row["generation_index"]),
                trial_index=int(row["trial_index"]),
                regime=str(row["regime"]),
                invasion_outcome=bool(row["invasion_outcome"]),
                survival_time=int(row["survival_time"]),
                resident_strategy=resident_strategy,
                mutant_strategy=mutant_strategy,
                strategy_hash=str(strategy_hash_raw),
                resident_strategy_hash=str(row["resident_strategy_hash"]),
                mutant_strategy_hash=str(row["mutant_strategy_hash"]),
            )
        )

    return ESSDataset(
        seed=seed,
        n_resident_generations=n_resident_generations,
        n_mutant_trials=n_mutant_trials,
        records=tuple(reconstructed),
        config_payload=config_payload,
        full_strategy_payload=full_strategy_payload,
    )
