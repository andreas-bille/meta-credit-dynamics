from __future__ import annotations

from typing import Any
from typing import Mapping

import numpy as np

from .genome import SelectorGenome
from .genome_validation import validate_selector_genome


def genome_to_dict(genome: SelectorGenome) -> dict[str, Any]:
    validate_selector_genome(genome)

    return {
        "flow_matrix": np.asarray(genome.flow_matrix, dtype=float).tolist(),
        "output_weights": np.asarray(genome.output_weights, dtype=float).tolist(),
        "lambda_risk": float(genome.lambda_risk),
        "selector_policy": str(genome.selector_policy),
        "settlement_params": {
            str(key): float(value)
            for key, value in sorted(genome.settlement_params.items(), key=lambda item: item[0])
        },
    }


def genome_from_dict(payload: Mapping[str, Any]) -> SelectorGenome:
    try:
        if not isinstance(payload, Mapping):
            raise ValueError("invalid payload")

        required = {
            "flow_matrix",
            "output_weights",
            "lambda_risk",
            "selector_policy",
            "settlement_params",
        }
        if set(payload.keys()) != required:
            raise ValueError("invalid payload keys")

        settlement_params_raw = payload["settlement_params"]
        if not isinstance(settlement_params_raw, Mapping):
            raise ValueError("invalid settlement params payload")

        genome = SelectorGenome(
            flow_matrix=np.asarray(payload["flow_matrix"], dtype=float),
            output_weights=np.asarray(payload["output_weights"], dtype=float),
            lambda_risk=float(payload["lambda_risk"]),
            selector_policy=str(payload["selector_policy"]),
            settlement_params={str(k): float(v) for k, v in settlement_params_raw.items()},
        )
        validate_selector_genome(genome)
        return genome
    except Exception:
        raise ValueError("invalid genome payload")
