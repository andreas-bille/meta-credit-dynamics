from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SelectorGenome:
    flow_matrix: np.ndarray
    output_weights: np.ndarray
    lambda_risk: float
    selector_policy: str
    settlement_params: dict[str, float]

    def __post_init__(self) -> None:
        flow_matrix = np.asarray(self.flow_matrix, dtype=float).copy()
        output_weights = np.asarray(self.output_weights, dtype=float).copy()
        flow_matrix.setflags(write=False)
        output_weights.setflags(write=False)

        object.__setattr__(self, "flow_matrix", flow_matrix)
        object.__setattr__(self, "output_weights", output_weights)
        object.__setattr__(self, "lambda_risk", float(self.lambda_risk))
        object.__setattr__(self, "selector_policy", str(self.selector_policy))

        settlement_params = self.settlement_params
        if isinstance(settlement_params, dict):
            settlement_params = dict(settlement_params)
        object.__setattr__(
            self,
            "settlement_params",
            settlement_params,
        )

    @property
    def n_inputs(self) -> int:
        return int(self.flow_matrix.shape[0])

    @property
    def m_outputs(self) -> int:
        return int(self.flow_matrix.shape[1])
