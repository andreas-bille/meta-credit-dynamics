import numpy as np

from .core import CapitalSelector
from .stats import EWMAStats
from .rebirth import RebirthPolicy
from .reweight import exp_reweight
from .config import ProfileAConfig, ProfileBConfig
from .genome import SelectorGenome
from .genome_validation import validate_selector_genome
from .phase_i_state import DEFAULT_LAMBDA_RISK, validate_lambda_risk
from .selector_policy import DEFAULT_SELECTOR_POLICY, SelectorPolicy, validate_selector_policy


class CapitalSelectorBuilder:
    def __init__(self):
        self._wealth = 1.0
        self._rebirth_threshold = 0.5
        # fixed hyperparameters (not part of genome)
        # intentionally constant for v0.9.x series
        # TODO(v1.0.0): GenomeHyperParams
        self._beta = 0.01
        self._eta = 1.0
        self._kind = "entrepreneur"
        self._rebirth_policy = None
        self._K: int = 0
        self._selector_policy: SelectorPolicy = DEFAULT_SELECTOR_POLICY
        self._lambda_risk: float = DEFAULT_LAMBDA_RISK
        self._resolved_config: dict[str, object] = {}

    @classmethod
    def from_profile(cls, profile: ProfileAConfig | ProfileBConfig):
        profile.validate_closed()
        builder = cls()
        builder._resolved_config = {
            "dt": profile.dt,
            "cost_distribution": profile.cost_distribution,
            "score_mode": profile.score_mode,
            "stats_signal": profile.stats_signal,
            "stack_weighting": profile.stack_weighting,
            "freeze_stats": profile.freeze_stats,
            "credit_condition_active": profile.credit_condition_active,
            "sparsity_active": profile.sparsity_active,
            "rebirth_pool_active": profile.rebirth_pool_active,
        }
        return builder

    def with_initial_wealth(self, w: float):
        self._wealth = float(w); return self

    def with_rebirth_threshold(self, t: float):
        self._rebirth_threshold = float(t); return self

    def with_stats(self, beta: float):
        self._beta = float(beta); return self

    def with_reweight_eta(self, eta: float):
        self._eta = float(eta); return self

    def with_kind(self, kind: str):
        self._kind = kind; return self

    def with_rebirth_policy(self, policy: RebirthPolicy):
        self._rebirth_policy = policy; return self

    def with_selector_policy(self, selector_policy: SelectorPolicy | str):
        self._selector_policy = validate_selector_policy(str(selector_policy)); return self

    def with_lambda_risk(self, lambda_risk: float):
        self._lambda_risk = validate_lambda_risk(lambda_risk); return self

    def with_K(self, K: int):
        """Setzt die Simplex-Dimension K des Selectors."""
        K = int(K)
        if K < 0:
            raise ValueError("K must be >= 0")
        self._K = K
        return self

    def build(self) -> CapitalSelector:
        stats = EWMAStats(beta=self._beta, seed_var=1.0)

        def reweight(w, adv):
            return exp_reweight(w, adv, self._eta)

        return CapitalSelector(
            wealth=self._wealth,
            rebirth_threshold=self._rebirth_threshold,
            stats=stats,
            reweight_fn=reweight,
            kind=self._kind,
            rebirth_policy=self._rebirth_policy,
            K=self._K,
            selector_policy=self._selector_policy,
            lambda_risk=self._lambda_risk,
        )


def _decode_genome_settlement_params(settlement_params: dict[str, float]) -> dict[str, float | bool | int]:
    decoded: dict[str, float | bool | int] = {}
    for key, raw in settlement_params.items():
        value = float(raw)
        if key == "accept_by_default":
            decoded[key] = bool(value > 0.0)
        elif key == "future_maturity_offset":
            decoded[key] = int(round(value))
        else:
            decoded[key] = value
    return decoded


def build_selector_from_genome(
    genome: SelectorGenome,
    *,
    seed: int | None = None,
    process_id: int = 0,
    generation_id: int = 0,
    initial_wealth: float = 1.0,
    rebirth_threshold: float = 0.5,
    kind: str = "entrepreneur",
) -> CapitalSelector:
    _ = seed

    try:
        validate_selector_genome(genome)

        flow_matrix = np.asarray(genome.flow_matrix, dtype=float).copy()
        output_weights = np.asarray(genome.output_weights, dtype=float).copy()
        n_inputs = int(flow_matrix.shape[0])
        m_outputs = int(flow_matrix.shape[1])

        selector = (
            CapitalSelectorBuilder()
            .with_K(n_inputs)
            .with_kind(str(kind))
            .with_selector_policy(str(genome.selector_policy))
            .with_lambda_risk(float(genome.lambda_risk))
            .with_initial_wealth(float(initial_wealth))
            .with_rebirth_threshold(float(rebirth_threshold))
            .build()
        )

        selector.process_id = int(process_id)
        selector.generation_id = int(generation_id)
        selector.flow_matrix = flow_matrix
        selector.output_weights = output_weights
        selector.settlement_config = _decode_genome_settlement_params(dict(genome.settlement_params))

        if selector.flow_matrix.shape != (n_inputs, m_outputs):
            raise ValueError("dimension mismatch after build")
        if np.asarray(selector.output_weights, dtype=float).shape != (m_outputs,):
            raise ValueError("output shape mismatch after build")

        return selector
    except Exception as exc:
        raise ValueError("invalid genome passed to builder") from exc
