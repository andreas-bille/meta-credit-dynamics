from .core import CapitalSelector, Channel
from .channels import DummyChannel
from .stats import EWMAStats
from .rebirth import RebirthPolicy
from .reweight import exp_reweight


class CapitalSelectorBuilder:
    def __init__(self):
        self._wealth = 1.0
        self._rebirth_threshold = 0.5
        self._beta = 0.01
        self._eta = 1.0
        self._kind = "entrepreneur"
        self._rebirth_policy = None
        self._channels: list[Channel] = []

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

    def with_channels(self, channels: list[Channel]):
        self._channels = channels; return self

    def with_K(self, K: int):
        """Setzt die Simplex-Dimension über Dummy-Kanäle ohne Semantik."""
        K = int(K)
        if K < 0:
            raise ValueError("K must be >= 0")
        self._channels = [DummyChannel() for _ in range(K)]
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
            channels=self._channels,
        )
