from dataclasses import dataclass, fields


def _validate_closed_bundle(cfg) -> None:
    for field in fields(cfg):
        if getattr(cfg, field.name) is None:
            raise ValueError(f"Missing required config field: {field.name}")


@dataclass(frozen=True)
class ProfileAConfig:
    """Profile A (Canonical Inhibition Mode) defaults.

    This is a single source of truth for v1/Prod-v0 configuration.
    """

    # D1: time discretization (internal)
    dt: float = 1.0

    # D3: cost distribution
    cost_distribution: str = "proportional"

    # D5: score definition
    score_mode: str = "net_contrib_minus_mu"

    # D7: statistics signal
    stats_signal: str = "pi"

    # D8: stack weighting
    stack_weighting: str = "equal"

    # D11: freeze semantics
    freeze_stats: bool = True

    # D13: credit condition (inactive in v1)
    credit_condition_active: bool = False

    # D14: sparsity (inactive in v1)
    sparsity_active: bool = False

    # D15: rebirth pool (inactive in v1)
    rebirth_pool_active: bool = False

    def validate_closed(self) -> None:
        _validate_closed_bundle(self)


@dataclass(frozen=True)
class ProfileBConfig:
    """Profile B (Non-canonical, analytical) defaults.

    Profile B is explicitly non-canonical and must be used only for analysis.
    """

    # D1: time discretization (internal)
    dt: float = 1.0

    # D3: cost distribution
    cost_distribution: str = "proportional"

    # D5: score definition
    score_mode: str = "net_contrib_minus_mu"

    # D7: statistics signal
    stats_signal: str = "pi"

    # D8: stack weighting
    stack_weighting: str = "equal"

    # D11: freeze semantics (Profile B may allow stats to continue)
    freeze_stats: bool = False

    # D13: credit condition (inactive in v1)
    credit_condition_active: bool = False

    # D14: sparsity (inactive in v1)
    sparsity_active: bool = False

    # D15: rebirth pool (inactive in v1)
    rebirth_pool_active: bool = False

    def validate_closed(self) -> None:
        _validate_closed_bundle(self)
