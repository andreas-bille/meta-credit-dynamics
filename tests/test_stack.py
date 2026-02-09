import numpy as np

from capitalmarket.capitalselector.channels import DeterministicExplorer
from capitalmarket.capitalselector.stack import StackChannel, StackConfig


def test_stack_equal_weighting_returns_average():
    members = {
        "a": DeterministicExplorer(r=1.0, c=0.0),
        "b": DeterministicExplorer(r=2.0, c=0.0),
    }
    stack = StackChannel(members, cfg=StackConfig(C_agg=0.0))
    r, c, alive, dt = stack.step(1.0)
    assert abs(r - 1.5) < 1e-12
    assert abs(c - 0.0) < 1e-12
    assert alive


def test_stack_stability_thresholds_respected():
    members = {
        "a": DeterministicExplorer(r=0.1, c=0.0),
        "b": DeterministicExplorer(r=0.1, c=0.0),
    }
    cfg = StackConfig(C_agg=0.0, tau_mu=0.05, tau_vol=1.0, tau_dd=1.0, use_cvar=False)
    stack = StackChannel(members, cfg=cfg)
    for _ in range(30):
        stack.step(1.0)
    assert stack.stable()

    cfg2 = StackConfig(C_agg=0.0, tau_mu=1.0, tau_vol=1.0, tau_dd=1.0, use_cvar=False)
    stack2 = StackChannel(members, cfg=cfg2)
    for _ in range(10):
        stack2.step(1.0)
    assert not stack2.stable()
