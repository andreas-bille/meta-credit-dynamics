from __future__ import annotations

import os
import random

import numpy as np
import torch

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import RegimeSwitchBanditWorld
from capitalmarket.capitalselector.cuda_state import to_device_state


def _build_selector(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"

    world = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=seed)
    selector = CapitalSelectorBuilder().with_K(0).build()

    out = world.step(0)
    r_vec, c_total = validate_world_output(out)
    if selector.w is None or len(selector.w) != len(r_vec):
        selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
        selector.K = len(r_vec)
    selector.feedback_vector(r_vec, c_total, trace=None, freeze=False)
    return selector


def test_device_state_shapes_and_dtype():
    selector = _build_selector(seed=0)
    state = to_device_state(selector)

    assert state.weights.shape == (1, selector.K)
    for t in (
        state.wealth,
        state.mean,
        state.var,
        state.drawdown,
        state.cum_pi,
        state.peak_cum_pi,
        state.rebirth_threshold,
    ):
        assert t.shape == (1, 1)

    assert state.weights.dtype == torch.float64
    assert state.wealth.dtype == torch.float64
    assert state.weights.device.type == "cpu"


def test_device_state_roundtrip_parity():
    selector = _build_selector(seed=0)
    state = to_device_state(selector)

    weights = state.weights.squeeze(0).cpu().numpy()
    wealth = float(state.wealth.item())
    mean = float(state.mean.item())
    var = float(state.var.item())
    cum_pi = float(state.cum_pi.item())
    peak_cum_pi = float(state.peak_cum_pi.item())
    drawdown = float(state.drawdown.item())
    rebirth_threshold = float(state.rebirth_threshold.item())

    np.testing.assert_allclose(weights, selector.w, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(wealth, selector.wealth, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(mean, selector.stats.mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(var, selector.stats.var, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(cum_pi, selector.stats.cum_pi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(peak_cum_pi, selector.stats.peak_cum_pi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(drawdown, selector.stats.dd, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(rebirth_threshold, selector.rebirth_threshold, rtol=0.0, atol=0.0)
