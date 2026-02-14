from __future__ import annotations

import os
import random

import numpy as np
import torch

from capitalmarket.capitalselector.runtime import run, RuntimeConfig
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import RegimeSwitchBanditWorld
REFERENCE_VALUES = {
    "final_wealth": 10.34759917461935,
    "final_weights": [
        0.6817048428403422,
        0.0032831251588185577,
        0.002067683558154122,
        0.3101920350289068,
        0.0027523134137782445,
    ],
    "final_cum_pi": 9.347599174619349,
    "final_drawdown": 0.0,
    "final_top2_share": 0.991896877869249,
}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def test_backend_cpu_parity_against_reference():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"
    os.environ["CAPM_DEVICE"] = "cpu"

    seed = 0
    _seed_all(seed)

    world = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=seed)
    result = run(world=world, steps=500, config=RuntimeConfig(profile="A"))
    history = result["history"]
    assert history, "runtime.run() returned empty history"
    last = history[-1]

    weights = np.asarray(last["weights"], dtype=float)

    np.testing.assert_allclose(last["wealth"], REFERENCE_VALUES["final_wealth"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights, np.asarray(REFERENCE_VALUES["final_weights"], dtype=float), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(last["cum_pi"], REFERENCE_VALUES["final_cum_pi"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(last["dd"], REFERENCE_VALUES["final_drawdown"], rtol=0.0, atol=0.0)


def test_backend_selects_cpu():
    os.environ["CAPM_DEVICE"] = "cpu"
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"

    seed = 0
    _seed_all(seed)

    world = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=seed)
    result = run(world=world, steps=10, config=RuntimeConfig(profile="A"))
    assert result["history"], "CPU backend run produced empty history"
