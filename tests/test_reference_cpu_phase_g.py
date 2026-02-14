from __future__ import annotations

import os
import random
from typing import Dict, Any

import numpy as np
import pytest
import torch

from capitalmarket.capitalselector.runtime import RuntimeConfig, run
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import RegimeSwitchBanditWorld


REFERENCE_VALUES: Dict[str, Any] = {
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


def _compute_reference() -> Dict[str, Any]:
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    world = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=seed)
    result = run(world=world, steps=500, config=RuntimeConfig(profile="A"))
    # Using runtime history state dict (wealth/weights/cum_pi/dd) as reference snapshot.
    history = result["history"]
    assert history, "runtime.run() returned empty history"
    last = history[-1]

    weights = np.asarray(last["weights"], dtype=float)
    top2_share = float(np.sort(weights)[-2:].sum())

    return {
        "final_wealth": float(last["wealth"]),
        "final_weights": weights,
        "final_cum_pi": float(last["cum_pi"]),
        "final_drawdown": float(last["dd"]),
        "final_top2_share": top2_share,
    }


def test_reference_cpu_phase_g(request):
    ref = _compute_reference()

    if request.config.getoption("--regenerate"):
        print("\nREFERENCE_VALUES = {")
        print(f"    'final_wealth': {ref['final_wealth']},")
        print(f"    'final_weights': {ref['final_weights'].tolist()},")
        print(f"    'final_cum_pi': {ref['final_cum_pi']},")
        print(f"    'final_drawdown': {ref['final_drawdown']},")
        print(f"    'final_top2_share': {ref['final_top2_share']},")
        print("}")
        pytest.skip("Reference values regenerated. Copy into REFERENCE_VALUES.")

    for key in ("final_wealth", "final_cum_pi", "final_drawdown", "final_top2_share"):
        assert REFERENCE_VALUES[key] is not None, f"REFERENCE_VALUES['{key}'] not set"

    assert REFERENCE_VALUES["final_weights"] is not None, "REFERENCE_VALUES['final_weights'] not set"

    np.testing.assert_allclose(
        ref["final_wealth"],
        REFERENCE_VALUES["final_wealth"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        ref["final_weights"],
        np.asarray(REFERENCE_VALUES["final_weights"], dtype=float),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        ref["final_cum_pi"],
        REFERENCE_VALUES["final_cum_pi"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        ref["final_drawdown"],
        REFERENCE_VALUES["final_drawdown"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        ref["final_top2_share"],
        REFERENCE_VALUES["final_top2_share"],
        rtol=0.0,
        atol=1e-15,
    )
