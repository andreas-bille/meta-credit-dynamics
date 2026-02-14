from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.experiments.g3_4_6_1_asym_drift import run_g3_4_6_1_sweep
from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.stack import StackManager
from capitalmarket.capitalselector.cuda_state import canonical_state_dump
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import (
    NonStationaryVolatilityBanditWorld,
    _generate_regime_sequence,
)


def _assert_dumps_equal(a, b, rtol=1e-12, atol=1e-12):
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_dumps_equal(va, vb, rtol=rtol, atol=atol)
        elif isinstance(va, np.ndarray):
            np.testing.assert_allclose(va, vb, rtol=rtol, atol=atol)
        elif isinstance(va, float):
            np.testing.assert_allclose(va, vb, rtol=rtol, atol=atol)
        else:
            assert va == vb


def test_g3_4_6_1_determinism():
    p_values = [0.01, 0.05]
    modes = ["stationary", "asym_drift"]
    seeds = [0, 1, 2, 3, 4]
    results1 = run_g3_4_6_1_sweep(p_values, modes, seeds, steps=500)
    results2 = run_g3_4_6_1_sweep(p_values, modes, seeds, steps=500)
    assert len(results1) == len(results2) == 20
    for r1, r2 in zip(results1, results2):
        assert r1["p"] == r2["p"]
        assert r1["volatility_mode"] == r2["volatility_mode"]
        assert r1["seed"] == r2["seed"]
        assert r1["result"]["observables"] == r2["result"]["observables"]
        _assert_dumps_equal(r1["result"]["dumps"][0], r2["result"]["dumps"][0])
        _assert_dumps_equal(r1["result"]["dumps"][500], r2["result"]["dumps"][500])


def test_g3_4_6_1_asym_drift_sanity():
    p = 0.05
    seed = 1
    steps = 500
    regime_sequence = _generate_regime_sequence(p=p, seed=seed, length=steps)
    world = NonStationaryVolatilityBanditWorld(
        p=p,
        sigma_stationary=0.01,
        sigma_low=0.005,
        sigma_high=0.02,
        volatility_mode="asym_drift",
        seed=seed,
        horizon=steps,
        regime_sequence=regime_sequence,
    )
    r_seq = []
    for t in range(steps):
        out = world.step(t)
        r_vec, _ = validate_world_output(out)
        r_seq.append(np.asarray(r_vec, dtype=float))

    r_arr = np.asarray(r_seq, dtype=float)
    means = np.zeros_like(r_arr)
    active_idx = np.zeros(steps, dtype=int)
    for t, regime in enumerate(regime_sequence):
        if regime == "A":
            means[t] = np.array([0.02, 0.0, 0.0, 0.0, 0.0], dtype=float)
            active_idx[t] = 0
        else:
            means[t] = np.array([0.0, 0.0, 0.0, 0.02, 0.0], dtype=float)
            active_idx[t] = 3
    residuals = r_arr - means

    early = slice(0, 100)
    late = slice(400, None)
    neutral_idx = [1, 2, 4]

    neutral_early = residuals[early][:, neutral_idx].ravel()
    neutral_late = residuals[late][:, neutral_idx].ravel()
    var_neutral_early = float(np.var(neutral_early))
    var_neutral_late = float(np.var(neutral_late))
    assert var_neutral_late > var_neutral_early * (1.0 - 1e-2)

    active_early = np.array([residuals[t, active_idx[t]] for t in range(0, 100)], dtype=float)
    active_late = np.array([residuals[t, active_idx[t]] for t in range(400, steps)], dtype=float)
    var_active_early = float(np.var(active_early))
    var_active_late = float(np.var(active_late))
    np.testing.assert_allclose(var_active_late, var_active_early, rtol=1e-2, atol=1e-3)


def test_g3_4_6_1_no_semantic_mutation():
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="regime", phase_id="G3_4_6_1", run_id="run_0")
    dump_before = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    world = NonStationaryVolatilityBanditWorld(
        p=0.05,
        sigma_stationary=0.01,
        sigma_low=0.005,
        sigma_high=0.02,
        volatility_mode="asym_drift",
        seed=1,
        horizon=500,
    )
    out = world.step(0)
    r_vec, c_total = validate_world_output(out)
    if selector.w is None or len(selector.w) != len(r_vec):
        selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
        selector.K = len(r_vec)
    selector.feedback_vector(r_vec, c_total, trace=None, freeze=False)

    _ = {
        "wealth": float(selector.wealth),
        "dominant_channel": int(np.argmax(selector.w)) if selector.w is not None else None,
        "stack_count": len(stack_manager.stacks),
        "rebirth_count": 0,
    }

    dump_after = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)
    assert dump_before != dump_after
    dump_after_2 = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)
    assert dump_after == dump_after_2
