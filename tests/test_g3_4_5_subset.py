from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.experiments.g3_4_5_subset import run_g3_4_5_sweep
from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.stack import StackManager
from capitalmarket.capitalselector.cuda_state import canonical_state_dump
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import SubsetRegimeBanditWorld


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


def test_g3_4_5_determinism():
    p_values = [0.01, 0.05]
    seeds = [0, 1, 2, 3, 4]
    results1 = run_g3_4_5_sweep(p_values, seeds, steps=500)
    results2 = run_g3_4_5_sweep(p_values, seeds, steps=500)
    assert len(results1) == len(results2) == 10
    for r1, r2 in zip(results1, results2):
        assert r1["p"] == r2["p"]
        assert r1["seed"] == r2["seed"]
        for key in ("subset", "baseline"):
            assert r1[key]["observables"] == r2[key]["observables"]
            _assert_dumps_equal(r1[key]["dumps"][0], r2[key]["dumps"][0])
            _assert_dumps_equal(r1[key]["dumps"][500], r2[key]["dumps"][500])


def test_g3_4_5_regime_sanity():
    p = 0.05
    seed = 1
    steps = 500
    world = SubsetRegimeBanditWorld(p=p, sigma=0.01, seed=seed)
    r_seq = []
    for t in range(steps):
        out = world.step(t)
        r_vec, _ = validate_world_output(out)
        r_seq.append(np.asarray(r_vec, dtype=float))
    r_arr = np.asarray(r_seq, dtype=float)
    means = np.mean(r_arr, axis=0)
    mean_01 = float(np.mean(means[[0, 1]]))
    mean_34 = float(np.mean(means[[3, 4]]))
    mean_2 = float(means[2])
    tol = 1e-2 + 1e-2 * abs(mean_2)
    assert (mean_01 - mean_2) > tol
    assert (mean_34 - mean_2) > tol


def test_g3_4_5_no_semantic_mutation():
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="regime", phase_id="G3_4_5", run_id="run_0")
    dump_before = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    world = SubsetRegimeBanditWorld(p=0.05, sigma=0.01, seed=1)
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
