from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.experiments.g3_4_12_adversarial_phase_shift import run_g3_4_12_sweep
from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.stack import StackManager
from capitalmarket.capitalselector.cuda_state import canonical_state_dump
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import AdversarialPhaseShiftBanditWorld


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


def _residuals(r_seq: np.ndarray, regime_seq: list[str]) -> np.ndarray:
    means = np.zeros_like(r_seq)
    for i, regime in enumerate(regime_seq):
        if regime == "A":
            means[i, 0] = 0.02
        else:
            means[i, 3] = 0.02
    return r_seq - means


def test_g3_4_12_determinism():
    seeds = [0, 1, 2, 3, 4]
    results1 = run_g3_4_12_sweep(seeds, steps=500)
    results2 = run_g3_4_12_sweep(seeds, steps=500)
    assert len(results1) == len(results2) == 5
    for r1, r2 in zip(results1, results2):
        assert r1["seed"] == r2["seed"]
        assert r1["observables"] == r2["observables"]
        _assert_dumps_equal(r1["dumps"][0], r2["dumps"][0])
        _assert_dumps_equal(r1["dumps"][500], r2["dumps"][500])


def test_g3_4_12_phase_variance_sanity():
    results = run_g3_4_12_sweep([0], steps=500)
    r_seq = results[0]["r_seq"]
    regime_seq = results[0]["regime_seq"]

    residuals = _residuals(r_seq, regime_seq)

    early = residuals[:100]
    late = residuals[400:]

    var_phase1 = float(np.var(early))
    np.testing.assert_allclose(var_phase1, 0.005**2, rtol=1e-2, atol=1e-3)

    active_res = []
    opposing_res = []
    for idx, t in enumerate(range(400, 500)):
        regime = regime_seq[t]
        if regime == "A":
            active_res.append(late[idx, 0])
            opposing_res.append(late[idx, 3])
        else:
            active_res.append(late[idx, 3])
            opposing_res.append(late[idx, 0])

    var_active = float(np.var(np.asarray(active_res)))
    var_opposing = float(np.var(np.asarray(opposing_res)))

    assert var_active > var_phase1
    assert var_opposing < var_active


def test_g3_4_12_no_semantic_mutation():
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="regime", phase_id="G3_4_12", run_id="run_0")
    dump_before = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    world = AdversarialPhaseShiftBanditWorld(p=0.001, seed=1, horizon=500)
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
