from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.experiments.g3_4_8_shocks import run_g3_4_8_sweep, SHOCK_TIMES
from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.stack import StackManager
from capitalmarket.capitalselector.cuda_state import canonical_state_dump
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import (
    RegimeSwitchBanditWorld,
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


def test_g3_4_8_determinism():
    p_values = [0.01, 0.05]
    modes = ["baseline", "shock"]
    seeds = [0, 1, 2, 3, 4]
    results1 = run_g3_4_8_sweep(p_values, modes, seeds, steps=500)
    results2 = run_g3_4_8_sweep(p_values, modes, seeds, steps=500)
    assert len(results1) == len(results2) == 20
    for r1, r2 in zip(results1, results2):
        assert r1["p"] == r2["p"]
        assert r1["mode"] == r2["mode"]
        assert r1["seed"] == r2["seed"]
        assert r1["observables"] == r2["observables"]
        _assert_dumps_equal(r1["dumps"][0], r2["dumps"][0])
        _assert_dumps_equal(r1["dumps"][500], r2["dumps"][500])


def test_g3_4_8_shock_sanity():
    p = 0.05
    seed = 1
    steps = 200
    shock_size = 0.2
    regime_sequence = _generate_regime_sequence(p=p, seed=seed, length=steps)
    rng = np.random.default_rng(int(seed))
    noise_sequence = rng.normal(0.0, 0.01, size=(steps, 5))

    baseline = RegimeSwitchBanditWorld(
        p=p,
        sigma=0.01,
        seed=seed,
        regime_sequence=regime_sequence,
        noise_sequence=noise_sequence,
        shock_times=set(),
        shock_size=shock_size,
    )
    shocked = RegimeSwitchBanditWorld(
        p=p,
        sigma=0.01,
        seed=seed,
        regime_sequence=regime_sequence,
        noise_sequence=noise_sequence,
        shock_times=SHOCK_TIMES,
        shock_size=shock_size,
    )

    for t in range(steps):
        out_base = baseline.step(t)
        out_shock = shocked.step(t)
        r_base, _ = validate_world_output(out_base)
        r_shock, _ = validate_world_output(out_shock)
        if t in SHOCK_TIMES:
            active_idx = 0 if regime_sequence[t] == "A" else 3
            expected = float(r_base[active_idx]) - shock_size
            np.testing.assert_allclose(r_shock[active_idx], expected, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_allclose(r_shock, r_base, rtol=1e-12, atol=1e-12)


def test_g3_4_8_no_semantic_mutation():
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="regime", phase_id="G3_4_8", run_id="run_0")
    dump_before = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    world = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=1, shock_times=SHOCK_TIMES, shock_size=0.2)
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
