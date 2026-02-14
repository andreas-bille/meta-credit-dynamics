from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import (
    RegimeSwitchBanditWorld,
    MarginalMatchedControlWorld,
)
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.stack import StackManager
from capitalmarket.capitalselector.cuda_state import canonical_state_dump


def _collect_world_outputs(world, steps: int):
    r_list = []
    for t in range(steps):
        out = world.step(t)
        r_vec, _ = validate_world_output(out)
        r_list.append(r_vec)
    return np.asarray(r_list)


def test_g3_2_determinism_fixed_seed():
    w1 = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=123)
    w2 = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=123)
    r1 = _collect_world_outputs(w1, steps=200)
    r2 = _collect_world_outputs(w2, steps=200)
    np.testing.assert_allclose(r1, r2, rtol=0.0, atol=0.0)


def test_g3_2_marginal_match():
    steps = 500
    w_a = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=7)
    w_b = MarginalMatchedControlWorld(sigma=0.01, seed=7)
    r_a = _collect_world_outputs(w_a, steps)
    r_b = _collect_world_outputs(w_b, steps)
    mean_a = r_a.mean(axis=0)
    mean_b = r_b.mean(axis=0)
    var_a = r_a.var(axis=0)
    var_b = r_b.var(axis=0)
    np.testing.assert_allclose(mean_a, mean_b, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(var_a, var_b, rtol=1e-2, atol=1e-2)


def test_g3_2_no_semantic_mutation():
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="regime", phase_id="G3_2", run_id="run_0")
    dump_before = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    world = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=1)
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
