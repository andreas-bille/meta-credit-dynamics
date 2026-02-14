from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.stack import StackManager
from capitalmarket.capitalselector.worlds.deterministic_script_world import DeterministicScriptWorld
from capitalmarket.capitalselector.cuda_state import canonical_state_dump


def _make_selector():
    return CapitalSelectorBuilder().with_K(0).build()


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


def _run_experiment(steps: int = 200):
    world = DeterministicScriptWorld()
    selector = _make_selector()
    stack_manager = StackManager(world_id="deterministic", phase_id="G3_1", run_id="run_0")
    rebirth_count = {"n": 0}

    original_rebirth = selector.rebirth

    def _rebirth_wrapper():
        rebirth_count["n"] += 1
        return original_rebirth()

    selector.rebirth = _rebirth_wrapper  # test-only hook

    observables = []
    dumps = {}

    # t=0 dump before first update
    dumps[0] = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    for t in range(steps):
        out = world.step(t)
        r_vec, c_total = validate_world_output(out)
        if selector.w is None or len(selector.w) != len(r_vec):
            selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
            selector.K = len(r_vec)
        selector.feedback_vector(r_vec, c_total, trace=None, freeze=False)

        obs = {
            "wealth": float(selector.wealth),
            "dominant_channel": int(np.argmax(selector.w)) if selector.w is not None else None,
            "stack_count": len(stack_manager.stacks),
            "rebirth_count": int(rebirth_count["n"]),
        }
        observables.append(obs)

        if t == 99:
            dumps[100] = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)
        if t == 199:
            dumps[200] = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    return observables, dumps


def test_g3_1_determinism():
    obs1, _ = _run_experiment(steps=200)
    obs2, _ = _run_experiment(steps=200)
    assert obs1 == obs2


def test_g3_1_state_equality():
    _, d1 = _run_experiment(steps=200)
    _, d2 = _run_experiment(steps=200)
    _assert_dumps_equal(d1[0], d2[0])
    _assert_dumps_equal(d1[100], d2[100])
    _assert_dumps_equal(d1[200], d2[200])


def test_g3_1_no_semantic_mutation():
    world = DeterministicScriptWorld()
    selector = _make_selector()
    stack_manager = StackManager()

    out = world.step(0)
    r_vec, c_total = validate_world_output(out)
    if selector.w is None or len(selector.w) != len(r_vec):
        selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
        selector.K = len(r_vec)
    selector.feedback_vector(r_vec, c_total, trace=None, freeze=False)

    dump_before = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    _ = {
        "wealth": float(selector.wealth),
        "dominant_channel": int(np.argmax(selector.w)) if selector.w is not None else None,
        "stack_count": len(stack_manager.stacks),
        "rebirth_count": 0,
    }

    dump_after = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)
    assert dump_before == dump_after
