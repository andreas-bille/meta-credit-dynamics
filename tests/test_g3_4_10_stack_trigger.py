from __future__ import annotations

import numpy as np

from capitalmarket.capitalselector.experiments.g3_4_10_stack_trigger import run_g3_4_10_stack_trigger
from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.stack import StackManager
from capitalmarket.capitalselector.cuda_state import canonical_state_dump
from capitalmarket.capitalselector.worlds.deterministic_cluster_world import DeterministicClusterWorld


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


def test_g3_4_10_determinism(monkeypatch):
    monkeypatch.setenv("CAPM_ENABLE_TOPOLOGY", "1")
    result1 = run_g3_4_10_stack_trigger(steps=500)
    result2 = run_g3_4_10_stack_trigger(steps=500)
    assert result1["observables"] == result2["observables"]
    _assert_dumps_equal(result1["dumps"][0], result2["dumps"][0])
    _assert_dumps_equal(result1["dumps"][500], result2["dumps"][500])


def test_g3_4_10_stack_formation(monkeypatch):
    monkeypatch.setenv("CAPM_ENABLE_TOPOLOGY", "1")
    result = run_g3_4_10_stack_trigger(steps=500)
    assert result["aggregates"]["mean_stack_count"] > 0.0


def test_g3_4_10_no_semantic_mutation():
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="cluster", phase_id="G3_4_10", run_id="run_0")
    dump_before = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    world = DeterministicClusterWorld()
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
