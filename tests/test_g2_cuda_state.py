from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from capitalmarket.capitalselector.core import CapitalSelector
from capitalmarket.capitalselector.stats import EWMAStats
from capitalmarket.capitalselector.channels import DummyChannel, DeterministicExplorer
from capitalmarket.capitalselector.stack import StackChannel, StackManager, StackConfig
from capitalmarket.capitalselector.cuda_state import (
    canonical_state_dump,
    toCuda,
    fromCuda,
)


def _make_selector(K: int = 3) -> CapitalSelector:
    stats = EWMAStats(beta=0.1, seed_var=1.0)
    selector = CapitalSelector(
        wealth=1.0,
        rebirth_threshold=0.5,
        stats=stats,
        reweight_fn=lambda w, adv: w,
        kind="entrepreneur",
        rebirth_policy=None,
        channels=[DummyChannel() for _ in range(K)],
    )
    selector.w = np.array([0.2, 0.3, 0.5], dtype=float)
    selector.K = K
    selector._last_r = 0.1
    selector._last_c = 0.05
    stats.mu = 0.01
    stats.var = 0.5
    stats.dd = 0.0
    stats.cum_pi = 0.2
    stats.peak_cum_pi = 0.3
    return selector


def _make_stack_manager() -> StackManager:
    members = {
        "e1": DeterministicExplorer(r=0.1, c=0.0),
        "e2": DeterministicExplorer(r=0.2, c=0.0),
    }
    stack = StackChannel(members, cfg=StackConfig(), stack_id="stack_1")
    sm = StackManager(stack_cfg=StackConfig())
    sm.stacks["stack_1"] = stack
    return sm


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


def test_g2_round_trip_identity_cpu():
    selector = _make_selector()
    sm = _make_stack_manager()
    dump = canonical_state_dump(selector, stack_manager=sm)
    cuda_state = toCuda(dump, device="cpu")
    dump_rt = fromCuda(cuda_state)
    _assert_dumps_equal(dump, dump_rt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_g2_round_trip_identity_cuda():
    selector = _make_selector()
    sm = _make_stack_manager()
    dump = canonical_state_dump(selector, stack_manager=sm)
    cuda_state = toCuda(dump, device="cuda")
    dump_rt = fromCuda(cuda_state)
    _assert_dumps_equal(dump, dump_rt)


def test_g2_shape_and_dtype_preservation():
    selector = _make_selector()
    sm = _make_stack_manager()
    dump = canonical_state_dump(selector, stack_manager=sm)
    cuda_state = toCuda(dump, device="cpu")
    w_t = cuda_state.tensors["selector.w"]
    assert w_t.shape == (3,)
    assert w_t.dtype == torch.float64


def test_g2_no_semantic_calls():
    selector = _make_selector()
    calls = {"reweight": 0, "rebirth": 0}

    def _reweight(*args, **kwargs):
        calls["reweight"] += 1
        return args[0]

    def _rebirth():
        calls["rebirth"] += 1

    selector.reweight_fn = _reweight
    selector.rebirth = _rebirth

    dump = canonical_state_dump(selector)
    _ = toCuda(dump, device="cpu")
    _ = fromCuda(toCuda(dump, device="cpu"))
    assert calls["reweight"] == 0
    assert calls["rebirth"] == 0


def test_g2_profile_agnostic():
    selector = _make_selector()
    dump = canonical_state_dump(selector)
    cuda_state = toCuda(dump, device="cpu")
    assert "profile" not in cuda_state.meta
    assert "profile" not in dump


def _selector_from_dump(dump) -> CapitalSelector:
    stats = EWMAStats(beta=dump["stats"]["beta"], seed_var=dump["stats"]["seed_var"])
    stats.mu = dump["stats"]["mu"]
    stats.var = dump["stats"]["var"]
    stats.dd = dump["stats"]["dd"]
    stats.cum_pi = dump["stats"]["cum_pi"]
    stats.peak_cum_pi = dump["stats"]["peak_cum_pi"]

    selector = CapitalSelector(
        wealth=dump["selector"]["wealth"],
        rebirth_threshold=dump["selector"]["rebirth_threshold"],
        stats=stats,
        reweight_fn=lambda w, adv: w,
        kind=dump["selector"]["kind"],
        rebirth_policy=None,
        channels=[DummyChannel() for _ in range(dump["selector"]["K"])],
    )
    selector.w = None if dump["selector"]["w"] is None else np.asarray(dump["selector"]["w"])
    selector.K = dump["selector"]["K"]
    selector._last_r = dump["selector"]["_last_r"]
    selector._last_c = dump["selector"]["_last_c"]
    return selector


def test_g2_determinism_parity_cpu_round_trip():
    selector = _make_selector()
    dump = canonical_state_dump(selector)

    r_vec = np.array([0.1, 0.2, 0.3], dtype=float)
    c_total = 0.05

    sel1 = _selector_from_dump(dump)
    sel1.feedback_vector(r_vec, c_total, trace=None, freeze=False)
    s1 = canonical_state_dump(sel1)

    dump_rt = fromCuda(toCuda(dump, device="cpu"))
    sel2 = _selector_from_dump(dump_rt)
    sel2.feedback_vector(r_vec, c_total, trace=None, freeze=False)
    s1_rt = canonical_state_dump(sel2)

    _assert_dumps_equal(s1, s1_rt)
