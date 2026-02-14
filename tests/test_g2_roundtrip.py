import pytest

torch = pytest.importorskip("torch")

from capitalmarket.capitalselector.cuda_state import canonical_state_dump, toCuda, fromCuda
from capitalmarket.capitalselector.stats import EWMAStats
from capitalmarket.capitalselector.core import CapitalSelector
from capitalmarket.capitalselector.channels import DummyChannel


def _make_selector():
    stats = EWMAStats(beta=0.1, seed_var=1.0)
    return CapitalSelector(
        wealth=1.0,
        rebirth_threshold=0.5,
        stats=stats,
        reweight_fn=lambda w, adv: w,
        kind="entrepreneur",
        rebirth_policy=None,
        channels=[DummyChannel() for _ in range(2)],
    )


def test_g2_roundtrip_cpu():
    selector = _make_selector()
    dump = canonical_state_dump(selector)
    dump_rt = fromCuda(toCuda(dump, device="cpu"))
    assert dump.keys() == dump_rt.keys()
