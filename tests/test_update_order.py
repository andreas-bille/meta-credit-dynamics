import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder


def test_update_order_trace():
    selector = CapitalSelectorBuilder().with_K(2).with_rebirth_threshold(-1e9).build()
    trace = []
    r_vec = np.array([0.1, 0.0], dtype=float)
    selector.feedback_vector(r_vec, c=0.0, trace=trace)

    assert trace == [
        "compute_pi",
        "update_stats",
        "reweight",
        "invariants",
    ]
