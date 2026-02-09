import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder


def test_compute_pi_identity():
    selector = CapitalSelectorBuilder().with_K(3).build()
    r_vec = np.array([0.1, 0.2, -0.05], dtype=float)
    c_total = 0.05
    R, C, Pi, pi_vec = selector.compute_pi(r_vec, c_total)
    assert abs(R - r_vec.sum()) < 1e-12
    assert abs(C - c_total) < 1e-12
    assert abs(Pi - (R - C)) < 1e-12
    assert abs(pi_vec.sum() - Pi) < 1e-12


def test_equal_scores_leave_weights_unchanged():
    selector = CapitalSelectorBuilder().with_K(3).build()
    w0 = selector.allocate().copy()
    r_vec = np.array([0.1, 0.1, 0.1], dtype=float)
    selector.feedback_vector(r_vec, c=0.0)
    w1 = selector.allocate()
    assert np.allclose(w0, w1)


def test_dominant_score_increases_weight():
    selector = CapitalSelectorBuilder().with_K(3).with_reweight_eta(1.0).build()
    w0 = selector.allocate().copy()
    # dominant channel 0
    r_vec = np.array([0.5, 0.0, 0.0], dtype=float)
    selector.feedback_vector(r_vec, c=0.0)
    w1 = selector.allocate()
    assert w1[0] > w0[0]
