import copy
import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder


def test_freeze_keeps_state_observationally_equal():
    selector = CapitalSelectorBuilder().with_K(2).with_initial_wealth(1.0).build()
    # drive to non-default state
    selector.feedback_vector(np.array([0.2, 0.0], dtype=float), c=0.0)

    stats_before = (selector.stats.mu, selector.stats.var, selector.stats.dd, selector.stats.cum_pi, selector.stats.peak_cum_pi)
    w_before = selector.allocate().copy()
    wealth_before = selector.wealth

    # freeze step: no state changes expected
    selector.feedback_vector(np.array([0.1, 0.1], dtype=float), c=0.0, freeze=True)

    stats_after = (selector.stats.mu, selector.stats.var, selector.stats.dd, selector.stats.cum_pi, selector.stats.peak_cum_pi)
    w_after = selector.allocate().copy()
    wealth_after = selector.wealth

    assert stats_before == stats_after
    assert np.allclose(w_before, w_after)
    assert wealth_before == wealth_after
