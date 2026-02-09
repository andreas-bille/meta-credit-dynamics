import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder


def test_rebirth_resets_behavioral_state():
    selector = (
        CapitalSelectorBuilder()
        .with_K(3)
        .with_initial_wealth(1.0)
        .with_rebirth_threshold(0.5)
        .build()
    )
    # drive stats and weights away from defaults
    selector.feedback_vector(np.array([0.2, 0.0, 0.0], dtype=float), c=0.0)
    selector.feedback_vector(np.array([-10.0, 0.0, 0.0], dtype=float), c=0.0)

    # rebirth should have triggered
    w = selector.allocate()
    assert np.allclose(w, np.ones(3) / 3.0)
    assert selector.stats.mu == 0.0
    assert selector.stats.var == selector.stats.seed_var
    assert selector.stats.dd == 0.0
    assert selector.stats.cum_pi == 0.0
    assert selector.stats.peak_cum_pi == 0.0
    assert selector.wealth >= selector.rebirth_threshold


def test_state_contains_behavioral_fields():
    selector = CapitalSelectorBuilder().with_K(2).build()
    state = selector.state()
    for key in ("mu", "var", "dd", "cum_pi", "peak_cum_pi", "weights", "wealth"):
        assert key in state
