import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.config import ProfileAConfig, ProfileBConfig


def _contains_profile_obj(obj) -> bool:
    seen = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, (ProfileAConfig, ProfileBConfig)):
            return True
        if isinstance(cur, dict):
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple, set, frozenset)):
            stack.extend(cur)
            continue
        if hasattr(cur, "__dict__"):
            stack.extend(cur.__dict__.values())
    return False


def _make_profiles_identical():
    return ProfileAConfig(), ProfileBConfig(freeze_stats=True)


def _run_steps(selector, steps=5):
    r_vec = np.array([0.1, 0.2, 0.3], dtype=float)
    c_total = 0.05
    for _ in range(steps):
        selector.feedback_vector(r_vec, c_total, trace=None, freeze=False)
    return selector.state()


def test_g1_profile_equivalence_identical_params():
    prof_a, prof_b = _make_profiles_identical()
    sel_a = CapitalSelectorBuilder.from_profile(prof_a).with_K(3).build()
    sel_b = CapitalSelectorBuilder.from_profile(prof_b).with_K(3).build()

    state_a = _run_steps(sel_a)
    state_b = _run_steps(sel_b)

    assert state_a.keys() == state_b.keys()
    for k in state_a:
        if isinstance(state_a[k], np.ndarray):
            np.testing.assert_allclose(state_a[k], state_b[k])
        else:
            assert state_a[k] == state_b[k]


def test_g1_no_profile_leakage_in_instance_or_state():
    prof_a, _ = _make_profiles_identical()
    sel = CapitalSelectorBuilder.from_profile(prof_a).with_K(2).build()

    assert not hasattr(sel, "profile")
    assert not _contains_profile_obj(sel)
    assert not _contains_profile_obj(sel.state())


def test_g1_rebirth_reset_invariance():
    prof_a, prof_b = _make_profiles_identical()
    sel_a = CapitalSelectorBuilder.from_profile(prof_a).with_K(2).build()
    sel_b = CapitalSelectorBuilder.from_profile(prof_b).with_K(2).build()

    r_vec = np.array([0.0, 0.0], dtype=float)
    c_total = 0.0
    sel_a.feedback_vector(r_vec, c_total, trace=None, freeze=False)
    sel_b.feedback_vector(r_vec, c_total, trace=None, freeze=False)

    sel_a.wealth = -1.0
    sel_b.wealth = -1.0
    sel_a.rebirth()
    sel_b.rebirth()

    np.testing.assert_allclose(sel_a.w, sel_b.w)
    assert sel_a.stats.mu == sel_b.stats.mu
    assert sel_a.stats.var == sel_b.stats.var
    assert sel_a.stats.dd == sel_b.stats.dd
    assert len(sel_a.channels) == len(sel_b.channels)
