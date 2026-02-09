import numpy as np

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.channels import GaussianExplorer
from capitalmarket.capitalselector.interfaces import validate_world_output


def test_simplex_invariant_after_feedback_vector():
    selector = CapitalSelectorBuilder().with_K(3).build()
    r_vec = np.array([0.1, -0.05, 0.02], dtype=float)
    selector.feedback_vector(r_vec, c=0.0)
    w = selector.allocate()
    assert w is not None
    assert np.all(w >= 0.0)
    assert abs(w.sum() - 1.0) < 1e-9


def test_determinism_fixed_seed():
    def run_once():
        ch = GaussianExplorer(mu=0.01, sigma=0.01, seed=123)
        selector = CapitalSelectorBuilder().with_K(1).build()
        out = []
        for _ in range(10):
            r, c, alive, dt = ch.step(1.0)
            selector.feedback_vector(np.array([r], dtype=float), c=float(c))
            out.append((r, c, alive, dt, float(selector.allocate()[0])))
        return out

    a = run_once()
    b = run_once()
    assert a == b


def test_world_output_validation():
    out = {"r": [0.1, -0.2], "c": 0.05}
    r_vec, c_total = validate_world_output(out)
    assert r_vec.shape == (2,)
    assert abs(c_total - 0.05) < 1e-12


def test_runtime_profile_a_only():
    from capitalmarket.capitalselector.runtime import run, RuntimeConfig

    class _World:
        def step(self, t: int):
            return {"r": [0.0], "c": 0.0}

    # Profile A works
    res = run(world=_World(), steps=2, config=RuntimeConfig(profile="A"))
    assert len(res["history"]) == 2

    # Profile B should be rejected in v1
    try:
        run(world=_World(), steps=1, config=RuntimeConfig(profile="B"))
        assert False, "Expected ValueError for Profile B"
    except ValueError:
        pass
