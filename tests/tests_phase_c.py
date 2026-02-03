
"""Phase C tests (A1-A5) for capitalmarket.capitalselector.

Run as:
- pytest -q
"""

import unittest

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.broker import Broker, BrokerConfig
from capitalmarket.capitalselector.channels import (
    GaussianExplorer,
    TailRiskExplorer,
    DeterministicExplorer,
)
from capitalmarket.capitalselector.stack import StackManager, StackConfig


def run_loop(channels: dict, selector, broker: Broker, steps: int, stack_mgr: StackManager | None = None):
    """Minimal Phase-C event loop.

    Returns history of applied weights per step (dict of id->weight).
    """
    ids = list(channels.keys())
    history = []

    for _t in range(steps):
        # Keep ids in sync (stacks may modify)
        ids = list(channels.keys())
        selector.channels = [None] * len(ids)  # dummy: we only use w length
        selector.K = len(ids)
        if selector.w is None or len(selector.w) != len(ids):
            selector.w = __import__("numpy").ones(len(ids)) / max(1, len(ids))

        w_raw = selector.allocate()
        raw = {eid: float(w_raw[i]) for i, eid in enumerate(ids)}

        broker.decide_limits()
        w = broker.apply_policies(raw)
        w = broker.apply_decorrelation_cap(w)

        # Step eligible channels
        r_vec = __import__("numpy").zeros(len(ids), dtype=float)
        c_total = 0.0
        for i, eid in enumerate(ids):
            wi = w.get(eid, 0.0)
            if wi <= 0.0:
                continue
            r, c, alive, dt = channels[eid].step(wi)
            broker.observe(eid, r, c, alive, dt)
            broker.mark_funded(eid)
            r_vec[i] = float(r)
            c_total += float(c)

        selector.feedback_vector(r_vec, c_total)

        # stack maintenance / formation (optional)
        if stack_mgr is not None:
            stack_mgr.maintain(channels)
            stack_mgr.try_form_stack(broker, channels)

        history.append(w)

    return history


class PhaseCTests(unittest.TestCase):
    def test_A1_high_risk_gets_defunded(self):
        # one stable, one tail-risk
        channels = {
            "good": GaussianExplorer(mu=0.1, sigma=0.01, cost=0.0, seed=1),
            "bad": TailRiskExplorer(mu=0.05, sigma=0.2, p_tail=0.2, mu_tail=-2.0, sigma_tail=0.5, seed=2),
        }
        broker = Broker(BrokerConfig(sigma_ref=0.1, hard_block_tail=True, gamma_tail=1.0, k_var=3.0, default_limit=1.0))
        selector = CapitalSelectorBuilder().with_K(len(channels)).with_initial_wealth(1.0).with_rebirth_threshold(-1e9).build()

        hist = run_loop(channels, selector, broker, steps=120)

        # bad should be close to zero in last steps
        last = hist[-1]
        self.assertLess(last.get("bad", 0.0), 0.05)

    def test_A2_stack_forms_and_is_less_volatile(self):
        channels = {
            "a": GaussianExplorer(mu=0.08, sigma=0.05, seed=1),
            "b": GaussianExplorer(mu=0.07, sigma=0.05, seed=2),
            "c": GaussianExplorer(mu=0.06, sigma=0.05, seed=3),
        }
        broker = Broker(BrokerConfig(sigma_ref=0.1, hard_block_tail=False, hard_block_dd=False, default_limit=1.0))
        selector = CapitalSelectorBuilder().with_K(len(channels)).with_initial_wealth(1.0).with_rebirth_threshold(-1e9).build()
        stack_mgr = StackManager(StackConfig(C_agg=0.001, min_size=2, max_size=3))

        run_loop(channels, selector, broker, steps=400, stack_mgr=stack_mgr)

        # should have at least one stack
        stacks = [k for k in channels.keys() if k.startswith("stack_")]
        self.assertTrue(len(stacks) >= 1)
        sid = stacks[0]
        st = channels[sid]
        vol_stack = st.state()["vol"]
        # stack vol should be lower than typical member vol (~0.05)
        self.assertLess(vol_stack, 0.05)

    def test_A3_decorrelation_cap(self):
        # Two identical deterministic explorers => high correlation, but tail-risk one makes cap reduce both
        channels = {
            "x": TailRiskExplorer(mu=0.01, sigma=0.01, p_tail=0.5, mu_tail=-1.0, sigma_tail=0.0, seed=1),
            "y": TailRiskExplorer(mu=0.01, sigma=0.01, p_tail=0.5, mu_tail=-1.0, sigma_tail=0.0, seed=2),
            "z": GaussianExplorer(mu=0.01, sigma=0.01, seed=3),
        }
        cfg = BrokerConfig(sigma_ref=0.1, tau_corr=0.1, gamma_tail=0.5, hard_block_tail=False, lambda_down=0.5)
        broker = Broker(cfg)
        selector = CapitalSelectorBuilder().with_K(len(channels)).with_initial_wealth(1.0).with_rebirth_threshold(-1e9).build()

        hist = run_loop(channels, selector, broker, steps=200)
        last = hist[-1]
        # At least one of x,y should not dominate simultaneously due to cap (both reduced)
        self.assertLess(last.get("x", 0.0) + last.get("y", 0.0), 0.9)

    def test_A4_alive_false_defunds_immediately(self):
        # alive_prob very low; should be blocked and weight -> 0
        channels = {
            "alive": GaussianExplorer(mu=0.05, sigma=0.01, alive_prob=1.0, seed=1),
            "dies": GaussianExplorer(mu=0.05, sigma=0.01, alive_prob=0.0, seed=2),
        }
        broker = Broker(BrokerConfig())
        selector = CapitalSelectorBuilder().with_K(len(channels)).with_initial_wealth(1.0).with_rebirth_threshold(-1e9).build()

        hist = run_loop(channels, selector, broker, steps=10)
        # after a few steps, dies should be 0
        self.assertEqual(hist[-1].get("dies", 0.0), 0.0)

    def test_A5_rebirth_resets_simplex(self):
        channels = {"a": DeterministicExplorer(r=0.0), "b": DeterministicExplorer(r=0.0), "c": DeterministicExplorer(r=0.0)}
        selector = CapitalSelectorBuilder().with_K(len(channels)).with_initial_wealth(1.0).with_rebirth_threshold(0.5).build()

        import numpy as np
        # Force massive loss through feedback_vector
        selector.feedback_vector(np.array([-10.0, 0.0, 0.0], dtype=float), c=0.0)

        self.assertAlmostEqual(selector.wealth, 0.5)
        w = selector.allocate()
        self.assertTrue(np.allclose(w, np.ones(3)/3))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
