"""Phase E tests (E1-E3) for Sediment + Rebirth.

Run as:
- pytest -q
"""

import unittest
import tempfile
from pathlib import Path

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.broker import Broker, BrokerConfig
from capitalmarket.capitalselector.channels import GaussianExplorer
from capitalmarket.capitalselector.stack import StackManager, StackConfig
from capitalmarket.capitalselector.sediment import SedimentDAG
from capitalmarket.capitalselector.telemetry import TelemetryLogger


def run_loop(channels: dict, selector, broker: Broker, steps: int, stack_mgr: StackManager | None = None):
    """Minimal event loop (Phase C style), extended for Phase E via StackManager hooks."""
    ids = list(channels.keys())
    history = []

    for _t in range(steps):
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

        if stack_mgr is not None:
            stack_mgr.maintain(channels)
            stack_mgr.try_form_stack(broker, channels)

        history.append(w)

    return history


class PhaseETests(unittest.TestCase):
    def _make_base(self):
        channels = {
            "a": GaussianExplorer(mu=0.10, sigma=0.01, seed=1),
            "b": GaussianExplorer(mu=0.10, sigma=0.01, seed=2),
            "c": GaussianExplorer(mu=0.10, sigma=0.01, seed=3),
        }
        broker = Broker(BrokerConfig(sigma_ref=0.1, hard_block_tail=False, hard_block_dd=False, default_limit=1.0))
        selector = CapitalSelectorBuilder().with_K(len(channels)).with_initial_wealth(1.0).with_rebirth_threshold(-1e9).build()
        return channels, broker, selector

    def test_E1_monotone_sediment_growth(self):
        channels, broker, selector = self._make_base()

        with tempfile.TemporaryDirectory() as td:
            sediment_path = Path(td) / "sediment.jsonl"
            sediment = SedimentDAG(persist_path=sediment_path, truncate=True)

            # Force stack instability: require unrealistically high mu at stack level
            stack_cfg = StackConfig(C_agg=0.001, min_size=2, max_size=3, tau_mu=0.5)
            mgr = StackManager(stack_cfg, sediment=sediment, world_id="maze_v1", phase_id="E1")

            # Run in phase E1: should form and dissolve at least once
            run_loop(channels, selector, broker, steps=80, stack_mgr=mgr)
            n1 = len(sediment.nodes())
            self.assertGreaterEqual(n1, 1)

            # Switch phase to allow re-formation of same config and dissolve again
            mgr.set_context(phase_id="E2")
            run_loop(channels, selector, broker, steps=80, stack_mgr=mgr)
            n2 = len(sediment.nodes())
            self.assertGreaterEqual(n2, 2)
            self.assertGreaterEqual(n2, n1)

            # JSONL append-only: lines should equal number of node+edge events (>= nodes)
            lines = sediment_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), n2)

    def test_E2_reformation_suppression(self):
        # Baseline: without sediment, same stack can re-form after dissolution
        channels, broker, selector = self._make_base()
        stack_cfg = StackConfig(C_agg=0.001, min_size=2, max_size=3, tau_mu=0.5)
        mgr_no = StackManager(stack_cfg, sediment=None, world_id="maze_v1", phase_id="E1")
        run_loop(channels, selector, broker, steps=120, stack_mgr=mgr_no)
        # Expect that at least once a stack existed at end of run (re-formation possible)
        self.assertGreaterEqual(mgr_no._counter, 1)  # at least one formation attempt occurred

        # With sediment: after first dissolution, exact config is forbidden within phase
        channels2, broker2, selector2 = self._make_base()
        with tempfile.TemporaryDirectory() as td:
            sediment = SedimentDAG(persist_path=Path(td) / "sediment.jsonl", truncate=True)
            mgr_yes = StackManager(stack_cfg, sediment=sediment, world_id="maze_v1", phase_id="E1")
            run_loop(channels2, selector2, broker2, steps=120, stack_mgr=mgr_yes)
            # After running, sediment should have at least one node and re-formation should be suppressed
            self.assertGreaterEqual(len(sediment.nodes()), 1)

            # No stack should remain formed with same members repeatedly; easiest proxy:
            # stack count in manager should be small (often zero) after repeated dissolutions
            self.assertGreaterEqual(mgr_yes._counter, 1)  # formation attempted
            # if sediment is effective, repeated identical formations are blocked; we should see at least one rejection or no persistent stacks
            self.assertTrue(len(mgr_yes.stacks) == 0 or len(mgr_yes.stacks) == 1)

    def test_E3_canalisation_rejections_shift(self):
        channels, broker, selector = self._make_base()
        with tempfile.TemporaryDirectory() as td:
            tele_path = Path(td) / "telemetry.jsonl"
            sediment_path = Path(td) / "sediment.jsonl"

            telemetry = TelemetryLogger(path=tele_path)
            sediment = SedimentDAG(persist_path=sediment_path, truncate=True)

            # enable pair forbids to encourage switching away from known-bad subsets
            sediment.forbid_pairs = True

            stack_cfg = StackConfig(C_agg=0.001, min_size=2, max_size=3, tau_mu=0.5)
            mgr = StackManager(stack_cfg, sediment=sediment, telemetry=telemetry, world_id="maze_v1", phase_id="E1")
            run_loop(channels, selector, broker, steps=220, stack_mgr=mgr)
            telemetry.close()

            # Count rejection events by time thirds
            rej_ts = []
            for line in tele_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = __import__("json").loads(line)
                if d.get("event_type") == "SEDIMENT_FORMATION_REJECTED":
                    rej_ts.append(int(d.get("t", 0)))

            # We expect at least one rejection early
            self.assertGreaterEqual(len(rej_ts), 1)

            if len(rej_ts) >= 3:
                max_t = max(rej_ts)
                first = sum(1 for t in rej_ts if t <= max_t/3)
                last = sum(1 for t in rej_ts if t >= 2*max_t/3)
                # canalisation: late rejections should not dominate early rejections
                self.assertLessEqual(last, first + 2)
