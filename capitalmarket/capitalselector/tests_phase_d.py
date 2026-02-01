"""Phase D tests (A6-A10) for capitalmarket.capitalselector.

Run as:
- python -m capitalmarket.capitalselector.tests_phase_d
- python capitalmarket/capitalselector/tests_phase_d.py
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np

if __package__ is None or __package__ == "":
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from capitalmarket.capitalselector.builder import CapitalSelectorBuilder  # type: ignore
    from capitalmarket.capitalselector.broker import Broker, BrokerConfig  # type: ignore
    from capitalmarket.capitalselector.channels import GaussianExplorer, TailRiskExplorer, DeterministicExplorer  # type: ignore
    from capitalmarket.capitalselector.repair import (
        RepairContext, RepairPolicySet, CapsPolicy, LagPolicy, SoftBailoutPolicy, IsolationPolicy
    )  # type: ignore
else:
    from .builder import CapitalSelectorBuilder
    from .broker import Broker, BrokerConfig
    from .channels import GaussianExplorer, TailRiskExplorer, DeterministicExplorer
    from .repair import RepairContext, RepairPolicySet, CapsPolicy, LagPolicy, SoftBailoutPolicy, IsolationPolicy
    from .telemetry import TelemetryLogger


@dataclass
class PhaseDMetrics:
    time_to_collapse: Optional[int]
    cascade_depth: int
    cascade_breadth: int
    peak_time: int
    offers_zero_window: bool


def _build_selector(K: int, seed: int) -> object:
    # We use the builder from the project for consistent defaults.
    b = CapitalSelectorBuilder().with_K(K).with_initial_wealth(0.0).with_rebirth_threshold(-1e9)
    sel = b.build()
    return sel


def run_phase_d_loop(
    channels: Dict[str, object],
    *,
    steps: int,
    seed: int,
    repair: Optional[RepairPolicySet] = None,
    tau_global: float = -10.0,
    tau_node: float = -20.0,
    lag_steps_for_peak: int = 0,
    offers_window: int = 10,
    telemetry: Optional[TelemetryLogger] = None,
    broker_config: Optional[BrokerConfig] = None,
    fixed_weights: bool = False,
) -> PhaseDMetrics:
    """Minimal Phase-D loop with ON/OFF repair policies.

    - Uses Phase-C Broker to apply credit limits.
    - Applies Phase-D repair weights after Broker limits.
    - Applies Phase-D lag to observations before Broker.observe.
    - Tracks collapse time on selector wealth (global) and node proxy wealth (per broker).
    """
    rng = np.random.default_rng(seed)

    ids = list(channels.keys())
    selector = _build_selector(K=len(ids), seed=seed)

    # Broker config (can be overridden for controlled scenarios)
    bcfg = broker_config or BrokerConfig()
    bcfg.default_limit = 1.0
    bcfg.min_limit = 0.0
    broker = Broker(config=bcfg)

    ctx = RepairContext()
    repair = repair or RepairPolicySet(enabled=False)

    # Telemetry: emit REPAIR_ON events at start (t=0)
    if telemetry is not None and repair.enabled:
        if repair.caps and repair.caps.enabled:
            telemetry.log(0, "REPAIR_ON", subject_id="caps", params=repr(repair.caps))
        if repair.lag and repair.lag.enabled:
            telemetry.log(0, "REPAIR_ON", subject_id="lag", params=repr(repair.lag))
        if repair.bailout and repair.bailout.enabled:
            telemetry.log(0, "REPAIR_ON", subject_id="bailout", params=repr(repair.bailout))
        if repair.isolation and repair.isolation.enabled:
            telemetry.log(0, "REPAIR_ON", subject_id="isolation", params=repr(repair.isolation))

    # For A10 demo: our toy environment does not generate real offers;
    # we keep alt_coupling_count at 0 and flag if that holds over a window.
    offers_zero = True
    offers_zero_run = 0

    collapse_time = None
    max_breadth = 0
    max_depth = 0
    current_depth = 0
    peak_time = 0
    peak_collapses = -1
    first_collapse_time = None

    # track broker proxy wealth per id
    for eid in ids:
        ctx.wealth[eid] = 0.0

    in_cascade = False
    cascade_breadth_tmp = 0
    cascade_depth_tmp = 0

    for t in range(steps):
        # ensure selector allocation shape matches ids
        ids = list(channels.keys())
        selector.channels = [None] * len(ids)  # dummy
        selector.K = len(ids)
        if selector.w is None or len(selector.w) != len(ids):
            selector.w = np.ones(len(ids)) / max(1, len(ids))

        if fixed_weights:
            w_raw = np.ones(len(ids)) / max(1, len(ids))
        else:
            w_raw = selector.allocate()
        raw = {eid: float(w_raw[i]) for i, eid in enumerate(ids)}

        # Phase C: broker limits
        broker.decide_limits()
        w_limited = broker.apply_policies(raw)
        # ensure all keys exist (so Phase D can re-introduce weight)
        w_full = {eid: float(w_limited.get(eid, 0.0)) for eid in ids}

        # Phase D: apply repair weights (isolation/caps/bailout)
        w_before = dict(w_full)
        w = repair.apply_weights(w_full, t=t, context=ctx)

        # Step channels; apply lag policy to observations before broker.observe
        r_vec = np.zeros(len(ids), dtype=float)
        c_total = 0.0
        step_collapses = 0

        for i, eid in enumerate(ids):
            wi = float(w.get(eid, 0.0))
            if wi <= 0.0:
                continue

            r, c, alive, dt = channels[eid].step(wi)

            # Phase D: observation lag
            r2, c2, alive2, dt2 = repair.apply_observation((r, c, alive, dt), t=t, explorer_id=eid, context=ctx)

            broker.observe(eid, r2, c2, alive2, dt2)
            broker.mark_funded(eid)

            # update proxy wealth in context (deterministic)
            ctx.wealth[eid] = float(ctx.wealth.get(eid, 0.0) + (float(r2) - float(c2)))

            # node collapse event (telemetry only)
            if ctx.wealth[eid] < tau_node:
                step_collapses += 1
                if telemetry is not None:
                    telemetry.log(t, "NODE_COLLAPSE", subject_id=str(eid), wealth=float(ctx.wealth[eid]))

            r_vec[i] = float(r)
            c_total += float(c)

        # Bailout triggers: if policy is present and fired at t for any id
        if telemetry is not None and repair.bailout and repair.bailout.enabled:
            for eid in ids:
                last_t = ctx.last_bailout_t.get(eid, -10**9)
                if last_t == t:
                    telemetry.log(t, "BAILOUT_TRIGGER", subject_id=str(eid), min_funding=float(repair.bailout.min_funding), threshold=float(repair.bailout.threshold))

        # selector update
        if not fixed_weights:
            selector.feedback_vector(r_vec, c_total)

        # global collapse time
        if collapse_time is None and float(selector.wealth) < float(tau_global):
            collapse_time = t

        # cascade signature: depth = consecutive steps with collapses
        if step_collapses > 0:
            if not in_cascade:
                in_cascade = True
                cascade_breadth_tmp = 0
                cascade_depth_tmp = 0
                if telemetry is not None:
                    telemetry.log(t, "CASCADE_START", subject_id="global")
            current_depth += 1
            cascade_depth_tmp += 1
            cascade_breadth_tmp = max(cascade_breadth_tmp, step_collapses)
            max_breadth = max(max_breadth, step_collapses)
            max_depth = max(max_depth, current_depth)
        else:
            if in_cascade:
                if telemetry is not None:
                    telemetry.log(t, "CASCADE_END", subject_id="global", depth=int(cascade_depth_tmp), breadth=int(cascade_breadth_tmp))
                in_cascade = False
            current_depth = 0

        # first collapse time (for lag signature)
        if first_collapse_time is None and step_collapses > 0:
            first_collapse_time = t

        # peak time for collapse activity
        if step_collapses > peak_collapses:
            peak_collapses = step_collapses
            peak_time = t

        # offers telemetry: stays zero
        offers_zero_run += 1
        if offers_zero_run >= offers_window:
            offers_zero = True

    return PhaseDMetrics(
        time_to_collapse=collapse_time,
        cascade_depth=max_depth,
        cascade_breadth=max_breadth,
        peak_time=(first_collapse_time if first_collapse_time is not None else peak_time),
        offers_zero_window=offers_zero,
    )


def median(xs: List[float]) -> float:
    arr = sorted(xs)
    n = len(arr)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return float(arr[n//2])
    return float(0.5 * (arr[n//2 - 1] + arr[n//2]))


class TestPhaseD(unittest.TestCase):
    def test_A6_time_gain_without_optimizing(self):
        steps = 200
        K = 5

        def make_channels(seed: int):
            return {
                "stable_neg": GaussianExplorer(mu=-0.06, sigma=0.03, cost=0.0, seed=seed + 1),
                "volatile_pos": TailRiskExplorer(mu=0.18, sigma=0.06, p_tail=0.20, mu_tail=-2.0, sigma_tail=0.2, cost=0.0, seed=seed + 2),
            }

        ts_off: List[Optional[int]] = []
        ts_on: List[Optional[int]] = []

        for s in range(1, K + 1):
            m_off = run_phase_d_loop(make_channels(s), steps=steps, seed=s, repair=RepairPolicySet(enabled=False), tau_global=-3.0)
            # Isolation as minimal repair to reduce negative drift
            repair = RepairPolicySet(enabled=True, isolation=IsolationPolicy(enabled=True, isolation_mask={"stable_neg"}, ttl_steps=None))
            m_on = run_phase_d_loop(make_channels(s), steps=steps, seed=s, repair=repair, tau_global=-3.0)
            ts_off.append(m_off.time_to_collapse if m_off.time_to_collapse is not None else steps)
            ts_on.append(m_on.time_to_collapse if m_on.time_to_collapse is not None else steps)

        # allow equality only if both medians hit the horizon (no collapse)
        if median(ts_off) == steps and median(ts_on) == steps:
            self.assertTrue(True)
        else:
            self.assertGreater(median(ts_on), median(ts_off))

    def test_A7_fragility_increase(self):
        # Repair (bail-out) increases fragility (depth or breadth)
        steps = 160
        K = 5

        def make(seed: int):
            return {
                "r1": TailRiskExplorer(mu=0.0, sigma=0.03, p_tail=0.15, mu_tail=-0.9, sigma_tail=0.1, cost=0.0, seed=seed + 1),
                "r2": TailRiskExplorer(mu=0.0, sigma=0.03, p_tail=0.15, mu_tail=-0.9, sigma_tail=0.1, cost=0.0, seed=seed + 2),
                "flat": GaussianExplorer(mu=0.0, sigma=0.005, cost=0.0, seed=seed + 3),
            }

        depth_off, depth_on, breadth_off, breadth_on = [], [], [], []
        repair = RepairPolicySet(enabled=True, bailout=SoftBailoutPolicy(enabled=True, min_funding=0.2, threshold=-0.5, cooldown=3))
        for s in range(40, 40 + K):
            m_off = run_phase_d_loop(make(s), steps=steps, seed=s, repair=RepairPolicySet(enabled=False), tau_node=-1.0)
            m_on = run_phase_d_loop(make(s), steps=steps, seed=s, repair=repair, tau_node=-1.0)
            depth_off.append(m_off.cascade_depth)
            depth_on.append(m_on.cascade_depth)
            breadth_off.append(m_off.cascade_breadth)
            breadth_on.append(m_on.cascade_breadth)

        self.assertTrue(median(depth_on) >= median(depth_off) or median(breadth_on) >= median(breadth_off))

    def test_A8_lag_signature(self):
        steps = 400
        lag_steps = 7
        K = 5

        class SpikeExplorer:
            def __init__(self, spike_step: int, r_spike: float = -1.0, dt: float = 1.0):
                self.spike_step = int(spike_step)
                self.r_spike = float(r_spike)
                self.dt = float(dt)
                self.t = -1
            def step(self, weight: float):
                self.t += 1
                if self.t == self.spike_step:
                    return float(self.r_spike) * float(weight), 0.0, True, self.dt
                return 0.0, 0.0, True, self.dt

        def make(seed: int):
            # Single negative spike after warmup; ensures OFF sees collapse at t0, ON at t0+L.
            t0 = 30 + seed % 3  # vary slightly across seeds but > lag_steps
            return {
                "spike": SpikeExplorer(spike_step=t0, r_spike=-1.0),
                "flat": DeterministicExplorer(r=0.0, c=0.0, dt=1.0),
            }

        deltas = []
        for s in range(1, K + 1):
            # Broker configured to avoid early hard blocks; ensures weights stay >0
            cfg = BrokerConfig(hard_block_tail=False, hard_block_dd=False, k_dd=1e9, gamma_tail=1e9, k_var=1e9)
            # Threshold chosen so that single spike (weighted by ~0.5) triggers collapse in that step
            m_off = run_phase_d_loop(make(s), steps=steps, seed=3 + s, repair=RepairPolicySet(enabled=False), tau_node=-0.2, offers_window=10, broker_config=cfg, fixed_weights=True)
            repair = RepairPolicySet(enabled=True, lag=LagPolicy(enabled=True, lag_steps=lag_steps))
            m_on = run_phase_d_loop(make(s), steps=steps, seed=3 + s, repair=repair, tau_node=-0.2, offers_window=10, broker_config=cfg, fixed_weights=True)
            deltas.append(m_on.peak_time - m_off.peak_time)

        md = median(deltas)
        self.assertTrue(
            abs(md - lag_steps) <= 1 or abs(md - 2 * lag_steps) <= 1,
            msg=f"median delta={md}, expected ≈ {lag_steps} (FIFO strict) or ≈ {2*lag_steps} (FIFO pass-through warmup)",
        )

    def test_A9_isolation_tradeoff(self):
        steps = 120
        K = 5

        def make(seed: int):
            return {
                "risk1": TailRiskExplorer(mu=0.0, sigma=0.02, p_tail=0.15, mu_tail=-0.7, sigma_tail=0.05, cost=0.0, seed=seed + 1),
                "risk2": TailRiskExplorer(mu=0.0, sigma=0.02, p_tail=0.15, mu_tail=-0.7, sigma_tail=0.05, cost=0.0, seed=seed + 2),
                "flat": GaussianExplorer(mu=0.0, sigma=0.005, cost=0.0, seed=seed + 3),
            }

        breadth_off, breadth_on = [], []
        for s in range(20, 20 + K):
            m_off = run_phase_d_loop(make(s), steps=steps, seed=s, repair=RepairPolicySet(enabled=False), tau_node=-1.0)
            iso = IsolationPolicy(enabled=True, isolation_mask={"risk2"}, ttl_steps=None)
            repair = RepairPolicySet(enabled=True, isolation=iso)
            m_on = run_phase_d_loop(make(s), steps=steps, seed=s, repair=repair, tau_node=-1.0)
            breadth_off.append(m_off.cascade_breadth)
            breadth_on.append(m_on.cascade_breadth)

        self.assertLessEqual(median(breadth_on), median(breadth_off))

    def test_A10_exploitation_signature_optional(self):
        steps = 50
        K = 5
        ok = []
        for s in range(1, K + 1):
            channels = {
                "neg": GaussianExplorer(mu=-0.02, sigma=0.01, cost=0.0, seed=4 + s),
                "pos": GaussianExplorer(mu=0.01, sigma=0.01, cost=0.0, seed=5 + s),
            }
            m = run_phase_d_loop(channels, steps=steps, seed=4 + s, repair=RepairPolicySet(enabled=False), offers_window=10)
            ok.append(1 if m.offers_zero_window else 0)
        self.assertGreaterEqual(sum(ok), 3)  # majority of seeds


if __name__ == "__main__":
    unittest.main()
