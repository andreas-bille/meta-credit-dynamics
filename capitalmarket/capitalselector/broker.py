
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Iterable
import math
import numpy as np

from .stats import EWMAStats


# -----------------------------
# Phase-C channel interface
# -----------------------------

class PhaseCChannel:
    """Phase-C Channel interface.

    Semantikfrei:
    - Input: weight (Kapitalzufuhr; beliebige Einheit)
    - Output: (r, c, alive, dt)
        r: Rückfluss in derselben Zeiteinheit
        c: Kosten in derselben Zeiteinheit
        alive: False signalisiert Marktaustritt
        dt: Zeit seit letzter Rückmeldung (freie Einheit, >0)
    """

    def step(self, weight: float) -> Tuple[float, float, bool, float]:
        raise NotImplementedError


class LegacyChannelAdapter(PhaseCChannel):
    """Wrap legacy Channel (step -> (r,c)) into Phase-C interface."""
    def __init__(self, legacy_channel):
        self.legacy_channel = legacy_channel
        self._alive = True

    def step(self, weight: float) -> Tuple[float, float, bool, float]:
        r, c = self.legacy_channel.step(weight)
        return float(r), float(c), self._alive, 1.0


# -----------------------------
# Credit policy
# -----------------------------

@dataclass
class CreditPolicy:
    limit: float = 1.0
    min_interval: float = 0.0
    blocked: bool = False


# -----------------------------
# EWMA helpers
# -----------------------------

@dataclass
class EWMAQuantile:
    """Online quantile estimate using stochastic approximation.

    q <- q + eta * (alpha - I[x < q])
    """
    alpha: float
    eta: float
    q: float = 0.0

    def update(self, x: float) -> float:
        ind = 1.0 if x < self.q else 0.0
        self.q = self.q + self.eta * (self.alpha - ind)
        return self.q


@dataclass
class EWMADrawdown:
    """Tracks max drawdown on cumulative net cashflow."""
    peak: float = 0.0
    value: float = 0.0
    max_dd: float = 0.0

    def update(self, net: float) -> float:
        self.value += net
        if self.value > self.peak:
            self.peak = self.value
        dd = self.peak - self.value
        if dd > self.max_dd:
            self.max_dd = dd
        return self.max_dd


@dataclass
class EWMACov:
    beta: float
    cov: float = 0.0

    def update(self, x: float, y: float) -> float:
        self.cov = (1 - self.beta) * self.cov + self.beta * (x * y)
        return self.cov


# -----------------------------
# Broker (Inhibitor)
# -----------------------------

@dataclass
class BrokerConfig:
    beta_mu: float = 0.1
    beta_var: float = 0.1
    alpha_tail: float = 0.1
    eta_q: float = 0.02  # quantile learning rate
    beta_cvar: float = 0.1
    beta_surv: float = 0.05
    beta_dt: float = 0.1

    # thresholds
    k_var: float = 3.0             # vol > k_var * |mu|
    gamma_tail: float = 1.0        # cvar < -gamma_tail * sigma_ref
    k_dd: float = 5.0              # dd > k_dd * sigma_ref
    tau_corr: float = 0.8          # |rho| cap threshold (for bad tail)
    sigma_ref: float = 1.0         # reference scale

    # actions
    lambda_down: float = 0.5
    hard_block_tail: bool = True
    hard_block_dd: bool = True

    # limits
    default_limit: float = 1.0
    min_limit: float = 0.0

    # min interval
    base_min_interval: float = 0.0
    cooldown_factor: float = 1.0


@dataclass
class ExplorerMetrics:
    mu: EWMAStats
    var: EWMAStats
    q: EWMAQuantile
    cvar: float = 0.0
    dd: EWMADrawdown = field(default_factory=EWMADrawdown)
    surv: float = 1.0
    dt_mu: float = 1.0

    # last observation
    last_r: float = 0.0
    last_c: float = 0.0
    last_alive: bool = True
    last_dt: float = 1.0


class Broker:
    """Phase-C Broker as Inhibitor.

    Observes only semantikfreie time-series metrics and produces CreditPolicy per explorer.
    """

    def __init__(self, config: Optional[BrokerConfig] = None):
        self.cfg = config or BrokerConfig()
        self.metrics: Dict[str, ExplorerMetrics] = {}
        self.policies: Dict[str, CreditPolicy] = {}
        self._time: float = 0.0
        self._last_funded_at: Dict[str, float] = {}

        # sparse correlation tracking
        self._cov: Dict[Tuple[str, str], EWMACov] = {}

    # ----- registration -----

    def _ensure(self, explorer_id: str):
        if explorer_id in self.metrics:
            return
        cfg = self.cfg
        self.metrics[explorer_id] = ExplorerMetrics(
            mu=EWMAStats(beta=cfg.beta_mu, mu=0.0, var=1.0),
            var=EWMAStats(beta=cfg.beta_var, mu=0.0, var=1.0),  # store mean in mu.mu, variance in mu.var
            q=EWMAQuantile(alpha=cfg.alpha_tail, eta=cfg.eta_q, q=0.0),
            cvar=0.0,
            dd=EWMADrawdown(),
            surv=1.0,
            dt_mu=1.0,
        )
        self.policies[explorer_id] = CreditPolicy(limit=cfg.default_limit, min_interval=cfg.base_min_interval, blocked=False)
        self._last_funded_at[explorer_id] = -float("inf")

    # ----- observation -----

    def observe(self, explorer_id: str, r: float, c: float, alive: bool, dt: float):
        self._ensure(explorer_id)

        m = self.metrics[explorer_id]
        cfg = self.cfg

        r = float(r); c = float(c); dt = float(dt) if dt and dt > 0 else 1.0
        net = r - c

        # store last obs
        m.last_r, m.last_c, m.last_alive, m.last_dt = r, c, bool(alive), dt

        # update time (global)
        self._time += dt

        # mean/var on r
        m.mu.update(r)
        m.var.update(r)  # uses EWMAStats.var as variance; mu inside m.var.mu unused
        vol = math.sqrt(max(m.var.var, 0.0))

        # tail quantile and cvar proxy
        q = m.q.update(r)
        loss = min(r - q, 0.0)
        m.cvar = (1 - cfg.beta_cvar) * m.cvar + cfg.beta_cvar * loss  # negative or 0

        # drawdown on net
        m.dd.update(net)

        # survival EWMA
        alive_f = 1.0 if alive else 0.0
        m.surv = (1 - cfg.beta_surv) * m.surv + cfg.beta_surv * alive_f

        # dt mean
        m.dt_mu = (1 - cfg.beta_dt) * m.dt_mu + cfg.beta_dt * dt

    # ----- correlations -----

    def update_correlations(self, top_ids: Iterable[str]):
        """Update sparse correlations between a set of ids using last_r - mu."""
        ids = list(top_ids)
        cfg = self.cfg
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                if a not in self.metrics or b not in self.metrics:
                    continue
                ma, mb = self.metrics[a], self.metrics[b]
                xa = ma.last_r - ma.mu.mu
                xb = mb.last_r - mb.mu.mu
                key = (a, b)
                cov = self._cov.get(key)
                if cov is None:
                    cov = EWMACov(beta=cfg.beta_var, cov=0.0)
                    self._cov[key] = cov
                cov.update(xa, xb)

    def rho(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        key = (a, b) if (a, b) in self._cov else (b, a)
        cov = self._cov.get(key)
        if cov is None:
            return 0.0
        ma, mb = self.metrics.get(a), self.metrics.get(b)
        if ma is None or mb is None:
            return 0.0
        va = max(ma.var.var, 1e-12)
        vb = max(mb.var.var, 1e-12)
        return float(cov.cov / (math.sqrt(va * vb) + 1e-12))

    # ----- decisions -----

    def decide_limits(self) -> Dict[str, CreditPolicy]:
        """Compute new policies based on current metrics.

        No positive judgement. Only inhibition: reduce limits, cooldown, block.
        """
        cfg = self.cfg
        for eid, m in self.metrics.items():
            pol = self.policies.get(eid, CreditPolicy(limit=cfg.default_limit, min_interval=cfg.base_min_interval, blocked=False))

            # immediate block on death
            if not m.last_alive:
                pol.blocked = True
                pol.limit = 0.0
                self.policies[eid] = pol
                continue

            mu = m.mu.mu
            vol = math.sqrt(max(m.var.var, 0.0))
            cvar = m.cvar
            dd = m.dd.max_dd

            # start from current
            limit = pol.limit
            min_interval = max(pol.min_interval, cfg.base_min_interval)

            # volatility inhibition
            if abs(mu) > 1e-12 and vol > cfg.k_var * abs(mu):
                limit = max(cfg.min_limit, limit * cfg.lambda_down)

            # tail risk inhibition (cvar negative)
            if cvar < -cfg.gamma_tail * cfg.sigma_ref:
                limit = max(cfg.min_limit, limit * cfg.lambda_down)
                if cfg.hard_block_tail:
                    pol.blocked = True
                    limit = 0.0

            # drawdown inhibition
            if dd > cfg.k_dd * cfg.sigma_ref:
                limit = max(cfg.min_limit, limit * cfg.lambda_down)
                if cfg.hard_block_dd:
                    pol.blocked = True
                    limit = 0.0

            # cooldown based on dt mean (simple)
            min_interval = max(min_interval, m.dt_mu * cfg.cooldown_factor)

            pol.limit = float(limit)
            pol.min_interval = float(min_interval)
            self.policies[eid] = pol

        return {k: CreditPolicy(v.limit, v.min_interval, v.blocked) for k, v in self.policies.items()}

    # ----- gating -----

    def is_eligible(self, explorer_id: str) -> bool:
        pol = self.policies.get(explorer_id)
        if pol is None:
            return True
        if pol.blocked or pol.limit <= 0.0:
            return False
        last = self._last_funded_at.get(explorer_id, -float("inf"))
        if (self._time - last) < pol.min_interval:
            return False
        return True

    def mark_funded(self, explorer_id: str):
        self._last_funded_at[explorer_id] = self._time

    def apply_policies(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply blocking, min_interval and limit caps to a weight dict.

        Returns renormalized weights (sum=1) for remaining active channels.
        """
        out = {}
        for eid, w in weights.items():
            self._ensure(eid)
            pol = self.policies.get(eid, CreditPolicy(limit=self.cfg.default_limit, min_interval=self.cfg.base_min_interval, blocked=False))
            if pol.blocked or not self.is_eligible(eid):
                continue
            w2 = min(float(w), float(pol.limit))
            if w2 > 0:
                out[eid] = w2

        s = sum(out.values())
        if s <= 0:
            return {eid: 0.0 for eid in weights.keys()}
        return {eid: w / s for eid, w in out.items()}

    def apply_decorrelation_cap(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Optional de-correlation cap: if two funded explorers are highly correlated
        and at least one has bad tail risk, reduce both weights.
        """
        cfg = self.cfg
        ids = [eid for eid, w in weights.items() if w > 0]
        if len(ids) < 2:
            return weights
        # update correlations among active ids
        self.update_correlations(ids)

        w = dict(weights)
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                rho = abs(self.rho(a, b))
                if rho <= cfg.tau_corr:
                    continue
                ma, mb = self.metrics[a], self.metrics[b]
                bad_tail = (ma.cvar < -cfg.gamma_tail * cfg.sigma_ref) or (mb.cvar < -cfg.gamma_tail * cfg.sigma_ref)
                if not bad_tail:
                    continue
                # cap both
                w[a] *= cfg.lambda_down
                w[b] *= cfg.lambda_down

        s = sum(w.values())
        if s <= 0:
            return weights
        return {eid: float(v/s) for eid, v in w.items()}

    # ----- introspection -----

    def metric_snapshot(self) -> Dict[str, Dict[str, float]]:
        snap = {}
        for eid, m in self.metrics.items():
            snap[eid] = {
                "mu": float(m.mu.mu),
                "vol": float(math.sqrt(max(m.var.var, 0.0))),
                "cvar": float(m.cvar),
                "dd": float(m.dd.max_dd),
                "surv": float(m.surv),
                "dt": float(m.dt_mu),
                "alive": float(1.0 if m.last_alive else 0.0),
            }
        return snap
