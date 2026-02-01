
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import math
import numpy as np

from .broker import PhaseCChannel, LegacyChannelAdapter, Broker, BrokerConfig


@dataclass
class StackConfig:
    C_agg: float = 0.01
    min_size: int = 2
    max_size: int = 5

    # stability thresholds (v0)
    tau_mu: float = 0.0
    tau_vol: float = 1.0e9  # disable by default
    tau_cvar: float = -1.0e9
    tau_dd: float = 1.0e9


class StackChannel(PhaseCChannel):
    """Aggregated identity (Stack) exported as a Phase-C Channel."""

    def __init__(self, members: Dict[str, PhaseCChannel], cfg: Optional[StackConfig] = None, stack_id: str = "stack"):
        if len(members) == 0:
            raise ValueError("Stack needs at least one member")
        self.members: Dict[str, PhaseCChannel] = dict(members)
        self.cfg = cfg or StackConfig()
        self.stack_id = stack_id

        # equal internal allocation
        self._w_internal: Dict[str, float] = {k: 1.0 / len(self.members) for k in self.members.keys()}
        self._alive = True
        self._dt = 1.0

        # internal stats
        self._mu = 0.0
        self._var = 0.0
        self._beta = 0.05
        self._dd_peak = 0.0
        self._dd_val = 0.0
        self._dd_max = 0.0
        self._cvar = 0.0

    def add(self, explorer_id: str, ch: PhaseCChannel):
        self.members[explorer_id] = ch
        self._w_internal = {k: 1.0 / len(self.members) for k in self.members.keys()}

    def remove(self, explorer_id: str):
        if explorer_id in self.members:
            del self.members[explorer_id]
        if len(self.members) == 0:
            self._alive = False
        else:
            self._w_internal = {k: 1.0 / len(self.members) for k in self.members.keys()}

    def step(self, weight: float) -> Tuple[float, float, bool, float]:
        if not self._alive:
            return 0.0, 0.0, False, self._dt

        rs, cs = [], []
        alive_all = True
        dts = []
        for eid, ch in list(self.members.items()):
            wi = weight * self._w_internal[eid]
            r, c, alive, dt = ch.step(wi)
            rs.append(r); cs.append(c); dts.append(dt)
            if not alive:
                alive_all = False
                # remove dead member
                self.remove(eid)

        r_total = float(sum(rs))
        c_total = float(sum(cs) + self.cfg.C_agg)
        dt_out = float(np.mean(dts)) if dts else self._dt

        # update internal stats on net (r-c)
        net = r_total - c_total
        self._mu = (1 - self._beta) * self._mu + self._beta * net
        self._var = (1 - self._beta) * self._var + self._beta * (net - self._mu) ** 2
        # drawdown
        self._dd_val += net
        self._dd_peak = max(self._dd_peak, self._dd_val)
        self._dd_max = max(self._dd_max, self._dd_peak - self._dd_val)
        # crude cvar proxy: EWMA of negative net
        self._cvar = (1 - self._beta) * self._cvar + self._beta * min(net, 0.0)

        self._dt = dt_out
        if len(self.members) < self.cfg.min_size:
            self._alive = False

        return r_total, c_total, self._alive, dt_out

    def state(self) -> Dict[str, float]:
        return {
            "mu": float(self._mu),
            "vol": float(math.sqrt(max(self._var, 0.0))),
            "dd": float(self._dd_max),
            "cvar": float(self._cvar),
            "size": float(len(self.members)),
            "alive": float(1.0 if self._alive else 0.0),
        }

    def stable(self) -> bool:
        s = self.state()
        return (
            s["alive"] > 0.5
            and s["mu"] >= self.cfg.tau_mu
            and s["vol"] <= self.cfg.tau_vol
            and s["cvar"] >= self.cfg.tau_cvar
            and s["dd"] <= self.cfg.tau_dd
        )


@dataclass
class StackFormationThresholds:
    tau_mu: float = 0.0
    tau_vol: float = 1.0e9
    tau_cvar: float = -1.0e9
    tau_surv: float = 0.6
    tau_corr: float = 0.3
    min_size: int = 2
    max_size: int = 5


class StackManager:
    """Forms and dissolves stacks based on broker metrics.

    v0: forms diversification-style stacks (low correlation) for stability.
    """

    def __init__(self, stack_cfg: Optional[StackConfig] = None, thresholds: Optional[StackFormationThresholds] = None):
        self.stack_cfg = stack_cfg or StackConfig()
        self.thresholds = thresholds or StackFormationThresholds(
            tau_mu=0.0, tau_vol=1.0e9, tau_cvar=-1.0e9, tau_surv=0.6, tau_corr=0.3,
            min_size=self.stack_cfg.min_size, max_size=self.stack_cfg.max_size
        )
        self.stacks: Dict[str, StackChannel] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"stack_{self._counter}"

    def try_form_stack(self, broker: Broker, channels: Dict[str, PhaseCChannel]) -> Optional[str]:
        """Form one stack if possible. Returns new stack_id or None."""
        th = self.thresholds
        ids = list(channels.keys())
        # Candidate filter by per-explorer metrics
        cand = []
        snap = broker.metric_snapshot()
        for eid in ids:
            m = snap.get(eid)
            if m is None:
                continue
            if m["alive"] < 0.5:
                continue
            if m["mu"] < th.tau_mu:
                continue
            if m["vol"] > th.tau_vol:
                continue
            if m["cvar"] < th.tau_cvar:
                continue
            if m["surv"] < th.tau_surv:
                continue
            cand.append(eid)

        if len(cand) < th.min_size:
            return None

        # Build low-correlation subset greedily
        chosen: List[str] = []
        for eid in cand:
            if len(chosen) == 0:
                chosen.append(eid); continue
            ok = True
            for other in chosen:
                if abs(broker.rho(eid, other)) > th.tau_corr:
                    ok = False
                    break
            if ok:
                chosen.append(eid)
            if len(chosen) >= th.max_size:
                break

        if len(chosen) < th.min_size:
            return None

        # Create stack, remove members from channels
        members = {eid: channels[eid] for eid in chosen}
        for eid in chosen:
            del channels[eid]
        sid = self._next_id()
        stack = StackChannel(members, cfg=self.stack_cfg, stack_id=sid)
        self.stacks[sid] = stack
        channels[sid] = stack
        return sid

    def maintain(self, channels: Dict[str, PhaseCChannel]):
        """Remove dead/unstable stacks."""
        to_remove = []
        for sid, st in self.stacks.items():
            if sid not in channels:
                to_remove.append(sid)
                continue
            if not st.stable():
                # dissolve: re-expose members if still alive
                for eid, ch in st.members.items():
                    channels[eid] = ch
                del channels[sid]
                to_remove.append(sid)
        for sid in to_remove:
            self.stacks.pop(sid, None)
