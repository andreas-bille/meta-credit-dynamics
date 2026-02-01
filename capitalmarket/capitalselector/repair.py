from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple, List
import math


def simplex_renorm(weights: Dict[str, float]) -> Dict[str, float]:
    """Project a nonnegative weight dict to the simplex (sum=1 over active ids).

    Active ids are those with weight > 0.
    If sum <= 0, returns all zeros (same keys as input).
    """
    out = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(out.values())
    if s <= 0.0:
        return {k: 0.0 for k in out.keys()}
    return {k: v / s for k, v in out.items()}


@dataclass
class RepairPolicy:
    """Base class for Phase-D repair policies.

    Policies are deterministic, reversible and non-optimizing.
    """
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def apply_weights(self, weights: Dict[str, float], *, t: int, context: "RepairContext") -> Dict[str, float]:
        return weights

    def apply_observation(
        self,
        obs: Tuple[float, float, bool, float],
        *,
        t: int,
        explorer_id: str,
        context: "RepairContext",
    ) -> Tuple[float, float, bool, float]:
        return obs

    def on_step_end(self, *, t: int, context: "RepairContext") -> None:
        return


@dataclass
class RepairContext:
    """Shared, explicit state for Phase-D repair.

    Important: This context is part of the deterministic simulation state
    (especially for lag buffers and previous weights).
    """
    # Per explorer cumulative "wealth" proxy (updated externally, e.g. by Broker)
    wealth: Dict[str, float] = field(default_factory=dict)

    # Policy-visible time series state
    prev_weights: Dict[str, float] = field(default_factory=dict)

    # Bail-out cooldown tracking
    last_bailout_t: Dict[str, int] = field(default_factory=dict)

    # Isolation TTL tracking (node_id -> t_end exclusive)
    isolation_until: Dict[str, int] = field(default_factory=dict)

    # Lag buffers (explorer_id -> list of observations)
    lag_buffers: Dict[str, List[Tuple[float, float, bool, float]]] = field(default_factory=dict)

    # EMA state (explorer_id -> r_tilde)
    ema_state: Dict[str, float] = field(default_factory=dict)


@dataclass
class CapsPolicy(RepairPolicy):
    """Caps on weight changes (cap_rate) and absolute magnitude (cap_magnitude)."""

    cap_rate: Optional[float] = None   # rho
    cap_magnitude: Optional[float] = None  # m

    def apply_weights(self, weights: Dict[str, float], *, t: int, context: RepairContext) -> Dict[str, float]:
        if not self.enabled:
            return weights

        rho = self.cap_rate
        m = self.cap_magnitude

        out: Dict[str, float] = {k: float(v) for k, v in weights.items()}
        prev = context.prev_weights or {k: float(v) for k, v in weights.items()}

        # cap magnitude first
        if m is not None:
            for k in out.keys():
                out[k] = min(out[k], float(m))

        # cap rate (per id)
        if rho is not None:
            rho = float(rho)
            for k in out.keys():
                p = float(prev.get(k, 0.0))
                w = float(out[k])
                lo = max(0.0, p - rho)
                hi = p + rho
                out[k] = min(max(w, lo), hi)

        # renorm (caps modify weights)
        out = simplex_renorm(out)
        return out


@dataclass
class SoftBailoutPolicy(RepairPolicy):
    """Soft bail-out: ensure minimal funding for explorers below a wealth threshold.

    Note: This intentionally breaks 'mass conservation' by injecting weight floor
    before simplex renormalization.
    """
    min_funding: float = 0.0   # epsilon
    threshold: float = -math.inf
    cooldown: int = 0

    def apply_weights(self, weights: Dict[str, float], *, t: int, context: RepairContext) -> Dict[str, float]:
        if not self.enabled:
            return weights

        eps = float(self.min_funding)
        thr = float(self.threshold)
        cd = int(self.cooldown)

        out = {k: float(v) for k, v in weights.items()}
        for k in out.keys():
            w = float(context.wealth.get(k, 0.0))
            last = context.last_bailout_t.get(k, -10**9)
            if w < thr and (t - last) >= cd:
                # trigger bail-out floor
                if out[k] < eps:
                    out[k] = eps
                context.last_bailout_t[k] = t

        out = simplex_renorm(out)
        return out


@dataclass
class IsolationPolicy(RepairPolicy):
    """Isolation (gating) of nodes and optional TTL."""

    isolation_mask: Set[str] = field(default_factory=set)
    ttl_steps: Optional[int] = None  # None means "permanent" for given mask
    scope: str = "local"

    def apply_weights(self, weights: Dict[str, float], *, t: int, context: RepairContext) -> Dict[str, float]:
        if not self.enabled:
            return weights

        out = {k: float(v) for k, v in weights.items()}

        # Activate TTL once per mask entry (no sliding extension)
        if self.ttl_steps is not None:
            ttl = int(self.ttl_steps)
            for nid in self.isolation_mask:
                if nid not in context.isolation_until:
                    context.isolation_until[nid] = t + ttl

        # Gate by either permanent mask or TTL window
        for k in list(out.keys()):
            if k in self.isolation_mask and self.ttl_steps is None:
                # Permanent isolation
                out[k] = 0.0
                continue

            until = context.isolation_until.get(k)
            if until is not None:
                if t < until:
                    # Within TTL window: gate
                    out[k] = 0.0
                else:
                    # TTL expired: un-gate and remove entry
                    context.isolation_until.pop(k, None)

        # isolation changes weights; renorm to simplex
        out = simplex_renorm(out)
        return out


@dataclass
class LagPolicy(RepairPolicy):
    """Lag injection for observations.

    Supports either discrete FIFO lag_steps or EMA smoothing on r.
    Exactly one of lag_steps or ema_alpha should be used.
    """
    lag_steps: Optional[int] = None
    ema_alpha: Optional[float] = None

    def apply_observation(
        self,
        obs: Tuple[float, float, bool, float],
        *,
        t: int,
        explorer_id: str,
        context: RepairContext,
    ) -> Tuple[float, float, bool, float]:
        if not self.enabled:
            return obs

        r, c, alive, dt = obs
        # FIFO lag
        if self.lag_steps is not None:
            L = int(self.lag_steps)
            buf = context.lag_buffers.setdefault(explorer_id, [])
            buf.append((float(r), float(c), bool(alive), float(dt)))
            # Until we have accumulated L observations, pass-through (no delay yet).
            # Once the buffer exceeds L, return the oldest and thus lag by ~L steps.
            if L <= 0 or len(buf) <= L:
                return (float(r), float(c), bool(alive), float(dt))
            return buf.pop(0)

        # EMA smoothing
        if self.ema_alpha is not None:
            a = float(self.ema_alpha)
            prev = context.ema_state.get(explorer_id)
            if prev is None:
                prev = float(r)
            r_tilde = a * float(r) + (1.0 - a) * float(prev)
            context.ema_state[explorer_id] = r_tilde
            return (float(r_tilde), float(c), bool(alive), float(dt))

        return obs


@dataclass
class RepairPolicySet:
    """A container applying policies in a fixed order.

    Order (weights): Isolation -> Caps -> Bailout
    Order (obs): Lag
    """
    caps: Optional[CapsPolicy] = None
    lag: Optional[LagPolicy] = None
    bailout: Optional[SoftBailoutPolicy] = None
    isolation: Optional[IsolationPolicy] = None
    enabled: bool = True

    def apply_weights(self, weights: Dict[str, float], *, t: int, context: RepairContext) -> Dict[str, float]:
        if not self.enabled:
            return weights
        w = weights
        if self.isolation is not None:
            w = self.isolation.apply_weights(w, t=t, context=context)
        if self.caps is not None:
            w = self.caps.apply_weights(w, t=t, context=context)
        if self.bailout is not None:
            w = self.bailout.apply_weights(w, t=t, context=context)
        # store prev weights for cap_rate
        context.prev_weights = dict(w)
        return w

    def apply_observation(
        self,
        obs: Tuple[float, float, bool, float],
        *,
        t: int,
        explorer_id: str,
        context: RepairContext,
    ) -> Tuple[float, float, bool, float]:
        if not self.enabled or self.lag is None:
            return obs
        return self.lag.apply_observation(obs, t=t, explorer_id=explorer_id, context=context)
