from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch


@dataclass(frozen=True)
class CudaState:
    """Passive CUDA representation (tensors + structural metadata)."""

    tensors: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


@dataclass(frozen=True)
class DeviceState:
    """CPU/GPU tensor state (batch-first)."""

    weights: torch.Tensor
    wealth: torch.Tensor
    mean: torch.Tensor
    var: torch.Tensor
    drawdown: torch.Tensor
    cum_pi: torch.Tensor
    peak_cum_pi: torch.Tensor
    rebirth_threshold: torch.Tensor

    def to(self, device: str | torch.device) -> "DeviceState":
        return DeviceState(
            weights=self.weights.to(device),
            wealth=self.wealth.to(device),
            mean=self.mean.to(device),
            var=self.var.to(device),
            drawdown=self.drawdown.to(device),
            cum_pi=self.cum_pi.to(device),
            peak_cum_pi=self.peak_cum_pi.to(device),
            rebirth_threshold=self.rebirth_threshold.to(device),
        )

    def to_cpu(self) -> "DeviceState":
        return self.to("cpu")


def to_device_state(selector, *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> DeviceState:
    """CPU selector -> DeviceState with batch dimension B=1."""
    if selector.w is None:
        raise ValueError("selector.w must be initialized before to_device_state()")

    w_np = np.asarray(selector.w, dtype=float)
    w_t = torch.as_tensor(w_np, device=device)
    base_dtype = dtype if dtype is not None else w_t.dtype
    w_t = w_t.to(dtype=base_dtype).unsqueeze(0)

    def _scalar(val: float) -> torch.Tensor:
        t = torch.as_tensor(np.array(val), device=device, dtype=base_dtype)
        return t.reshape(1, 1)

    return DeviceState(
        weights=w_t,
        wealth=_scalar(float(selector.wealth)),
        mean=_scalar(float(selector.stats.mu)),
        var=_scalar(float(selector.stats.var)),
        drawdown=_scalar(float(selector.stats.dd)),
        cum_pi=_scalar(float(selector.stats.cum_pi)),
        peak_cum_pi=_scalar(float(selector.stats.peak_cum_pi)),
        rebirth_threshold=_scalar(float(selector.rebirth_threshold)),
    )


def canonical_state_dump(
    selector,
    *,
    stack_manager: Any | None = None,
    sediment: Any | None = None,
) -> Dict[str, Any]:
    """Return a canonical, next-step-sufficient dump for test equivalence."""
    stats = selector.stats
    return {
        "selector": {
            "wealth": float(selector.wealth),
            "rebirth_threshold": float(selector.rebirth_threshold),
            "kind": str(selector.kind),
            "K": int(selector.K),
            "w": None if selector.w is None else np.asarray(selector.w, dtype=float),
            "_last_r": float(selector._last_r),
            "_last_c": float(selector._last_c),
        },
        "stats": {
            "beta": float(stats.beta),
            "mu": float(stats.mu),
            "var": float(stats.var),
            "dd": float(stats.dd),
            "cum_pi": float(stats.cum_pi),
            "peak_cum_pi": float(stats.peak_cum_pi),
            "seed_var": float(stats.seed_var),
        },
        "stack_manager": _dump_stack_manager(stack_manager),
        "sediment": _dump_sediment(sediment),
    }


def _dump_stack_manager(stack_manager: Any | None) -> Dict[str, Any] | None:
    if stack_manager is None:
        return None
    stacks_dump = []
    for sid, st in stack_manager.stacks.items():
        stacks_dump.append(
            {
                "stack_id": str(sid),
                "members": list(st.members.keys()),
                "_w_internal_keys": list(st._w_internal.keys()),
                "_w_internal_vals": [float(v) for v in st._w_internal.values()],
                "_alive": bool(st._alive),
                "_dt": float(st._dt),
                "_mu": float(st._mu),
                "_var": float(st._var),
                "_beta": float(st._beta),
                "_dd_peak": float(st._dd_peak),
                "_dd_val": float(st._dd_val),
                "_dd_max": float(st._dd_max),
                "_cvar": float(st._cvar),
            }
        )
    return {
        "stack_cfg": {
            "C_agg": float(stack_manager.stack_cfg.C_agg),
            "min_size": int(stack_manager.stack_cfg.min_size),
            "max_size": int(stack_manager.stack_cfg.max_size),
            "stack_weighting": str(stack_manager.stack_cfg.stack_weighting),
            "tau_mu": float(stack_manager.stack_cfg.tau_mu),
            "tau_vol": float(stack_manager.stack_cfg.tau_vol),
            "tau_cvar": float(stack_manager.stack_cfg.tau_cvar),
            "tau_dd": float(stack_manager.stack_cfg.tau_dd),
            "use_cvar": bool(stack_manager.stack_cfg.use_cvar),
        },
        "thresholds": {
            "tau_mu": float(stack_manager.thresholds.tau_mu),
            "tau_vol": float(stack_manager.thresholds.tau_vol),
            "tau_cvar": float(stack_manager.thresholds.tau_cvar),
            "tau_surv": float(stack_manager.thresholds.tau_surv),
            "tau_corr": float(stack_manager.thresholds.tau_corr),
            "min_size": int(stack_manager.thresholds.min_size),
            "max_size": int(stack_manager.thresholds.max_size),
        },
        "stacks": stacks_dump,
        "world_id": str(stack_manager.world_id),
        "phase_id": str(stack_manager.phase_id),
        "run_id": str(stack_manager.run_id),
        "_counter": int(stack_manager._counter),
        "_t_counter": int(stack_manager._t_counter),
        "_last_reject_t": int(stack_manager._last_reject_t),
    }


def _dump_sediment(sediment: Any | None) -> Dict[str, Any] | None:
    if sediment is None:
        return None
    nodes = []
    for node in sediment.nodes():
        nodes.append(
            {
                "node_id": int(node.node_id),
                "members": list(node.members),
                "mask": dict(node.mask),
                "world_id": str(node.world_id),
                "phase_id": str(node.phase_id),
                "t": int(node.t),
                "run_id": str(node.run_id),
            }
        )
    return {
        "forbid_pairs": bool(sediment.forbid_pairs),
        "_last_node_id_by_run": dict(sediment._last_node_id_by_run),
        "_next_node_id": int(sediment._next_node_id),
        "nodes": nodes,
    }


def toCuda(state_dump: Dict[str, Any], device: str, dtype: torch.dtype | None = None) -> CudaState:
    """CPU/Python -> CUDA tensor representation. No semantic changes."""
    selector = state_dump["selector"]
    stats = state_dump["stats"]

    tensors: Dict[str, torch.Tensor] = {}

    w = selector["w"]
    if w is not None:
        w_arr = np.asarray(w)
        w_t = torch.as_tensor(w_arr, device=device)
        if dtype is not None:
            w_t = w_t.to(dtype)
        tensors["selector.w"] = w_t
        if (w_t < 0).any() or not torch.isclose(w_t.sum(), torch.tensor(1.0, device=device, dtype=w_t.dtype)):
            raise ValueError("Invariant violation: weights not on simplex")

    for key in ("wealth", "rebirth_threshold", "_last_r", "_last_c"):
        val = np.array(selector[key])
        t = torch.as_tensor(val, device=device)
        if dtype is not None:
            t = t.to(dtype)
        tensors[f"selector.{key}"] = t

    for key in ("beta", "mu", "var", "dd", "cum_pi", "peak_cum_pi", "seed_var"):
        val = np.array(stats[key])
        t = torch.as_tensor(val, device=device)
        if dtype is not None:
            t = t.to(dtype)
        tensors[f"stats.{key}"] = t
        if key in ("var", "dd", "seed_var") and t.item() < 0:
            raise ValueError(f"Invariant violation: {key} < 0")

    meta = {
        "selector.kind": selector["kind"],
        "selector.K": int(selector["K"]),
        "stack_manager": state_dump.get("stack_manager"),
        "sediment": state_dump.get("sediment"),
    }

    # Stack internal state (if present)
    sm = state_dump.get("stack_manager")
    if sm is not None:
        meta["stack_manager"] = dict(sm)
        for st in sm.get("stacks", []):
            sid = st["stack_id"]
            meta[f"stack.{sid}.members"] = list(st["members"])
            meta[f"stack.{sid}._w_internal_keys"] = list(st["_w_internal_keys"])

            for key in ("_w_internal_vals", "_dt", "_mu", "_var", "_beta", "_dd_peak", "_dd_val", "_dd_max", "_cvar"):
                val = np.asarray(st[key], dtype=float)
                t = torch.as_tensor(val, device=device)
                if dtype is not None:
                    t = t.to(dtype)
                tensors[f"stack.{sid}.{key}"] = t

            alive_val = np.asarray(1.0 if st["_alive"] else 0.0, dtype=float)
            t_alive = torch.as_tensor(alive_val, device=device)
            if dtype is not None:
                t_alive = t_alive.to(dtype)
            tensors[f"stack.{sid}._alive"] = t_alive

    return CudaState(tensors=tensors, meta=meta)


def fromCuda(cuda_state: CudaState) -> Dict[str, Any]:
    """CUDA tensor representation -> CPU/Python state dump."""
    t = cuda_state.tensors
    meta = cuda_state.meta

    w_t = t.get("selector.w")
    w = None if w_t is None else w_t.detach().cpu().numpy()

    selector = {
        "wealth": float(t["selector.wealth"].item()),
        "rebirth_threshold": float(t["selector.rebirth_threshold"].item()),
        "kind": str(meta["selector.kind"]),
        "K": int(meta["selector.K"]),
        "w": w,
        "_last_r": float(t["selector._last_r"].item()),
        "_last_c": float(t["selector._last_c"].item()),
    }

    stats = {
        "beta": float(t["stats.beta"].item()),
        "mu": float(t["stats.mu"].item()),
        "var": float(t["stats.var"].item()),
        "dd": float(t["stats.dd"].item()),
        "cum_pi": float(t["stats.cum_pi"].item()),
        "peak_cum_pi": float(t["stats.peak_cum_pi"].item()),
        "seed_var": float(t["stats.seed_var"].item()),
    }

    stack_manager = meta.get("stack_manager")
    if stack_manager is not None:
        stacks = []
        for st in stack_manager.get("stacks", []):
            sid = st["stack_id"]
            keys = meta.get(f"stack.{sid}._w_internal_keys", [])
            vals_t = t.get(f"stack.{sid}._w_internal_vals")
            vals = [] if vals_t is None else [float(x) for x in vals_t.detach().cpu().numpy().tolist()]
            w_internal = dict(zip(keys, vals))
            stacks.append(
                {
                    "stack_id": str(sid),
                    "members": list(meta.get(f"stack.{sid}.members", [])),
                    "_w_internal_keys": list(keys),
                    "_w_internal_vals": vals,
                    "_alive": bool(t[f"stack.{sid}._alive"].item() >= 0.5),
                    "_dt": float(t[f"stack.{sid}._dt"].item()),
                    "_mu": float(t[f"stack.{sid}._mu"].item()),
                    "_var": float(t[f"stack.{sid}._var"].item()),
                    "_beta": float(t[f"stack.{sid}._beta"].item()),
                    "_dd_peak": float(t[f"stack.{sid}._dd_peak"].item()),
                    "_dd_val": float(t[f"stack.{sid}._dd_val"].item()),
                    "_dd_max": float(t[f"stack.{sid}._dd_max"].item()),
                    "_cvar": float(t[f"stack.{sid}._cvar"].item()),
                }
            )
        stack_manager = dict(stack_manager)
        stack_manager["stacks"] = stacks

    return {
        "selector": selector,
        "stats": stats,
        "stack_manager": stack_manager,
        "sediment": meta.get("sediment"),
    }
