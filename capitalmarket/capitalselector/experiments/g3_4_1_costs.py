from __future__ import annotations

from typing import Any, Dict, Iterable, List
import numpy as np

from ..builder import CapitalSelectorBuilder
from ..interfaces import validate_world_output
from ..stack import StackManager
from .topology_activation import topology_enabled, ensure_topology_state, update_topology_state
from ..cuda_state import canonical_state_dump
from ..worlds.regime_switch_bandit_world import RegimeSwitchBanditWorld


def _run_single(p: float, c_high: float, seed: int, steps: int, enable_topology: bool | None = None) -> Dict[str, Any]:
    world = RegimeSwitchBanditWorld(p=p, sigma=0.01, seed=int(seed), c_high=float(c_high))
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="regime", phase_id="G3_4_1", run_id=f"run_{seed}_{p}_{c_high}")

    rebirth_count = {"n": 0}
    original_rebirth = selector.rebirth

    def _rebirth_wrapper():
        rebirth_count["n"] += 1
        return original_rebirth()

    selector.rebirth = _rebirth_wrapper  # test/experiment hook

    dumps: Dict[int, Any] = {}
    dumps[0] = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    observables: List[Dict[str, Any]] = []
    topology_state: Dict[str, object] = {}
    for t in range(int(steps)):
        out = world.step(t)
        r_vec, c_total = validate_world_output(out)
        if selector.w is None or len(selector.w) != len(r_vec):
            selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
            selector.K = len(r_vec)
        selector.feedback_vector(r_vec, c_total, trace=None, freeze=False)
        if topology_enabled(enable_topology):
            ensure_topology_state(topology_state, len(r_vec))
            update_topology_state(topology_state, r_vec, stack_manager)

        w = selector.w
        if w is None:
            weight_entropy = 0.0
            dominant_channel = None
            weight_share_0 = 0.0
            weight_share_3 = 0.0
        else:
            w_clip = np.clip(w, 1e-12, 1.0)
            weight_entropy = float(-np.sum(w_clip * np.log(w_clip)))
            dominant_channel = int(np.argmax(w))
            weight_share_0 = float(w[0]) if len(w) > 0 else 0.0
            weight_share_3 = float(w[3]) if len(w) > 3 else 0.0

        observables.append(
            {
                "wealth": float(selector.wealth),
                "dominant_channel": dominant_channel,
                "stack_count": len(stack_manager.stacks),
                "rebirth_count": int(rebirth_count["n"]),
                "weight_entropy": weight_entropy,
                "weight_share_0": weight_share_0,
                "weight_share_3": weight_share_3,
            }
        )

    dumps[steps] = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    final_wealth = observables[-1]["wealth"] if observables else float(selector.wealth)
    mean_stack_count = float(np.mean([o["stack_count"] for o in observables])) if observables else 0.0
    total_rebirths = int(observables[-1]["rebirth_count"]) if observables else int(rebirth_count["n"])

    return {
        "p": float(p),
        "c_high": float(c_high),
        "seed": int(seed),
        "observables": observables,
        "aggregates": {
            "final_wealth": final_wealth,
            "mean_stack_count": mean_stack_count,
            "total_rebirths": total_rebirths,
            "weight_entropy": float(np.mean([o["weight_entropy"] for o in observables])) if observables else 0.0,
            "weight_share_0": float(np.mean([o["weight_share_0"] for o in observables])) if observables else 0.0,
            "weight_share_3": float(np.mean([o["weight_share_3"] for o in observables])) if observables else 0.0,
        },
        "dumps": dumps,
    }


def run_g3_4_1_sweep(
    p_values: Iterable[float], c_high_values: Iterable[float], seeds: Iterable[int], steps: int = 500, enable_topology: bool | None = None
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for p in p_values:
        for c_high in c_high_values:
            for seed in seeds:
                results.append(_run_single(float(p), float(c_high), int(seed), int(steps), enable_topology=enable_topology))
    return results
