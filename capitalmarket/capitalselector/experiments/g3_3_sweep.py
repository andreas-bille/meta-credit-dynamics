from __future__ import annotations

from typing import Any, Dict, Iterable, List
import numpy as np

from ..builder import CapitalSelectorBuilder
from ..interfaces import validate_world_output
from ..stack import StackManager
from ..cuda_state import canonical_state_dump
from ..worlds.regime_switch_bandit_world import RegimeSwitchBanditWorld


def _run_single(p: float, seed: int, steps: int) -> Dict[str, Any]:
    world = RegimeSwitchBanditWorld(p=p, sigma=0.01, seed=int(seed))
    selector = CapitalSelectorBuilder().with_K(0).build()
    stack_manager = StackManager(world_id="regime", phase_id="G3_3", run_id=f"run_{seed}_{p}")

    rebirth_count = {"n": 0}
    original_rebirth = selector.rebirth

    def _rebirth_wrapper():
        rebirth_count["n"] += 1
        return original_rebirth()

    selector.rebirth = _rebirth_wrapper  # test/experiment hook

    dumps: Dict[int, Any] = {}
    dumps[0] = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    observables: List[Dict[str, Any]] = []
    for t in range(int(steps)):
        out = world.step(t)
        r_vec, c_total = validate_world_output(out)
        if selector.w is None or len(selector.w) != len(r_vec):
            selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
            selector.K = len(r_vec)
        selector.feedback_vector(r_vec, c_total, trace=None, freeze=False)

        observables.append(
            {
                "wealth": float(selector.wealth),
                "dominant_channel": int(np.argmax(selector.w)) if selector.w is not None else None,
                "stack_count": len(stack_manager.stacks),
                "rebirth_count": int(rebirth_count["n"]),
                "weight_entropy": float(
                    -np.sum(
                        np.clip(selector.w, 1e-12, 1.0) * np.log(np.clip(selector.w, 1e-12, 1.0))
                    )
                )
                if selector.w is not None
                else 0.0,
                "weight_share_0": float(selector.w[0]) if selector.w is not None else 0.0,
                "weight_share_3": float(selector.w[3]) if selector.w is not None and len(selector.w) > 3 else 0.0,
            }
        )

    dumps[steps] = canonical_state_dump(selector, stack_manager=stack_manager, sediment=stack_manager.sediment)

    final_wealth = observables[-1]["wealth"] if observables else float(selector.wealth)
    mean_stack_count = float(np.mean([o["stack_count"] for o in observables])) if observables else 0.0
    total_rebirths = int(observables[-1]["rebirth_count"]) if observables else int(rebirth_count["n"])

    return {
        "p": float(p),
        "seed": int(seed),
        "observables": observables,
        "aggregates": {
            "final_wealth": final_wealth,
            "mean_stack_count": mean_stack_count,
            "total_rebirths": total_rebirths,
        },
        "dumps": dumps,
    }


def run_g3_3_sweep(p_values: Iterable[float], seeds: Iterable[int], steps: int = 500) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for p in p_values:
        for seed in seeds:
            results.append(_run_single(p=float(p), seed=int(seed), steps=int(steps)))
    return results
