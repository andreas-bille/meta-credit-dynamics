from __future__ import annotations

from typing import Callable
import numpy as np
import torch


def _extract_eta(reweight_fn: Callable) -> float:
    closure = getattr(reweight_fn, "__closure__", None)
    if not closure:
        raise ValueError("Cannot extract eta from reweight_fn closure")
    for cell in closure:
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, (int, float)):
            return float(val)
        if hasattr(val, "_eta"):
            eta_val = getattr(val, "_eta")
            if isinstance(eta_val, (int, float)):
                return float(eta_val)
    raise ValueError("No eta found in reweight_fn closure")


class CudaCore:
    """CUDA backend skeleton with reweight port (Step 3)."""

    def step(self, selector, r_vec, c_total, *, freeze: bool) -> None:
        if freeze:
            selector._enforce_invariants()
            return

        if selector.w is None:
            return

        r_vec = np.asarray(r_vec, dtype=float)
        c_total = float(c_total)
        w_np = np.asarray(selector.w, dtype=float)

        r_total = float(r_vec.sum())
        pi_vec = r_vec - w_np * c_total
        pi_total = r_total - c_total

        selector._last_r = r_total
        selector._last_c = c_total
        selector.wealth += pi_total

        selector.stats.update(pi_total)
        score = pi_vec - float(selector.stats.mu)
        eta = _extract_eta(selector.reweight_fn)

        with torch.no_grad():
            w = torch.as_tensor(w_np).unsqueeze(0)
            g = torch.zeros_like(w)
            g[:] = torch.as_tensor(score, dtype=w.dtype)
            w_new = w * torch.exp(eta * g)
            w_new = torch.clamp(w_new, min=0.0)
            s = w_new.sum(dim=1, keepdim=True)
            w_new = torch.where(
                s == 0.0,
                torch.ones_like(w_new) / w_new.shape[1],
                w_new / s,
            )

        selector.w = w_new.squeeze(0).detach().cpu().numpy()

        if selector.wealth < selector.rebirth_threshold:
            selector.rebirth()

        selector._enforce_invariants()
