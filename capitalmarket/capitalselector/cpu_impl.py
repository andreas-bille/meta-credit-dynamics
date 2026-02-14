from __future__ import annotations


class CpuCore:
    """Thin wrapper around existing CPU step logic."""

    def step(self, selector, r_vec, c_total, *, freeze: bool) -> None:
        selector.feedback_vector(r_vec, c_total, trace=None, freeze=freeze)
