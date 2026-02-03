"""Lightweight smoke test for CapitalSelector.

Supports execution as pytest import or script.
"""

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.rebirth import SwitchTypePolicy
import numpy as np


def main():
    cs = CapitalSelectorBuilder().build()
    cs.feedback(r=0.1, c=0.05)
    print(cs.state())

    # Simple two-cell stack with type-switch policy
    a = CapitalSelectorBuilder().with_kind("entrepreneur").build()
    b = CapitalSelectorBuilder().with_kind("banker").build()
    stack = (
        CapitalSelectorBuilder()
        .with_channels([a, b])
        .with_rebirth_policy(SwitchTypePolicy())
        .build()
    )

    for t in range(10):
        a.feedback(r=0.1, c=0.02)
        b.feedback(r=0.05, c=0.01)
        stack.stack_step()
        print(t, stack.state())

    # Meta selector over a small grid of cells
    cells = [CapitalSelectorBuilder().with_initial_wealth(1.0).build() for _ in range(9)]

    meta = (
        CapitalSelectorBuilder()
        .with_channels(cells)
        .with_kind("banker")
        .build()
    )

    for t in range(20):
        for c in cells:
            c.feedback(r=float(np.random.randn() * 0.01), c=0.01)
        meta.stack_step()

    print(meta.state())


if __name__ == "__main__":
    main()
