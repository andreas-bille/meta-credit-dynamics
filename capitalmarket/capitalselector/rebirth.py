from __future__ import annotations

from typing import Optional


class RebirthPolicy:
    """Defines what happens on selector rebirth.

    Note: Phase E resets most behavioural state outside the selector.
    This policy remains a hook for lightweight selector-local resets.
    """

    def on_rebirth(self, selector: "CapitalSelector"):
        pass


class SwitchTypePolicy(RebirthPolicy):
    """Example: switch kind on rebirth (banker <-> entrepreneur)."""

    def on_rebirth(self, selector: "CapitalSelector"):
        selector.kind = "banker" if selector.kind == "entrepreneur" else "entrepreneur"


class SedimentAwareRebirthPolicy(RebirthPolicy):
    """Delegating rebirth policy that preserves Sediment by design.

    Phase E v0 requirement: SedimentDAG persists across rebirths.
    Sediment lives outside the selector; this class is a semantic marker and
    a safe wrapper to avoid accidental resets in custom policies.
    """

    def __init__(self, inner: Optional[RebirthPolicy] = None):
        self.inner = inner

    def on_rebirth(self, selector: "CapitalSelector"):
        if self.inner is not None:
            self.inner.on_rebirth(selector)
