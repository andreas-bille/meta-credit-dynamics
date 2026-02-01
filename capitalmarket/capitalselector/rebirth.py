from __future__ import annotations

class RebirthPolicy:
    """
    Legt fest, was bei Rebirth passiert.
    """

    def on_rebirth(self, selector: "CapitalSelector"):
        pass


class SwitchTypePolicy(RebirthPolicy):
    """
    Beispiel: Typwechsel bei Rebirth (Banker <-> Unternehmer)
    """

    def on_rebirth(self, selector: "CapitalSelector"):
        selector.kind = "banker" if selector.kind == "entrepreneur" else "entrepreneur"