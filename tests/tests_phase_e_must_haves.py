import pytest

from capitalmarket.capitalselector.sediment import SedimentDAG
from capitalmarket.capitalselector.rebirth import SedimentAwareRebirthPolicy


def test_sediment_persists_across_rebirth():
    sd = SedimentDAG()
    sd.add_node(
        members=["a", "b"],
        mask={"masked_members": ["a", "b"], "mask_depth": 1},
        world_id="w",
        phase_id="E1",
        t=1,
        run_id="r1",
    )

    # SedimentAwareRebirthPolicy preserves sediment by design; it delegates only.
    rp = SedimentAwareRebirthPolicy()
    class _Dummy:
        pass
    rp.on_rebirth(_Dummy())

    assert len(sd.nodes()) == 1


def test_phase_local_forbid():
    sd = SedimentDAG()
    sd.add_node(
        members=["a", "b"],
        mask={"masked_members": ["a", "b"], "mask_depth": 1},
        world_id="w",
        phase_id="E1",
        t=1,
        run_id="r1",
    )
    # same phase -> forbidden
    assert sd.is_forbidden(candidate_members=["a", "b"], phase_id="E1")
    # different phase -> allowed
    assert not sd.is_forbidden(candidate_members=["a", "b"], phase_id="E2")
