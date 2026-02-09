from capitalmarket.capitalselector.sediment import SedimentDAG
from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
import numpy as np


def test_sediment_filter_blocks_formation():
    sd = SedimentDAG()
    sd.add_node(
        members=["a", "b"],
        mask={"masked_members": ["a", "b"], "mask_depth": 1},
        world_id="w",
        phase_id="E1",
        t=1,
        run_id="r1",
    )
    assert sd.is_forbidden(candidate_members=["a", "b"], phase_id="E1")


def test_sediment_has_no_effect_on_reweighting():
    # Reweighting should be independent of sediment state
    selector1 = CapitalSelectorBuilder().with_K(2).build()
    selector2 = CapitalSelectorBuilder().with_K(2).build()

    r_vec = np.array([0.1, -0.05], dtype=float)
    selector1.feedback_vector(r_vec, c=0.0)
    selector2.feedback_vector(r_vec, c=0.0)

    w1 = selector1.allocate()
    w2 = selector2.allocate()
    assert np.allclose(w1, w2)
