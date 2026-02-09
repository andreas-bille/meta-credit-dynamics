from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import json


@dataclass(frozen=True)
class SedimentNode:
    node_id: int
    members: List[str]
    mask: Dict[str, Any]
    world_id: str
    phase_id: str
    t: int
    run_id: str


class SedimentDAG:
    """Passive, append-only record of dead mediation configurations.

    Phase E v0:
    - nodes inserted on Stack dissolution
    - edges encode temporal order (chain per run_id)
    - filter is hard (is_forbidden)

    Canonical (v1):
    - Sediment is non-causal for weights/stats/wealth.
    - Its only effect is structural exclusion during stack formation.
    """

    def __init__(self, persist_path: Optional[Path] = None, *, forbid_pairs: bool = False, truncate: bool = False):
        self.persist_path = persist_path
        self.forbid_pairs = bool(forbid_pairs)

        self._nodes: Dict[int, SedimentNode] = {}
        self._last_node_id_by_run: Dict[str, int] = {}
        self._next_node_id = 1

        if self.persist_path is not None:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            if truncate:
                # create/clear file
                self.persist_path.write_text("", encoding="utf-8")

    # ---------- persistence helpers ----------

    def _append_event(self, event: str, payload: Dict[str, Any]):
        if self.persist_path is None:
            return
        with open(self.persist_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "payload": payload}, ensure_ascii=False) + "\n")

    # ---------- public API ----------

    def add_node(self, *, members: List[str], mask: Dict[str, Any], world_id: str, phase_id: str, t: int, run_id: str) -> int:
        members_sorted = sorted(list(members))
        node_id = self._next_node_id
        self._next_node_id += 1

        node = SedimentNode(
            node_id=node_id,
            members=members_sorted,
            mask=dict(mask),
            world_id=str(world_id),
            phase_id=str(phase_id),
            t=int(t),
            run_id=str(run_id),
        )
        self._nodes[node_id] = node

        # persist node
        self._append_event("SEDIMENT_NODE_ADDED", {
            "node_id": node.node_id,
            "members": node.members,
            "mask": node.mask,
            "world_id": node.world_id,
            "phase_id": node.phase_id,
            "t": node.t,
            "run_id": node.run_id,
        })

        # chain edge for v0
        last = self._last_node_id_by_run.get(node.run_id)
        if last is not None:
            self.add_edge(last, node_id, run_id=node.run_id, t=node.t)
        self._last_node_id_by_run[node.run_id] = node_id

        return node_id

    def add_edge(self, a: int, b: int, *, run_id: str, t: int):
        # We do not store full adjacency for v0; ordering is represented in the log.
        self._append_event("SEDIMENT_EDGE_ADDED", {"from": int(a), "to": int(b), "run_id": str(run_id), "t": int(t)})

    def nodes(self) -> List[SedimentNode]:
        return [self._nodes[k] for k in sorted(self._nodes.keys())]

    def is_forbidden(self, *, candidate_members: List[str], phase_id: str) -> bool:
        cand = set(candidate_members)
        phase_id = str(phase_id)

        for node in self._nodes.values():
            if node.phase_id != phase_id:
                continue
            M = set(node.members)
            if cand == M:
                return True
            if self.forbid_pairs:
                # forbid any pair from known-bad clique
                if len(M) >= 2:
                    mlist = list(M)
                    # check if any pair subset is contained in candidate
                    for i in range(len(mlist)):
                        for j in range(i+1, len(mlist)):
                            if {mlist[i], mlist[j]}.issubset(cand):
                                return True
        return False
