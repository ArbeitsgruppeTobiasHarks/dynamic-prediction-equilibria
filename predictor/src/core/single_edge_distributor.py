from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.distributor import Distributor
from core.graph import Node
from utilities.interpolate import LinearlyInterpolatedFunction


class SingleEdgeDistributor(Distributor):
    def type(self) -> str:
        return "Single-Edge Distributor"

    def distribute(
            self,
            phi: float,
            node_inflow: Dict[Node, float],
            sink: Node,
            past_queues: List[np.ndarray],
            labels: Dict[Node, LinearlyInterpolatedFunction],
            costs: List[LinearlyInterpolatedFunction]
    ) -> np.ndarry:
        m = len(self.network.graph.edges)
        new_inflow = np.zeros(m)
        for index in self.network.graph.nodes:
            v = self.network.graph.nodes[index]
            found_active_edge = False
            for e in v.outgoing_edges:
                is_active = labels[e.node_to](phi + costs[e.id](phi)) <= labels[v](phi)
                if is_active:
                    new_inflow[e.id] = node_inflow[v]
                    found_active_edge = True
                    break
            assert v == sink or found_active_edge
        return new_inflow
