from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.distributor import Distributor
from core.dynamic_flow import PartialDynamicFlow
from core.graph import Node
from utilities.interpolate import LinearlyInterpolatedFunction


class SingleEdgeDistributor(Distributor):
    def distribute(
            self,
            flow: PartialDynamicFlow,
            labels: Dict[Node, LinearlyInterpolatedFunction],
            costs: List[LinearlyInterpolatedFunction]
    ) -> np.ndarry:
        m = len(self.network.graph.edges)
        phi = flow.times[-1]
        new_inflow = np.zeros(m)
        for index in self.network.graph.nodes:
            v = self.network.graph.nodes[index]
            inflow = sum(flow.curr_outflow[e.id] for e in v.incoming_edges)
            # Todo: Remove this in favor of a network attribute
            if v.id == 0:
                inflow += 3
            found_active_edge = False
            for e in v.outgoing_edges:
                is_active = labels[e.node_to](phi + costs[e.id](phi)) <= labels[v](phi)
                if is_active:
                    new_inflow[e.id] = inflow
                    found_active_edge = True
                    break
            assert v == self.network.sink or found_active_edge
        return new_inflow
