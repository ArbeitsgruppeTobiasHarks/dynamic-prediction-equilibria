from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.distributor import Distributor
from core.graph import Node
from utilities.interpolate import LinearlyInterpolatedFunction


class SingleEdgeDistributor(Distributor):
    def type(self) -> str:
        return "Single-Edge Distributor"

    def supports_const(self) -> bool:
        return True

    def distribute_const(self, phi: float, node_inflow: Dict[Node, float], sink: Node, past_queues: List[np.ndarray],
                         labels: Dict[Node, float], costs: np.ndarray) -> np.ndarray:
        m = len(self.network.graph.edges)
        new_inflow = np.zeros(m)
        for v in labels.keys():
            if v == sink:
                continue
            found_active_edge = False
            for e in v.outgoing_edges:
                w = e.node_to
                if w not in labels.keys():
                    continue
                is_active = costs[e.id] + labels[w] <= labels[v]
                if is_active:
                    new_inflow[e.id] = node_inflow[v]
                    found_active_edge = True
                    break
            assert found_active_edge
        return new_inflow

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
        for v in labels.keys():
            found_active_edge = False
            for e in v.outgoing_edges:
                w = e.node_to
                if w not in labels.keys():
                    continue
                is_active = labels[w](phi + costs[e.id](phi)) <= labels[v](phi)
                if is_active:
                    new_inflow[e.id] = node_inflow[v]
                    found_active_edge = True
                    break
            assert v == sink or found_active_edge
        return new_inflow
