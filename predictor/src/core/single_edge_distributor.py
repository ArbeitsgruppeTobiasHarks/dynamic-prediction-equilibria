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

    def distribute_const(
            self, phi: float, node_inflow: Dict[Node, float], sink: Node,
            past_queues: List[LinearlyInterpolatedFunction], labels: Dict[Node, float], costs: np.ndarray
    ) -> Dict[int, float]:
        new_inflow: Dict[int, float] = {}
        for v in node_inflow.keys():
            if v == sink:
                continue
            found_active_edge = False
            for e in v.outgoing_edges:
                w = e.node_to
                if w not in labels.keys():
                    continue
                is_active = costs[e.id] + labels[w] <= labels[v]
                if is_active and not found_active_edge:
                    new_inflow[e.id] = node_inflow[v]
                    found_active_edge = True
                else:
                    new_inflow[e.id] = 0.
            assert found_active_edge
        return new_inflow

    def distribute(
            self,
            phi: float,
            node_inflow: Dict[Node, float],
            sink: Node,
            queues: np.ndarray,
            labels: Dict[Node, LinearlyInterpolatedFunction],
            costs: List[LinearlyInterpolatedFunction]
    ) -> Dict[int, float]:
        new_inflow: Dict[int, float] = {}
        for v in node_inflow.keys():
            if v == sink:
                continue
            found_active_edge = False
            for e in v.outgoing_edges:
                w = e.node_to
                if w not in labels.keys():
                    continue
                is_active = labels[w](phi + costs[e.id](phi)) <= labels[v](phi)
                if is_active and not found_active_edge:
                    new_inflow[e.id] = node_inflow[v]
                    found_active_edge = True
                else:
                    new_inflow[e.id] = 0.
            assert found_active_edge
        return new_inflow
