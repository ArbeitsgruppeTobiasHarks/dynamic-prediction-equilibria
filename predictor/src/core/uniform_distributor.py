from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.distributor import Distributor
from core.graph import Node
from core.machine_precision import eps
from utilities.piecewise_linear import PiecewiseLinear


class UniformDistributor(Distributor):
    def type(self) -> str:
        return "Uniform Distributor"

    def supports_const(self) -> bool:
        return True

    def needs_queues(self) -> bool:
        return False

    def distribute_const(
            self, phi: float, node_inflow: Dict[Node, float], sink: Node,
            past_queues: List[PiecewiseLinear], labels: Dict[Node, float], costs: np.ndarray
    ) -> Dict[int, float]:
        new_inflow: Dict[int, float] = {}
        for v in node_inflow.keys():
            if v == sink:
                continue
            active_edges = []
            for e in v.outgoing_edges:
                w = e.node_to
                if w not in labels.keys():
                    continue
                is_active = costs[e.id] + labels[w] <= labels[v]
                if is_active:
                    active_edges.append(e)
                else:
                    new_inflow[e.id] = 0.
            assert len(active_edges) > 0
            distribution = node_inflow[v] / len(active_edges)
            for e in active_edges:
                new_inflow[e.id] = distribution
        return new_inflow

    def distribute(
            self,
            phi: float,
            node_inflow: Dict[Node, float],
            sink: Node,
            queues: np.ndarray,
            labels: Dict[Node, PiecewiseLinear],
            costs: List[PiecewiseLinear]
    ) -> Dict[int, float]:
        new_inflow: Dict[int, float] = {}
        for v in node_inflow.keys():
            if v == sink:
                continue
            if node_inflow[v] == 0:
                for e in v.outgoing_edges:
                    new_inflow[e.id] = 0.
                continue

            active_edges = []
            for e in v.outgoing_edges:
                w = e.node_to
                if w not in labels.keys():
                    continue
                is_active = labels[w](phi + costs[e.id](phi)) <= labels[v](phi) + eps
                if is_active:
                    active_edges.append(e)
                else:
                    new_inflow[e.id] = 0.

            assert len(active_edges) > 0
            distribution = node_inflow[v] / len(active_edges)
            for e in active_edges:
                new_inflow[e.id] = distribution
        return new_inflow
