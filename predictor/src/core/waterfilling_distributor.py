from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.distributor import Distributor
from core.graph import Node
from core.machine_precision import eps
from core.waterfilling_procedure import waterfilling_procedure
from utilities.interpolate import LinearlyInterpolatedFunction


class WaterfillingDistributor(Distributor):
    def type(self) -> str:
        return "Waterfilling Distributor"

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
        capacity = self.network.capacity
        new_inflow = np.zeros(m)
        identity = LinearlyInterpolatedFunction([phi, phi + 1], [phi, phi + 1], (phi, float('inf')))
        for v in node_inflow.keys():
            if v == sink:
                continue

            active_edges = []
            for e in v.outgoing_edges:
                is_active = labels[e.node_to](phi + costs[e.id](phi)) <= labels[v](phi) + eps
                if is_active:
                    active_edges.append(e)
            assert len(active_edges) > 0, "No active edges have been found."
            if len(active_edges) == 1:
                new_inflow[active_edges[0].id] = node_inflow[v]
                continue

            a = []
            for e in active_edges:
                composition = labels[e.node_to].compose(identity.plus(costs[e.id]).ensure_monotone())
                assert composition.domain[0] == phi and composition.times[0] == phi
                a.append(composition.gradient(0))
            beta = [
                a[index] - 1 if past_queues[-1][e.id] > 0 else a[index] for index, e in enumerate(active_edges)
            ]
            alpha = [capacity[e.id] for e in active_edges]
            gamma = [0 if past_queues[-1][e.id] > 0 else capacity[e.id] for e in active_edges]
            h = [
                LinearlyInterpolatedFunction([0, gamma[index], gamma[index] + 1],
                                             [beta[index], beta[index], beta[index] + 1. / capacity[e.id]],
                                             (0, float('inf')))
                if gamma[index] > 0 else
                LinearlyInterpolatedFunction([gamma[index], gamma[index] + 1],
                                             [beta[index], beta[index] + 1. / capacity[e.id]],
                                             (0, float('inf')))
                for index, e in enumerate(active_edges)
            ]
            z = waterfilling_procedure(node_inflow[v], h, alpha, beta)
            for ind, e in enumerate(active_edges):
                new_inflow[e.id] = z[ind]
        return new_inflow
