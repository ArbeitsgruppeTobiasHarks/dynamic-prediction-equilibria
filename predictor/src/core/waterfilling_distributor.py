from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.distributor import Distributor
from core.dynamic_flow import PartialDynamicFlow
from core.graph import Node
from core.machine_precision import eps
from core.waterfilling_procedure import waterfilling_procedure
from utilities.interpolate import LinearlyInterpolatedFunction


class WaterfillingDistributor(Distributor):
    def type(self) -> str:
        return "Waterfilling Distributor"

    def distribute(
            self,
            flow: PartialDynamicFlow,
            labels: Dict[Node, LinearlyInterpolatedFunction],
            costs: List[LinearlyInterpolatedFunction]
    ) -> np.ndarry:
        m = len(self.network.graph.edges)
        phi = flow.times[-1]
        capacity = self.network.capacity
        new_inflow = np.zeros(m)
        identity = LinearlyInterpolatedFunction([phi, phi + 1], [phi, phi + 1], (phi, float('inf')))
        for i in self.network.graph.nodes.keys():
            v = self.network.graph.nodes[i]
            if v == self.network.sink:
                continue
            inflow = sum(flow.curr_outflow[e.id] for e in v.incoming_edges)
            # Todo: Remove this in favor of a network attribute
            if v.id == 0:
                inflow += 3

            active_edges = []
            for e in v.outgoing_edges:
                is_active = labels[e.node_to](phi + costs[e.id](phi)) <= labels[v](phi) + eps
                if is_active:
                    active_edges.append(e)
            assert len(active_edges) > 0, "No active edges have been found."
            if len(active_edges) == 1:
                new_inflow[active_edges[0].id] = inflow
                continue

            a = []
            for e in active_edges:
                composition = labels[e.node_to].compose(identity.plus(costs[e.id]).ensure_monotone())
                assert composition.domain[0] == phi and composition.times[0] == phi
                a.append(composition.gradient(0))
            beta = [
                a[index] - 1 if flow.queues[-1][e.id] > 0 else a[index] for index, e in enumerate(active_edges)
            ]
            alpha = [capacity[e.id] for e in active_edges]
            gamma = [0 if flow.queues[-1][e.id] > 0 else capacity[e.id] for e in active_edges]
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
            z = waterfilling_procedure(inflow, h, alpha, beta)
            for ind, e in enumerate(active_edges):
                new_inflow[e.id] = z[ind]
        return new_inflow
