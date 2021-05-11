from __future__ import annotations

from typing import Generator

import numpy as np

from core.dynamic_flow import PartialDynamicFlow
from core.network import Network
from core.predictor import Predictor
from core.time_refinement import time_refinement
from utilities.interpolate import LinearlyInterpolatedFunction


class FlowBuilder:
    network: Network
    predictor: Predictor
    max_extension_length: float

    def __init__(self, network: Network, predictor: Predictor, max_extension_length: float):
        self.network = network
        self.predictor = predictor
        self.max_extension_length = max_extension_length

    def build_flow(self) -> Generator[PartialDynamicFlow, None, None]:
        flow = PartialDynamicFlow(self.network)
        phi = flow.times[-1]
        m = len(self.network.graph.edges)
        travel_time = self.network.travel_time
        capacity = self.network.capacity
        while phi < float('inf'):
            prediction = self.predictor.predict(flow.times, flow.queues)
            pred_times = prediction.times
            pred_queues = np.asarray(prediction.queues)
            pred_costs = [travel_time[e] + pred_queues[:, e] / capacity[e] for e in range(m)]

            costs = [
                LinearlyInterpolatedFunction(pred_times, pred_costs[e], (phi, float('inf')))
                for e in range(m)
            ]

            # Calculate dynamic shortest paths
            labels = time_refinement(self.network.graph, self.network.sink, costs, phi)

            # Determine a good flow-split on active outgoing edges

            # Todo: Will just put everything in one edge for now
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
            flow.extend(new_inflow, self.max_extension_length)
            phi = flow.times[-1]

            yield flow
