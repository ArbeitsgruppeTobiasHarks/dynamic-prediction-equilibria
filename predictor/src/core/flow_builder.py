from __future__ import annotations

from typing import Generator

import numpy as np

from core.distributor import Distributor
from core.dynamic_flow import PartialDynamicFlow
from core.network import Network
from core.predictor import Predictor
from core.time_refinement import time_refinement
from utilities.interpolate import LinearlyInterpolatedFunction


class FlowBuilder:
    network: Network
    predictor: Predictor
    distributor: Distributor
    max_extension_length: float

    def __init__(self, network: Network, predictor: Predictor, distributor: Distributor, max_extension_length: float):
        self.network = network
        self.predictor = predictor
        self.max_extension_length = max_extension_length
        self.distributor = distributor

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
            new_inflow = self.distributor.distribute(flow, labels, costs)

            flow.extend(new_inflow, self.max_extension_length)
            phi = flow.times[-1]

            yield flow
