from __future__ import annotations

from typing import Generator, Optional

import numpy as np

from core.distributor import Distributor
from core.dynamic_flow import PartialDynamicFlow
from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network
from core.predictor import Predictor
from core.time_refinement import time_refinement
from utilities.interpolate import LinearlyInterpolatedFunction


class MultiComFlowBuilder:
    network: Network
    predictor: Predictor
    distributor: Distributor
    max_extension_length: float

    def __init__(self, network: Network,
                 predictor: Predictor,
                 distributor: Distributor,
                 max_extension_length: float,
                 stop_at_queue_changes: Optional[bool] = False,
                 reroute_on_changes: Optional[bool] = False):
        self.network = network
        self.predictor = predictor
        self.max_extension_length = max_extension_length
        self.distributor = distributor
        self.stop_at_queue_changes = stop_at_queue_changes

    def build_flow(self) -> Generator[PartialDynamicFlow, None, None]:
        flow = MultiComPartialDynamicFlow(self.network)
        phi = flow.times[-1]
        n = len(self.network.commodities)
        m = len(self.network.graph.edges)
        travel_time = self.network.travel_time
        capacity = self.network.capacity
        last_prediction_time = float('-inf')
        while phi < float('inf'):

            prediction = self.predictor.predict(flow.times, flow.queues)
            pred_times = prediction.times
            pred_queues = np.asarray(prediction.queues)
            pred_costs = [travel_time[e] + pred_queues[:, e] / capacity[e] for e in range(m)]

            costs = [
                LinearlyInterpolatedFunction(pred_times, pred_costs[e], (phi, float('inf')))
                for e in range(m)
            ]

            new_inflow = np.zeros((n, m))
            for i, sink in enumerate(self.network.commodities):

                # Calculate dynamic shortest paths
                labels = time_refinement(self.network.graph, sink, costs, phi)

                # Determine a good flow-split on active outgoing edges
                new_inflow[i, :] = self.distributor.distribute(phi, flow.curr_outflow[i, :], flow.queues, labels, costs)

            flow.extend(new_inflow, self.max_extension_length, self.stop_at_queue_changes)
            phi = flow.times[-1]

            yield flow
