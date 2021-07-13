from __future__ import annotations

from typing import Generator, Optional

import numpy as np

from core.distributor import Distributor
from core.dynamic_flow import PartialDynamicFlow
from core.network import Network
from core.predictor import Predictor
from core.time_refinement import time_refinement
from utilities.piecewise_linear import PiecewiseLinear


class FlowBuilder:
    network: Network
    predictor: Predictor
    distributor: Distributor
    max_extension_length: float

    def __init__(self, network: Network,
                 predictor: Predictor,
                 distributor: Distributor,
                 max_extension_length: float,
                 stop_at_queue_changes: Optional[bool] = False):
        self.network = network
        assert len(self.network.commodities) == 1
        self.predictor = predictor
        self.max_extension_length = max_extension_length
        self.distributor = distributor
        self.stop_at_queue_changes = stop_at_queue_changes

    def build_flow(self) -> Generator[PartialDynamicFlow, None, None]:
        flow = PartialDynamicFlow(self.network)
        phi = flow.times[-1]
        m = len(self.network.graph.edges)
        source = self.network.commodities[0].source
        sink = self.network.commodities[0].sink
        demand = self.network.commodities[0].demand
        travel_time = self.network.travel_time
        capacity = self.network.capacity
        while phi < float('inf'):
            prediction = self.predictor.predict(flow.times, flow.queues)
            pred_times = prediction.times
            pred_queues = np.asarray(prediction.queues)
            assert np.min(pred_queues) >= 0.
            pred_costs = [travel_time[e] + pred_queues[:, e] / capacity[e] for e in range(m)]

            costs = [
                PiecewiseLinear(pred_times, pred_costs[e], (phi, float('inf')))
                for e in range(m)
            ]

            # Calculate dynamic shortest paths
            labels = time_refinement(self.network.graph, sink, costs, phi)

            node_inflow = {
                v: sum(flow.curr_outflow[e.id] for e in v.incoming_edges)
                for v in self.network.graph.nodes.values()
            }
            node_inflow[source] += demand
            # Determine a good flow-split on active outgoing edges
            new_inflow = self.distributor.distribute(phi, node_inflow, sink, flow.queues, labels, costs)

            flow.extend(new_inflow, self.max_extension_length, self.stop_at_queue_changes)
            phi = flow.times[-1]

            yield flow
