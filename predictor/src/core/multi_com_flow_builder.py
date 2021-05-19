from __future__ import annotations

from typing import Generator, Optional, Dict

import numpy as np

from core.bellman_ford import bellman_ford
from core.constant_predictor import ConstantPredictor
from core.dijkstra import dijkstra
from core.distributor import Distributor
from core.graph import Node
from core.machine_precision import eps
from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network
from core.predictor import Predictor
from utilities.interpolate import LinearlyInterpolatedFunction


class MultiComFlowBuilder:
    network: Network
    predictor: Predictor
    distributor: Distributor
    reroute_interval: Optional[float]

    def __init__(self, network: Network,
                 predictor: Predictor,
                 distributor: Distributor,
                 reroute_interval: Optional[float],  # None means rerouting every time some outflow changes
                 stop_at_queue_changes: Optional[bool] = False):
        self.network = network
        self.predictor = predictor
        self.reroute_interval = reroute_interval
        self.distributor = distributor
        self.stop_at_queue_changes = stop_at_queue_changes

    def build_flow(self) -> Generator[MultiComPartialDynamicFlow, None, None]:
        flow = MultiComPartialDynamicFlow(self.network)
        phi = flow.times[-1]
        n = len(self.network.commodities)
        m = len(self.network.graph.edges)
        travel_time = self.network.travel_time
        capacity = self.network.capacity

        # Preprocessing...
        # For each commodity find the nodes that reach the sink
        reaching_nodes = [
            self.network.graph.get_nodes_reaching(commodity.sink) for commodity in self.network.commodities
        ]
        assert all(c.source in reaching_nodes[i] for i, c in enumerate(self.network.commodities))

        next_reroute_time = phi
        costs = []
        labels = {}
        const_labels = {}
        const_costs = {}

        while phi < float('inf'):
            if self.reroute_interval is None or phi >= next_reroute_time - eps:
                # PREDICT NEW QUEUES
                prediction = self.predictor.predict(flow.times, flow.queues)
                pred_queues = np.asarray(prediction.queues)
                pred_costs = [travel_time[e] + pred_queues[:, e] / capacity[e] for e in range(m)]

                costs = [
                    LinearlyInterpolatedFunction(prediction.times, pred_costs[e], (phi, float('inf')))
                    for e in range(m)
                ]
                if isinstance(self.predictor, ConstantPredictor):
                    const_costs = travel_time + pred_queues[0, :] / capacity

                # CALCULATE NEW SHORTEST PATHS
                for i, commodity in enumerate(self.network.commodities):
                    if isinstance(self.predictor, ConstantPredictor):
                        const_labels[i] = dijkstra(commodity.sink, const_costs)
                        if self.distributor.supports_const():
                            continue

                        labels[i] = {
                            v: LinearlyInterpolatedFunction([phi, phi + 1], [label, label], (phi, float('inf')))
                            for v, label in const_labels[i].items()
                        }
                    else:
                        labels[i] = bellman_ford(commodity.sink, costs, phi)
                next_reroute_time += self.reroute_interval

            # DETERMINE OUTFLOW SPLIT
            new_inflow = np.zeros((n, m))
            for i, commodity in enumerate(self.network.commodities):
                node_inflow: Dict[Node, float] = {
                    v: sum(flow.curr_outflow[i, e.id] for e in v.incoming_edges)
                    for v in reaching_nodes[i]
                }
                node_inflow[commodity.source] += commodity.demand

                # Filter nodes with zero inflow
                node_inflow = {
                    v: inflow for v, inflow in node_inflow.items() if inflow > 0
                }

                if self.distributor.supports_const():
                    new_inflow[i, :] = self.distributor.distribute_const(
                        phi, node_inflow, commodity.sink, flow.queues, const_labels[i], const_costs
                    )
                else:
                    new_inflow[i, :] = self.distributor.distribute(
                        phi, node_inflow, commodity.sink, flow.queues, labels[i], costs
                    )

            max_ext_length = next_reroute_time - phi
            flow.extend(new_inflow, max_ext_length, self.stop_at_queue_changes)
            phi = flow.times[-1]

            yield flow
