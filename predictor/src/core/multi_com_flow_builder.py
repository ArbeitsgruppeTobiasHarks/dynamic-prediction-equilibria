from __future__ import annotations

from typing import Generator, Optional, Dict, List

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
    predictors: List[Predictor]
    distributor: Distributor
    reroute_interval: Optional[float]

    def __init__(self, network: Network,
                 predictors: List[Predictor],
                 distributor: Distributor,
                 reroute_interval: Optional[float]  # None means rerouting every time some outflow changes
                 ):
        self.network = network
        self.predictors = predictors
        self.reroute_interval = reroute_interval
        self.distributor = distributor

    def build_flow(self) -> Generator[MultiComPartialDynamicFlow, None, None]:
        flow = MultiComPartialDynamicFlow(self.network)
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

        next_reroute_time = flow.phi
        costs = []
        labels: Dict[int, Dict[Node, LinearlyInterpolatedFunction]] = {}
        const_labels = {}
        const_costs = {}
        handle_nodes = set(self.network.graph.nodes.values())
        queues = None

        while flow.phi < float('inf'):
            if self.reroute_interval is None or flow.phi >= next_reroute_time - eps:
                # PREDICT NEW QUEUES
                predictions = [predictor.predict_from_fcts(flow.queues, flow.phi) for predictor in self.predictors]
                pred_queues_list = [np.asarray(prediction.queues) for prediction in predictions]
                pred_costs = [[travel_time[e] + pred_queues[:, e] / capacity[e] for e in range(m)]
                              for pred_queues in pred_queues_list]
                costs = [
                    [
                        LinearlyInterpolatedFunction(predictions[k].times, pred_costs[k][e], (flow.phi, float('inf')))
                        for e in range(m)
                    ]
                    for k in range(len(self.predictors))]

                const_costs = None
                # CALCULATE NEW SHORTEST PATHS
                for i, commodity in enumerate(self.network.commodities):
                    if isinstance(self.predictors[commodity.predictor], ConstantPredictor):
                        # Handle constant predictor separately for better performance
                        if const_costs is None:
                            const_costs = travel_time + pred_queues_list[commodity.predictor][0, :] / capacity
                        const_labels[i] = dijkstra(commodity.sink, const_costs)
                        if self.distributor.supports_const():
                            continue

                        labels[i] = {
                            v: LinearlyInterpolatedFunction([flow.phi, flow.phi + 1], [label, label],
                                                            (flow.phi, float('inf')))
                            for v, label in const_labels[i].items()
                        }
                    else:
                        labels[i] = bellman_ford(commodity.sink, costs[commodity.predictor], flow.phi)
                next_reroute_time += self.reroute_interval
                handle_nodes = set(self.network.graph.nodes.values())

            # DETERMINE OUTFLOW SPLIT
            if self.distributor.needs_queues():
                queues = np.asarray([queue(flow.phi) for queue in flow.queues])

            inflow_per_comm: List[Dict[int, float]] = []
            for i, commodity in enumerate(self.network.commodities):
                node_inflow: Dict[Node, float] = {
                    v: sum(flow.outflow[e.id][i](flow.phi) for e in v.incoming_edges)
                    for v in reaching_nodes[i].intersection(handle_nodes)
                }
                if commodity.source in handle_nodes:
                    node_inflow[commodity.source] += commodity.demand

                if isinstance(self.predictors[commodity.predictor], ConstantPredictor) and \
                        self.distributor.supports_const():
                    new_inflow_i = self.distributor.distribute_const(
                        flow.phi, node_inflow, commodity.sink, queues, const_labels[i], const_costs
                    )
                    inflow_per_comm.append(new_inflow_i)
                else:
                    new_inflow_i = self.distributor.distribute(flow.phi, node_inflow, commodity.sink, queues, labels[i],
                                                               costs[commodity.predictor])
                    inflow_per_comm.append(new_inflow_i)

            new_inflow = {
                e: np.asarray([
                    inflow_per_comm[i][e] if e in inflow_per_comm[i].keys() else 0. for i in range(n)
                ]) for e in inflow_per_comm[0].keys()
            }

            max_ext_length = next_reroute_time - flow.phi
            changed_edges = flow.extend(new_inflow, max_ext_length)

            yield flow

            handle_nodes = set()
            for e in changed_edges:
                handle_nodes.add(self.network.graph.edges[e].node_to)
