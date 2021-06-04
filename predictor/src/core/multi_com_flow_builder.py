from __future__ import annotations

from typing import Generator, Optional, Dict, List, Set

import numpy as np
from functools import reduce

from core.predictors.constant_predictor import ConstantPredictor
from core.dijkstra import dijkstra, realizing_dijkstra
from core.graph import Node, Edge
from core.machine_precision import eps
from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network
from core.predictors.predictor import Predictor
from core.predictors.zero_predictor import ZeroPredictor
from utilities.interpolate import LinearlyInterpolatedFunction


class MultiComFlowBuilder:
    network: Network
    predictors: List[Predictor]
    reroute_interval: Optional[float]
    _active_edges: List[Dict[Node, List[Edge]]]

    def __init__(self,
                 network: Network,
                 predictors: List[Predictor],
                 reroute_interval: Optional[float]  # None means rerouting every time some outflow changes
                 ):
        self.network = network
        self.predictors = predictors
        self.reroute_interval = reroute_interval

    def build_flow(self) -> Generator[MultiComPartialDynamicFlow, None, None]:
        flow = MultiComPartialDynamicFlow(self.network)
        n = len(self.network.commodities)
        m = len(self.network.graph.edges)
        travel_time = self.network.travel_time
        capacity = self.network.capacity
        self._active_edges = [{} for i in range(n)]

        # Preprocessing...
        # For each commodity find the nodes that reach the sink
        reaching_nodes = [
            self.network.graph.get_nodes_reaching(commodity.sink) for commodity in self.network.commodities
        ]
        reachable_nodes = [
            self.network.graph.get_reachable_nodes(commodity.source) for commodity in self.network.commodities
        ]
        important_nodes = [
            reaching_nodes[i].intersection(reachable_nodes[i]) for i in range(len(self.network.commodities))
        ]
        assert all(c.source in reaching_nodes[i] for i, c in enumerate(self.network.commodities))

        next_reroute_time = flow.phi
        costs = []
        handle_nodes = set(self.network.graph.nodes.values())

        yield flow
        while flow.phi < float('inf'):
            if self.reroute_interval is None or flow.phi >= next_reroute_time - eps:
                # PREDICT NEW QUEUES
                self._active_edges = [{} for i in range(n)]
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

                const_costs = {}
                for k, predictor in enumerate(self.predictors):
                    if isinstance(predictor, ConstantPredictor) or isinstance(predictor, ZeroPredictor):
                        const_costs[k] = travel_time + pred_queues_list[k][0, :] / capacity

                # CALCULATE NEW SHORTEST PATHS
                for i, commodity in enumerate(self.network.commodities):
                    if isinstance(self.predictors[commodity.predictor], ConstantPredictor) \
                            or isinstance(self.predictors[commodity.predictor], ZeroPredictor):
                        # PRECALCULATE FOR CONSTANT PREDICTORS
                        const_labels = dijkstra(commodity.sink, const_costs[commodity.predictor])
                        for v in important_nodes[i]:
                            if v == commodity.sink:
                                continue
                            active_edges = []
                            for e in v.outgoing_edges:
                                w = e.node_to
                                if w not in const_labels.keys():
                                    continue
                                if const_costs[commodity.predictor][e.id] + const_labels[w] <= const_labels[v] + eps:
                                    active_edges.append(e)
                            assert len(active_edges) > 0
                            self._active_edges[i][v] = active_edges

                next_reroute_time += self.reroute_interval
                handle_nodes = set(self.network.graph.nodes.values())

            # DETERMINE OUTFLOW SPLIT
            inflow_per_comm: List[Dict[int, float]] = []
            for i, commodity in enumerate(self.network.commodities):
                node_inflow: Dict[Node, float] = {
                    v: sum(flow.outflow[e.id][i](flow.phi) for e in v.incoming_edges)
                    for v in important_nodes[i].intersection(handle_nodes)
                }
                if commodity.source in handle_nodes:
                    node_inflow[commodity.source] += commodity.demand

                new_inflow_i = self.distribute(i, next_reroute_time - self.reroute_interval, node_inflow,
                                               commodity.sink,
                                               important_nodes[i], costs[commodity.predictor])
                inflow_per_comm.append(new_inflow_i)

            all_keys = reduce(lambda acc, item: acc.union(item.keys()), inflow_per_comm, set())
            new_inflow = {
                e: np.asarray([
                    inflow_per_comm[i][e]
                    if e in inflow_per_comm[i].keys() else
                    flow.inflow[e][i].values[-1]
                    for i in range(n)
                ]) for e in all_keys
            }

            max_ext_length = next_reroute_time - flow.phi
            changed_edges = flow.extend(new_inflow, max_ext_length)

            yield flow

            handle_nodes = set()
            for e in changed_edges:
                handle_nodes.add(self.network.graph.edges[e].node_to)

    def distribute(self, i: int, phi: float, node_inflow: Dict[Node, float], sink: Node, interesting_nodes: Set[Node],
                   costs: List[LinearlyInterpolatedFunction]) -> Dict[int, float]:
        new_inflow = {}
        for s in node_inflow.keys():
            if s == sink:
                continue

            if node_inflow[s] < eps:
                for e in s.outgoing_edges:
                    new_inflow[e.id] = 0.
                continue

            if s not in self._active_edges[i].keys():
                # Do Time-Dependent dijkstra from s to t to find active outgoing edges of s
                arrival_times, realised_cost = realizing_dijkstra(phi, s, sink, interesting_nodes, costs)

                # Now search all active edges leading to t.
                active_edges = []
                touched_nodes = {sink}
                queue: List[Node] = [sink]
                while queue:
                    w = queue.pop()
                    for e in w.incoming_edges:
                        if e not in realised_cost.keys():
                            continue
                        v: Node = e.node_from
                        if arrival_times[v] + realised_cost[e] <= arrival_times[w] + eps:
                            if v == s:
                                active_edges.append(e)
                            if v not in touched_nodes:
                                touched_nodes.add(v)
                                queue.append(v)

                assert len(active_edges) > 0
                self._active_edges[i][s] = active_edges

            active_edges = self._active_edges[i][s]
            distribution = node_inflow[s] / len(active_edges)
            for e in s.outgoing_edges:
                if e in active_edges:
                    new_inflow[e.id] = distribution
                else:
                    new_inflow[e.id] = 0.

        return new_inflow
