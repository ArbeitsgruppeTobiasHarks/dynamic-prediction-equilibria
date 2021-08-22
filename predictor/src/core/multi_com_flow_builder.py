from __future__ import annotations

from functools import reduce
from typing import Generator, Optional, Dict, List, Set

import numpy as np

from core.dijkstra import dijkstra, dynamic_dijkstra
from core.distributor import Distributor
from core.graph import Node, Edge
from core.machine_precision import eps
from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network
from core.predictor import Predictor
from utilities.arrays import elem_lrank
from utilities.piecewise_linear import PiecewiseLinear


class MultiComFlowBuilder:
    network: Network
    predictors: List[Predictor]
    distributor: Distributor
    reroute_interval: Optional[float]
    _active_edges: List[Dict[Node, List[Edge]]]

    def __init__(self,
                 network: Network,
                 predictors: List[Predictor],
                 distributor: Distributor,
                 reroute_interval: Optional[float]  # None means rerouting every time some outflow changes
                 ):
        self.network = network
        self.predictors = predictors
        self.reroute_interval = reroute_interval
        self.distributor = distributor

    def calc_next_inflow_change(self, phi: float):
        next_change = float('inf')
        for c in self.network.commodities:
            rnk = elem_lrank(c.net_inflow.times, phi) + 1
            if rnk < len(c.net_inflow.times) and c.net_inflow.times[rnk] < next_change:
                next_change = c.net_inflow.times[rnk]
        return next_change

    def build_flow(self) -> Generator[MultiComPartialDynamicFlow, None, None]:
        flow = MultiComPartialDynamicFlow(self.network)
        graph = self.network.graph
        n = len(self.network.commodities)
        m = len(graph.edges)
        travel_time = self.network.travel_time
        capacity = self.network.capacity
        self._active_edges = [{} for _ in range(n)]

        # Preprocessing...
        # For each commodity find the nodes that reach the sink
        important_nodes = [
            graph.get_nodes_reaching(commodity.sink).intersection(graph.get_reachable_nodes(commodity.source))
            for commodity in self.network.commodities
        ]
        assert all(c.source in important_nodes[i] for i, c in enumerate(self.network.commodities))

        next_reroute_time = flow.phi
        next_net_inflow_change = self.calc_next_inflow_change(flow.phi)
        costs = []
        handle_nodes = set(self.network.graph.nodes.values())

        yield flow
        while flow.phi < float('inf'):
            if flow.phi >= next_net_inflow_change:
                next_net_inflow_change = self.calc_next_inflow_change(flow.phi)
                for i in self.network.commodities:
                    handle_nodes.add(i.source)
            if self.reroute_interval is None or flow.phi >= next_reroute_time - eps:
                # PREDICT NEW QUEUES
                self._active_edges = [{} for _ in range(n)]
                predictions = [predictor.predict_from_fcts(flow.queues, flow.phi) for predictor in self.predictors]
                costs = [
                    [
                        PiecewiseLinear(
                            prediction[e].times,
                            [travel_time[e] + value / capacity[e] for value in prediction[e].values],
                            prediction[e].first_slope / capacity[e],
                            prediction[e].last_slope / capacity[e],
                            (flow.phi, float('inf'))
                        )
                        for e in range(m)
                    ]
                    for prediction in predictions
                ]

                const_costs = {}
                for k, predictor in enumerate(self.predictors):
                    if predictor.is_constant():
                        const_costs[k] = [cost.values[0] for cost in costs[k]]

                # CALCULATE NEW SHORTEST PATHS
                """
                PRECALCULATE FOR CONSTANT PREDICTORS
                """
                for i, commodity in enumerate(self.network.commodities):
                    if self.predictors[commodity.predictor].is_constant():
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
                    node_inflow[commodity.source] += commodity.net_inflow(flow.phi)

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

            max_ext_length = min(next_reroute_time - flow.phi, next_net_inflow_change - flow.phi)
            changed_edges = flow.extend(new_inflow, max_ext_length)

            yield flow

            handle_nodes = set()
            for e in changed_edges:
                handle_nodes.add(self.network.graph.edges[e].node_to)

    def distribute(self, i: int, phi: float, node_inflow: Dict[Node, float], sink: Node, interesting_nodes: Set[Node],
                   costs: List[PiecewiseLinear]) -> Dict[int, float]:
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
                arrival_times, realised_cost = dynamic_dijkstra(phi, s, sink, interesting_nodes, costs)

                # Dijkstra done. Now searching all active edges leading to t.
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
