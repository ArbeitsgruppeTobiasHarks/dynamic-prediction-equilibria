from __future__ import annotations

from typing import Generator, Optional, Dict, List, Set, Tuple

from core.dijkstra import reverse_dijkstra, dynamic_dijkstra
from core.graph import Node, Edge
from core.machine_precision import eps
from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from core.network import Network, Commodity
from core.predictor import Predictor
from core.predictors.predictor_type import PredictorType
from utilities.piecewise_linear import PiecewiseLinear
from utilities.queues import PriorityQueue


class MultiComFlowBuilder:
    network: Network
    predictors: Dict[PredictorType, Predictor]
    reroute_interval: Optional[float]
    _active_edges: List[Dict[Node, List[Edge]]]
    _built: bool
    _handle_nodes: Set[Node]
    _flow: MultiComPartialDynamicFlow
    _next_reroute_time: float
    _route_time: float
    _costs: Dict[PredictorType, List[PiecewiseLinear]]
    _network_inflow_changes: PriorityQueue[Tuple[Commodity, float]]

    def __init__(self,
                 network: Network,
                 predictors: Dict[PredictorType, Predictor],
                 reroute_interval: Optional[float]  # None means rerouting every time some outflow changes
                 ):
        self.network = network
        self.predictors = predictors
        self.reroute_interval = reroute_interval
        self._built = False
        self._active_edges = [{} for _ in network.commodities]
        self._handle_nodes = set()
        self._flow = MultiComPartialDynamicFlow(network)
        self._next_reroute_time = self._route_time = self._flow.phi
        self._important_nodes = [
            network.graph.get_nodes_reaching(commodity.sink).intersection(
                network.graph.get_reachable_nodes(commodity.source))
            for commodity in self.network.commodities
        ]
        assert all(c.source in self._important_nodes[i] for i, c in enumerate(self.network.commodities))
        self._network_inflow_changes = PriorityQueue([
            ((c, t), t) for c in network.commodities for t in c.net_inflow.times
        ])
        self._costs = {}

    def build_flow(self) -> Generator[MultiComPartialDynamicFlow, None, None]:
        if self._built:
            raise RuntimeError("Flow was already built. Initialize a new builder.")
        self._built = True
        graph = self.network.graph
        travel_time = self.network.travel_time
        capacity = self.network.capacity

        yield self._flow
        while self._flow.phi < float('inf'):
            while self._flow.phi >= self._network_inflow_changes.min_key():
                c, t = self._network_inflow_changes.pop()
                self._handle_nodes.add(c.source)
            if self.reroute_interval is None or self._flow.phi >= self._next_reroute_time:
                predictions = {
                    key: predictor.predict_from_fcts(self._flow.queues, self._flow.phi)
                    for (key, predictor) in self.predictors.items()
                }
                self._costs = {
                    key: [
                        PiecewiseLinear(
                            prediction[e].times,
                            [travel_time[e] + value / capacity[e] for value in prediction[e].values],
                            prediction[e].first_slope / capacity[e],
                            prediction[e].last_slope / capacity[e],
                            (self._flow.phi, float('inf'))
                        )
                        for e in range(len(graph.edges))
                    ]
                    for (key, prediction) in predictions.items()
                }

                self._active_edges = [{} for _ in self.network.commodities]
                self._route_time = self._next_reroute_time
                self._next_reroute_time += self.reroute_interval
                self._handle_nodes = set(self.network.graph.nodes.values())

            new_inflow = self._determine_new_inflow()
            max_ext_time = min(self._next_reroute_time, self._network_inflow_changes.min_key())
            edges_with_outflow_change = self._flow.extend(new_inflow, max_ext_time)
            self._handle_nodes = set(self.network.graph.edges[e].node_to for e in edges_with_outflow_change)

            yield self._flow

    def _get_active_edges(self, i: int, s: Node) -> List[Edge]:
        if s in self._active_edges[i]:
            return self._active_edges[i][s]

        commodity = self.network.commodities[i]
        com_nodes = self._important_nodes[i]
        sink = commodity.sink

        if self.predictors[commodity.predictor_type].is_constant():
            const_costs = [c.values[0] for c in self._costs[commodity.predictor_type]]
            distances = reverse_dijkstra(commodity.sink, const_costs, com_nodes)
            for v in com_nodes:
                if v == commodity.sink:
                    continue
                active_edges = []
                for e in v.outgoing_edges:
                    w = e.node_to
                    if w not in distances.keys():
                        continue
                    if const_costs[e.id] + distances[w] <= distances[v] + eps:
                        active_edges.append(e)
                assert len(active_edges) > 0
                self._active_edges[i][v] = active_edges
        else:
            # Do Time-Dependent dijkstra from s to t to find active outgoing edges of s
            arrival_times, realised_cost = dynamic_dijkstra(self._route_time, s, sink, com_nodes,
                                                            self._costs[commodity.predictor_type])

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
        return self._active_edges[i][s]

    def _determine_new_inflow(self) -> Dict[int, List[float]]:
        new_inflow = {}
        m = len(self.network.commodities)
        for v in self._handle_nodes:
            new_inflow.update({e.id: [0.] * m for e in v.outgoing_edges})
            for i, commodity in enumerate(self.network.commodities):
                if v not in self._important_nodes[i] or v == commodity.sink:
                    continue
                inflow = sum(self._flow.outflow[e.id][i](self._flow.phi) for e in v.incoming_edges)
                if v == commodity.source:
                    inflow += commodity.net_inflow(self._flow.phi)
                if inflow < eps:
                    continue

                active_edges = self._get_active_edges(i, v)
                distribution = inflow / len(active_edges)
                for e in active_edges:
                    new_inflow[e.id][i] = distribution
        return new_inflow
