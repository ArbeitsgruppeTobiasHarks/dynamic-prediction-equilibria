from __future__ import annotations

from functools import lru_cache
from typing import Dict, Generator, List, Optional, Set, Tuple

from core.dijkstra import dynamic_dijkstra, get_active_edges_from_dijkstra, reverse_dijkstra
from core.dynamic_flow import DynamicFlow, FlowRatesCollection
from core.graph import Edge, Node
from core.machine_precision import eps
from core.network import Commodity, Network
from core.predictor import Predictor
from core.predictors.predictor_type import PredictorType
from utilities.piecewise_linear import PiecewiseLinear
from utilities.queues import PriorityQueue


class FlowBuilder:
    network: Network
    predictors: Dict[PredictorType, Predictor]
    reroute_interval: Optional[float]
    _active_edges: List[Dict[Node, List[Edge]]]
    _built: bool
    _handle_nodes: Set[Node]
    _flow: DynamicFlow
    _next_reroute_time: float
    _route_time: float
    _net_inflow_by_node: Dict[Node, FlowRatesCollection]
    _commodity_ids_by_sink: Dict[Node, Set[int]]
    _network_inflow_changes: PriorityQueue[Tuple[Commodity, Node, float]]

    def __init__(
        self,
        network: Network,
        predictors: Dict[PredictorType, Predictor],
        # None means rerouting every time some outflow changes
        reroute_interval: Optional[float],
    ):
        self.network = network
        self.predictors = predictors
        self.reroute_interval = reroute_interval
        self._built = False
        self._active_edges = [{} for _ in network.commodities]
        self._handle_nodes = set()
        self._flow = DynamicFlow(network)
        self._next_reroute_time = self._route_time = self._flow.phi
        self._important_nodes = [
            network.graph.get_nodes_reaching(commodity.sink).intersection(
                set(
                    node
                    for source in commodity.sources
                    for node in network.graph.get_reachable_nodes(source)
                )
            )
            for commodity in self.network.commodities
        ]
        self._net_inflow_by_node = {
            v: FlowRatesCollection(
                {
                    i: c.sources[v]
                    for i, c in enumerate(self.network.commodities)
                    if v in c.sources
                }
            )
            for v in network.graph.nodes.values()
        }
        self._commodity_ids_by_sink = {
            v: set(i for i, c in enumerate(self.network.commodities) if c.sink == v)
            for v in network.graph.nodes.values()
        }
        assert all(
            source in self._important_nodes[i]
            for i, c in enumerate(self.network.commodities)
            for source in c.sources
        )
        self._network_inflow_changes = PriorityQueue(
            [
                ((c, s, t), t)
                for c in network.commodities
                for s, inflow in c.sources.items()
                for t in inflow.times
            ]
        )

    @lru_cache(maxsize=None)
    def _get_costs(self, predictor_type: PredictorType) -> List[PiecewiseLinear]:
        prediction_time = self._next_reroute_time - self.reroute_interval
        graph = self.network.graph
        travel_time = self.network.travel_time
        capacity = self.network.capacity
        predictions = self.predictors[predictor_type].predict(
            prediction_time, self._flow
        )
        costs = [
            PiecewiseLinear(
                predictions[e].times,
                [
                    travel_time[e] + value / capacity[e]
                    for value in predictions[e].values
                ],
                predictions[e].first_slope / capacity[e],
                predictions[e].last_slope / capacity[e],
                (prediction_time, float("inf")),
            )
            for e in range(len(graph.edges))
        ]
        return costs

    def build_flow(self) -> Generator[DynamicFlow, None, None]:
        if self._built:
            raise RuntimeError("Flow was already built. Initialize a new builder.")
        self._built = True

        yield self._flow
        while self._flow.phi < float("inf"):
            while self._flow.phi >= self._network_inflow_changes.min_key():
                c, s, t = self._network_inflow_changes.pop()
                self._handle_nodes.add(s)
            if (
                self.reroute_interval is None
                or self._flow.phi >= self._next_reroute_time
            ):
                self._get_costs.cache_clear()

                self._active_edges = [{} for _ in self.network.commodities]
                self._route_time = self._next_reroute_time
                self._next_reroute_time += self.reroute_interval
                self._handle_nodes = set(self.network.graph.nodes.values())

            new_inflow = self._determine_new_inflow()
            max_ext_time = min(
                self._next_reroute_time, self._network_inflow_changes.min_key()
            )
            edges_with_outflow_change = self._flow.extend(new_inflow, max_ext_time)
            self._handle_nodes = set(
                self.network.graph.edges[e].node_to for e in edges_with_outflow_change
            )

            yield self._flow

    def _get_active_edges(self, i: int, s: Node) -> List[Edge]:
        if s in self._active_edges[i]:
            return self._active_edges[i][s]

        commodity = self.network.commodities[i]
        com_nodes = self._important_nodes[i]
        sink = commodity.sink

        if self.predictors[commodity.predictor_type].is_constant():
            const_costs = [
                c.values[0] for c in self._get_costs(commodity.predictor_type)
            ]
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
                for j in self._commodity_ids_by_sink[commodity.sink]:
                    if (
                        self.network.commodities[j].predictor_type
                        == commodity.predictor_type
                    ):
                        self._active_edges[j][v] = active_edges
        else:
            # Do Time-Dependent dijkstra from s to t to find active outgoing edges of s
            result = dynamic_dijkstra(
                self._route_time,
                s,
                sink,
                com_nodes,
                self._get_costs(commodity.predictor_type),
            )
            self._active_edges[i][s] = get_active_edges_from_dijkstra(result, s, sink)
        return self._active_edges[i][s]

    def _determine_new_inflow(self) -> Dict[int, Dict[int, float]]:
        new_inflow = {}
        for v in self._handle_nodes:
            new_inflow.update({e.id: {} for e in v.outgoing_edges})

            outflows = {
                e.id: self._flow.outflow[e.id].get_values_at_time(self._flow.phi)
                for e in v.incoming_edges
            }

            net_inflow_by_com = self._net_inflow_by_node[v].get_values_at_time(
                self._flow.phi
            )

            used_commodities = (
                set(key for outflow in outflows.values() for key in outflow)
                .union(net_inflow_by_com.keys())
                .difference(self._commodity_ids_by_sink[v])
            )

            for i in used_commodities:
                inflow = sum(
                    outflow[i] for outflow in outflows.values() if i in outflow
                )
                if i in net_inflow_by_com:
                    inflow += net_inflow_by_com[i]
                if inflow < eps:
                    continue

                active_edges = self._get_active_edges(i, v)
                distribution = inflow / len(active_edges)
                for e in active_edges:
                    new_inflow[e.id][i] = distribution
        return new_inflow
