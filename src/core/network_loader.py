from __future__ import annotations

from typing import Generator, Dict, List, Set, Tuple

from core.dijkstra import reverse_dijkstra, dynamic_dijkstra
from core.graph import Node, Edge
from core.machine_precision import eps
from core.dynamic_flow import DynamicFlow, FlowRatesCollection
from core.network import Network
from utilities.queues import PriorityQueue
from utilities.right_constant import RightConstant

Path = List[Edge]

class NetworkLoader:
    network: Network
    path_inflows: List[Tuple[Path, RightConstant]] # (path, inflow)
    _network_inflow_changes: PriorityQueue[Tuple[int, Node, float]] # (path_id, source, time)
    _net_inflow_by_node: Dict[Node, FlowRatesCollection]
    _path_indexes_by_sink: Dict[Node, Set[int]]

    def __init__(self, network: Network,
                 path_inflows: List[Tuple[Path, RightConstant]]):
        self.network = network
        self.path_inflows = path_inflows
        self._built = False
        self._flow = DynamicFlow(network)
        self._net_inflow_by_node = {
            v: FlowRatesCollection({
                i: path_inflow[1]
                for i, path_inflow in enumerate(self.path_inflows)
                if v == path_inflow[0][0].node_from
            })
            for v in network.graph.nodes.values()
        }
        self._network_inflow_changes = PriorityQueue([
            ((i, path[0].node_from, time), time)
            for i, (path, inflow) in enumerate(path_inflows)
            for time in inflow.times
        ])

        self._path_indexes_by_sink = {
            v: set(
                i for i, path_inflow in enumerate(self.path_inflows)
                if path_inflow[0][-1].node_to == v
            )
            for v in network.graph.nodes.values()
        }
        self._handle_nodes = set()


    def build_flow(self) -> Generator[DynamicFlow, None, None]:
        if self._built:
            raise RuntimeError(
                "Flow was already built. Initialize a new builder.")
        self._built = True

        yield self._flow
        while self._flow.phi < float('inf'):
            while self._flow.phi >= self._network_inflow_changes.min_key():
                _, s, _ = self._network_inflow_changes.pop()
                self._handle_nodes.add(s)

            new_inflow = self._determine_new_inflow()
            max_ext_time = self._network_inflow_changes.min_key()
            edges_with_outflow_change = self._flow.extend(
                new_inflow, max_ext_time)
            self._handle_nodes = set(
                self.network.graph.edges[e].node_to for e in edges_with_outflow_change)

            yield self._flow

    def _get_active_edges(self, i: int, s: Node) -> List[Edge]:
        path = self.path_inflows[i][0]
        edge = None
        for e in path:
            if e.node_from == s:
                edge = e
                break
        if edge is None:
            raise RuntimeError("Node does not appear in path p (or is the last node of p).")
        return [edge]

    def _determine_new_inflow(self) -> Dict[int, Dict[int, float]]:
        new_inflow = {}
        for v in self._handle_nodes:
            new_inflow.update({e.id: {} for e in v.outgoing_edges})

            outflows = {
                e.id: self._flow.outflow[e.id].get_values_at_time(
                    self._flow.phi)
                for e in v.incoming_edges
            }

            net_inflow_by_com = self._net_inflow_by_node[v].get_values_at_time(self._flow.phi)

            used_commodities = set(
                key
                for outflow in outflows.values()
                for key in outflow
            ).union(net_inflow_by_com.keys()).difference(self._path_indexes_by_sink[v])

            for i in used_commodities:
                inflow = sum(
                    outflow[i]
                    for outflow in outflows.values()
                    if i in outflow
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
