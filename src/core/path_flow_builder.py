from __future__ import annotations

from typing import Dict, Generator, List, Optional, Set, Tuple

from core.active_paths import Path
from core.dynamic_flow import DynamicFlow, FlowRatesCollection
from core.graph import Edge, Node
from core.machine_precision import eps
from core.network import Commodity, Network
from utilities.queues import PriorityQueue
from utilities.right_constant import RightConstant


class PathFlowBuilder:
    network: Network
    paths: Dict[int, Path]  # each commodity is associated with a single path
    _built: bool
    _handle_nodes: Set[Node]
    _flow: DynamicFlow
    _net_inflow_by_node: Dict[Node, FlowRatesCollection]
    _commodity_ids_by_sink: Dict[Node, Set[int]]
    _network_inflow_changes: PriorityQueue[Tuple[Commodity, Node, float]]

    def __init__(
        self,
        network: Network,
        paths: Dict[int, Path],
    ):
        self.network = network
        self.paths = paths
        self._built = False
        self._handle_nodes = set()
        self._flow = DynamicFlow(network)
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
        self._network_inflow_changes = PriorityQueue(
            [
                ((c, s, t), t)
                for c in network.commodities
                for s, inflow in c.sources.items()
                for t in inflow.times
            ]
        )

    def build_flow(self) -> Generator[DynamicFlow, None, None]:
        if self._built:
            raise RuntimeError("Flow was already built. Initialize a new builder.")
        self._built = True

        yield self._flow
        while self._flow.phi < float("inf"):
            while self._flow.phi >= self._network_inflow_changes.min_key():
                c, s, t = self._network_inflow_changes.pop()
                self._handle_nodes.add(s)

            new_inflow = self._determine_new_inflow()
            max_ext_time = self._network_inflow_changes.min_key()
            edges_with_outflow_change = self._flow.extend(new_inflow, max_ext_time)
            self._handle_nodes = set(
                self.network.graph.edges[e].node_to for e in edges_with_outflow_change
            )

            yield self._flow

    def _get_next_edge(self, com_id: int, v: Node):
        path = self.paths[com_id]
        edge = None
        for e in path.edges:
            if e.node_from == v:
                edge = e
                break
        return edge

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

                edge = self._get_next_edge(i, v)
                if edge is not None:
                    new_inflow[edge.id][i] = inflow

        return new_inflow
