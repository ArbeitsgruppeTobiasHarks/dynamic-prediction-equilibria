from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from core.graph import DirectedGraph, Node, Edge


@dataclass
class Commodity:
    source: Node
    sink: Node
    demand: float
    predictor: int


class Network:
    graph: DirectedGraph
    capacity: np.ndarray[float]
    travel_time: np.ndarray[float]
    commodities: List[Commodity]

    def __init__(self):
        self.graph = DirectedGraph()
        self.capacity = np.array([])
        self.travel_time = np.array([])
        self.commodities = []

    def add_edge(self, node_from: int, node_to: int, travel_time: float, capacity: float):
        self.graph.add_edge(node_from, node_to)
        self.travel_time = np.append(self.travel_time, travel_time)
        self.capacity = np.append(self.capacity, capacity)

    def add_commodity(self, source: int, sink: int, demand: float, predictor_id: int):
        nodes = self.graph.nodes
        assert source in nodes.keys(), f"No node with id#{sink} in the graph!"
        assert sink in nodes.keys(), f"No node with id#{sink} in the graph!"
        self.commodities.append(Commodity(nodes[source], nodes[sink], demand, predictor_id))

    def _remove_edge(self, edge: Edge):
        edge.node_to.incoming_edges.remove(edge)
        edge.node_from.outgoing_edges.remove(edge)
        del self.graph.edges[edge.id]
        self.capacity = np.delete(self.capacity, edge.id)
        self.travel_time = np.delete(self.travel_time, edge.id)
        for i in range(edge.id, len(self.graph.edges)):
            self.graph.edges[i].id = i

    def compress_lonely_nodes(self):
        """
        A node is lonely, if it is no source or sink and if it has a single incoming and a single outgoing edge.
        This function removes these useless nodes to speed up computation
        """
        remove_nodes = []
        for v in self.graph.nodes.values():
            if len(v.outgoing_edges) == 1 == len(v.incoming_edges) and \
                    all(c.source != v != c.sink for c in self.commodities):
                edge1 = v.incoming_edges[0]
                edge2 = v.outgoing_edges[0]
                new_travel_time = self.travel_time[edge1.id] + self.travel_time[edge2.id]
                new_capacity = min(self.capacity[edge1.id], self.capacity[edge2.id])
                self._remove_edge(edge1)
                self._remove_edge(edge2)
                if edge1.node_from != edge2.node_to:
                    self.add_edge(edge1.node_from.id, edge2.node_to.id, new_travel_time, new_capacity)
                remove_nodes.append(v)
        for v in remove_nodes:
            self.graph.nodes.pop(v.id)

    def remove_unnecessary_nodes(self):
        """
        This functions checks whether a node is necessary for any commodity.
        A node v is necessary for a commodity, if there is a path from its source to its sink passing through v.
        """
        important_nodes = set()
        for commodity in self.commodities:
            reaching_t = self.graph.get_nodes_reaching(commodity.sink)
            reachable_from_s = self.graph.get_reachable_nodes(commodity.source)
            important_nodes = important_nodes.union(reaching_t.intersection(reachable_from_s))

        remove_nodes = set(self.graph.nodes.values()).difference(important_nodes)

        print(f"\rRemoving {len(remove_nodes)} unnecessary nodes.", end="\r")
        for v in remove_nodes:
            for edge in v.outgoing_edges:
                self._remove_edge(edge)
            for edge in v.incoming_edges:
                self._remove_edge(edge)
            self.graph.nodes.pop(v.id)
        print(f"\rRemoved {len(remove_nodes)} unnecessary nodes.")

    def remove_unnecessary_commodities(self, selected_commodity: int) -> int:
        """
        Remove all commodities that cannot interfere with commodity i.
        """
        selected_comm = self.commodities[selected_commodity]
        important_nodes = [
            self.graph.get_reachable_nodes(i.source).intersection(
                self.graph.get_nodes_reaching(i.sink)
            )
            for i in self.commodities
        ]
        remove_commodities = []
        for i, comm in enumerate(self.commodities):
            if len(important_nodes[i].intersection(important_nodes[selected_commodity])) == 0:
                remove_commodities.append(comm)

        print(f"\rRemoving {len(remove_commodities)} non-interfering commodities.", end="\r")
        for comm in remove_commodities:
            self.commodities.remove(comm)
        print(f"Removed {len(remove_commodities)} non-interfering commodities.")

        return self.commodities.index(selected_comm)
