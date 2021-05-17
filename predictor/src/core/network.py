from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from core.graph import DirectedGraph, Node


@dataclass
class Commodity:
    source: Node
    sink: Node
    demand: float


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

    def add_commodity(self, source: int, sink: int, demand: float):
        nodes = self.graph.nodes
        assert source in nodes.keys(), f"No node with id#{sink} in the graph!"
        assert sink in nodes.keys(), f"No node with id#{sink} in the graph!"
        self.commodities.append(
            Commodity(nodes[source], nodes[sink], demand)
        )
