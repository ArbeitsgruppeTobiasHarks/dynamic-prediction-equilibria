from __future__ import annotations

from typing import Dict
from core.graph import DirectedGraph


class Network:
    graph: DirectedGraph
    capacity: Dict[int]
    travel_time: Dict[int]

    def __init__(self):
        self.graph = DirectedGraph()
        self.capacity = {}
        self.travel_time = {}

    def add_edge(self, node_from: int, node_to: int, id: int, capacity: float, travel_time: float):
        self.graph.add_edge(node_from, node_to, id)
        self.capacity[id] = capacity
        self.travel_time[id] = travel_time
