from __future__ import annotations

import numpy as np

from core.graph import DirectedGraph


class Network:
    graph: DirectedGraph
    capacity: np.ndarray[float]
    travel_time: np.ndarray[int]  # We use integers for a simpler discretization of time

    def __init__(self):
        self.graph = DirectedGraph()
        self.capacity = np.array([])
        self.travel_time = np.array([])

    def add_edge(self, node_from: int, node_to: int, travel_time: int, capacity: float):
        self.graph.add_edge(node_from, node_to)
        self.travel_time = np.append(self.travel_time, travel_time)
        self.capacity = np.append(self.capacity, capacity)
