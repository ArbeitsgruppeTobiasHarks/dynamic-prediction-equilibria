from typing import Dict

from core.edge import Edge
from core.node import Node


class DirectedGraph:
    edges: Dict[int, Edge]
    nodes: Dict[int, Node]

    def __init__(self):
        self.edges = {}
        self.nodes = {}

    def add_edge(self, node_from: int, node_to: int, id: int):
        if node_from not in self.nodes:
            self.nodes[node_from] = Node(node_from)
        if node_to not in self.nodes:
            self.nodes[node_to] = Node(node_to)
        edge = Edge(self.nodes[node_from], self.nodes[node_to], id)
        self.nodes[node_from].outgoing_edges.append(edge)
        self.nodes[node_to].incoming_edges.append(edge)
