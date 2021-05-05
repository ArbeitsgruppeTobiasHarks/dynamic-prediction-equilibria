from __future__ import annotations

from typing import Dict, List


class Edge:
    node_from: Node
    node_to: Node
    id: int

    def __init__(self, node_from: Node, node_to: Node, id: int):
        self.node_from = node_from
        self.node_to = node_to
        self.id = id


class Node:
    id: int
    incoming_edges: List[Edge]
    outgoing_edges: List[Edge]

    def __init__(self, id: int):
        self.id = id
        self.incoming_edges = []
        self.outgoing_edges = []


class DirectedGraph:
    edges: List[Edge]
    nodes: Dict[int, Node]

    def __init__(self):
        self.edges = []
        self.nodes = {}

    def add_edge(self, node_from: int, node_to: int):
        if node_from not in self.nodes.keys():
            self.nodes[node_from] = Node(node_from)
        if node_to not in self.nodes.keys():
            self.nodes[node_to] = Node(node_to)
        index = len(self.edges)
        edge = Edge(self.nodes[node_from], self.nodes[node_to], index)
        self.edges.append(edge)
        self.nodes[node_from].outgoing_edges.append(edge)
        self.nodes[node_to].incoming_edges.append(edge)
