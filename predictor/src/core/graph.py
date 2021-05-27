from __future__ import annotations

from typing import Dict, List, Set


class Edge:
    node_from: Node
    node_to: Node
    id: int

    def __init__(self, node_from: Node, node_to: Node, id: int):
        self.node_from = node_from
        self.node_to = node_to
        self.id = id

    def __str__(self):
        return str(self.id)


class Node:
    id: int
    incoming_edges: List[Edge]
    outgoing_edges: List[Edge]

    def __init__(self, id: int):
        self.id = id
        self.incoming_edges = []
        self.outgoing_edges = []

    def __str__(self):
        return str(self.id)


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

    def get_nodes_reaching(self, node: Node) -> Set[Node]:
        assert node in self.nodes.values()
        nodes_found: Set[Node] = {node}
        queue = [node]
        while queue:
            v = queue.pop()
            for e in v.incoming_edges:
                if e.node_from not in nodes_found:
                    nodes_found.add(e.node_from)
                    queue.append(e.node_from)
        return nodes_found

    def get_reachable_nodes(self, source: Node) -> Set[Node]:
        """
        Returns all nodes that are reachable from source
        """
        nodes_found: Set[Node] = {source}
        queue = [source]
        while queue:
            v = queue.pop()
            for e in v.outgoing_edges:
                if e.node_to not in nodes_found:
                    nodes_found.add(e.node_to)
                    queue.append(e.node_to)
        return nodes_found
