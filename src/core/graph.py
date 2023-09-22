from __future__ import annotations

from typing import Dict, List, Set, Tuple


class Edge:
    _node_from: Node
    _node_to: Node
    _graph: DirectedGraph
    id: int

    def __init__(self, node_from: Node, node_to: Node, id: int, graph: DirectedGraph):
        self._node_from = node_from
        self._node_to = node_to
        self.id = id
        self._graph = graph

    @property
    def node_from(self) -> Node:
        return self._node_from if not self._graph.reversed else self._node_to

    @property
    def node_to(self) -> Node:
        return self._node_to if not self._graph.reversed else self._node_from

    def __str__(self):
        return str(self.id)


class Node:
    id: int
    _incoming_edges: List[Edge]
    _outgoing_edges: List[Edge]
    _graph: DirectedGraph

    def __init__(self, id: int, graph: DirectedGraph):
        self.id = id
        self._incoming_edges = []
        self._outgoing_edges = []
        self._graph = graph

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def incoming_edges(self):
        return (
            self._incoming_edges if not self._graph.reversed else self._outgoing_edges
        )

    @incoming_edges.setter
    def incoming_edges(self, value: List[Edge]):
        if not self._graph.reversed:
            self._incoming_edges = value
        else:
            self._outgoing_edges = value

    @property
    def outgoing_edges(self):
        return (
            self._outgoing_edges if not self._graph.reversed else self._incoming_edges
        )

    @outgoing_edges.setter
    def outgoing_edges(self, value: List[Edge]):
        if not self._graph.reversed:
            self._outgoing_edges = value
        else:
            self._incoming_edges = value

    def __str__(self):
        return str(self.id)


class DirectedGraph:
    edges: List[Edge]
    nodes: Dict[int, Node]
    reversed: bool
    positions: Dict[int, Tuple[float, float]]

    def __init__(self):
        self.edges = []
        self.nodes = {}
        self.reversed = False
        self.positions = {}

    def __setstate__(self, state):
        self.reversed = state["reversed"]
        self.edges = []
        self.nodes = {}
        self.positions = state["positions"]
        for [node_from, node_to] in state["edges"]:
            self.add_edge(node_from, node_to)

    def __getstate__(self):
        return {
            "edges": [[e.node_from.id, e.node_to.id] for e in self.edges],
            "positions": self.positions,
            "reversed": self.reversed,
        }

    def add_edge(self, node_from: int, node_to: int):
        if node_from not in self.nodes.keys():
            self.nodes[node_from] = Node(node_from, self)
        if node_to not in self.nodes.keys():
            self.nodes[node_to] = Node(node_to, self)
        index = len(self.edges)
        edge = Edge(self.nodes[node_from], self.nodes[node_to], index, self)
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

    def reverse(self):
        self.reversed = not self.reversed
