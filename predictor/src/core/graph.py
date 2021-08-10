from __future__ import annotations

from typing import Callable, Dict, List, Set


class Edge:
    _node_from: Node
    _node_to: Node
    _is_reversed: Callable[[], bool]
    id: int

    def __init__(
        self, node_from: Node, node_to: Node, id: int, is_reversed: Callable[[], bool]
        ):
        self._node_from = node_from
        self._node_to = node_to
        self.id = id
        self._is_reversed = is_reversed

    @property
    def node_from(self) -> Node:
        return self._node_from if not self._is_reversed() else self._node_to

    @property
    def node_to(self) -> Node:
        return self._node_to if not self._is_reversed() else self._node_from

    def __str__(self):
        return str(self.id)


class Node:
    id: int
    _incoming_edges: List[Edge]
    _outgoing_edges: List[Edge]
    _is_reversed: Callable[[], bool]

    def __init__(self, id: int, is_reversed: Callable[[], bool]):
        self.id = id
        self._incoming_edges = []
        self._outgoing_edges = []
        self._is_reversed = is_reversed

    
    @property
    def incoming_edges(self):
        return self._incoming_edges if not self._is_reversed() else self._outgoing_edges
    @incoming_edges.setter
    def incoming_edges(self, value: List[Edge]):
        if not self._is_reversed():
            self._incoming_edges = value
        else:
            self._outgoing_edges = value
    
    @property
    def outgoing_edges(self):
        return self._outgoing_edges if not self._is_reversed() else self._incoming_edges
    @outgoing_edges.setter
    def outgoing_edges(self, value: List[Edge]):
        if not self._is_reversed():
            self._outgoing_edges = value
        else:
            self._incoming_edges = value

    def __str__(self):
        return str(self.id)


class DirectedGraph:
    edges: List[Edge]
    nodes: Dict[int, Node]
    reversed: bool
    _is_reversed: Callable[[], bool]

    def __init__(self):
        self.edges = []
        self.nodes = {}
        self.reversed = False
        self._is_reversed = lambda: self.reversed

    def add_edge(self, node_from: int, node_to: int):
        if node_from not in self.nodes.keys():
            self.nodes[node_from] = Node(node_from, self._is_reversed)
        if node_to not in self.nodes.keys():
            self.nodes[node_to] = Node(node_to, self._is_reversed)
        index = len(self.edges)
        edge = Edge(self.nodes[node_from], self.nodes[node_to], index, self._is_reversed)
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