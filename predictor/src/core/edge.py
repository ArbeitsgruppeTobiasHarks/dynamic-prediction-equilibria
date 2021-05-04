from core.node import Node


class Edge:
    node_from: Node
    node_to: Node
    id: int

    def __init__(self, node_from: Node, node_to: Node, id: int):
        self.node_from = node_from
        self.node_to = node_to
        self.id = id
