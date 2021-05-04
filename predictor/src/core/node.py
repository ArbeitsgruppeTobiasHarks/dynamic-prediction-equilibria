from typing import List

from core.edge import Edge


class Node:
    id: int
    incoming_edges: List[Edge] = []
    outgoing_edges: List[Edge] = []

    def __init__(self, id: int):
        self.id = id
