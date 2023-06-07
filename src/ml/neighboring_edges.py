import numpy as np
from typing import Set
from core.graph import Edge, Node
from core.network import Network


def get_neighboring_edges_forward(edge: Edge, max_distance: int) -> Set[Edge]:
    discovered: Set[Node] = set([edge.node_to])
    queue = [edge.node_to]
    result = set([edge])
    for _ in range(max_distance):
        new_queue = []
        for v in queue:
            for e in v.outgoing_edges:
                result.add(e)
                if e.node_to not in discovered:
                    discovered.add(e.node_to)
                    new_queue.append(e.node_to)
        queue = new_queue
    return result


def get_neighboring_edges_backward(edge: Edge, max_distance: int) -> Set[Edge]:
    discovered: Set[Node] = set([edge.node_from])
    queue = [edge.node_from]
    result = set([edge])
    for _ in range(max_distance):
        new_queue = []
        for v in queue:
            for e in v.incoming_edges:
                result.add(e)
                if e.node_from not in discovered:
                    discovered.add(e.node_from)
                    new_queue.append(e.node_from)
    return result


def get_neighboring_edges_directed(edge: Edge, max_distance: int) -> Set[Edge]:
    return get_neighboring_edges_backward(edge, max_distance).union(
        get_neighboring_edges_forward(edge, max_distance)
    )


def get_neighboring_edges_undirected(edge: Edge, max_distance: int) -> Set[Edge]:
    discovered: Set[Node] = set([edge.node_from, edge.node_to])
    queue = [edge.node_from, edge.node_to]
    result = set([edge])
    for _ in range(max_distance):
        new_queue = []
        for v in queue:
            for e in v.incoming_edges:
                result.add(e)
                if e.node_from not in discovered:
                    discovered.add(e.node_from)
                    new_queue.append(e.node_from)
            for e in v.outgoing_edges:
                result.add(e)
                if e.node_to not in discovered:
                    discovered.add(e.node_to)
                    new_queue.append(e.node_to)
    return result


def get_neighboring_edges_mask_directed(
    edge: Edge, network: Network, max_distance: int
) -> np.ndarray:
    neighboring_edges = get_neighboring_edges_directed(edge, max_distance)
    return np.array([e in neighboring_edges for e in network.graph.edges])


def get_neighboring_edges_mask_undirected(
    edge: Edge, network: Network, max_distance: int
) -> np.ndarray:
    neighboring_edges = get_neighboring_edges_undirected(edge, max_distance)
    return np.array([e in neighboring_edges for e in network.graph.edges])
