from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, NamedTuple, Set, Tuple

from core.graph import Edge, Node
from core.machine_precision import eps
from utilities.queues import PriorityQueue


def reverse_dijkstra(
    sink: Node, costs: List[float], nodes: Set[Node]
) -> Dict[Node, float]:
    dist: Dict[Node, float] = {sink: 0}
    queue = PriorityQueue([(sink, dist[sink])])

    assert sink in nodes

    while len(queue) > 0:
        w = queue.pop()
        for edge in w.incoming_edges:
            v = edge.node_from
            if v not in nodes:
                continue
            relaxation = costs[edge.id] + dist[w]
            if v not in dist.keys():
                dist[v] = relaxation
                queue.push(v, dist[v])
            elif relaxation < dist[v] - eps:
                dist[v] = relaxation
                if queue.has(v):
                    queue.decrease_key(v, relaxation)
                else:
                    queue.push(v, relaxation)

    return dist


class DynamicDijkstraResult(NamedTuple):
    arrival_times: Dict[Node, float]
    realized_cost: Dict[Edge, float]


def dynamic_dijkstra(
    phi: float,
    source: Node,
    sink: Node,
    relevant_nodes: Set[Node],
    cost: Callable[[int, float], float],
) -> DynamicDijkstraResult:
    """
    Assumes costs to follow the FIFO rule and relevant_nodes to contain
    all nodes that lie on a path from source to sink.
    Returns the earliest arrival times when departing from source at
    time phi for nodes that source can reach up to the arrival at sink.
    """
    arrival_times: Dict[Node, float] = {}
    queue: PriorityQueue[Node] = PriorityQueue([(source, phi)])
    realized_cost = {}
    while len(queue) > 0:
        arrival_time, v = queue.min_key(), queue.pop()
        arrival_times[v] = arrival_time
        if v == sink:
            break
        for e in v.outgoing_edges:
            w = e.node_to
            if w in arrival_times or w not in relevant_nodes:
                continue
            realized_cost[e] = cost(e.id, arrival_times[v])
            relaxation = arrival_times[v] + realized_cost[e]
            if not queue.has(w):
                queue.push(w, relaxation)
            elif relaxation < queue.key_of(w):
                queue.decrease_key(w, relaxation)
    return DynamicDijkstraResult(arrival_times, realized_cost)


def get_active_edges_from_dijkstra(
    dijkstra_result: DynamicDijkstraResult,
    source: Node,
    sink: Node,
) -> List[Edge]:
    arrival_times, realised_cost = dijkstra_result
    active_edges = []
    touched_nodes = {sink}
    queue: List[Node] = [sink]
    while queue:
        w = queue.pop()
        for e in w.incoming_edges:
            if e not in realised_cost.keys():
                continue
            v: Node = e.node_from
            if arrival_times[v] + realised_cost[e] <= arrival_times[w] + eps:
                if v == source:
                    active_edges.append(e)
                if v not in touched_nodes:
                    touched_nodes.add(v)
                    queue.append(v)

    assert len(active_edges) > 0
    return active_edges
