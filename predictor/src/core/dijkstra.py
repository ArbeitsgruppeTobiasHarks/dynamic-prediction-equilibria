from __future__ import annotations

from typing import Callable, Dict, List, Set, Tuple
from core.graph import Node, Edge
from core.machine_precision import eps
from utilities.queues import PriorityQueue


def dijkstra(
        sink: Node,
        costs: List[float]
) -> Dict[Node, float]:
    dist: Dict[Node, float] = {sink: 0}

    queue = PriorityQueue([(sink, dist[sink])])

    while len(queue) > 0:
        w = queue.pop()
        for edge in w.incoming_edges:
            v = edge.node_from
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


def realizing_dijkstra(
        phi: float, source: Node, sink: Node, relevant_nodes: Set[Node], costs: List[Callable[[float], float]]
) -> Tuple[Dict[Node, float], Dict[Edge, float]]:
    '''
    Assumes costs to follow the FIFO rule and relevant_nodes to contain
    all nodes that lie on a path from source to sink.
    Returns the earliest arrival times when departing from source at
    time phi for nodes that source can reach up to the arrival at sink.
    '''
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
            if w in arrival_times.keys() or w not in relevant_nodes:
                continue
            realized_cost[e] = costs[e.id](arrival_times[v])
            relaxation = arrival_times[v] + realized_cost[e]
            if not queue.has(w):
                queue.push(w, relaxation)
            elif relaxation < queue.key_of(w):
                queue.decrease_key(w, relaxation)
    return arrival_times, realized_cost
