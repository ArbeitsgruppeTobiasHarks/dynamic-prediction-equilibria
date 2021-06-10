from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from core.graph import Node, Edge
from core.machine_precision import eps
from utilities.interpolate import LinearlyInterpolatedFunction
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
        phi: float, source: Node, sink: Node, interesting_nodes: Set[Node], costs: List[LinearlyInterpolatedFunction]
) -> Tuple[Dict[Node, float], Dict[Edge, float]]:
    arrival_times: Dict[Node, float] = {source: phi}
    queue: PriorityQueue[Node] = PriorityQueue([(source, phi)])
    realised_cost = {}
    stop_after = float('inf')
    while len(queue) > 0:
        arrival_time = queue.min_key()
        v = queue.pop()
        if v == sink:
            stop_after = arrival_time
        if arrival_time > stop_after:
            break

        for e in v.outgoing_edges:
            w = e.node_to
            if w not in interesting_nodes:
                continue
            realised_cost[e] = costs[e.id](arrival_time)
            relaxation = arrival_times[v] + realised_cost[e]
            if w not in arrival_times.keys():
                arrival_times[w] = relaxation
                queue.push(w, relaxation)
            elif relaxation < arrival_times[w]:
                arrival_times[w] = relaxation
                queue.decrease_key(w, relaxation)
    return arrival_times, realised_cost
