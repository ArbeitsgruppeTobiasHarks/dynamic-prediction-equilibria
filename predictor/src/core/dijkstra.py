from __future__ import annotations

from typing import Dict

import numpy as np

from core.graph import Node
from core.machine_precision import eps
from utilities.queues import PriorityQueue


def dijkstra(
        sink: Node,
        costs: np.ndarray
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
