from __future__ import annotations

from typing import Dict

import numpy as np

from core.graph import Node
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
            elif dist[v] > relaxation:
                dist[v] = relaxation
                queue.decrease_key(v, relaxation)

    return dist
