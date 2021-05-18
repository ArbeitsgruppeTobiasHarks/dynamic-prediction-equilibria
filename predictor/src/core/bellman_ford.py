from __future__ import annotations

from typing import List, Dict

from core.graph import Node
from utilities.interpolate import LinearlyInterpolatedFunction
from utilities.queues import PriorityQueue


def bellman_ford(
        sink: Node,
        costs: List[LinearlyInterpolatedFunction],
        phi: float
) -> Dict[Node, LinearlyInterpolatedFunction]:
    identity = LinearlyInterpolatedFunction([phi, phi + 1], [phi, phi + 1], (phi, float('inf')))
    g: Dict[Node, LinearlyInterpolatedFunction] = {
        sink: identity
    }
    node_distance: Dict[Node, int] = {
        sink: 0
    }

    edge_arrival_times = [identity.plus(cost).ensure_monotone() for cost in costs]

    changes_detected_at = PriorityQueue([(sink, 0.)])

    while len(changes_detected_at) > 0:
        changed_nodes = changes_detected_at
        changes_detected_at = PriorityQueue([])

        for w in changed_nodes.sorted():
            for edge in w.incoming_edges:
                v = edge.node_from
                relaxation = g[w].compose(edge_arrival_times[edge.id])
                if v not in g.keys():
                    node_distance[v] = node_distance[w] + 1
                    changes_detected_at.push(v, node_distance[v])
                    g[v] = relaxation
                elif not g[v].smaller_equals(relaxation):
                    changes_detected_at.push(v, node_distance[v])
                    g[v] = g[v].minimum(relaxation)
    return g
