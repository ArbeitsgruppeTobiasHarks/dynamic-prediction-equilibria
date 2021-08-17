from __future__ import annotations

from typing import List, Dict, Set

from core.graph import Node
from core.machine_precision import eps
from utilities.piecewise_linear import PiecewiseLinear
from utilities.queues import PriorityQueue


def bellman_ford(
        sink: Node,
        costs: List[PiecewiseLinear],
        interesting_nodes: Set[Node],
        phi: float
) -> Dict[Node, PiecewiseLinear]:
    identity = PiecewiseLinear([phi], [phi], 1., 1., (phi, float('inf')))
    g: Dict[Node, PiecewiseLinear] = {
        sink: identity
    }
    node_distance: Dict[Node, int] = {
        sink: 0
    }

    def get_fifo_arrival_time(traversal: PiecewiseLinear):
        new_values = traversal.values.copy()
        for i in range(len(new_values) - 1):
            assert new_values[i] <= new_values[i + 1] + eps
            new_values[i + 1] = max(new_values[i], new_values[i + 1], traversal.times[i + 1])

        new_traversal = PiecewiseLinear(traversal.times, new_values, traversal.first_slope, traversal.last_slope,
                                        traversal.domain)
        if new_traversal.last_slope < 1:
            new_traversal.last_slope = 1
        return new_traversal

    edge_arrival_times = [get_fifo_arrival_time(identity.plus(cost)).simplify() for cost in costs]

    changes_detected_at = PriorityQueue([(sink, 0.)])

    while len(changes_detected_at) > 0:
        changed_nodes = changes_detected_at
        changes_detected_at = PriorityQueue([])

        for w in changed_nodes.sorted():
            for edge in w.incoming_edges:
                v = edge.node_from
                if v not in interesting_nodes:
                    continue
                relaxation = g[w].compose(edge_arrival_times[edge.id])
                if v not in g.keys():
                    node_distance[v] = node_distance[w] + 1
                    if not changes_detected_at.has(v):
                        changes_detected_at.push(v, node_distance[v])
                    g[v] = relaxation.simplify()
                elif not g[v].smaller_equals(relaxation):
                    if not changes_detected_at.has(v):
                        changes_detected_at.push(v, node_distance[v])
                    g[v] = g[v].minimum(relaxation).simplify()
    return g
