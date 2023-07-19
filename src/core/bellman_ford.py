from __future__ import annotations
import array

from typing import Dict, List, Set

from core.graph import Node
from core.machine_precision import eps
from src.cython_test.piecewise_linear import PiecewiseLinear
from utilities.queues import PriorityQueue


def bellman_ford(
    sink: Node,
    costs: List[PiecewiseLinear],
    interesting_nodes: Set[Node],
    phi: float,
    horizon: float = float("inf"),
) -> Dict[Node, PiecewiseLinear]:
    """
    Calculates the earliest arrival time at `sink` as functions (l_v).
    """
    identity = PiecewiseLinear(array.array("d", (phi,)), array.array("d", (phi,)), 1.0, 1.0, (phi, horizon))
    # g_v(t) = earliest arrival at sink when starting in v at time t
    g: Dict[Node, PiecewiseLinear] = {sink: identity}
    node_distance: Dict[Node, int] = {sink: 0}

    def make_fifo(traversal: PiecewiseLinear):
        new_values = traversal.values
        for i in range(len(new_values) - 1):
            assert new_values[i] <= new_values[i + 1] + eps
            new_values[i + 1] = max(
                new_values[i], new_values[i + 1], traversal.times[i + 1]
            )

        if traversal.last_slope < 1:
            traversal.last_slope = 1
        return traversal

    edge_arrival_times = [
        make_fifo(identity.plus(cost)).simplify() for cost in costs
    ]

    changes_detected_at = PriorityQueue([(sink, 0.0)])

    while len(changes_detected_at) > 0:
        changed_nodes = changes_detected_at
        changes_detected_at = PriorityQueue([])

        for w in changed_nodes.sorted():
            for edge in w.incoming_edges:
                v = edge.node_from
                if v not in interesting_nodes:
                    continue
                T = edge_arrival_times[edge.id]
                restr_domain = (
                    T.min_t_above(g[w].domain[0]),
                    T.max_t_below(g[w].domain[1]),
                )
                if restr_domain[0] is None or restr_domain[1] is None:
                    continue
                relaxation = g[w].compose(T.restrict(restr_domain))  # type: ignore
                if v not in g.keys():
                    node_distance[v] = node_distance[w] + 1
                    if v not in changes_detected_at:
                        changes_detected_at.push(v, node_distance[v])
                    g[v] = relaxation.simplify()
                elif not g[v].smaller_equals(relaxation):
                    if not changes_detected_at.has(v):
                        changes_detected_at.push(v, node_distance[v])
                    g[v] = g[v].outer_minimum(relaxation).simplify()
    return g
