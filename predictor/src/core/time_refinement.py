from __future__ import annotations

from typing import List, Dict

from core.graph import DirectedGraph, Node
from utilities.piecewise_linear import PiecewiseLinear
from utilities.queues import PriorityQueue, PriorityItem

"""
An adjusted version of the Time Refinement Algorithm from the Paper
"Finding Time-Dependent Shortest Paths over Large Graphs"
by Bolin Ding, Jeffrey Xu Yu, Lu Qin
"""


def time_refinement(graph: DirectedGraph, sink: Node, costs: List[PiecewiseLinear], phi: float):
    nodes = graph.nodes.values()
    identity = PiecewiseLinear([phi, phi + 1], [phi, phi + 1], (phi, float('inf')))
    g: Dict[Node, PiecewiseLinear] = {
        sink: identity
    }
    tau: Dict[Node, float] = {v: phi for v in nodes}

    queue: PriorityQueue[Node] = PriorityQueue([
        PriorityItem(phi if v == sink else float('inf'), v) for v in nodes
    ])

    nodes_left = len(nodes)

    while len(queue) >= 2 and nodes_left > 0:
        i = queue.pop()
        if tau[i] == phi:
            nodes_left -= 1
        g_k_of_tau_k = queue.min_time()
        #  We probably need to adjust the bound g_k_of_tau_k + delta from the paper:
        #  delta = min([costs[e.id](g_k_of_tau_k) for e in i.outgoing_edges], default=float('inf'))

        #  Lets see... max t s.t. for all outgoing edges i,w:  g_i(t) <= g_w(T_{i,w}(t))
        #  i.e. s.t. g_i(T_{i,w}^{-1} (t) ) <= g_w(t) (<= g_k(tau_k))
        #  First get max t' s.t. g_i( t' ) <= g_k(tau_k)
        #  Then get for all edges iw: max t s.t. T_{i,w}^{-1} ( t) <= t'  (i.e. t = T_{i,w}(t') )
        #  and then take the minimum as bound.

        t_prime = g[i].max_t_below_bound(g_k_of_tau_k)
        if t_prime == float('inf'):
            tau[i] = float('inf')
        else:
            tau[i] = t_prime + min([costs[e.id](t_prime) for e in i.outgoing_edges], default=float('inf'))

        for e in i.incoming_edges:  # Difference to paper: Incoming instead of outgoing edges
            j: Node = e.node_from
            # Difference to paper: Other computation of bound
            bound_for_g_j = g[i].compose(identity.plus(costs[e.id]).ensure_monotone())
            g[j] = g[j].minimum(bound_for_g_j) if j in g.keys() else bound_for_g_j
            new_val = g[j](tau[j])
            queue.decrease_key(j, new_val)

        if tau[i] < float('inf'):
            queue.push(PriorityItem(g[i](tau[i]), i))

    return g
