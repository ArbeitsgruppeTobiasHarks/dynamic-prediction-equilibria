from __future__ import annotations

from typing import List, Dict

from core.graph import DirectedGraph, Node
from utilities.interpolate import LinearlyInterpolatedFunction
from utilities.queues import PriorityQueue, PriorityItem

"""
An adjusted version of the Time Refinement Algorithm from the Paper
"Finding Time-Dependent Shortest Paths over Large Graphs"
by Bolin Ding, Jeffrey Xu Yu, Lu Qin
"""


def time_refinement(graph: DirectedGraph, sink: Node, costs: List[LinearlyInterpolatedFunction], phi: float):
    nodes = graph.nodes.values()
    identity = LinearlyInterpolatedFunction([phi, phi + 1], [phi, phi + 1], (phi, float('inf')))
    g: Dict[Node, LinearlyInterpolatedFunction] = {
        sink: identity
    }
    tau: Dict[Node, float] = {sink: phi}
    for v in nodes:
        if v != sink:
            tau[v] = phi

    queue: PriorityQueue[Node] = PriorityQueue([
        PriorityItem(phi if v == sink else float('inf'), v) for v in nodes
    ])
    while len(queue) >= 2:
        i = queue.pop()
        g_k_of_tau_k = queue.min_time()
        #  We probably need to adjust the bound g_k_of_tau_k + delta from the paper:
        #  delta = min([costs[e.id](g_k_of_tau_k) for e in i.outgoing_edges], default=float('inf'))

        #  Lets see... max t s.t. for all outgoing edges i,w:  g_i(t) <= g_w(T_{i,w}(t))
        #  i.e. s.t. g_i(T_{i,w}^{-1} (t) ) <= g_w(t) (<= g_k(tau_k))
        #  First get max t' s.t. g_i( t' ) <= g_k(tau_k)
        #  Then get for all edges iw: max t s.t. T_{i,w}^{-1} ( t) <= t'  (i.e. t = T_{i,w}(t') )
        #  and then take the minimum as bound.

        index = None
        for index_j in range(len(g[i].times)):
            if g[i].values[index_j] > g_k_of_tau_k:
                index = index_j - 1
                break
        if index is None:
            t_prime = float('inf')
        else:
            t_prime = g[i].inverse(g_k_of_tau_k, index)

        tau[i] = t_prime + min([costs[e.id](t_prime) for e in i.outgoing_edges], default=float('inf'))

        for e in i.incoming_edges:  # Different to algorithm from paper
            j: Node = e.node_from
            bound_for_g_j = g[i].compose(identity.plus(costs[e.id]))  # Different to algorithm from paper
            g[j] = g[j].minimum(bound_for_g_j) if j in g.keys() else bound_for_g_j
            new_val = g[j](tau[j])
            queue.decrease_key(j, new_val)

        if tau[i] < float('inf'):
            queue.push(PriorityItem(g[i](tau[i]), i))

    return g
