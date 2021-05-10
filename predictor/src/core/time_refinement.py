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


def time_refinement(graph: DirectedGraph, source: Node, weights: List[LinearlyInterpolatedFunction], phi: float):
    nodes = graph.nodes.values()
    g: Dict[Node, LinearlyInterpolatedFunction] = {
        source: LinearlyInterpolatedFunction([phi, phi + 1], [phi, phi + 1]),
    }
    tau: Dict[Node, float] = {source: phi}
    for v in nodes:
        if v != source:
            tau[v] = phi

    queue: PriorityQueue[Node] = PriorityQueue([
        PriorityItem(phi if v == source else float('inf'), v) for v in nodes
    ])
    while len(queue) >= 2:
        i = queue.pop()
        k = queue.next()

        g_k_of_tau_k = g[k](tau[k]) if k in g.keys() else float('inf')
        delta = min([weights[e.id](g_k_of_tau_k) for e in i.incoming_edges],
                    default=float('inf'))
        bound = g_k_of_tau_k + delta

        #  Calculate new_tau_i = max t s.t. g_i(t) <= g_k(tau_k) + delta
        index = None
        for index_j in range(len(g[i].times)):
            if g[i].values[index_j] > bound:
                index = index_j - 1
                break
        if index is None:
            tau[i] = float('inf')
        else:
            tau[i] = g[i].inverse(bound, index)

        for e in i.outgoing_edges:
            j: Node = e.node_to
            bound_for_g_j = g[i].plus(weights[e.id].compose(g[i]))
            g[j] = g[j].minimum(bound_for_g_j) if j in g.keys() else bound_for_g_j
            new_val = g[j](tau[j])
            queue.decrease_key(j, new_val)

        # if tau[i] >=

    return g, tau
