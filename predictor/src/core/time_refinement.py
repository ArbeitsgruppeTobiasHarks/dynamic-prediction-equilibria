from __future__ import annotations

from typing import List, Dict

from core.graph import DirectedGraph, Node
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
            g[v] = LinearlyInterpolatedFunction([phi, phi + 1], [float('inf'), float('inf')])
            tau[v] = phi

    queue: PriorityQueue[Node] = PriorityQueue([
        PriorityItem(g[v].eval(tau[v]), v) for v in nodes
    ])
    while len(queue) >= 2:
        i = queue.pop()
        k = queue.next()

        delta = min([weights[e.id](g[k](tau[k])) for e in i.incoming_edges], default=float('inf'))
        bound = g[k](tau[k]) + delta

        #  Calculate new_tau_i = max t s.t. g_i(t) <= g_k(tau_k) + delta
        index = None
        for index_j in range(len(g[i].times)):
            if g[i].values[index_j] > bound:
                index = index_j - 1
                break
        if index is None:
            new_tau_i = float('inf')
        else:
            new_tau_i = g[i].inverse(bound, index - 1)

        for e in i.outgoing_edges:
            j: Node = e.node_to
            g_j = g[i].plus(weights[e.id].compose(g[i]))
            for t in range(pair_i.tau[pair_i.v], tau_i_first + 1):
                gj_first[t] = pair_i.g[t] + Gt.weights[e](pair_i.g[t])
                g[e[1]][t] = min(g[e[1]][t], gj_first[t])

            tmpQ = PriorityQueue()
            for p in Q.queue:
                tmpQ.put(p)
            Q = tmpQ

        tau[i] = new_tau_i
    return g
