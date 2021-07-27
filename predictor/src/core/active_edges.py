


from typing import Callable, List, Set
from core.dijkstra import dynamic_dijkstra
from core.graph import DirectedGraph, Edge, Node

from utilities.piecewise_linear import PiecewiseLinear

identity = PiecewiseLinear([0., 1.], [0.,1.])

def get_active_edges(
    costs: List[PiecewiseLinear], theta: float, source: Node, sink: Node,
    relevant_nodes: Set[Node], graph: DirectedGraph, strong_fifo: bool
) -> Set[Edge]:
    if len(e for e in source.outgoing_edges if e.node_to in relevant_nodes) <= 1:
        return source.outgoing_edges
    arrivals = dynamic_dijkstra(
        theta, source, sink, relevant_nodes, costs
    )
    if strong_fifo: # Simple search on active edges leading to t.
        active_edges = []
        queue: List[Node] = [sink]
        nodes_enqueued: Set[Node] = {sink}
        while len(queue) > 0:
            w = queue.pop()
            for e in w.incoming_edges:
                v = e.node_from
                if v not in arrivals.keys():
                    continue
                if arrivals[v] + costs[e.id](arrivals[v]) <= arrivals[w]:
                    if v == source:
                        active_edges.append(e)
                    if v not in nodes_enqueued:
                        queue.append(v)
                        nodes_enqueued.add(v)
        return active_edges
    # Second run of the Dynamic Dijkstra Algorithm on the reverse graph.
    graph.reverse()
    traversals = [cost + identity for cost in costs]
    new_costs: List[Callable[[float], float]] = [
        lambda t: -trav.reversal(-t) - t for trav in traversals
    ]
    neg_departures = dynamic_dijkstra(
        arrivals[sink], sink, source, relevant_nodes, new_costs
    )
    graph.reverse()
    active_edges = []
    for e in source.outgoing_edges:
        if traversals[e.id](theta) <= -neg_departures[e.node_to]:
            active_edges.append(e)
    return active_edges