import unittest
from test.sample_network import build_sample_network

from core.bellman_ford import bellman_ford
from utilities.piecewise_linear import PiecewiseLinear


def test_flow_bellman_ford():
    network = build_sample_network()
    costs = [
        PiecewiseLinear(
            [2.05, 2.55, 3.55], [1.025, 1.0, 1.0], 0.0, 0.0, (2.05, float("inf"))
        ),
        PiecewiseLinear(
            [2.05, 2.55, 3.55], [3.075, 3.7, 3.7], 0.0, 0.0, (2.05, float("inf"))
        ),
        PiecewiseLinear(
            [2.05, 2.55, 3.55], [1.0, 1.0, 1.0], 0.0, 0.0, (2.05, float("inf"))
        ),
        PiecewiseLinear(
            [2.05, 2.55, 3.55], [1.0, 1.0, 1.0], 0.0, 0.0, (2.05, float("inf"))
        ),
        PiecewiseLinear(
            [2.05, 2.55, 3.55], [1.05, 1.55, 1.55], 0.0, 0.0, (2.05, float("inf"))
        ),
    ]
    phi = 2.075
    labels = bellman_ford(
        network.graph.nodes[2], costs, set(network.graph.nodes.values()), 2.075
    )
    for v in network.graph.nodes.values():
        has_active_edge = False
        for e in v.outgoing_edges:
            if labels[e.node_to](phi + costs[e.id](phi)) <= labels[v](phi):
                has_active_edge = True
                break
        assert has_active_edge or len(v.outgoing_edges) == 0
