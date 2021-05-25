import unittest

from core.bellman_ford import bellman_ford
from test.sample_network import build_sample_network
from utilities.interpolate import LinearlyInterpolatedFunction


class TestBellmanFord(unittest.TestCase):

    def test_flow_bellman_ford(self):
        network = build_sample_network()
        costs = [
            LinearlyInterpolatedFunction([2.05, 2.55, 3.55], [1.025, 1., 1.], (2.05, float('inf'))),
            LinearlyInterpolatedFunction([2.05, 2.55, 3.55], [3.075, 3.7, 3.7], (2.05, float('inf'))),
            LinearlyInterpolatedFunction([2.05, 2.55, 3.55], [1., 1., 1.], (2.05, float('inf'))),
            LinearlyInterpolatedFunction([2.05, 2.55, 3.55], [1., 1., 1.], (2.05, float('inf'))),
            LinearlyInterpolatedFunction([2.05, 2.55, 3.55], [1.05, 1.55, 1.55], (2.05, float('inf')))
        ]
        phi = 2.075
        labels = bellman_ford(network.graph.nodes[2], costs, 2.075)
        for v in network.graph.nodes.values():
            has_active_edge = False
            for e in v.outgoing_edges:
                if labels[e.node_to](phi + costs[e.id](phi)) <= labels[v](phi):
                    has_active_edge = True
                    break
            assert has_active_edge or len(v.outgoing_edges) == 0
