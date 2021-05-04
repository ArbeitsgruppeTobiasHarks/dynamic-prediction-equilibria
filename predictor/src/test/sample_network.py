import unittest

from core.network import Network


def build_sample_network() -> Network:
    """
    Builds the sample network taken from the paper "Dynamic Flows with Adaptive Route Choice"
    Notation: (travel time, capacity)
            (1,2)
         0 → → → → 1
         ↓ ↘       ↓
    (3,1)↓   ↘(1,1)↓ (1,2)
         ↓     ↘   ↓
         ↓       ↘ ↓
         2 ← ← ← ← 3
            (1,1)
    """
    network = Network()
    network.add_edge(0, 1, 0, 2, 1)
    network.add_edge(0, 3, 1, 1, 1)
    network.add_edge(0, 2, 2, 1, 3)
    network.add_edge(1, 3, 3, 2, 1)
    network.add_edge(3, 2, 4, 1, 1)

    return network


class TestNetwork(unittest.TestCase):
    def test_build(self):
        network = build_sample_network()
        self.assertEqual(len(network.graph.nodes), 4)
        self.assertEqual(len(network.graph.edges), 5)
        self.assertEqual(len(network.graph.nodes[0].outgoing_edges), 3)
        self.assertEqual(len(network.graph.nodes[2].incoming_edges), 2)


if __name__ == '__main__':
    unittest.main()
