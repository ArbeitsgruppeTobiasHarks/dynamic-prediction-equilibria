import unittest

from core.network import Network


def build_sample_network() -> Network:
    """
    Builds the sample network taken from the paper "Dynamic Flows with Adaptive Route Choice"
    Notation: index (travel time, capacity)
            0 (1,2)
           0 → → → → 1
           ↓ ↖       ↓
    1 (3,1)↓  3↖(1,1)↓ 2 (1,2)
           ↓     ↖   ↓
           ↓       ↖ ↓
           2 ← ← ← ← 3
             4 (1,1)
    """
    network = Network()
    network.add_edge(0, 1, 1, 2)
    network.add_edge(0, 2, 3, 1)
    network.add_edge(1, 3, 1, 2)
    network.add_edge(3, 0, 1, 1)
    network.add_edge(3, 2, 1, 1)

    return network


class TestNetwork(unittest.TestCase):
    def test_build(self):
        network = build_sample_network()
        self.assertEqual(len(network.graph.nodes), 4)
        self.assertEqual(len(network.graph.edges), 5)
        self.assertEqual(len(network.graph.nodes[0].outgoing_edges), 2)
        self.assertEqual(len(network.graph.nodes[2].incoming_edges), 2)


if __name__ == '__main__':
    unittest.main()
