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

    network.graph.positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 1.0),
    }

    return network
