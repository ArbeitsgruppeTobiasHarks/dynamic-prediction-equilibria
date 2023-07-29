from typing import List, Tuple

from core.network import Network
from core.network_loader import NetworkLoader, Path
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def run_scenario():
    """
    We confirm the counter-example of uniqueness of (multi-commodity!) dynamic nash equilibria outlined in
    https://link.springer.com/article/10.1007/s11067-013-9206-6
    """
    # We map the nodes a,b,c,d in the paper to 0,1,2,3.
    # We map the edges 1,2,3,4,5,6 in the paper to 0,1,2,3,4,5.

    network = Network()
    infinite_capacity = 40
    # Link 1: a -> b; transit time 10; capacity 10
    network.add_edge(0, 1, 10, 10)
    # Link 2: b -> c; transit time 10; capacity infty
    network.add_edge(1, 2, 10, infinite_capacity)
    # Link 3: c -> d; transit time 10; capacity 10
    network.add_edge(2, 3, 10, 10)
    # Link 4: d -> a; transit time 10; capacity infty
    network.add_edge(3, 0, 10, infinite_capacity)
    # Link 5: a -> d; transit time 40; capacity infty
    network.add_edge(0, 3, 40, infinite_capacity)
    # Link 6: c -> b; transit time 40; capacity infty
    network.add_edge(2, 1, 40, infinite_capacity)

    network.graph.positions = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}

    # There are two OD-pairs, OD pair 1 from a to d, and OD pair 2 from c to b.
    # Both have a demand of 20 on the interval [0, 15].

    edges = network.graph.edges

    # FIRST EQUILIBRIUM
    path_inflows_1: List[Tuple[Path, RightConstant]] = [
        # Path A1: a -> b -> c -> d
        ([edges[0], edges[1], edges[2]], RightConstant([0, 5, 15], [20, 10, 0])),
        # Path A2: a -> d
        ([edges[4]], RightConstant([0, 5, 15], [0, 10, 0])),
    ]
    path_inflows_2: List[Tuple[Path, RightConstant]] = [
        # Path B1: c -> d -> a -> b
        ([edges[2], edges[3], edges[0]], RightConstant([0, 10, 15], [20, 10, 0])),
        # Path B2: c -> b
        ([edges[5]], RightConstant([0, 10, 15], [0, 10, 0])),
    ]

    path_inflows = path_inflows_1 + path_inflows_2

    loader = NetworkLoader(network, path_inflows)
    build = loader.build_flow()
    flow = next(build)
    while flow.phi < float("inf"):
        flow = next(build)
    print("Equilibrium 1 computed.")
    to_visualization_json(
        "./counter-example.json",
        flow,
        network,
        color_by_comm_idx={0: "red", 1: "red", 2: "blue", 3: "blue"},
    )


if __name__ == "__main__":
    run_scenario()
