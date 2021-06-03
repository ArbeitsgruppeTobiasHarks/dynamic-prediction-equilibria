import numpy as np
from core.network import Network


def show_net_info(network: Network):
    print(f"The network contains {len(network.graph.nodes)} nodes and {len(network.graph.edges)} edges.")
    print(f"Moreover, there are {len(network.commodities)} commodities.")
    print(f"Minimum/Maximum capacity: {np.min(network.capacity)}/{np.max(network.capacity)}")
    print(f"Minimum/Maximum transit time: {np.min(network.travel_time)}/{np.max(network.travel_time)}")
    max_in_degree = 0
    max_out_degree = 0
    max_degree = 0
    for node in network.graph.nodes.values():
        max_degree = max(max_degree, len(node.incoming_edges) + len(node.outgoing_edges))
        max_in_degree = max(max_in_degree, len(node.incoming_edges))
        max_out_degree = max(max_out_degree, len(node.outgoing_edges))
    print(f"Maximum indgree: {max_in_degree}")
    print(f"Maximum outdegree: {max_out_degree}")
    print(f"Maximum degree: {max_degree}")
