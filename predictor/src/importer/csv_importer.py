import random

from numpy import genfromtxt
from core.network import Network


def network_from_csv(path: str) -> Network:
    np_data = genfromtxt(path, delimiter=' ', skip_header=1)
    # subsequent rows are structured as:
    # from_node to_node length_in_meters freeflow_speed_kph num_lanes
    # freeflow_speed_kph is -1 when the value is absent in the data.
    # We default to 20 for now.

    network = Network()
    for i, row in enumerate(np_data):
        freeflow_speed = row[3] if row[3] > 0 else 20.
        travel_time = row[2] / freeflow_speed
        capacity = row[4] * freeflow_speed

        if row[0] == row[1]:
            print("Edge #{i} has not been added, as it's a loop from and to node #{row[0]}!")
            continue
        network.add_edge(row[0], row[1], travel_time, capacity)
    return network


def add_demands_to_network(network: Network, demands_path: str, use_default_demands: bool,
                           suppress_ignored: bool = False, upscale: bool = True):
    if not use_default_demands:
        with open("./seed.txt", "r") as file:
            seed = int(file.read())
        with open("./seed.txt", "w") as file:
            file.write(str(seed + 1))
        random.seed(seed)
    np_data = genfromtxt(demands_path, delimiter=' ')
    for i, row in enumerate(np_data):
        # Filter unrealistic / impractical commodities where the source cant reach the sink
        source = network.graph.nodes[row[0]]
        sink = network.graph.nodes[row[1]]
        if source not in network.graph.get_nodes_reaching(sink):
            if not suppress_ignored:
                print(f"Did not add the commodity of row {i}! The source #{source} can not reach the sink #{sink}!")
        else:
            if use_default_demands:
                # after upscaling: between 20 and 100
                demand = 20 + (row[2] - 10) / 20. * 80. if upscale else row[2]
            else:
                demand = random.randint(20, 100)
            network.add_commodity(row[0], row[1], demand, 0)
    return None if use_default_demands else seed
