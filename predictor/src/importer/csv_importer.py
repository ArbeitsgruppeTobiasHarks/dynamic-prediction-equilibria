import pickle
import random

from numpy import genfromtxt

from core.constant_predictor import ConstantPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network
from core.single_edge_distributor import SingleEdgeDistributor


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


def add_demands_to_network(network: Network, demands_path: str):
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
            print(f"Did not add the commodity of row {i}! The source #{source} can not reach the sink #{sink}!")
        else:
            demand = random.randint(20, 100)
            network.add_commodity(row[0], row[1], demand)
    return seed


if __name__ == '__main__':
    assert False, "Turn on the optimizer for the real speeeeeeeed"
    while True:
        network_path = '/home/michael/Nextcloud2/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_small.arcs'
        network = network_from_csv(network_path)
        demands_path = '/home/michael/Nextcloud2/Universität/2021-SS/softwareproject/data/from-kostas/tokyo.demands'
        seed = add_demands_to_network(network, demands_path)

        print(f"Generating flow with seed {seed}...")
        max_in_degree = 0
        max_out_degree = 0
        for node in network.graph.nodes.values():
            max_in_degree = max(max_in_degree, len(node.incoming_edges))
            max_out_degree = max(max_out_degree, len(node.outgoing_edges))
        print(f"Maximum indgree: {max_in_degree}")
        print(f"Maximum outdegree: {max_out_degree}")

        predictor = ConstantPredictor(network)
        distributor = SingleEdgeDistributor(network)
        reroute_interval = 5
        horizon = 400

        flow_builder = MultiComFlowBuilder(network, predictor, distributor, reroute_interval)
        generator = flow_builder.build_flow()
        flow = next(generator)
        next_milestone = reroute_interval
        while flow.phi < horizon:
            flow = next(generator)
            if flow.phi >= next_milestone:
                print(f"phi={flow.phi}")
                next_milestone += reroute_interval
        print(f"Successfully built flow up to time {flow.phi}!")
        with open(f"./{seed}.flow.pickle", "wb") as file:
            pickle.dump(flow, file)
        print(f"Successfully written flow to disk!")
        print("\n")
