import pickle

from core.constant_predictor import ConstantPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.single_edge_distributor import SingleEdgeDistributor
from importer.csv_importer import network_from_csv, add_demands_to_network

if __name__ == '__main__':
    save_flows_to_file = False
    check_for_optimizations = False
    use_default_demands = True

    if check_for_optimizations:
        assert (lambda: False)(), "Turn on the optimizer for the real speeeeeeeed"
    while True:
        network_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_small.arcs'
        network = network_from_csv(network_path)
        demands_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo.demands'
        seed = add_demands_to_network(network, demands_path, use_default_demands)

        print(f"Generating flow with seed {seed}...")
        max_in_degree = 0
        max_out_degree = 0
        for node in network.graph.nodes.values():
            max_in_degree = max(max_in_degree, len(node.incoming_edges))
            max_out_degree = max(max_out_degree, len(node.outgoing_edges))
        print(f"Maximum indgree: {max_in_degree}")
        print(f"Maximum outdegree: {max_out_degree}")

        predictors = [ConstantPredictor(network)]
        distributor = SingleEdgeDistributor(network)
        reroute_interval = 5
        horizon = 400

        flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)
        generator = flow_builder.build_flow()
        flow = next(generator)
        next_milestone = reroute_interval
        while flow.phi < horizon:
            flow = next(generator)
            if flow.phi >= next_milestone:
                print(f"phi={flow.phi}")
                next_milestone += reroute_interval

        print(f"Successfully built flow up to time {flow.phi}!")
        if save_flows_to_file:
            with open(f"./{seed}.flow.pickle", "wb") as file:
                pickle.dump(flow, file)
            print(f"Successfully written flow to disk!")
        else:
            print("Did not write flow to disk.")
        print("\n")
