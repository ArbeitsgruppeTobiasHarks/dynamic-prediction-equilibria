import pickle

import numpy as np
from core.constant_predictor import ConstantPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.single_edge_distributor import SingleEdgeDistributor
from importer.csv_importer import network_from_csv, add_demands_to_network
from importer.show_net_info import show_net_info


def build_flows_from_demand(network_path: str, demands_path: str, check_for_optimizations: bool = True):
    save_flows_to_file = False
    use_default_demands = True

    if check_for_optimizations:
        assert (lambda: False)(), "Use PYTHONOPTIMIZE=TRUE for a faster generation."
    while True:
        network = network_from_csv(network_path)
        show_net_info(network)
        seed = add_demands_to_network(network, demands_path, use_default_demands)
        print(f"Generating flow with seed {seed}...")

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


if __name__ == '__main__':
    network_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_tiny.arcs'
    demands_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_tiny.demands'

    build_flows_from_demand(network_path, demands_path)
