import os
import pickle

from core.constant_predictor import ConstantPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.uniform_distributor import UniformDistributor
from importer.csv_importer import network_from_csv, add_demands_to_network
from utilities.build_with_times import build_with_times


def build_flows_from_demand(network_path: str, demands_path: str, out_directory: str, number_flows: int,
                            check_for_optimizations: bool = True):
    if check_for_optimizations:
        assert (lambda: False)(), "Use PYTHONOPTIMIZE=TRUE for a faster generation."
    os.makedirs(out_directory, exist_ok=True)
    print()
    print("Will save the current state in ./next_random_seed.txt. " + \
          "You can start multiple processes with this command to speed up the generation. " + \
          "Make sure to delete the file next_random_seed.txt if you want to do another round of generations.")
    print()
    while True:

        network = network_from_csv(network_path)
        if os.path.exists("./next_random_seed.txt"):
            with open("./next_random_seed.txt", "r") as file:
                random_seed = int(file.read())
        else:
            random_seed = 0

        with open("./next_random_seed.txt", "w") as file:
            file.write(str(random_seed + 1))
        if random_seed >= number_flows:
            break

        add_demands_to_network(network, demands_path, random_seed=random_seed)
        print(f"Generating flow with seed {random_seed}...")

        predictors = [ConstantPredictor(network)]
        distributor = UniformDistributor(network)
        reroute_interval = 2.5
        horizon = 100

        flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)
        flow = build_with_times(flow_builder, random_seed, reroute_interval, horizon)

        print(f"Successfully built flow up to time {flow.phi}!")
        with open(os.path.join(out_directory, f"{random_seed}.flow.pickle"), "wb") as file:
            pickle.dump(flow, file)
        print(f"Successfully written flow to disk!")
        print("\n")


if __name__ == '__main__':
    network_path = '/home/michael/Nextcloud2/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_tiny.arcs'
    demands_path = '/home/michael/Nextcloud2/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_tiny.demands'

    build_flows_from_demand(network_path, demands_path, "../../out/generated_flows", 200)
