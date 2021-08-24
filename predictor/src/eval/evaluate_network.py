import json
import os

from core.network import Network
from eval.evaluate import evaluate_single_run
from utilities.right_constant import RightConstant


def eval_network(network_path: str, output_folder: str, check_for_optimizations: bool = True):
    if check_for_optimizations:
        assert (lambda: False)(), "Use PYTHONOPTIMIZE=TRUE for a faster evaluation."
    print("Evaluating the network for all (possible) commodities given in the demands file.")
    print("For each of these commodities, we add an additional commodity - one for each predictor - " +
          "with small inflow rate.")
    print("All other commodities will run the constant predictor.")
    print()
    print("Will save the next working step in ./next_commodity.txt. " +
          "You can start multiple processes with this command to speed up the evaluation. " +
          "Make sure to delete the file next_commodity.txt if you want to do another round of evaluations.")
    print()
    while True:
        network = Network.from_file(network_path)

        if os.path.exists("./next_commodity.txt"):
            with open("./next_commodity.txt", "r") as file:
                original_commodity = int(file.read())
        else:
            original_commodity = 0
        with open("./next_commodity.txt", "w") as file:
            file.write(str(original_commodity + 1))
        if original_commodity >= len(network.commodities):
            break

        print()
        print(f"Computing Flow#{original_commodity}...")
        selected_commodity = network.remove_unnecessary_commodities(original_commodity)
        opt_net_inflow = RightConstant([0.], [1.], (0, float('inf')))
        evaluate_single_run(network, flow_id=original_commodity, opt_net_inflow=opt_net_inflow,
                            focused_commodity=selected_commodity, horizon=100., reroute_interval=2.5,
                            split=False, output_folder=output_folder)


def network_results_from_file_to_tikz():
    directory = "../../out/lol"
    files = os.listdir(directory)
    times = [[], [], [], [], []]  # Zero, Constant, Linear, RegularizedLinear, ML
    means = [0, 0, 0, 0, 0]
    num = 0
    for file_path in files:
        if not file_path.endswith(".json"):
            continue
        with open(os.path.join(directory, file_path), "r") as file:
            res_dict = json.load(file)
            travel_times = res_dict['avg_travel_times']
            if any(travel_times[j] != travel_times[0] for j in range(len(travel_times) - 1)):
                for i in range(len(times)):
                    times[i].append(travel_times[i] / travel_times[5])
                for j in range(len(means)):
                    means[j] += travel_times[j]
                num += 1
    for i in range(len(times)):
        tikz = "data \\\\\n"
        for y in times[i]:
            tikz += f"{y}\\\\\n"
        print(tikz)

    print("Means:")
    for j in range(len(means)):
        print(means[j] / num)


if __name__ == '__main__':
    network_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/tokyo_tiny/default_demands.pickle"
    eval_network(network_path, "../../out/lol", check_for_optimizations=False)
