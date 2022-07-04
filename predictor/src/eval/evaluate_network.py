import json
import os
import random
from typing import Optional

from core.network import Network, Commodity
from core.predictors.predictor_type import PredictorType
from eval.evaluate import COLORS, evaluate_single_run, PredictorBuilder
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def eval_network(network_path: str, output_folder: str, inflow_horizon: float,
                 future_timesteps: int, prediction_interval: float, reroute_interval: float, horizon: float,
                 split: bool = False, random_commodities: bool = False, suppress_log=True,
                 build_predictors: Optional[PredictorBuilder] = None, check_for_optimizations: bool = True):
    if check_for_optimizations:
        assert (lambda: False)(
        ), "Use PYTHONOPTIMIZE=TRUE for a faster evaluation."
    print("Evaluating the network for all (possible) commodities given in the demands file.")
    print("For each of these commodities, we add an additional commodity - one for each predictor - " +
          "with small inflow rate.")
    print("All other commodities will run the constant predictor.")
    print()
    print("You can start multiple processes with this command to speed up the evaluation. " +
          "Make sure to delete the output folder if you want to do another round of evaluations.")
    print()
    num_commodities = len(Network.from_file(network_path).commodities)
    os.makedirs(output_folder, exist_ok=True)
    for k in range(num_commodities):
        eval_path = os.path.join(output_folder, f"{k}.json")
        lock_path = os.path.join(output_folder, f".lock.{k}.json")
        if os.path.exists(eval_path):
            print(f"Commodity {k} already evaluated. Skipping...")
            continue
        elif os.path.exists(lock_path):
            print(f"Detected lock file for commodity {k}. Skipping...")
            continue

        with open(lock_path, "w") as file:
            file.write("")

        network = Network.from_file(network_path)

        print()
        print(f"Evaluating Commodity {k}...")
        if random_commodities:
            nodes = list(network.graph.nodes.values())
            random.seed(-k)
            while True:
                source = random.choice(nodes)
                sink = random.choice(nodes)
                if sink in network.graph.get_reachable_nodes(source):
                    break
            commodity = Commodity(source, sink,
                                  net_inflow=RightConstant([0, inflow_horizon], [
                                                           1, 0], (0, float('inf'))),
                                  predictor_type=PredictorType.CONSTANT)
            network.commodities.append(commodity)
            selected_commodity = network.remove_unnecessary_commodities(
                len(network.commodities) - 1)
        else:
            selected_commodity = network.remove_unnecessary_commodities(k)
        _, _, flow = evaluate_single_run(network, flow_id=k, focused_commodity=selected_commodity,
                                         horizon=horizon, reroute_interval=reroute_interval,
                                         inflow_horizon=inflow_horizon, future_timesteps=future_timesteps, prediction_interval=prediction_interval,
                                         suppress_log=suppress_log, split=split, output_folder=output_folder, build_predictors=build_predictors)

        to_visualization_json(
            output_folder + "/visualization/ " +
            f"{k}.vis.json", flow, network,
            {id: COLORS[comm.predictor_type]
                for (id, comm) in enumerate(network.commodities)}
        )
        os.remove(lock_path)

    network_results_from_file_to_tikz(output_folder)


def network_results_from_file_to_tikz(directory: str):
    files = os.listdir(directory)
    # Zero, Constant, Linear, RegularizedLinear, ML
    times = [[], [], [], [], []]
    means = [0, 0, 0, 0, 0, 0]
    num = 0
    for file_path in files:
        if file_path.startswith(".lock"):
            raise ValueError("Detected lock file. Will not create tikz file.")
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
    configs = [
        {"label": "$\\hat q^{\\text{Z}}$", "color": "blue"},
        {"label": "$\\hat q^{\\text{C}}$", "color": "red"},
        {"label": "$\\hat q^{\\text{L}}$",
            "color": "{rgb,255:red,0; green,128; blue,0}"},
        {"label": "$\\hat q^{\\text{RL}}$", "color": "orange"},
        {"label": "$\\hat q^{\\text{ML}}$", "color": "black"},
    ]
    tikz = """\\begin{tikzpicture}
        \\begin{axis}
  [
  width=.5\\textwidth,
  boxplot/draw direction = y,
  ylabel = {$T_i^{\\mathrm{avg}} / T^{\\mathrm{avg}}_{\\text{OPT}}$},
  xtick = {1, 2, 3, 4, 5},
  xticklabels = {""" + ",".join([c["label"] for c in configs]) + """},
  every axis plot/.append style = {fill, fill opacity = .1},
  ]
    """
    for i in range(len(times)):
        tikz += """\\addplot + [
          mark = *,
          boxplot,
          color="""
        tikz += configs[i]["color"]
        tikz += """]
          table [row sep = \\\\, y index = 0] {
        data \\\\
        """
        for y in times[i]:
            tikz += f"{y}\\\\\n"
        tikz += "};\n"

    tikz += """\\end{axis}
    \\end{tikzpicture}
    """

    with open(os.path.join(directory, "boxplot.tikz"), "w") as file:
        file.write(tikz)
    print("Successfully saved a tikz boxplot in the output directory.")

    print("Means:")
    for j in range(len(means)):
        print(means[j] / num)


if __name__ == '__main__':
    def main():
        # network_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/tokyo_tiny/default_demands.pickle"
        network_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/random-demands.pickle"
        Network.from_file(network_path).print_info()
        # eval_network(network_path, "../../out/sioux-falls-3", check_for_optimizations=False)
        # network_results_from_file_to_tikz("../../out/sioux-falls-3")

    main()
