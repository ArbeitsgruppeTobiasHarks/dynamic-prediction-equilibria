from dataclasses import dataclass
import json
from math import ceil, log10
import numpy as np
import os
import random
from typing import Dict, List, Optional, Tuple

from core.network import Network, Commodity
from core.predictors.predictor_type import PredictorType
from eval.evaluate import COLORS, evaluate_single_run, PredictorBuilder
from ml.build_test_flows import generate_network_demands
from utilities.file_lock import wait_for_locks, with_file_lock
from utilities.json_encoder import JSONEncoder
from utilities.right_constant import RightConstant
from utilities.status_logger import StatusLogger
from visualization.make_tikz_boxplot import BoxPlot, make_tikz_boxplot
from visualization.to_json import merge_commodities, to_visualization_json


def eval_network_demand(
    network_path: str,
    number_flows: int,
    out_dir: str,
    inflow_horizon: float,
    future_timesteps: int,
    prediction_interval: float,
    reroute_interval: float,
    horizon: float,
    build_predictors: PredictorBuilder,
    demand_sigma: Optional[float],
    visualization_config: Dict[PredictorType, Tuple[str, str]],
    split: bool = False,
    suppress_log=True,
    check_for_optimizations: bool = True,
    generate_flow_visualization: bool = True,
):
    """
    Evaluates a single (randomly chosen) commodity.
    """
    if check_for_optimizations:
        assert (lambda: False)(), "Use PYTHONOPTIMIZE=TRUE for a faster evaluation."
    os.makedirs(out_dir, exist_ok=True)

    for flow_id in range(number_flows):
        flow_path = os.path.join(
            out_dir, f"{str(flow_id).zfill(ceil(log10(number_flows)))}.pickle"
        )
        json_eval_path = os.path.join(
            out_dir, f"{str(flow_id).zfill(ceil(log10(number_flows)))}.json"
        )

        visualization_json_path = os.path.join(
            out_dir,
            "visualization",
            f"{str(flow_id).zfill(ceil(log10(number_flows)))}.vis.json",
        )

        def handle(_):
            network = Network.from_file(network_path)
            original_num_commodities = len(network.commodities)
            seed = -flow_id - 1
            print()
            print(f"Building Evaluation Flow#{flow_id} with seed {seed}...")
            generate_network_demands(network, seed, inflow_horizon, demand_sigma)
            focused_commodity_index = random.randrange(0, len(network.commodities))
            print(
                f"Focused Commodity {focused_commodity_index} with source {next(iter(network.commodities[focused_commodity_index].sources))} and sink {network.commodities[focused_commodity_index].sink}."
            )
            _, _, flow = evaluate_single_run(
                network,
                flow_id=flow_id,
                focused_commodity_index=focused_commodity_index,
                horizon=horizon,
                reroute_interval=reroute_interval,
                flow_path=flow_path,
                json_eval_path=json_eval_path,
                inflow_horizon=inflow_horizon,
                future_timesteps=future_timesteps,
                prediction_interval=prediction_interval,
                suppress_log=suppress_log,
                split=split,
                build_predictors=build_predictors,
            )

            if generate_flow_visualization:
                with StatusLogger("Generating flow visualization..."):
                    merged_flow = merge_commodities(
                        flow, network, range(original_num_commodities)
                    )
                    to_visualization_json(
                        visualization_json_path,
                        merged_flow,
                        network,
                        {
                            id: COLORS[comm.predictor_type]
                            for (id, comm) in enumerate(network.commodities)
                        },
                    )

        expect_exists = [flow_path, json_eval_path]
        if generate_flow_visualization:
            expect_exists.append(visualization_json_path)
        with_file_lock(flow_path, handle, expect_exists)

    wait_for_locks(out_dir)

    generate_slowdown_boxplot(out_dir, visualization_config)
    generate_mae_boxplot(out_dir, visualization_config)
    eval_jsons_to_avg_slowdowns(out_dir, visualization_config)
    compare_mae_with_perf(out_dir, visualization_config)


def eval_network_for_commodities(
    network_path: str,
    out_dir: str,
    inflow_horizon: float,
    future_timesteps: int,
    prediction_interval: float,
    reroute_interval: float,
    horizon: float,
    split: bool = False,
    random_commodities: bool = False,
    suppress_log=True,
    build_predictors: Optional[PredictorBuilder] = None,
    check_for_optimizations: bool = True,
    visualization_config: Optional[Dict[PredictorType, Tuple[str, str]]] = None,
):
    if check_for_optimizations:
        assert (lambda: False)(), "Use PYTHONOPTIMIZE=TRUE for a faster evaluation."
    print(
        "Evaluating the network for all (possible) commodities given in the demands file."
    )
    print(
        "For each of these commodities, we add an additional commodity - one for each predictor - "
        + "with small inflow rate."
    )
    print("All other commodities will run the constant predictor.")
    print()
    print(
        "You can start multiple processes with this command to speed up the evaluation. "
        + "Make sure to delete the output folder if you want to do another round of evaluations."
    )
    print()
    num_commodities = len(Network.from_file(network_path).commodities)
    os.makedirs(out_dir, exist_ok=True)
    for k in range(num_commodities):
        eval_path = os.path.join(out_dir, f"{k}.json")
        lock_path = os.path.join(out_dir, f".lock.{k}.json")
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
            commodity = Commodity(
                source,
                sink,
                net_inflow=RightConstant(
                    [0, inflow_horizon], [1, 0], (0, float("inf"))
                ),
                predictor_type=PredictorType.CONSTANT,
            )
            network.commodities.append(commodity)
            selected_commodity = network.remove_unnecessary_commodities(
                len(network.commodities) - 1
            )
        else:
            selected_commodity = network.remove_unnecessary_commodities(k)
        _, _, flow = evaluate_single_run(
            network,
            flow_id=k,
            focused_commodity_index=selected_commodity,
            horizon=horizon,
            reroute_interval=reroute_interval,
            inflow_horizon=inflow_horizon,
            future_timesteps=future_timesteps,
            prediction_interval=prediction_interval,
            suppress_log=suppress_log,
            split=split,
            out_dir=out_dir,
            build_predictors=build_predictors,
        )

        to_visualization_json(
            out_dir + "/visualization/ " + f"{k}.vis.json",
            flow,
            network,
            {
                id: visualization_config[comm.predictor_type][0]
                for (id, comm) in enumerate(network.commodities)
            },
        )
        os.remove(lock_path)

    wait_for_locks(out_dir)

    generate_slowdown_boxplot(out_dir, visualization_config)

    generate_mae_boxplot(out_dir, visualization_config)

    eval_jsons_to_avg_slowdowns(out_dir, visualization_config)

    compare_mae_with_perf(out_dir, visualization_config)


def compare_mae_with_perf(dir: str, visualization_config):
    files = sorted([file for file in os.listdir(dir) if file.endswith(".json")])
    # Zero, Constant, Linear, RegularizedLinear, ML
    colors = [t[0] for t in visualization_config.values()]
    coordinates = [[] for _ in colors]
    for file_path in files:
        with open(os.path.join(dir, file_path), "r") as file:
            res_dict = json.load(file)
            mean_absolute_errors = res_dict["mean_absolute_errors"]
            travel_times = res_dict["avg_travel_times"]
            if any(
                travel_times[j] != travel_times[0] for j in range(len(travel_times) - 1)
            ):
                for i, err in enumerate(mean_absolute_errors):
                    coordinates[i].append((err, travel_times[i] - travel_times[-1]))

    tikz = r"""
    \begin{tikzpicture}
    \begin{axis}
    """

    for i, pairs in enumerate(coordinates):
        tikz += (
            r"""
        \addplot+[only marks, color="""
            + colors[i]
            + """, mark=x] coordinates {
        """
        )
        for x, y in pairs:
            tikz += f"({x}, {y})\n"

        tikz += r"""};"""
    tikz += r"""
    \end{axis}
    \end{tikzpicture}
    """

    with open(os.path.join(dir, "time-loss-by-mae.tikz"), "w") as file:
        file.write(tikz)


def generate_slowdown_boxplot(dir: str, visualization_config):
    files = [file for file in os.listdir(dir) if file.endswith(".json")]

    colors = [t[0] for t in visualization_config.values()]
    labels = [t[1] for t in visualization_config.values()]

    slowdowns = [[] for _ in visualization_config]
    means = [0.0 for _ in visualization_config]
    for file_path in files:
        with open(os.path.join(dir, file_path), "r") as file:
            res_dict = json.load(file)
            travel_times = res_dict["avg_travel_times"]
            if any(
                travel_times[j] != travel_times[0] for j in range(len(travel_times) - 1)
            ):
                for i in range(len(slowdowns)):
                    slowdowns[i].append(travel_times[i] / travel_times[-1] - 1)
                for j in range(len(means)):
                    means[j] += travel_times[j]

    tikz = make_tikz_boxplot(
        ylabel=r"\begin{tabular}{c}Slowdown\\\small$T_i^{\mathrm{avg}} / T^{\mathrm{avg}}_{\text{OPT}} - 1$\end{tabular}",
        plots=[
            BoxPlot(labels[i], colors[i], slowdowns[i]) for i in range(len(slowdowns))
        ],
    )

    with open(os.path.join(dir, "slowdown-boxplot.tikz"), "w") as file:
        file.write(tikz)
    print("Successfully saved a Slowdown boxplot in the output directory.")

    print("Mean average travel time:")
    for j in range(len(means)):
        print(means[j] / len(files))


def generate_mae_boxplot(dir: str, visualization_config):
    files = [file for file in os.listdir(dir) if file.endswith(".json")]

    colors = [t[0] for t in visualization_config.values()]
    labels = [t[1] for t in visualization_config.values()]

    maes = [[] for _ in visualization_config]
    for file_path in files:
        with open(os.path.join(dir, file_path), "r") as file:
            res_dict = json.load(file)
            mean_absolute_errors = res_dict["mean_absolute_errors"]

            for i in range(len(maes)):
                maes[i].append(mean_absolute_errors[i])

    tikz = make_tikz_boxplot(
        ylabel=r"$\mathrm{MAE}_i$",
        plots=[BoxPlot(labels[i], colors[i], maes[i]) for i in range(len(maes))],
    )

    with open(os.path.join(dir, "mae-boxplot.tikz"), "w") as file:
        file.write(tikz)
    print("Successfully saved a MAE boxplot in the output directory.")


def eval_jsons_to_avg_slowdowns(dir: str, visualization_config):
    files = [file for file in os.listdir(dir) if file.endswith(".json")]

    colors = [t[0] for t in visualization_config.values()]
    labels = [t[1] for t in visualization_config.values()]

    slowdowns_by_predictor = [[] for _ in visualization_config]
    means = [0.0 for _ in visualization_config]
    for file_path in files:
        with open(os.path.join(dir, file_path), "r") as file:
            res_dict = json.load(file)
            travel_times = res_dict["avg_travel_times"]
            if any(
                travel_times[j] != travel_times[0] for j in range(len(travel_times) - 1)
            ):
                for i in range(len(slowdowns_by_predictor)):
                    slowdowns_by_predictor[i].append(
                        travel_times[i] / travel_times[-1] - 1
                    )
    avg_slowdowns_by_predictor = [
        np.average(slowdowns) for slowdowns in slowdowns_by_predictor
    ]
    with open(os.path.join(dir, "../average_slowdowns.json"), "w") as file:
        JSONEncoder.dump(avg_slowdowns_by_predictor, file)


if __name__ == "__main__":

    def main():
        # network_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/tokyo_tiny/default_demands.pickle"
        network_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/random-demands.pickle"
        Network.from_file(network_path).print_info()
        # eval_network(network_path, "../../out/sioux-falls-3", check_for_optimizations=False)
        # network_results_from_file_to_tikz("../../out/sioux-falls-3")

    main()
