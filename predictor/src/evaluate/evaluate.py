import datetime

import json

import os
from matplotlib import pyplot as plt

from core.constant_predictor import ConstantPredictor
from core.linear_predictor import LinearPredictor
from core.linear_regression_predictor import LinearRegressionPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network, Commodity
from core.reg_linear_predictor import RegularizedLinearPredictor
from core.uniform_distributor import UniformDistributor
from core.zero_predictor import ZeroPredictor
from importer.csv_importer import network_from_csv, add_demands_to_network
from utilities.build_with_times import build_with_times


def evaluate_single_run(network: Network, original_commodity: int, split_commodity: int, horizon: float,
                        reroute_interval: float,
                        suppress_log: bool = False, save: bool = True):
    prediction_horizon = 10.
    predictors = [
        ConstantPredictor(network),
        ZeroPredictor(network),
        LinearRegressionPredictor(network),
        LinearPredictor(network, prediction_horizon),
        RegularizedLinearPredictor(network, prediction_horizon, delta=5.),
    ]

    commodity = network.commodities[split_commodity]
    # network.commodities.remove(commodity)
    new_commodities = range(len(network.commodities), len(network.commodities) + len(predictors))
    demand_per_comm = 0.125
    for i in range(len(predictors)):
        network.commodities.append(Commodity(commodity.source, commodity.sink, demand_per_comm, i))

    distributor = UniformDistributor(network)
    flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)

    flow = build_with_times(flow_builder, original_commodity, reroute_interval, horizon, new_commodities, suppress_log)
    travel_times = [flow.avg_travel_time(i, horizon) for i in new_commodities]
    save_dict = {
        "prediction_horizon": prediction_horizon,
        "horizon": horizon,
        "original_commodity": original_commodity,
        "avg_travel_times": travel_times
    }

    print(f"The following average travel times were computed for flow#{original_commodity}:")
    print(travel_times)

    if save:
        now = datetime.datetime.now()
        os.makedirs("../../out/tiny-new-scenario", exist_ok=True)
        with open(f"../../out/tiny-new-scenario/{original_commodity}.{str(now)}.json", "w") as file:
            json.dump(save_dict, file)
    return travel_times


def eval_tokyo():
    plot = False
    y = [[], [], [], [], []]
    #original_commodity = 3
    while True:
        network_path = '/home/michael/Nextcloud2/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_tiny.arcs'
        network = network_from_csv(network_path)
        demands_path = '/home/michael/Nextcloud2/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_tiny.demands'
        add_demands_to_network(network, demands_path, use_default_demands=True, suppress_log=False, upscale=True)
        network.remove_unnecessary_nodes()
        with open("./next_commodity.txt", "r") as file:
            original_commodity = int(file.read())
        with open("./next_commodity.txt", "w") as file:
            file.write(str(original_commodity + 1))
        if original_commodity >= len(network.commodities):
            break
        #network.commodities = [network.commodities[original_commodity]]
        #selected_commodity = original_commodity
        #network.commodities[0].demand *= 5
        selected_commodity = network.remove_unnecessary_commodities(original_commodity)
        times = evaluate_single_run(network, original_commodity, selected_commodity, 100., 2.5)
        for i, value in enumerate(times):
            y[i].append(value)

        if plot:
            for i in range(len(y)):
                plt.plot(range(len(y[0])), y[i], label=[
                    "Constant Predictor",
                    "Zero Predictor",
                    "Linear Regression Predictor",
                    "Linear Predictor",
                    "Regularized Linear Predictor"
                ][i])
            plt.title("Avg travel time when splitting commodity x uniformly")
            plt.legend()
            plt.grid(which='both')
            plt.show()

        #original_commodity += 1


def tokyo_from_file_to_tikz():
    directory = "../../out/tiny-new-scenario/"
    files = os.listdir(directory)
    times = [[], [], [], []]  # Zero, LinearRegression, Linear, RegularizedLinear
    means = [0, 0, 0, 0, 0]
    num = 0
    for file_path in files:
        with open(os.path.join(directory, file_path), "r") as file:
            res_dict = json.load(file)
            travel_times = res_dict['avg_travel_times']
            if any(travel_times[j] != travel_times[0] for j in range(len(travel_times))):
                for i in range(len(times)):
                        times[i].append(travel_times[i + 1] / travel_times[0])
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
    tokyo_from_file_to_tikz()
