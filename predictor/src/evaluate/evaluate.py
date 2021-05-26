import datetime

import pickle

import os
from matplotlib import pyplot as plt
import time

from core.constant_predictor import ConstantPredictor
from core.linear_predictor import LinearPredictor
from core.linear_regression_predictor import LinearRegressionPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network, Commodity
from core.reg_linear_predictor import RegularizedLinearPredictor
from core.uniform_distributor import UniformDistributor
from core.zero_predictor import ZeroPredictor
from importer.csv_importer import network_from_csv, add_demands_to_network
from utilities.right_constant import RightConstantFunction


def evaluate_single_run(network: Network, split_commodity: int, horizon: float, reroute_interval: float):
    prediction_horizon = 0.05 * horizon

    predictors = [
        ConstantPredictor(network),
        ZeroPredictor(network),
        LinearRegressionPredictor(network),
        LinearPredictor(network, prediction_horizon),
        RegularizedLinearPredictor(network, prediction_horizon, delta=5.),
    ]

    commodity = network.commodities[split_commodity]
    network.commodities.remove(commodity)
    new_commodities = range(len(network.commodities), len(network.commodities) + len(predictors))
    for i in range(len(predictors)):
        network.commodities.append(Commodity(commodity.source, commodity.sink, commodity.demand / len(predictors), i))

    demand_per_comm = commodity.demand / len(predictors)
    distributor = UniformDistributor(network)
    flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)

    generator = flow_builder.build_flow()
    start_time = time.time()
    print("\r Flow built until phi=0.", end="\r")
    flow = next(generator)
    milestone = reroute_interval
    while flow.phi < horizon:
        flow = next(generator)
        if flow.phi >= milestone:
            elapsed = time.time() - start_time
            remaining_time = (horizon - flow.phi) * elapsed / flow.phi
            print(f"\r Flow built until phi={flow.phi}. Est. remaining time={remaining_time}", end="\r")
            milestone += reroute_interval
    print()
    travel_times = []

    for i in new_commodities:
        net_outflow: RightConstantFunction = sum(flow.outflow[e.id][i] for e in commodity.sink.incoming_edges)
        accum_net_outflow = net_outflow.integral()
        avg_travel_time = horizon / 2 - \
                          accum_net_outflow.integrate(0., horizon) / (horizon * demand_per_comm)
        travel_times.append(avg_travel_time)

    save_dict = {
        "flow": flow,
        "commodities": network.commodities,
        "prediction_horizon": prediction_horizon,
        "horizon": horizon,
        "selected_commodity": split_commodity,
        "avg_travel_times": travel_times
    }

    now = datetime.datetime.now()
    os.makedirs("../../out/evaluation", exist_ok=True)
    with open(f"../../out/evaluation/{str(now)}.pickle", "wb") as file:
        pickle.dump(save_dict, file)
    return travel_times


if __name__ == '__main__':

    y = [[], [], [], [], []]
    selected_commodity = 0
    while True:
        network_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_small.arcs'
        network = network_from_csv(network_path)
        demands_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo.demands'
        add_demands_to_network(network, demands_path, True, suppress_ignored=True)
        network.remove_unnecessary_nodes()
        if selected_commodity >= len(network.commodities):
            break
        times = evaluate_single_run(network, selected_commodity, 300, 5)
        for i, time in enumerate(times):
            y[i].append(time)
        selected_commodity += 1

        for i in range(len(y)):
            plt.plot(range(len(y)), y[i], label=[
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
