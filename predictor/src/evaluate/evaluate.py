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
from test.sample_network import build_sample_network
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
    demand_per_comm = commodity.demand / len(predictors)
    for i in range(len(predictors)):
        network.commodities.append(Commodity(commodity.source, commodity.sink, demand_per_comm, i))

    distributor = UniformDistributor(network)
    flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)

    generator = flow_builder.build_flow()
    start_time = last_milestone_time = time.time()
    flow = next(generator)
    print(f"\rFlow built until phi={flow.phi}.", end="\r")
    milestone = reroute_interval
    while flow.phi < horizon:
        flow = next(generator)
        if flow.phi >= milestone:
            new_milestone_time = time.time()
            elapsed = new_milestone_time - start_time
            remaining_time = (horizon - flow.phi) * (new_milestone_time - last_milestone_time) / reroute_interval
            finish_time = (
                    datetime.datetime(1970, 1, 1) +
                    datetime.timedelta(seconds=round(new_milestone_time + remaining_time))
            ).replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).time()
            print(f"\rFlow built until phi={flow.phi:.1f}. Time Elapsed={elapsed:.1f}s; " +
                  f"Estimated Remaining Time={datetime.timedelta(seconds=round(remaining_time))}; " +
                  f"Finished at {finish_time}", end="\r")
            milestone += reroute_interval
            last_milestone_time = new_milestone_time
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
        "prediction_horizon": prediction_horizon,
        "horizon": horizon,
        "selected_commodity": split_commodity,
        "avg_travel_times": travel_times
    }

    now = datetime.datetime.now()
    os.makedirs("../../out/evaluation", exist_ok=True)
    with open(f"../../out/evaluation/{split_commodity}.{str(now)}.pickle", "wb") as file:
        pickle.dump(save_dict, file)
    return travel_times


def main():
    with open("../../out/evaluation/0.2021-05-27 00:51:09.797508.pickle", "rb") as file:
        result_dict = pickle.load(file)

    plt.ion()
    y = [[], [], [], [], []]
    selected_commodity = 0
    while True:
        # network = build_sample_network()
        # network.add_commodity(0, 2, 3, 0)
        # network.add_commodity(3, 2, 1, 0)
        network_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_small.arcs'
        network = network_from_csv(network_path)
        demands_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo.demands'
        add_demands_to_network(network, demands_path, True, suppress_ignored=True)
        network.remove_unnecessary_nodes()
        if selected_commodity >= len(network.commodities):
            break
        times = evaluate_single_run(network, selected_commodity, 100, 5)
        for i, value in enumerate(times):
            y[i].append(value)
        selected_commodity += 1

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


if __name__ == '__main__':
    main()
