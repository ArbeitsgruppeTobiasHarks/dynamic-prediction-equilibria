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


def evaluate_single_run(network: Network, split_commodity: int, horizon: float, reroute_interval: float,
                        suppress_log: bool = False):
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
    start_date_time = (
            datetime.datetime(1970, 1, 1) +
            datetime.timedelta(seconds=round(start_time))
    ).replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).time()
    if not suppress_log:
        print(f"Flow built until phi={flow.phi}; Started At={start_date_time}")
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
            if not suppress_log:
                print(f"Flow built until phi={flow.phi:.1f}; " +
                      f"Time Elapsed={datetime.timedelta(seconds=round(elapsed))}; " +
                      f"Estimated Remaining Time={datetime.timedelta(seconds=round(remaining_time))}; " +
                      f"Finished at {finish_time}")
            milestone += reroute_interval
            last_milestone_time = new_milestone_time
    print()
    travel_times = [flow.avg_travel_time(i, horizon) for i in new_commodities]
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


def eval_tokyo():
    plot = False
    y = [[], [], [], [], []]
    # selected_commodity = -1
    while True:
        # network = build_sample_network()
        # network.add_commodity(0, 2, 15, 0)
        # network.add_commodity(3, 2, 1, 0)
        # selected_commodity += 1
        network_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo_small.arcs'
        network = network_from_csv(network_path)
        demands_path = '/home/michael/Nextcloud/Universität/2021-SS/softwareproject/data/from-kostas/tokyo.demands'
        add_demands_to_network(network, demands_path, True, suppress_ignored=True)
        network.remove_unnecessary_nodes()
        with open("./next_commodity.txt", "r") as file:
            selected_commodity = int(file.read())
        with open("./next_commodity.txt", "w") as file:
            file.write(str(selected_commodity + 1))
        if selected_commodity >= len(network.commodities):
            break
        times = evaluate_single_run(network, selected_commodity, 400, 5)
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
        else:
            print("The following average travel times were computed:")
            print(y)


def eval_sample():
    max_demand = 30.
    demand = 0.
    step_size = 0.25
    while demand < max_demand:
        network = build_sample_network()
        network.add_commodity(0, 2, demand, 0)
        times = evaluate_single_run(network, 0, 100, 0.25, suppress_log=True)
        print(f"Calculated for demand={demand}. times={times}")
        demand += step_size


if __name__ == '__main__':
    network = build_sample_network()
    network.add_commodity(0, 2, 1.75, 0)
    times = evaluate_single_run(network, 0, 100, 0.25, suppress_log=True)
    eval_sample()
