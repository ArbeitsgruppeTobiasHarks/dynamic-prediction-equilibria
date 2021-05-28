import json

import datetime
import os

import time

from core.constant_predictor import ConstantPredictor
from core.linear_predictor import LinearPredictor
from core.linear_regression_predictor import LinearRegressionPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network, Commodity
from core.reg_linear_predictor import RegularizedLinearPredictor
from core.uniform_distributor import UniformDistributor
from core.zero_predictor import ZeroPredictor
from test.sample_network import build_sample_network

def evaluate_single_run(network: Network, original_commodity: int, split_commodity: int, horizon: float,
                        reroute_interval: float,
                        suppress_log: bool = False, save: bool = True):
    # Use a better prediction_horizon next time, i.e. 10
    prediction_horizon = 10.

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
        print(f"Flow#{original_commodity} built until phi={flow.phi}; Started At={start_date_time}")
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
                print(f"Flow#{original_commodity} built until phi={flow.phi:.1f}; " +
                      f"Time Elapsed={datetime.timedelta(seconds=round(elapsed))}; " +
                      f"Estimated Remaining Time={datetime.timedelta(seconds=round(remaining_time))}; " +
                      f"Finished at {finish_time}; " +
                      f"TravelTimes={[flow.avg_travel_time(i, flow.phi) for i in new_commodities]}")
            milestone += reroute_interval
            last_milestone_time = new_milestone_time
    print()
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
        os.makedirs("../../out/evaluation-without-bg", exist_ok=True)
        with open(f"../../out/evaluation-without-bg/{original_commodity}.{str(now)}.json", "w") as file:
            json.dump(save_dict, file)
    return travel_times



def eval_sample():
    max_demand = 30.
    demand = 0.
    step_size = 0.25
    avg_times = [[], [], [], [], []]
    while demand < max_demand:
        network = build_sample_network()
        network.add_commodity(0, 2, demand, 0)
        times = evaluate_single_run(network, 0, 0, 100, 0.25, suppress_log=True, save=False)
        for i, val in enumerate(times):
            avg_times[i].append(val)
        print(f"Calculated for demand={demand}. times={times}")
        demand += step_size
    print(avg_times)
    with open("./avg_times_sample.json", "w") as file:
        json.dump(avg_times, file)


def sample_from_file_to_tikz():
    with open("./avg_times_sample.json", "r") as file:
        avg_times = json.load(file)
    for values in avg_times:
        tikz = ""
        for i, y in enumerate(values):
            x = i * 0.25
            tikz += f"({x}, {y})"

        print(tikz)


if __name__ == '__main__':
    sample_from_file_to_tikz()
