import datetime
import json
import os
from typing import Optional

from core.constant_predictor import ConstantPredictor
from core.linear_predictor import LinearPredictor
from core.linear_regression_predictor import LinearRegressionPredictor
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network, Commodity
from core.reg_linear_predictor import RegularizedLinearPredictor
from core.uniform_distributor import UniformDistributor
from core.zero_predictor import ZeroPredictor
from utilities.build_with_times import build_with_times


def evaluate_single_run(network: Network, focused_commodity: int, split: bool, horizon: float,
                        reroute_interval: float, flow_id: Optional[int] = None, output_folder: Optional[str] = None,
                        suppress_log: bool = False):
    if output_folder is not None and flow_id is None:
        raise ValueError("You specified an output folder, but no flow_id. Specify flow_id to save the flow.")
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    prediction_horizon = 10.
    predictors = [
        ConstantPredictor(network),
        ZeroPredictor(network),
        LinearRegressionPredictor(network),
        LinearPredictor(network, prediction_horizon),
        RegularizedLinearPredictor(network, prediction_horizon, delta=5.),
    ]

    commodity = network.commodities[focused_commodity]
    if split:
        network.commodities.remove(commodity)
        demand_per_comm = commodity.demand / len(predictors)
    else:
        demand_per_comm = 0.125

    new_commodities = range(len(network.commodities), len(network.commodities) + len(predictors))
    for i in range(len(predictors)):
        network.commodities.append(Commodity(commodity.source, commodity.sink, demand_per_comm, i))

    distributor = UniformDistributor(network)
    flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)

    flow = build_with_times(flow_builder, flow_id, reroute_interval, horizon, new_commodities, suppress_log)
    travel_times = [flow.avg_travel_time(i, horizon) for i in new_commodities]
    save_dict = {
        "prediction_horizon": prediction_horizon,
        "horizon": horizon,
        "original_commodity": flow_id,
        "avg_travel_times": travel_times
    }

    if not suppress_log:
        print(f"The following average travel times were computed for flow#{flow_id}:")
        print(travel_times)

    if output_folder is not None:
        now = datetime.datetime.now()
        with open(os.path.join(output_folder, f"{flow_id}.{str(now)}.json"), "w") as file:
            json.dump(save_dict, file)
    return travel_times
