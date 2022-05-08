import json
import os
import pickle
from pathlib import Path
from typing import Optional, Dict, Callable

from core.bellman_ford import bellman_ford
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network, Commodity
from core.predictor import Predictor
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.linear_regression_predictor import LinearRegressionPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from core.uniform_distributor import UniformDistributor
from utilities.build_with_times import build_with_times
from utilities.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant

COLORS = {
    PredictorType.ZERO: "blue",
    PredictorType.CONSTANT: "red",
    PredictorType.LINEAR: "green",
    PredictorType.REGULARIZED_LINEAR: "orange",
    PredictorType.MACHINE_LEARNING: "black",
}


def _build_default_predictors(network: Network) -> Dict[PredictorType, Predictor]:
    prediction_horizon = 10.
    return {
        PredictorType.ZERO: ZeroPredictor(network),
        PredictorType.CONSTANT: ConstantPredictor(network),
        PredictorType.LINEAR: LinearPredictor(network, prediction_horizon),
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, prediction_horizon, delta=5.),
        PredictorType.MACHINE_LEARNING: LinearRegressionPredictor(network),
    }


PredictorBuilder = Callable[[Network], Dict[PredictorType, Predictor]]


def evaluate_single_run(network: Network, focused_commodity: int, split: bool, horizon: float,
                        reroute_interval: float, inflow_horizon: float, flow_id: Optional[int] = None,
                        output_folder: Optional[str] = None, suppress_log: bool = False,
                        build_predictors: PredictorBuilder = _build_default_predictors):
    if output_folder is not None and flow_id is None:
        raise ValueError("You specified an output folder, but no flow_id. Specify flow_id to save the flow.")
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        pickle_path = os.path.join(output_folder, f"{flow_id}.pickle")
        json_path = os.path.join(output_folder, f"{flow_id}.json")
    else:
        pickle_path = None
        json_path = None

    predictors = build_predictors(network)

    commodity = network.commodities[focused_commodity]
    if split:
        network.commodities.remove(commodity)
        demand_per_comm = RightConstant(
            commodity.net_inflow.times,
            [v / len(predictors) for v in commodity.net_inflow.values],
            commodity.net_inflow.domain
        )
    else:
        demand_per_comm = RightConstant(
            [0., inflow_horizon], [commodity.net_inflow.values[0] / 16 / len(predictors), 0.], (0., float('inf'))
        )

    new_commodities = range(len(network.commodities), len(network.commodities) + len(predictors))
    for i in predictors:
        network.commodities.append(Commodity(commodity.source, commodity.sink, demand_per_comm, i))

    if output_folder is not None and Path(pickle_path).exists():
        with open(pickle_path, "rb") as file:
            flow = pickle.load(file)
        flow._network = network
    else:
        flow_builder = MultiComFlowBuilder(network, predictors, reroute_interval)
        flow, elapsed = build_with_times(flow_builder, flow_id, reroute_interval, horizon, new_commodities, suppress_log)

        if pickle_path is not None:
            with open(pickle_path, "wb") as file:
                pickle.dump(flow, file)

    #  Calculating optimal predictor travel times
    costs = [
        PiecewiseLinear(
            flow.queues[e].times,
            [network.travel_time[e] + v / network.capacity[e] for v in flow.queues[e].values],
            flow.queues[e].first_slope / network.capacity[e],
            flow.queues[e].last_slope / network.capacity[e],
            domain=(0., horizon)
        ).simplify() for e in range(len(flow.queues))]
    labels = bellman_ford(
        commodity.sink,
        costs,
        network.graph.get_nodes_reaching(commodity.sink).intersection(
            network.graph.get_reachable_nodes(commodity.source)),
        0.,
        horizon
    )

    def integrate_opt(label: PiecewiseLinear):
        assert label.is_monotone()
        travel_time = label.plus(PiecewiseLinear([0], [0.], -1, -1))
        # Last time h for starting at source to arrive at sink before horizon.
        if inflow_horizon < float('inf'):
            h = min(inflow_horizon, label.max_t_below(horizon))
            inflow_until = min(horizon, inflow_horizon)
        else:
            h = label.max_t_below(horizon)
            inflow_until = horizon
        integral_travel_time = travel_time.integrate(0, h)
        if h < inflow_until:
            integral_travel_time += horizon * (inflow_until - h) - (inflow_until ** 2 - h ** 2) / 2
        avg_travel_time = integral_travel_time / inflow_until
        return avg_travel_time

    travel_times = [flow.avg_travel_time(i, horizon) for i in new_commodities] + \
                   [
                       integrate_opt(labels[commodity.source])
                       if commodity.source in labels
                       else inflow_horizon*horizon - inflow_horizon**2 / 2
                   ]
    save_dict = {
        "horizon": horizon,
        "original_commodity": flow_id,
        "avg_travel_times": travel_times,
        "comp_time": elapsed
    }

    if not suppress_log:
        print(f"The following average travel times were computed for flow#{flow_id}:")
        print(travel_times)

    if json_path is not None:
        with open(json_path, "w") as file:
            json.dump(save_dict, file)
    return travel_times, elapsed, flow
