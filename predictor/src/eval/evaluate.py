import json
import os
import pickle
from pathlib import Path
from typing import Optional

from core.bellman_ford import bellman_ford
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network, Commodity
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


def evaluate_single_run(network: Network, focused_commodity: int, split: bool, horizon: float,
                        reroute_interval: float, opt_net_inflow: RightConstant, flow_id: Optional[int] = None,
                        output_folder: Optional[str] = None, suppress_log: bool = False):
    assert opt_net_inflow.domain == (0, float('inf'))
    assert len(opt_net_inflow.values) == 2
    assert opt_net_inflow.values[0] > 0 and opt_net_inflow.values[1] == 0.
    if output_folder is not None and flow_id is None:
        raise ValueError("You specified an output folder, but no flow_id. Specify flow_id to save the flow.")
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        pickle_path = os.path.join(output_folder, f"{flow_id}.pickle")
        json_path = os.path.join(output_folder, f"{flow_id}.json")
    else:
        pickle_path = None
        json_path = None

    prediction_horizon = 10.
    predictors = {
        PredictorType.ZERO: ZeroPredictor(network),
        PredictorType.CONSTANT: ConstantPredictor(network),
        PredictorType.LINEAR: LinearPredictor(network, prediction_horizon),
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, prediction_horizon, delta=5.),
        PredictorType.MACHINE_LEARNING: LinearRegressionPredictor(network),
    }

    commodity = network.commodities[focused_commodity]
    if split:
        network.commodities.remove(commodity)
        demand_per_comm = RightConstant(
            commodity.net_inflow.times,
            [v / len(predictors) for v in commodity.net_inflow.values],
            commodity.net_inflow.domain
        )
    else:
        demand_per_comm = RightConstant([0.], [0.125], (0., float('inf')))

    new_commodities = range(len(network.commodities), len(network.commodities) + len(predictors))
    for i in predictors:
        network.commodities.append(Commodity(commodity.source, commodity.sink, demand_per_comm, i))

    if output_folder is not None and Path(pickle_path).exists():
        with open(pickle_path, "rb") as file:
            flow = pickle.load(file)
        flow._network = network
    else:
        distributor = UniformDistributor(network)
        flow_builder = MultiComFlowBuilder(network, predictors, distributor, reroute_interval)
        flow = build_with_times(flow_builder, flow_id, reroute_interval, horizon, new_commodities, suppress_log)

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
        if len(opt_net_inflow.times) == 2:
            h = min(opt_net_inflow.times[1], label.max_t_below(horizon))
            inflow_until = min(horizon, opt_net_inflow.times[1])
        else:
            h = label.max_t_below(horizon)
            inflow_until = horizon
        integral_travel_time = travel_time.integrate(0, h)
        if h < inflow_until:
            integral_travel_time += horizon * (inflow_until - h) - (inflow_until ** 2 - h ** 2) / 2
        avg_travel_time = integral_travel_time / inflow_until
        return avg_travel_time

    travel_times = [flow.avg_travel_time(i, horizon) for i in new_commodities] + \
                   [integrate_opt(labels[commodity.source])]
    save_dict = {
        "prediction_horizon": prediction_horizon,
        "horizon": horizon,
        "original_commodity": flow_id,
        "avg_travel_times": travel_times
    }

    if not suppress_log:
        print(f"The following average travel times were computed for flow#{flow_id}:")
        print(travel_times)

    if json_path is not None:
        with open(json_path, "w") as file:
            json.dump(save_dict, file)
    return travel_times
