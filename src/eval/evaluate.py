import os
import pickle
from math import floor
from typing import Callable, Dict, Optional

import numpy as np

from core.bellman_ford import bellman_ford
from core.dpe_flow_builder import FlowBuilder
from core.dynamic_flow import DynamicFlow
from core.network import Commodity, Network
from core.predictor import Predictor
from core.predictors.predictor_type import PredictorType
from eval.predicted_delay import approximate_max_predicted_delay
from utilities.arrays import elem_lrank, elem_rank
from utilities.build_with_times import build_with_times
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.json_encoder import JSONEncoder
from utilities.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant
from utilities.status_logger import StatusLogger

COLORS = {
    PredictorType.ZERO: "blue",
    PredictorType.CONSTANT: "red",
    PredictorType.LINEAR: "green",
    PredictorType.REGULARIZED_LINEAR: "orange",
    PredictorType.MACHINE_LEARNING_TF_FULL_NET: "black",
    PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: "black",
    PredictorType.MACHINE_LEARNING_SK_FULL_NET: "black",
    PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: "black",
}


PredictorBuilder = Callable[[Network], Dict[PredictorType, Predictor]]


def calculate_optimal_average_travel_time(
    flow: DynamicFlow,
    network: Network,
    inflow_horizon: float,
    horizon: float,
    commodity: Commodity,
):
    with StatusLogger("Computing optimal average travel time..."):
        if len(commodity.sources) != 1:
            raise ValueError("Expected a single source!")
        source = next(iter(commodity.sources))
        costs = [
            PiecewiseLinear(
                flow.queues[e].times,
                [
                    network.travel_time[e] + v / network.capacity[e]
                    for v in flow.queues[e].values
                ],
                flow.queues[e].first_slope / network.capacity[e],
                flow.queues[e].last_slope / network.capacity[e],
                domain=(0.0, horizon),
            ).simplify()
            for e in range(len(flow.queues))
        ]
        labels = bellman_ford(
            commodity.sink,
            costs,
            network.graph.get_nodes_reaching(commodity.sink).intersection(
                network.graph.get_reachable_nodes(source)
            ),
            0.0,
            horizon,
        )

        def integrate_opt(label: PiecewiseLinear) -> float:
            assert label.is_monotone()
            travel_time = label.plus(PiecewiseLinear([0], [0.0], -1, -1))
            # Last time h for starting at source to arrive at sink before horizon.
            if inflow_horizon < float("inf"):
                h = min(inflow_horizon, label.max_t_below(horizon))
                inflow_until = min(horizon, inflow_horizon)
            else:
                h = label.max_t_below(horizon)
                inflow_until = horizon
            integral_travel_time = travel_time.integrate(0, h)
            if h < inflow_until:
                integral_travel_time += (
                    horizon * (inflow_until - h) - (inflow_until**2 - h**2) / 2
                )
            avg_travel_time = integral_travel_time / inflow_until
            return avg_travel_time

        if source in labels:
            return integrate_opt(labels[source])
        else:
            return inflow_horizon * horizon - inflow_horizon**2 / 2


def evaluate_single_run(
    network: Network,
    focused_commodity_index: int,
    split: bool,
    horizon: float,
    reroute_interval: float,
    inflow_horizon: float,
    future_timesteps: int,
    prediction_interval: float,
    build_predictors: PredictorBuilder,
    flow_path: Optional[str] = None,
    json_eval_path: Optional[str] = None,
    flow_id: Optional[int] = None,
    suppress_log: bool = False,
):
    if flow_path is not None:
        os.makedirs(os.path.dirname(flow_path), exist_ok=True)
    if json_eval_path is not None:
        os.makedirs(os.path.dirname(json_eval_path), exist_ok=True)

    with StatusLogger("Building predictors..."):
        predictors = build_predictors(network)

    commodity = network.commodities[focused_commodity_index]
    if len(commodity.sources) != 1:
        raise ValueError("Expected a single source.")
    focused_source, focused_inflow = next(iter(commodity.sources.items()))

    if split:
        network.commodities.remove(commodity)
        demand_per_comm = RightConstant(
            focused_inflow.times,
            [v / len(predictors) for v in focused_inflow.values],
            focused_inflow.domain,
        )
    else:
        test_demand = max(
            min(network.capacity) / 256, focused_inflow.values[0] / 16 / len(predictors)
        )
        demand_per_comm = RightConstant(
            [0.0, inflow_horizon], [test_demand, 0.0], (0.0, float("inf"))
        )

    combine_commodities_with_same_sink(network)

    new_commodities_indices = range(
        len(network.commodities), len(network.commodities) + len(predictors)
    )
    for i in predictors:
        network.commodities.append(
            Commodity({focused_source: demand_per_comm}, commodity.sink, i)
        )

    if flow_path is None or not os.path.exists(flow_path):
        flow_builder = FlowBuilder(network, predictors, reroute_interval)
        flow, computation_time = build_with_times(
            flow_builder,
            flow_id,
            reroute_interval,
            horizon,
            new_commodities_indices,
            suppress_log,
        )

        if flow_path is not None:
            with StatusLogger(
                "Writing flow to disk...", "Succesfully written flow to disk."
            ):
                with open(flow_path, "wb") as file:
                    pickle.dump(
                        {"flow": flow, "computation_time": computation_time}, file
                    )
    else:
        with StatusLogger(
            "Flow already exists. Loading flow from disk...",
            "Succesfully loaded flow from disk.",
        ):
            with open(flow_path, "rb") as file:
                box = pickle.load(file)
            computation_time: float = box["computation_time"]
            flow: DynamicFlow = box["flow"]
            flow._network = network

    travel_times = [
        flow.avg_travel_time(i, horizon) for i in new_commodities_indices
    ] + [
        calculate_optimal_average_travel_time(
            flow, network, inflow_horizon, horizon, commodity
        )
    ]

    if not suppress_log:
        print(f"The following average travel times were computed for flow#{flow_id}:")
        print(travel_times)

    max_pred_delays = approximate_max_predicted_delay(
        flow,
        network,
        predictors,
        new_commodities_indices,
        future_timesteps,
        reroute_interval,
        prediction_interval,
        horizon,
    )
    mean_absolute_errors = evaluate_mean_absolute_error(
        flow,
        predictors,
        future_timesteps,
        reroute_interval,
        prediction_interval,
        horizon,
    )

    save_dict = {
        "horizon": horizon,
        "original_commodity": flow_id,
        "avg_travel_times": travel_times,
        "computation_time": computation_time,
        "mean_absolute_errors": [mae for mae in mean_absolute_errors.values()],
        "max_predicted_delays": max_pred_delays,
    }

    if json_eval_path is not None:
        with open(json_eval_path, "w") as file:
            JSONEncoder().dump(save_dict, file)

    return travel_times, computation_time, flow


def evaluate_mean_absolute_error(
    flow: DynamicFlow,
    predictors: Dict[PredictorType, Predictor],
    future_timesteps: int,
    reroute_interval: float,
    prediction_interval: float,
    horizon: float,
):
    with StatusLogger("Evaluating prediction accuracy MAE...") as status:
        eval_horizon = horizon - (future_timesteps + 1) * prediction_interval
        predictions = {}
        diffs = {}
        pred_times = [
            i * reroute_interval
            for i in range(0, floor(eval_horizon / reroute_interval) + 1)
        ]

        stride = round(prediction_interval / reroute_interval)
        queue_values = np.array(
            [
                [queue(i * reroute_interval) for queue in flow.queues]
                for i in range(0, floor(horizon / reroute_interval) + 1)
            ]
        )

        for predictor_type, predictor in predictors.items():
            predictor_predictions = predictor.batch_predict(pred_times, flow)

            def to_samples(i, pred_time):
                pred_queues = predictor_predictions[i]
                return [
                    [
                        pred_queue(pred_time + (k + 1) * prediction_interval)
                        for pred_queue in pred_queues
                    ]
                    for k in range(future_timesteps)
                ]

            predictions[predictor_type] = np.array(
                [to_samples(i, pred_time) for i, pred_time in enumerate(pred_times)]
            )
            diffs[predictor_type] = np.array(
                [
                    [
                        predictions[predictor_type][pred_ind, k, :]
                        - queue_values[pred_ind + (k + 1) * stride, :]
                        for k in range(future_timesteps)
                    ]
                    for pred_ind, _ in enumerate(pred_times)
                ]
            )

        mean_absolute_errors = {
            predictor_type: np.average(np.abs(diff))
            for predictor_type, diff in diffs.items()
        }
        status.finish_msg = "MAE. " + "; ".join(
            [
                f"{predictor_type.name}: {round(mae, 4)}"
                for predictor_type, mae in mean_absolute_errors.items()
            ]
        )
        return mean_absolute_errors
