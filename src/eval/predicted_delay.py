from dataclasses import dataclass
from math import floor
from typing import Callable, Dict, Iterable, List, Literal, Optional, Set

from core.dijkstra import dynamic_dijkstra, get_active_edges_from_dijkstra
from core.dynamic_flow import DynamicFlow
from core.graph import Node
from core.machine_precision import eps
from core.network import Network
from core.predictor import Predictor
from core.predictors.predictor_type import PredictorType
from utilities.arrays import elem_lrank, elem_rank
from src.cython_test.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant
from utilities.status_logger import TimedStatusLogger


@dataclass
class DelayWitness:
    interval_start: float
    interval_end: float
    occured_at: Literal["start", "end"]
    edge_idx: int


@dataclass
class Delay:
    eps: float
    witness: Optional[DelayWitness]
    predictor_type: PredictorType
    is_eval_commodity: bool


DelayByCommodity = Dict[int, Delay]


def approximate_max_predicted_delay(
    flow: DynamicFlow,
    network: Network,
    predictors: Dict[PredictorType, Predictor],
    new_commodities_indices: Iterable[int],
    future_timesteps: int,
    reroute_interval: float,
    prediction_interval: float,
    horizon: float,
) -> DelayByCommodity:
    with TimedStatusLogger("Evaluating prediction accuracy eps...") as status:
        eps_by_comm: DelayByCommodity = {}

        for comm_idx in range(len(network.commodities)):
            max_eps = 0.0
            witness = None
            commodity = network.commodities[comm_idx]
            predictor = predictors[commodity.predictor_type]

            com_nodes = set(
                network.graph.get_nodes_reaching(commodity.sink).intersection(
                    set(
                        node
                        for source in commodity.sources
                        for node in network.graph.get_reachable_nodes(source)
                    )
                )
            )

            eval_horizon = horizon - (future_timesteps + 1) * prediction_interval

            measurement_interval = reroute_interval / 2
            pred_times = [
                i * measurement_interval
                for i in range(0, floor(eval_horizon / measurement_interval) + 1)
            ]

            predictor_predictions = predictor.batch_predict(pred_times, flow)

            interval_start = pred_times[0]
            costs_at_start = costs_from_preds(
                network, predictor_predictions[0]
            )
            delays_at_start: Dict[int, float] = {}
            for k in range(len(pred_times) - 1):
                interval_start = pred_times[k]
                interval_end = pred_times[k + 1]

                costs_at_end = costs_from_preds(
                    network, predictor_predictions[k + 1]
                )
                delays_at_end = {}

                for edge_idx, edge_inflow in enumerate(flow.inflow):
                    if comm_idx in edge_inflow._functions_dict:
                        inflow = edge_inflow._functions_dict[comm_idx]
                        if is_positive_during(inflow, interval_start, interval_end):
                            if edge_idx not in delays_at_start:
                                delays_at_start[edge_idx] = get_pred_edge_delay(
                                    interval_start,
                                    edge_idx,
                                    network,
                                    commodity.sink,
                                    com_nodes,
                                    costs_at_start,
                                )
                            if delays_at_start[edge_idx] > max_eps:
                                max_eps = delays_at_start[edge_idx]
                                witness = DelayWitness(
                                    interval_start,
                                    interval_end,
                                    "start",
                                    edge_idx,
                                )
                            delays_at_end[edge_idx] = get_pred_edge_delay(
                                interval_end,
                                edge_idx,
                                network,
                                commodity.sink,
                                com_nodes,
                                costs_at_end,
                            )
                            if delays_at_end[edge_idx] > max_eps:
                                max_eps = delays_at_end[edge_idx]
                                witness = DelayWitness(
                                    interval_start,
                                    interval_end,
                                    "end",
                                    edge_idx,
                                )

                costs_at_start = costs_at_end
                delays_at_start = delays_at_end

            eps_by_comm[comm_idx] = Delay(
                max_eps,
                witness,
                commodity.predictor_type,
                comm_idx in new_commodities_indices,
            )

        status.finish_msg = f"Max eps: {eps_by_comm}"
        return eps_by_comm


def costs_from_preds(
    network: Network, predictions: List[PiecewiseLinear]
) -> List[PiecewiseLinear]:
    travel_time = network.travel_time
    capacity = network.capacity

    return lambda e_id, t: travel_time[e_id] + predictions[e_id](t) / capacity[e_id]


def get_pred_edge_delay(
    at: float,
    edge_idx: int,
    network: Network,
    sink: Node,
    com_nodes: Set[Node],
    costs: Callable[[int, float], float],
):
    edge = network.graph.edges[edge_idx]
    result_from_sink = dynamic_dijkstra(at, edge.node_from, sink, com_nodes, costs)
    active_edges = get_active_edges_from_dijkstra(
        result_from_sink, edge.node_from, sink
    )
    if edge in active_edges:
        return 0.0
    else:
        arrival_times_using_e, _ = dynamic_dijkstra(
            at + costs(edge_idx, at), edge.node_to, sink, com_nodes, costs
        )
        return arrival_times_using_e[sink] - result_from_sink.arrival_times[sink]


def is_positive_during(f: RightConstant, start: float, end: float):
    """
    Returns true, if f is postive at any point during the interval [start, end).
    """
    start_rank = elem_lrank(f.times, start)
    end_rank = elem_rank(f.times, end)
    return any(f.values[max(0, rank)] > eps for rank in range(start_rank, end_rank + 1))
