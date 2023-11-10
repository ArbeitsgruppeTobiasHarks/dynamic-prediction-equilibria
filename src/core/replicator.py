from copy import deepcopy
from typing import Callable, Dict, List, Set, Tuple

import numpy as np

from core.active_paths import Path
from core.bellman_ford import bellman_ford
from core.dijkstra import reverse_dijkstra
from core.dynamic_flow import DynamicFlow
from core.graph import Edge
from core.machine_precision import eps
from core.network import Commodity, Network
from core.path_flow_builder import PathFlowBuilder
from core.predictors.predictor_type import PredictorType
from utilities.piecewise_linear import PiecewiseLinear, zero
from utilities.right_constant import Indicator, RightConstant
from visualization.to_json import merge_commodities


class DynamicReplicator:
    """ """

    network: Network
    reroute_interval: float
    horizon: float
    inflow: RightConstant
    replication_window: float
    _last_replication: float
    _flow: DynamicFlow
    _inflow_distribution: List[RightConstant]
    _comm_to_path: Dict[int, Path]
    _path_to_comm: Dict[Path, int]

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        initial_distribution: List[Tuple[List[int], float]],
        replication_window: float,
    ):
        self.network = network
        self.reroute_interval = reroute_interval
        self.horizon = horizon
        self._last_replication = 0.0

        t = network.commodities[0].sink
        s, self.inflow = next(iter(network.commodities[0].sources.items()))
        network.commodities = []
        self._comm_to_path = dict()
        self._path_to_comm = dict()
        self._inflow_distribution = []
        for com_id, (e_ids, h) in enumerate(initial_distribution):
            network.add_commodity({s.id: h * self.inflow}, t.id, PredictorType.CONSTANT)
            path = Path([self.network.graph.edges[e_id] for e_id in e_ids])
            self._comm_to_path[com_id] = path
            self._path_to_comm[path] = com_id
            self._inflow_distribution.append(
                RightConstant([0.0], [h], (0, float("inf")))
            )

    def compute_avg_travel_times(self):
        avg_travel_times = np.zeros(len(self.network.commodities))
        for com_id, com in enumerate(self.network.commodities):
            last_edge = self._comm_to_path[com_id].edges[-1]
            accum_net_outflow = (
                self._flow.outflow[last_edge.id]._functions_dict[com_id].integral()
                if com_id in self._flow.outflow[last_edge.id]._functions_dict
                else zero
            )
            accum_net_inflow = next(iter(com.sources.values())).integral()
            avg_travel_times[com_id] = (
                accum_net_inflow.integrate(0, self._flow.phi)
                - accum_net_outflow.integrate(0, self._flow.phi)
            ) / (accum_net_inflow(self._flow.phi))
        return avg_travel_times

    def replicate(self):
        phi = self._flow.phi
        avg_travel_times = self.compute_avg_travel_times()
        h = np.array([d(phi) for d in self._inflow_distribution])
        h *= np.exp(
            (np.sum(avg_travel_times * h) - avg_travel_times)
            * (phi - self._last_replication)
        )
        h /= h.sum()

        for com_id in range(len(self.network.commodities)):
            s = next(iter(self.network.commodities[com_id].sources.keys()))
            self._inflow_distribution[com_id].extend(phi, h[com_id])
            self.network.commodities[com_id].sources[s] = (
                self.inflow * self._inflow_distribution[com_id]
            )
        self._last_replication = phi

    def run(self):
        flow_builder = PathFlowBuilder(
            self.network, self._comm_to_path, self.reroute_interval
        )
        generator = flow_builder.build_flow()
        self._flow = next(generator)
        while self._flow.phi < self.horizon:
            self._flow = next(generator)
            self.replicate()

        return self._flow, self._inflow_distribution

    # def _merge_commodities(self):
    #     merged_flow = self._flow
    #     merged_network = deepcopy(self.network)
    #     merged_network.commodities = []
    #
    #     merged_flow = merge_commodities(merged_flow, self.network, self.network.commodities)
    #
    #     merged_network.add_commodity(
    #             {s.id: self.inflow, t.id, PredictorType.CONSTANT
    #         )
    #
    #     return merged_flow, merged_network
