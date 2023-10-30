from copy import deepcopy
from typing import Dict, List, Tuple, Generator

from core.active_paths import Path
from core.dynamic_flow import DynamicFlow
from core.graph import Edge, Node
from core.network import Network
from core.path_flow_builder import PathFlowBuilder
from core.predictors.predictor_type import PredictorType
from utilities.right_constant import RightConstant
import numpy as np


class ReplicatorFlowBuilder(PathFlowBuilder):
    inflow: RightConstant
    _source: Node
    _path_distribution: Dict[int, RightConstant]

    def __init__(self,
                 network: Network,
                 reroute_interval: float,
                 initial_distribution: List[Tuple[List[int], float]],
                 ):

        network_copy = deepcopy(network)
        self._source, self.inflow = next(iter(network.commodities[0].sources.items()))
        t = network.commodities[0].sink
        network_copy.commodities = []
        paths = dict()
        self.path_distribution = dict()

        for i, (e_ids, path_prob) in enumerate(initial_distribution):
            path = Path([network.graph.edges[e_id] for e_id in e_ids])
            self.path_distribution[i] = RightConstant([0.0], [path_prob], (0, float('inf')))
            network_copy.add_commodity({self._source.id: self.inflow * self.path_distribution[i]}, t.id, PredictorType.CONSTANT)
            paths[i] = path

        super().__init__(network_copy, paths, reroute_interval)

    def _compute_path_fitnesses(self) -> np.ndarray:  #insert proper metric
        pass

    def _replicate(self):
        """Update path distribution"""

        path_probs = np.array([self.path_distribution[i](self._route_time) for i in range(len(self.paths))])
        path_fitnesses = self.network.capacity / path_probs  # self._compute_path_fitnesses()
        log_der = path_fitnesses - np.sum(path_probs * path_fitnesses)
        path_probs *= np.exp(log_der * self.reroute_interval)
        path_probs /= path_probs.sum()

        for i in range(len(self.paths)):
            self.path_distribution[i].extend(self._next_reroute_time, path_probs[i])
            self.network.commodities[i].sources[self._source] = self.inflow * self.path_distribution[i]

    def build_flow(self) -> Generator[DynamicFlow, None, None]:
        if self._built:
            raise RuntimeError("Flow was already built. Initialize a new builder.")
        self._built = True

        yield self._flow
        while self._flow.phi < float("inf"):
            while self._flow.phi >= self._network_inflow_changes.min_key():
                c, s, t = self._network_inflow_changes.pop()
                self._handle_nodes.add(s)
            if (
                self.reroute_interval is None
                or self._flow.phi >= self._next_reroute_time
            ):
                self._route_time = self._next_reroute_time
                self._next_reroute_time += self.reroute_interval
                self._handle_nodes = set(self.network.graph.nodes.values())

                # redistribute inflow and update the functions dict
                self._replicate()
                self._net_inflow_by_node[self._source].extend(
                    self._next_reroute_time,
                    {com_id: com.sources[self._source](self._next_reroute_time) for com_id, com in enumerate(self.network.commodities)},
                    self.inflow(self._next_reroute_time)
                )

            new_inflow = self._determine_new_inflow()
            max_ext_time = min(
                self._next_reroute_time, self._network_inflow_changes.min_key()
            )
            edges_with_outflow_change = self._flow.extend(new_inflow, max_ext_time)
            self._handle_nodes = set(
                self.network.graph.edges[e].node_to for e in edges_with_outflow_change
            )

            yield self._flow

    def run(self, horizon: float):
        generator = self.build_flow()
        flow = next(generator)
        while flow.phi < horizon:
            flow = next(generator)

        return flow
