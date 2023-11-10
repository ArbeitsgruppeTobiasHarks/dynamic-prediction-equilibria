from copy import deepcopy
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from core.active_paths import Path, compute_path_travel_time
from core.convergence import approximate_linear, integrate_with_weights
from core.dynamic_flow import DynamicFlow
from core.graph import Edge, Node
from core.machine_precision import eps
from core.network import Network
from core.path_flow_builder import PathFlowBuilder
from core.predictors.predictor_type import PredictorType
from utilities.piecewise_linear import identity, zero
from utilities.right_constant import RightConstant


class ReplicatorFlowBuilder(PathFlowBuilder):
    inflow: RightConstant
    horizon: float
    fitness: str
    rep_window: float
    rep_coef: float
    _source: Node
    _path_distribution: Dict[int, RightConstant]
    _path_fitnesses: Dict[int, RightConstant]

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        initial_distribution: List[Tuple[List[int], float]],
        fitness: str,
        rep_coef: float = 1.0,
        rep_window: Optional[float] = None,
    ):
        network_copy = deepcopy(network)
        self._source, self.inflow = next(iter(network.commodities[0].sources.items()))
        t = network.commodities[0].sink
        network_copy.commodities = []
        paths = dict()

        self.horizon = horizon
        self.fitness = fitness
        self.rep_coef = rep_coef
        self.rep_window = rep_window if rep_window is not None else float("inf")
        self._path_distribution = dict()
        self._path_fitnesses = dict()

        for i, (e_ids, path_prob) in enumerate(initial_distribution):
            path = Path([network.graph.edges[e_id] for e_id in e_ids])
            # self._free_flow_dist = min(
            #    sum(network.travel_time[e_id] for e_id in e_ids), self._free_flow_dist
            # )
            self._path_distribution[i] = RightConstant(
                [0.0], [path_prob], (0, float("inf"))
            )
            self._path_fitnesses[i] = RightConstant([0.0], [0.0], (0, float("inf")))
            network_copy.add_commodity(
                {self._source.id: self.inflow * self._path_distribution[i]},
                t.id,
                PredictorType.CONSTANT,
            )
            paths[i] = path

        super().__init__(network_copy, paths, reroute_interval)

    def _get_pred_travel_times(self):
        tt = {i: 0.0 for i in range(len(self.paths))}

        queues = [q(self._route_time) for q in self._flow.queues]

        for com_id, path in self.paths.items():
            tt[com_id] = sum(
                self.network.travel_time[e.id]
                + queues[e.id] / self.network.capacity[e.id]
                for e in path.edges
            )

        return tt

    def _get_last_travel_times(self):
        """Compute travel time of the last particle that completed the route."""

        tt = {i: 0.0 for i in range(len(self.paths))}
        if self._route_time < eps:
            return tt

        costs = self._flow.get_edge_costs()

        for com_id, path in self.paths.items():
            particle_entry = self._route_time
            for e in path.edges[::-1]:
                particle_entry = identity.plus(costs[e.id]).max_t_below(
                    particle_entry, 0.0
                )

            tt[com_id] = self._route_time - particle_entry

        return tt

    def _get_avg_travel_times_in_window(self, window_size: float):
        avg_tt = {i: 0.0 for i in range(len(self.paths))}
        if self._route_time < eps:
            return avg_tt

        window_start = max(self._route_time - window_size, 0.0)
        costs = self._flow.get_edge_costs()

        for com_id, path in self.paths.items():
            # need to count outflow only from window inflow
            outflow_start = window_start
            for e in path.edges:
                outflow_start += costs[e.id](outflow_start)
            outflow_start = min(outflow_start, self._route_time)

            accum_outflow = (
                self._flow.outflow[path.edges[-1].id]._functions_dict[com_id].integral()
                if com_id in self._flow.outflow[path.edges[-1].id]._functions_dict
                else zero
            )
            accum_inflow = (
                self.network.commodities[com_id].sources[self._source].integral()
            )

            if accum_inflow(self._route_time) - accum_inflow(window_start) > 1000 * eps:
                avg_tt[com_id] = (
                    accum_inflow.integrate(window_start, self._route_time)
                    - accum_inflow(window_start) * (self._route_time - window_start)
                    - accum_outflow.integrate(outflow_start, self._route_time)
                ) / (accum_inflow(self._route_time) - accum_inflow(window_start))

        return avg_tt

    def _compute_path_fitnesses(self):
        if self.fitness == "neg_avg_tt":
            fitnesses = {
                k: -v
                for k, v in self._get_avg_travel_times_in_window(
                    self.rep_window
                ).items()
            }
        elif self.fitness == "neg_last_tt":
            fitnesses = {k: -v for k, v in self._get_last_travel_times().items()}
        elif self.fitness == "neg_pred_tt":
            fitnesses = {k: -v for k, v in self._get_pred_travel_times().items()}

        for com_id, fitness in fitnesses.items():
            self._path_fitnesses[com_id].extend(self._route_time, fitness)

    def _replicate(self):
        """Update path distribution"""

        path_probs = np.array(
            [
                self._path_distribution[i](self._route_time)
                for i in range(len(self.paths))
            ]
        )
        path_fitnesses = np.array(
            [self._path_fitnesses[i](self._route_time) for i in range(len(self.paths))]
        )
        log_der = self.rep_coef * (path_fitnesses - np.sum(path_probs * path_fitnesses))
        path_probs *= np.exp(log_der * self.reroute_interval)
        path_probs /= (
            path_probs.sum()
        )  # normalization required, since path_probs.sum() slightly differs from 1

        for i in range(len(self.paths)):
            self._path_distribution[i].extend(self._next_reroute_time, path_probs[i])
            self.network.commodities[i].sources[self._source] = (
                self.inflow * self._path_distribution[i]
            )

    def build_flow(self) -> Generator[DynamicFlow, None, None]:
        if self._built:
            raise RuntimeError("Flow was already built. Initialize a new builder.")
        self._built = True

        yield self._flow
        while self._flow.phi < float("inf"):
            while self._flow.phi >= self._network_inflow_changes.min_key():
                c, s, t = self._network_inflow_changes.pop()
                self._handle_nodes.add(s)
            if self._flow.phi >= self._next_reroute_time:
                self._route_time = self._next_reroute_time
                self._next_reroute_time += self.reroute_interval
                self._handle_nodes = set(self.network.graph.nodes.values())

                # redistribute inflow and update the functions dict
                self._compute_path_fitnesses()
                self._replicate()
                self._net_inflow_by_node[self._source].extend(
                    self._next_reroute_time,
                    {
                        com_id: com.sources[self._source](self._next_reroute_time)
                        for com_id, com in enumerate(self.network.commodities)
                    },
                    self.inflow(self._next_reroute_time),
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

    def run(self):
        generator = self.build_flow()
        flow = next(generator)
        while flow.phi < self.horizon:
            flow = next(generator)

        costs = flow.get_edge_costs()

        paths_dict = {
            i: {
                "path": str(self.paths[i]),
                "inflow share": self._path_distribution[i],
                "fitness": self._path_fitnesses[i],
                "travel time": approximate_linear(
                    compute_path_travel_time(self.paths[i], costs),
                    self.reroute_interval,
                    self.horizon,
                ),
            }
            for i in range(len(self.paths))
        }

        return flow, paths_dict
