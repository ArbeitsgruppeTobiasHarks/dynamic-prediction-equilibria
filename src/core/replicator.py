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
from utilities.piecewise_linear import PiecewiseLinear, identity, zero
from utilities.right_constant import RightConstant


class ReplicatorFlowBuilder(PathFlowBuilder):
    inflow: RightConstant
    horizon: float
    fitness: Optional[str]
    replication_coef: float
    window_size: float
    regularization: Optional[str]
    regularization_coef: float
    regularization_decay: float
    _source: Node
    _path_distribution: Dict[int, RightConstant]
    _path_fitnesses: Dict[int, RightConstant]

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        initial_distribution: List[Tuple[List[int], float]],
        fitness: Optional[str] = None,
        replication_coef: float = 1.0,
        window_size: Optional[float] = None,
        regularization: Optional[str] = None,
        regularization_coef: float = 1.0,
        regularization_decay: float = 1.0,
    ):
        network_copy = deepcopy(network)
        self._source, self.inflow = next(iter(network.commodities[0].sources.items()))
        t = network.commodities[0].sink
        network_copy.commodities = []
        paths = dict()

        self.horizon = horizon
        self.fitness = fitness
        self.replication_coef = replication_coef
        self.window_size = window_size if window_size is not None else float("inf")
        self.regularization = regularization
        self.regularization_coef = regularization_coef
        self.regularization_decay = regularization_decay

        self._path_distribution = dict()
        self._path_fitnesses = dict()

        for i, (e_ids, path_prob) in enumerate(initial_distribution):
            path = Path([network.graph.edges[e_id] for e_id in e_ids])
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

    def _get_tt_grad_approx(self):
        grads = {i: 0.0 for i in self.paths.keys()}

        def grad_approx(queue: PiecewiseLinear, delta: float):
            if self._route_time < delta + eps:
                return 0.0
            return (queue(self._route_time) - queue(self._route_time - delta)) / delta

        avg_grads = [
            grad_approx(q, 2 * self.reroute_interval) for q in self._flow.queues
        ]

        for com_id, path in self.paths.items():
            grads[com_id] = sum(
                avg_grads[e.id] / self.network.capacity[e.id] for e in path.edges
            )

        return grads

    def _get_pred_travel_times(self):
        tt = {i: 0.0 for i in self.paths.keys()}

        queues = [q(self._route_time) for q in self._flow.queues]

        for com_id, path in self.paths.items():
            tt[com_id] = sum(
                self.network.travel_time[e.id]
                + queues[e.id] / self.network.capacity[e.id]
                for e in path.edges
            )

        return tt

    def _get_proj_travel_times(self, proj_horizon: float):
        tt = {i: 0.0 for i in self.paths.keys()}

        def predictor(queue: PiecewiseLinear, delta: float):
            if self._route_time < delta + eps:
                return zero
            avg_grad = (
                queue(self._route_time) - queue(self._route_time - delta)
            ) / delta
            return PiecewiseLinear(
                [self._route_time],
                [queue(self._route_time)],
                0.0,
                avg_grad,
            )

        queues = [predictor(q, 2 * self.reroute_interval) for q in self._flow.queues]

        for com_id, path in self.paths.items():
            theta = theta_start = self._route_time + proj_horizon
            for e in path.edges:
                theta += (
                    self.network.travel_time[e.id]
                    + queues[e.id](theta) / self.network.capacity[e.id]
                )
            tt[com_id] = theta - theta_start
        return tt

    def _get_last_travel_times(self):
        """Compute travel time of the last particle that completed the route."""

        tt = {i: 0.0 for i in self.paths.keys()}
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

    def _get_avg_travel_times_in_window(self, rep_window: float):
        avg_tt = {i: 0.0 for i in self.paths.keys()}
        if self._route_time < eps:
            return avg_tt

        window_start = max(self._route_time - rep_window, 0.0)
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

    def _get_mixed_travel_times(self, history_coef: float = 0.5):
        """Computes history_coef * avg_tt + (1-history_coef) * pred_tt"""

        avg_tt = self._get_avg_travel_times_in_window(self.window_size)
        pred_tt = self._get_pred_travel_times()

        tt = {
            k: history_coef * avg_tt[k] + (1 - history_coef) * pred_tt[k]
            for k in avg_tt.keys()
        }

        return tt

    def _get_avg_shares(self, window: float):
        avg_shares = {i: 0.0 for i in range(len(self.paths))}
        if self._route_time < eps:
            return avg_shares

        window_start = max(0.0, self._route_time - window)
        for com_id in self.paths.keys():
            share_int = self._path_distribution[com_id].integral()
            avg_shares[com_id] = (
                share_int(self._route_time) - share_int(window_start)
            ) / (self._route_time - window_start)

        return avg_shares

    # def _get_fluctuation_regularizations(self):
    #     return {
    #         com_id: abs(self._path_distribution[com_id](self._route_time) - avg_share)
    #         for com_id, avg_share in self._get_avg_shares(self.window_size).items()
    #     }

    def _get_capacity_regularizations(self):
        edge_loads = self._flow.get_edge_loads()
        regularizations = {
            com_id: sum(
                edge_loads[e.id](self._route_time) - self.network.capacity[e.id]
                for e in path.edges
            )
            / len(path)
            for com_id, path in self.paths.items()
        }
        return regularizations

    def _get_logit_regularizations(self):
        regulazitations = {
            com_id: np.log(prob_func(self._route_time))
            for com_id, prob_func in self._path_distribution.items()
        }
        return regulazitations

    def _compute_path_fitnesses(self):
        if self.fitness == "neg_avg_tt":
            fitnesses = {
                k: -v
                for k, v in self._get_avg_travel_times_in_window(
                    rep_window=self.window_size
                ).items()
            }
        elif self.fitness == "neg_last_tt":
            fitnesses = {k: -v for k, v in self._get_last_travel_times().items()}
        elif self.fitness == "neg_pred_tt":
            fitnesses = {k: -v for k, v in self._get_pred_travel_times().items()}
        elif self.fitness == "neg_proj_tt":
            fitnesses = {
                k: -v
                for k, v in self._get_proj_travel_times(
                    proj_horizon=self.window_size
                ).items()
            }
        elif self.fitness == "neg_grad_tt":
            fitnesses = {k: -v for k, v in self._get_tt_grad_approx().items()}
        else:
            fitnesses = {i: 0.0 for i in self.paths.keys()}

        if self.regularization == "capacity":
            regularization = self._get_capacity_regularizations()
        elif self.regularization == "logit":
            regularization = self._get_logit_regularizations()
        else:
            regularization = {i: 0.0 for i in self.paths.keys()}

        for com_id, fitness in fitnesses.items():
            fitness -= (
                regularization[com_id]
                * self.regularization_coef
                * np.exp(-self.regularization_decay * self._route_time)
            )
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
        log_der = self.replication_coef * (
            path_fitnesses - np.sum(path_probs * path_fitnesses)
        )
        path_probs *= np.exp(log_der * self.reroute_interval)
        path_probs /= (
            path_probs.sum()
        )  # normalization required, since sum slightly differs from 1

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

        dynamics_out = {
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
            for i in self.paths.keys()
        }
        #  add last value in case function was constant in the end
        for i in self.paths.keys():
            for k in ["inflow share", "fitness"]:
                if dynamics_out[i][k].times[-1] < self.horizon - eps:
                    dynamics_out[i][k].times.append(self.horizon)
                    dynamics_out[i][k].values.append(dynamics_out[i][k].values[-1])

        return flow, dynamics_out
