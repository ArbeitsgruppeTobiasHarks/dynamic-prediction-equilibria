import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Callable, Dict, List, Set, Tuple

from core.active_paths import (
    Path,
    PathsOverTime,
    all_active_paths_parallel,
    compute_all_active_paths,
    compute_path_travel_time,
    get_activity_indicator,
    get_one_path_coverage,
)
from core.bellman_ford import bellman_ford
from core.dijkstra import reverse_dijkstra
from core.dynamic_flow import DynamicFlow
from core.graph import Edge, Node
from core.machine_precision import eps
from core.network import Network
from core.path_flow_builder import PathFlowBuilder
from core.predictors.predictor_type import PredictorType
from utilities.arrays import elem_lrank
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.piecewise_linear import PiecewiseLinear, identity
from utilities.right_constant import Indicator, RightConstant
from utilities.status_logger import TimedStatusLogger
from visualization.to_json import merge_commodities, to_visualization_json


def integrate_with_weights(
    lin: PiecewiseLinear, weights: RightConstant, start: float, end: float
):
    assert weights.domain[0] <= start < end <= weights.domain[1]

    value = 0.0
    rnk = elem_lrank(weights.times, start)

    if rnk == len(weights.times) - 1:
        return weights.values[rnk] * lin.integrate(start, end)

    value += weights.values[rnk] * lin.integrate(start, weights.times[rnk + 1])
    rnk += 1
    while rnk < len(weights.times) - 1 and weights.times[rnk + 1] <= end:
        value += weights.values[rnk] * lin.integrate(
            weights.times[rnk], weights.times[rnk + 1]
        )
        rnk += 1

    value += weights.values[rnk] * lin.integrate(weights.times[rnk], end)

    return value


class BaseFlowIterator:
    """
    Base
    """

    network: Network
    reroute_interval: float
    horizon: float
    inflow_horizon: float
    delay_threshold: float  # max value of normalized delay to consider path active
    parallelize: bool
    _iter: int
    _flow: DynamicFlow
    _flows_history: List[DynamicFlow]
    _comm_to_path: Dict[int, Path]
    _path_to_comm: Dict[Path, int]
    _paths: Dict[Tuple[Node, Node], List[Path]]  # all used s-t paths
    _metrics: Dict[str, Dict[str, List]]
    _important_nodes: Dict[Tuple[Node, Node], Set[Node]]
    _free_flow_travel_times: Dict[
        Tuple[Node, Node], float
    ]  # travel times without queues
    _earliest_arrivals_to: Dict[
        Node, Dict[Node, PiecewiseLinear]
    ]  # earliest arrivals to sinks
    _inflows: Dict[Tuple[Node, Node], RightConstant]

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        inflow_horizon: float,
        delay_threshold: float,
        parallelize: bool,
    ):
        assert all(len(c.sources) == 1 for c in network.commodities)
        self.network = network
        self.reroute_interval = reroute_interval
        self.horizon = horizon
        self.inflow_horizon = inflow_horizon
        self.delay_threshold = delay_threshold
        self.parallelize = parallelize

        self._iter = 0
        self._flow = DynamicFlow(self.network)
        self._flows_history = []
        self._paths = dict()
        self._metrics = dict()
        self._important_nodes = dict()
        self._free_flow_travel_times = dict()
        self._earliest_arrivals_to = dict()
        self._inflows = dict()

        for i, com in enumerate(network.commodities):
            s, inflow = next(iter(com.sources.items()))
            t = com.sink
            self._paths[(s, t)] = []
            self._metrics[str((s.id, t.id))] = {"avg_delays": [], "path_metrics": []}
            self._important_nodes[(s, t)] = network.graph.get_nodes_reaching(
                t
            ) & network.graph.get_reachable_nodes(s)
            self._earliest_arrivals_to[t] = dict()
            self._inflows[(s, t)] = inflow

        self._comm_to_path = {i: None for i in range(len(network.commodities))}
        self._path_to_comm = dict()
        assert len(self._paths) == len(self._comm_to_path)

    def _initialize_paths(self):
        """
        Assigns each commodity the shortest path under assumption there are no queues
        """
        for com_id, com in enumerate(self.network.commodities):
            s = next(iter(com.sources.keys()))
            t = com.sink

            dist = reverse_dijkstra(
                t, self.network.travel_time, self.network.graph.get_nodes_reaching(t)
            )
            v = s
            edges = []
            while v != t:
                for e in v.outgoing_edges:
                    new_node = e._node_to
                    if new_node not in dist:
                        continue
                    if dist[v] - dist[new_node] == self.network.travel_time[e.id]:
                        edges.append(e)
                        v = new_node
                        break
            path = Path(edges)

            self._paths[(s, t)].append(path)
            self._comm_to_path[com_id] = path
            self._path_to_comm[path] = com_id
            self._free_flow_travel_times[(s, t)] = dist[s]

    def _run_bellman_ford(self):
        costs = self._flow.get_edge_costs()
        for s, t in self._paths.keys():
            earliest_arrivals = bellman_ford(
                t, costs, self._important_nodes[(s, t)], 0.0
            )
            self._earliest_arrivals_to[t].update(earliest_arrivals)

    def _compute_flow(self):
        flow_builder = PathFlowBuilder(
            self.network, self._comm_to_path, self.reroute_interval
        )
        generator = flow_builder.build_flow()
        flow = next(generator)
        while flow.phi < self.horizon:
            flow = next(generator)

        self._flows_history.append(flow)
        self._flow = flow

    def _get_path_inflow(self, path: Path) -> RightConstant:
        if path in self._path_to_comm:
            com_id = self._path_to_comm[path]
            return self.network.commodities[com_id].sources[path.start]
        else:
            return RightConstant([0.0], [0.0], (0, float("inf")))

    def _set_path_inflow(self, path: Path, new_inflow: RightConstant):
        s = path.start
        t = path.end

        if path in self._path_to_comm:
            com_id = self._path_to_comm[path]
            self.network.commodities[com_id].sources[s] = new_inflow
        else:
            new_com_id = len(self.network.commodities)
            self.network.add_commodity({s.id: new_inflow}, t.id, PredictorType.CONSTANT)
            self._comm_to_path[new_com_id] = path
            self._path_to_comm[path] = new_com_id
            self._paths[(s, t)].append(path)

    def _reassign_inflows(self):
        raise NotImplementedError

    def _compute_route_avg_delays(self, normalize: bool = True):
        """
        Computes average delays for all present s-t pairs.
        If parameter normalize is set to True, delays are divided by travel times without queues.
        """
        for (s, t), paths in self._paths.items():
            accum_net_outflow = sum(
                (
                    self._flow.outflow[p.edges[-1].id]._functions_dict[
                        self._path_to_comm[p]
                    ]
                    for p in paths
                    if self._path_to_comm[p]
                    in self._flow.outflow[p.edges[-1].id]._functions_dict
                ),
                start=RightConstant([0.0], [0.0], (0, float("inf"))),
            ).integral()
            accum_net_inflow = sum(
                (self._get_path_inflow(p) for p in paths),
                start=RightConstant([0.0], [0.0], (0, float("inf"))),
            ).integral()
            # assert abs(1 - accum_net_inflow(self.horizon)/accum_net_outflow(self.horizon)) < eps
            avg_travel_time = (
                accum_net_inflow.integrate(0.0, self.horizon)
                - accum_net_outflow.integrate(0.0, self.horizon)
            ) / accum_net_inflow(self.horizon)

            optimal_travel_time = self._earliest_arrivals_to[t][s] - identity
            opt_avg_travel_time = integrate_with_weights(
                optimal_travel_time, self._inflows[(s, t)], 0, self.inflow_horizon
            ) / (self._inflows[(s, t)].integral()(self.inflow_horizon))

            assert opt_avg_travel_time > 0

            avg_delay = avg_travel_time - opt_avg_travel_time
            if normalize:
                avg_delay /= self._free_flow_travel_times[(s, t)]

            self._metrics[str((s.id, t.id))]["avg_delays"].append(avg_delay)

    def _compute_path_metrics(self):
        for s, t in self._paths.keys():
            network_data = (
                self._flow.get_edge_costs(),
                self._earliest_arrivals_to[t][s] - identity,
                self._free_flow_travel_times[(s, t)],
                self._inflows[(s, t)],
                self.inflow_horizon,
                self.delay_threshold,
            )

            if self.parallelize:
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    path_metrics = pool.starmap(
                        compute_path_metrics,
                        [
                            (path, self._get_path_inflow(path), network_data)
                            for path in self._paths[(s, t)]
                        ],
                    )
            else:
                path_metrics = [
                    compute_path_metrics(
                        path, self._get_path_inflow(path), network_data
                    )
                    for path in self._paths[(s, t)]
                ]

            self._metrics[str((s.id, t.id))]["path_metrics"].append(path_metrics)

    def _iteration(self):
        if self._iter == 0:
            self._initialize_paths()
        else:
            self._reassign_inflows()

        self._compute_flow()
        self._run_bellman_ford()

        self._compute_route_avg_delays(normalize=True)
        self._compute_path_metrics()

    def _merge_commodities(self):
        merged_flow = self._flow
        merged_network = deepcopy(self.network)
        merged_network.commodities = []

        for (s, t), paths in self._paths.items():
            commodities = [self._path_to_comm[p] for p in paths]
            merged_flow = merge_commodities(merged_flow, self.network, commodities)
            merged_network.add_commodity(
                {s.id: self._inflows[(s, t)]}, t.id, PredictorType.CONSTANT
            )

        return merged_flow, merged_network

    def run(self, num_iterations, log_every=10):
        """
        Main cycle
        """
        with TimedStatusLogger(
            f"Performing {num_iterations} iterations", finish_msg=""
        ):
            for i in range(num_iterations):
                self._iteration()

                if i % log_every == 0:
                    print()
                    print(f"Iteration: {self._iter}")
                    print(f"Normalized average delays:")
                    for s, t in self._paths.keys():
                        avg_delay = self._metrics[str((s.id, t.id))]["avg_delays"][
                            self._iter
                        ]
                        print(f"({s.id} -> {t.id}): {round(avg_delay, 4)}")

                self._iter += 1

            merged_flow, merged_network = self._merge_commodities()

        return merged_flow, merged_network, self._metrics


def compute_path_metrics(path, path_inflow, network_data):
    (
        costs,
        optimal_travel_time,
        free_flow_travel_time,
        total_inflow,
        inflow_horizon,
        delay_threshold,
    ) = network_data

    metrics = {"path": str(path)}

    accum_path_inflow = path_inflow.integral()

    metrics["share_of_total_inflow"] = accum_path_inflow(
        inflow_horizon
    ) / total_inflow.integral()(inflow_horizon)

    if metrics["share_of_total_inflow"] < eps:
        metrics["avg_delay_normalized"] = 0.0
        metrics["active_inflow_share"] = 0.0
        return metrics

    path_delay = compute_path_travel_time(path, costs) - optimal_travel_time
    activity_ind = get_activity_indicator(
        path_delay, threshold=delay_threshold * free_flow_travel_time
    )

    metrics["avg_delay_normalized"] = integrate_with_weights(
        path_delay, path_inflow, 0, inflow_horizon
    ) / (accum_path_inflow(inflow_horizon) * free_flow_travel_time)

    metrics["active_inflow_share"] = (path_inflow * activity_ind).integral()(
        inflow_horizon
    ) / accum_path_inflow(inflow_horizon)

    return metrics


# class AlphaFlowIterator(BaseFlowIterator):  # old version, uncommenting will probably result in errors
#     alpha: float    # fraction of inflow to be redistributed
#     approx_inflows: bool    # option to project inflows to regular grid after each iteration
#     """
#     At each iteration, a constant fraction of inflow is redistributed to optimal path from previous iteration
#     """
#
#     def __init__(self,
#                  network: Network,
#                  reroute_interval: float,
#                  horizon: float,
#                  num_iterations: int = 100,
#                  alpha: float = 0.01,
#                  approx_inflows: bool = True):
#
#         super().__init__(network, reroute_interval, horizon, num_iterations)
#         self.alpha = alpha
#         self.approx_inflows = approx_inflows
#
#     def _assign_new_paths(self, route, p_o_t):
#
#         path_commodity = {i: None for i in range(len(p_o_t))}
#         for com_id in self._route_users[route]:
#             path = self._comm_to_path[com_id]
#             if path in p_o_t.paths:
#                 path_commodity[p_o_t.paths.index(path)] = com_id
#
#         for com_id in self._route_users[route]:
#             self.network.commodities[com_id].sources[route[0]] *= 1 - self.alpha
#
#         for i in range(len(p_o_t.paths)):
#             new_path = p_o_t.paths[i]
#             indicator = p_o_t.activity_indicators[i]
#
#             new_inflow = self.alpha * self._inflows[route] * indicator
#             if new_inflow.integral()(self.horizon) < eps:
#                 continue
#
#             if path_commodity[i] is not None:
#                 com_id = path_commodity[i]
#                 self.network.commodities[com_id].sources[route[0]] += new_inflow
#             else:
#                 new_com_id = len(self.network.commodities)
#                 self._comm_to_path[new_com_id] = new_path
#                 self._route_users[route].append(new_com_id)
#                 self.network.add_commodity(
#                     {route[0].id: new_inflow},
#                     route[1].id,
#                     PredictorType.CONSTANT,
#                 )
#
#     def _reassign_inflows(self, flow):
#
#         costs = flow.get_edge_costs()
#
#         def process_route(route):
#             s, t = route
#             earliest_arrivals = bellman_ford(
#                 t,
#                 costs,
#                 self._important_nodes[route],
#                 0.0,
#                 float("inf"),
#             )
#             p_o_t = compute_shortest_paths_over_time(earliest_arrivals, costs, s, t)
#
#             self._assign_new_paths(route, p_o_t)
#
#             if self.approx_inflows:
#                 for com_id in self._route_users[route]:
#                     inflow = self.network.commodities[com_id].sources[route[0]]
#                     self.network.commodities[com_id].sources[route[0]] = inflow.project_to_grid(
#                         self.reroute_interval).simplify()
#
#         with ThreadPoolExecutor() as executor:
#             executor.map(process_route, self._route_users.keys())
#
#         # for route in self._route_users.keys():
#         #     process_route(route)


def approximate_linear(
    lin: PiecewiseLinear, delta: float, horizon: float
) -> RightConstant:
    """
    Returns a RightConstant approximation of a given PiecewiseLinear with grid size delta
    """
    n_nodes = ceil(horizon / delta) + 1
    new_times = [delta * n for n in range(n_nodes)]
    new_values = [0.0] * n_nodes
    for i in range(n_nodes - 1):
        new_values[i] = lin.integrate(new_times[i], new_times[i + 1]) / delta
    new_values[-1] = lin.values[-1]

    return RightConstant(new_times, new_values, lin.domain)


class AlphaFlowIterator(BaseFlowIterator):
    alpha_fun: Callable[
        [float], float
    ]  # fraction of inflow to redistribute as function of relative delay
    min_path_active_time: float
    approx_inflows: bool
    """ 
    For each path, an approximation of relative delay d is computed. 
    Then, a fraction alpha_fun(d) of its inflow is redistributed to active paths.
    """

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        inflow_horizon: float,
        alpha_fun: Callable[[float], float] = lambda d: 0.01,
        delay_threshold: float = eps,
        min_path_active_time: float = 1000 * eps,
        approx_inflows: bool = True,
        parallelize: bool = False,
    ):
        super().__init__(
            network,
            reroute_interval,
            horizon,
            inflow_horizon,
            delay_threshold,
            parallelize,
        )
        self.alpha_fun = alpha_fun
        self.min_path_active_time = min_path_active_time
        self.approx_inflows = approx_inflows

        for s, t in self._paths.keys():
            self._metrics[str((s.id, t.id))]["inflow_changes"] = [1.0]

    def _determine_new_inflow(
        self,
        route: Tuple[Node, Node],
        costs: List[PiecewiseLinear],
        new_paths_coverage: Indicator,
    ) -> RightConstant:
        total_inflow = self._inflows[route].integral()(self.inflow_horizon)
        new_inflow = RightConstant([0.0], [0.0], (0, float("inf")))

        opt_travel_time = self._earliest_arrivals_to[route[1]][route[0]] - identity

        for path in self._paths[route]:
            path_inflow = self._get_path_inflow(path)

            if path_inflow.integral()(self.inflow_horizon) < eps * total_inflow:
                self._set_path_inflow(
                    path, RightConstant([0.0], [0.0], (0, float("inf")))
                )
                new_inflow += path_inflow
                continue

            path_delay = compute_path_travel_time(path, costs) - opt_travel_time
            path_delay_approx = approximate_linear(
                path_delay, self.reroute_interval, self.inflow_horizon
            )  # if path inflows are defined on same grid, this yields avg experienced delay on time intervals
            alpha_vals = [
                self.alpha_fun(d / self._free_flow_travel_times[route])
                if d > 0
                else self.alpha_fun(0)
                for d in path_delay_approx.values
            ]
            alpha = RightConstant(
                path_delay_approx.times, alpha_vals, (0, float("inf"))
            )

            inflow_change = alpha * path_inflow * new_paths_coverage
            new_inflow += inflow_change
            self._set_path_inflow(path, path_inflow - inflow_change)

        return new_inflow.simplify()

    def _assign_new_paths(self, active_paths: PathsOverTime, new_inflow: RightConstant):
        n_active_paths = RightConstant.sum(list(active_paths.values()))
        uniform_factor = n_active_paths.invert()

        for path, indicator in active_paths.items():
            current_inflow = self._get_path_inflow(path)

            path_prob = indicator.restrict((0, self.inflow_horizon)) * uniform_factor
            new_path_inflow = current_inflow + new_inflow * path_prob
            self._set_path_inflow(path, new_path_inflow)

    def _reassign_inflows(self):
        costs = self._flow.get_edge_costs()

        def process_route(route):
            s, t = route
            compute_active_paths_fun = (
                all_active_paths_parallel
                if self.parallelize
                else compute_all_active_paths
            )
            active_paths = compute_active_paths_fun(
                self._earliest_arrivals_to[t],
                costs,
                s,
                t,
                self.inflow_horizon,
                delay_threshold=self.delay_threshold
                * self._free_flow_travel_times[route],
                min_active_time=self.min_path_active_time,
            )

            # active_paths = get_one_path_coverage(active_paths, self.inflow_horizon)

            new_inflow = self._determine_new_inflow(
                route, costs, active_paths.coverage()
            )
            self._assign_new_paths(active_paths, new_inflow)

            if self.approx_inflows:
                for path in self._paths[route]:
                    inflow = self._get_path_inflow(path)
                    approximation = inflow.project_to_grid(
                        self.reroute_interval, self.inflow_horizon
                    )
                    self._set_path_inflow(path, approximation)

            redistributed = new_inflow.integral()(self.inflow_horizon) / self._inflows[
                (s, t)
            ].integral()(self.inflow_horizon)
            self._metrics[str((s.id, t.id))]["inflow_changes"].append(redistributed)
            # print(f"({s.id} -> {t.id}): {round(100 * redistributed, 2)}% of inflow redistributed to {len(active_paths)} active paths")

        if self.parallelize:
            with ThreadPoolExecutor() as executor:
                executor.map(process_route, self._paths.keys())
        else:
            for route in self._paths.keys():
                process_route(route)
