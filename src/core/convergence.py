import os
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Tuple, Collection, Callable

from core.bellman_ford import bellman_ford
from core.dijkstra import reverse_dijkstra
from core.dynamic_flow import DynamicFlow
from core.graph import Edge, Node
from core.machine_precision import eps
from core.network import Network
from core.path_flow_builder import PathFlowBuilder
from core.predictors.predictor_type import PredictorType
from eval.evaluate import calculate_optimal_average_travel_time
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.piecewise_linear import PiecewiseLinear, identity
from utilities.right_constant import RightConstant, Indicator
from utilities.arrays import elem_lrank
from visualization.to_json import merge_commodities, to_visualization_json

from concurrent.futures import ThreadPoolExecutor

Path = List[Edge]

#
# @dataclass
# class LabelledPathEntry:
#     """
#     This class represents one edge of a labelled path.
#     Each such edge is labelled with a starting time and an active_until time.
#     """
#
#     edge: Edge
#     starting_time: float
#     active_until: float


class PathsOverTime:
    """
    Collection of paths with same source and sink.
    Each path is equipped with intervals of activity represented by indicator
    """

    paths: List[Path]
    activity_indicators: List[Indicator]

    def __init__(self, paths, activity_indicators):
        self.paths = paths
        self.activity_indicators = activity_indicators

    def __len__(self):
        return len(self.paths)

    def __add__(self, other):
        return PathsOverTime(
            self.paths + other.paths,
            self.activity_indicators + other.activity_indicators
        )

    def add_path(self, path, activity_indicator):
        if path in self.paths:
            i = self.paths.index(path)
            self.activity_indicators[i] += activity_indicator
        else:
            self.paths += [path]
            self.activity_indicators += [activity_indicator]


def integrate_with_weights(lin: PiecewiseLinear, weights: RightConstant, start: float, end: float):
    assert weights.domain[0] <= start < end <= weights.domain[1]

    value = 0.0
    rnk = elem_lrank(weights.times, start)

    if rnk == len(weights.times) - 1:
        return weights.values[rnk] * lin.integrate(start, end)

    value += weights.values[0] * lin.integrate(start, weights.times[rnk + 1])
    rnk += 1
    while rnk < len(weights.times) - 1 and weights.times[rnk + 1] <= end:
        value += weights.values[rnk] * lin.integrate(weights.times[rnk], weights.times[rnk + 1])
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
    _iter: int
    _flow: DynamicFlow
    _flows_history: List[DynamicFlow]
    _comm_to_path: Dict[int, Path]
    _path_to_comm: Dict[Tuple, int]
    _paths: Dict[Tuple[Node, Node], List[Path]]  # all used s-t paths
    _path_metrics: Dict[Tuple[Node, Node], List[List[Tuple[Path, Dict]]]]
    _important_nodes: Dict[Tuple[Node, Node], Collection[Node]]
    _fastest_travel_times: Dict[Tuple[Node, Node], float]  # travel times without queues
    _earliest_arrivals_to: Dict[Node, Dict[Node, PiecewiseLinear]] # earliest arrivals to sinks
    _inflows: Dict[Tuple[Node, Node], float]

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        inflow_horizon: float,
        delay_threshold: float
    ):
        assert all(len(c.sources) == 1 for c in network.commodities)
        self.network = network
        self.reroute_interval = reroute_interval
        self.horizon = horizon
        self.inflow_horizon = inflow_horizon
        self.delay_threshold = delay_threshold

        self._iter = 0
        self._flow = DynamicFlow(self.network)
        self._flows_history = []
        self._paths = dict()
        self._path_metrics = dict()
        self._important_nodes = dict()
        self._fastest_travel_times = dict()
        self._earliest_arrivals_to = dict()
        self._inflows = dict()

        for i, com in enumerate(network.commodities):
            s, inflow = next(iter(com.sources.items()))
            t = com.sink
            self._paths[(s, t)] = []
            self._path_metrics[(s, t)] = []
            self._important_nodes[(s, t)] = (
                    network.graph.get_nodes_reaching(t) & network.graph.get_reachable_nodes(s)
            )
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
            path = []
            while v != t:
                for e in v.outgoing_edges:
                    if dist[v] - dist[e._node_to] == self.network.travel_time[e.id]:
                        path.append(e)
                        v = e._node_to
                        break

            self._paths[(s, t)].append(path)
            self._comm_to_path[com_id] = path
            self._path_to_comm[tuple(path)] = com_id
            self._fastest_travel_times[(s, t)] = dist[s]

    def _run_bellman_ford(self):
        costs = self._flow.get_edge_costs()
        for s, t in self._paths.keys():
            earliest_arrivals = bellman_ford(
                t,
                costs,
                self._important_nodes[(s, t)],
                0.0
            )
            self._earliest_arrivals_to[t] = earliest_arrivals

    def _compute_flow(self):
        flow_builder = PathFlowBuilder(self.network, self._comm_to_path, self.reroute_interval)
        generator = flow_builder.build_flow()
        flow = next(generator)
        while flow.phi < self.horizon:
            flow = next(generator)

        self._flows_history.append(flow)
        self._flow = flow

    def _get_path_inflow(self, path: Path) -> RightConstant:
        s = path[0]._node_from
        path_key = tuple(path)

        if path_key in self._path_to_comm:
            com_id = self._path_to_comm[path_key]
            return self.network.commodities[com_id].sources[s]
        else:
            return RightConstant([0.0], [0.0], (0, float('inf')))

    def _set_path_inflow(self, path: Path, new_inflow: RightConstant):
        s = path[0]._node_from
        t = path[-1]._node_to
        path_key = tuple(path)

        if path_key in self._path_to_comm:
            com_id = self._path_to_comm[path_key]
            self.network.commodities[com_id].sources[s] = new_inflow
        else:
            new_com_id = len(self.network.commodities)
            self.network.add_commodity(
                {s.id: new_inflow},
                t.id,
                PredictorType.CONSTANT
            )
            self._comm_to_path[new_com_id] = path
            self._path_to_comm[path_key] = new_com_id
            self._paths[(s, t)].append(path)

    def _reassign_inflows(self):
        raise NotImplementedError

    def _get_route_avg_delays(self, normalize: bool = True):
        """
        Computes average delays for all present s-t pairs.
        If parameter normalize is set to True, delays are divided by travel times without queues.
        """
        avg_delays = {}
        for (s, t), paths in self._paths.items():
            accum_net_outflow = sum(
                (
                    self._flow.outflow[p[-1].id]._functions_dict[self._path_to_comm[tuple(p)]]
                    for p in paths
                    if self._path_to_comm[tuple(p)] in self._flow.outflow[p[-1].id]._functions_dict
                ),
                start=RightConstant([0.0], [0.0], (0, float("inf"))),
            ).integral()
            accum_net_inflow = sum(
                (
                    self._get_path_inflow(p)
                    for p in paths
                ),
                start=RightConstant([0.0], [0.0], (0, float("inf"))),
            ).integral()
            assert abs(1 - accum_net_inflow(self.horizon)/accum_net_outflow(self.horizon)) < eps
            avg_travel_time = (
                                      accum_net_inflow.integrate(0.0, self.horizon) -
                                      accum_net_outflow.integrate(0.0, self.horizon)
                              ) / accum_net_inflow(self.horizon)

            optimal_travel_time = self._earliest_arrivals_to[t][s] - identity
            opt_avg_travel_time = (integrate_with_weights(
                optimal_travel_time, self._inflows[(s, t)], 0, self.inflow_horizon
            ) / (self._inflows[(s, t)].integral()(self.inflow_horizon)))

            assert opt_avg_travel_time > 0

            avg_delays[(s, t)] = avg_travel_time - opt_avg_travel_time
            if normalize:
                avg_delays[(s, t)] /= self._fastest_travel_times[(s, t)]

        return avg_delays

    def _compute_path_metrics(self, path):
        metrics = dict()

        s = path[0]._node_from
        t = path[-1]._node_to

        costs = self._flow.get_edge_costs()
        optimal_travel_time = self._earliest_arrivals_to[t][s] - identity

        path_inflow = self._get_path_inflow(path).simplify()
        accum_path_inflow = path_inflow.integral()
        path_delay = compute_path_travel_time(path, costs) - optimal_travel_time

        metrics['avg_delay_normalized'] = (
                integrate_with_weights(path_delay, path_inflow, 0, self.inflow_horizon) /
                (accum_path_inflow(self.inflow_horizon) * self._fastest_travel_times[(s, t)])
        )

        metrics['share_of_total_inflow'] = (
                accum_path_inflow(self.inflow_horizon) /
                self._inflows[(s, t)].integral()(self.inflow_horizon)
        )

        activity_ind = get_activity_indicator(
            path_delay,
            threshold=self.delay_threshold * self._fastest_travel_times[(s, t)]
        )
        metrics['active_inflow_share'] = (
                (path_inflow * activity_ind).integral()(self.inflow_horizon) /
                accum_path_inflow(self.inflow_horizon)
        )

        return metrics

    def run(self, num_iterations, eval_every=10):
        """
        Main cycle
        """

        if self._iter == 0:
            self._initialize_paths()
            self._compute_flow()

            self._run_bellman_ford()

            print(f"Initial normalized average delays:")
            for (s, t), avg_delay in self._get_route_avg_delays(normalize=True).items():
                print(f"({s.id} -> {t.id}): {round(avg_delay, 4)}")

        while self._iter < num_iterations:

            self._reassign_inflows()
            self._compute_flow()
            self._iter += 1

            self._run_bellman_ford()

            if self._iter % eval_every == 0:
                print()
                print(f"Iterations completed: {self._iter}/{num_iterations}")
                print(f"Normalized average delays:")
                for (s, t), avg_delay in self._get_route_avg_delays(normalize=True).items():
                    print(f"({s.id} -> {t.id}): {round(avg_delay, 4)}")
                    self._path_metrics[(s, t)].append([(path, self._compute_path_metrics(path)) for path in self._paths[(s, t)]])

        merged_flow = self._flow
        # combine_commodities_with_same_sink(self.network)
        for paths in self._paths.values():
            commodities = [self._path_to_comm[tuple(p)] for p in paths]
            merged_flow = merge_commodities(merged_flow, self.network, commodities)

        return merged_flow, self._path_metrics


# class AlphaFlowIterator(BaseFlowIterator):
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


# def compute_shortest_paths_over_time(
#     earliest_arrivals: Dict[Node, PiecewiseLinear],
#     edge_costs: List[PiecewiseLinear],
#     source: Node,
#     sink: Node,
# ) -> PathsOverTime:
#     """
#     Returns the shortest paths from source to sink in the network over time.
#     """
#     shortest_paths_computed_until = 0.0
#     identity = PiecewiseLinear(
#         [shortest_paths_computed_until],
#         [shortest_paths_computed_until],
#         1.0,
#         1.0,
#         (shortest_paths_computed_until, float("inf")),
#     )
#     edge_exit_times: Dict[Edge, PiecewiseLinear] = {}
#
#     paths_over_time: PathsOverTime = PathsOverTime([], [])
#
#     while shortest_paths_computed_until < float("inf"):
#         labelled_path: List[LabelledPathEntry] = []
#
#         path_start = departure = shortest_paths_computed_until
#         # We want to find a path that is active starting from time `departure` and that is active for as long as possible (heuristically?).
#
#         # We start at the source and iteratively select the next edge of the path.
#         v = source
#         while v != sink:
#             # Select the next outgoing edge of the current node v:
#             # the edge e that is active for the longest period (starting from the arrival/departure time at v `departure`).
#             best_edge = None
#             best_active_until = None
#
#             for edge in v.outgoing_edges:
#                 edge_exit_times[edge] = identity.plus(
#                     edge_costs[edge.id]
#                 ).ensure_monotone(True)
#                 edge_delay = (
#                         earliest_arrivals[edge.node_to].compose(edge_exit_times[edge])
#                         - earliest_arrivals[v]
#                 ).simplify()
#
#                 # it can happen that delay is negative - just computational issue?
#
#                 # edge_delay is (close to) zero at times when the edge is active; otherwise it is positive.
#                 if edge_delay(departure) > eps:
#                     continue
#
#                 active_until = edge_delay.next_change_time(departure)
#
#                 if best_edge is None or active_until > best_active_until:
#                     best_edge = edge
#                     best_active_until = active_until
#
#             assert best_edge is not None
#             labelled_path.append(
#                 LabelledPathEntry(best_edge, departure, best_active_until)
#             )
#             v = best_edge.node_to
#             departure = edge_exit_times[best_edge](departure)
#
#         # Compute path_active_until
#         rest_of_path_active_until = float("inf")
#         for labelled_edge in reversed(labelled_path):
#             last_enter_time_st_rest_path_active = edge_exit_times[
#                 labelled_edge.edge
#             ].max_t_below(rest_of_path_active_until)
#             rest_of_path_active_until = min(
#                 last_enter_time_st_rest_path_active, labelled_edge.active_until
#             )
#
#         paths_over_time.add_path(
#             [label.edge for label in labelled_path],
#             Indicator.from_interval(path_start, rest_of_path_active_until)
#         )
#
#         assert shortest_paths_computed_until < rest_of_path_active_until
#
#         shortest_paths_computed_until = rest_of_path_active_until
#     return paths_over_time


def get_activity_indicator(
        delay: PiecewiseLinear,
        threshold: float = eps,
        min_interval_length: float = eps
) -> Indicator:
    """
    Returns function I(t) with value 1 if delay(t) < threshold and 0 otherwise.
    Intervals of length less than min_interval_length are avoided.
    """

    times = []
    values = []

    is_active = False
    for i in range(len(delay.times)):
        if delay.values[i] < threshold and not is_active:
            if i > 0:
                offset = (delay.values[i] - threshold) / delay.gradient(i - 1)
                interval_start = delay.times[i] - offset
            else:
                interval_start = delay.domain[0]

            is_active = True
        elif delay.values[i] > threshold and is_active:
            offset = (threshold - delay.values[i - 1]) / delay.gradient(i - 1)
            interval_end = delay.times[i-1] + offset

            is_active = False

            if interval_end - interval_start > min_interval_length:
                times += [interval_start, interval_end]
                values += [1.0, 0.0]

    if is_active:  # assuming last slope for delay can only equal to 0
        times += [interval_start]
        values += [1.0]

    if len(times) == 0 or times[0] > delay.domain[0] + eps:
        times = [delay.domain[0]] + times
        values = [0.0] + values

    return Indicator(times, values, delay.domain)


def approximate_linear(lin: PiecewiseLinear, delta: float, horizon: float) -> RightConstant:
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


def compute_path_travel_time(path: Path, costs: List[PiecewiseLinear]) -> PiecewiseLinear:
    path_exit_time = identity

    for edge in path[::-1]:
        path_exit_time = path_exit_time.compose(
            identity.plus(costs[edge.id]).ensure_monotone(True)
        )

    return path_exit_time - identity


def compute_all_active_paths(
    earliest_arrival_fcts: Dict[Node, PiecewiseLinear],
    edge_costs: List[PiecewiseLinear],
    source: Node,
    sink: Node,
    inflow_horizon: float,
    delay_threshold: float = eps,
    min_active_time: float = 1000*eps
) -> PathsOverTime:
    """
    Constructs all simple s-t paths which are active for longer than min_active_time
    """

    shortest_paths = PathsOverTime([], [])

    initial_paths = PathsOverTime(
            paths=[[]],
            activity_indicators=[Indicator.from_interval(0.0, float('inf'))]
        )
    paths_to = {source: (initial_paths, [identity])}  # stores constructed subpaths grouped by end node

    while len(paths_to) > 0:
        destinations = set(e._node_to for v in paths_to.keys() for e in v.outgoing_edges if v != sink)
        extended_paths_to = {
            v: (PathsOverTime([], []), [])
            for v in destinations
        }

        for v, (paths_to_v, exit_times) in paths_to.items():
            if v == sink:
                shortest_paths += paths_to_v
                continue
            for e in v.outgoing_edges:
                for i in range(len(paths_to_v)):  # extend each path by adding an edge
                    if e in paths_to_v.paths[i]:    # do not allow loops
                        continue
                    path = paths_to_v.paths[i] + [e]
                    exit_time = (identity + edge_costs[e.id]).compose(exit_times[i])
                    delay = earliest_arrival_fcts[e._node_to].compose(exit_time) - earliest_arrival_fcts[source]
                    indicator = get_activity_indicator(delay, threshold=delay_threshold)
                    if indicator.integral()(inflow_horizon) > min_active_time:
                        extended_paths_to[e._node_to][0].add_path(path, indicator)
                        extended_paths_to[e._node_to][1].append(exit_time)

        paths_to = {v: (paths_to_v, exit_times)
                    for v, (paths_to_v, exit_times) in extended_paths_to.items()
                    if len(paths_to_v) > 0}

    return shortest_paths


class AlphaFlowIterator(BaseFlowIterator):
    alpha_fun: Callable[[float], float]  # fraction of inflow to redistribute as function of relative delay
    approx_inflows: bool
    """ 
    For each path, an approximation of relative delay d is computed. 
    Then, a fraction alpha_fun(d) of its inflow is redistributed to active paths.
    """
    def __init__(self,
                 network: Network,
                 reroute_interval: float,
                 horizon: float,
                 inflow_horizon: float,
                 alpha_fun: Callable[[float], float] = lambda d: 0.01,
                 delay_threshold: float = eps,
                 approx_inflows: bool = True):
        super().__init__(network, reroute_interval, horizon, inflow_horizon, delay_threshold)
        self.alpha_fun = alpha_fun
        self.approx_inflows = approx_inflows

    def _determine_new_inflow(self,
                              route: Tuple[Node, Node],
                              costs: List[PiecewiseLinear]
                              ) -> RightConstant:

        new_inflow = RightConstant([0.0], [0.0], (0, float('inf')))

        opt_travel_time = self._earliest_arrivals_to[route[1]][route[0]] - identity
        # opt_travel_time_approx = approximate_linear(opt_travel_time, self.reroute_interval, self.inflow_horizon)

        for path in self._paths[route]:
            path_inflow = self._get_path_inflow(path)

            # travel_time = compute_path_travel_time(path, costs)
            path_delay = compute_path_travel_time(path, costs) - opt_travel_time
            path_delay_approx = approximate_linear(path_delay, self.reroute_interval, self.inflow_horizon)
            alpha_vals = [
                self.alpha_fun(d / self._fastest_travel_times[route]) if d > 0 else self.alpha_fun(0)
                for d in path_delay_approx.values
            ]
            alpha = RightConstant(path_delay_approx.times, alpha_vals, (0, float('inf')))

            inflow_change = alpha * path_inflow
            new_inflow += inflow_change
            self._set_path_inflow(path, path_inflow - inflow_change)

        return new_inflow.simplify()

    def _assign_new_paths(self, active_paths: PathsOverTime, new_inflow: RightConstant):

        # could be not 1, shouldn't be 0
        n_active_paths = RightConstant.sum(active_paths.activity_indicators)
        uniform_factor = n_active_paths.invert().restrict((0, self.inflow_horizon))
        assert float('inf') not in uniform_factor.values

        for i in range(len(active_paths)):
            path = active_paths.paths[i]
            current_inflow = self._get_path_inflow(path)

            path_prob = active_paths.activity_indicators[i] * uniform_factor
            new_path_inflow = current_inflow + new_inflow * path_prob
            self._set_path_inflow(path, new_path_inflow)

    def _reassign_inflows(self):

        costs = self._flow.get_edge_costs()

        def process_route(route):
            s, t = route
            active_paths = compute_all_active_paths(
                self._earliest_arrivals_to[t],
                costs,
                s,
                t,
                self.inflow_horizon,
                delay_threshold=self.delay_threshold * self._fastest_travel_times[route]
            )
            new_inflow = self._determine_new_inflow(route, costs)
            self._assign_new_paths(active_paths, new_inflow)

            if self.approx_inflows:
                for path in self._paths[route]:
                    inflow = self._get_path_inflow(path)
                    approximation = inflow.project_to_grid(self.reroute_interval, self.inflow_horizon)
                    self._set_path_inflow(path, approximation)

        for route in self._paths.keys():
            process_route(route)

        # with ThreadPoolExecutor() as executor:
        #     executor.map(process_route, self._paths.keys())