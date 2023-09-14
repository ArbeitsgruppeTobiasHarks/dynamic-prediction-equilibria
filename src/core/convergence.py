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
from visualization.to_json import merge_commodities, to_visualization_json

from concurrent.futures import ThreadPoolExecutor


Path = List[Edge]


@dataclass
class LabelledPathEntry:
    """
    This class represents one edge of a labelled path.
    Each such edge is labelled with a starting time and an active_until time.
    """

    edge: Edge
    starting_time: float
    active_until: float


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


class BaseFlowIterator:
    network: Network
    reroute_interval: float
    horizon: float
    inflow_horizon: float
    _iter: int
    _flows: List[DynamicFlow]
    _comm_to_path: Dict[int, Path]
    _path_to_comm: Dict[Tuple, int]
    _paths: Dict[Tuple[Node, Node], List[Path]] # s-t paths
    _important_nodes: Dict[Tuple[Node, Node], Collection[Node]]

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        inflow_horizon: float
    ):
        assert all(len(c.sources) == 1 for c in network.commodities)
        self.network = network
        self.reroute_interval = reroute_interval
        self.horizon = horizon
        self.inflow_horizon = inflow_horizon
        self._iter = 0
        self._flows = []
        self._paths = dict()
        self._important_nodes = dict()

        for i, com in enumerate(network.commodities):
            s = next(iter(com.sources.keys()))
            t = com.sink
            self._paths[(s,t)] = []
            self._important_nodes[(s, t)] = (
                    network.graph.get_nodes_reaching(t) & network.graph.get_reachable_nodes(s)
            )

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

            self._paths[(s,t)].append(path)
            self._comm_to_path[com_id] = path
            self._path_to_comm[tuple(path)] = com_id

    def _compute_flow(self):
        flow_builder = PathFlowBuilder(self.network, self._comm_to_path, self.reroute_interval)
        generator = flow_builder.build_flow()
        flow = next(generator)
        while flow.phi < self.horizon:
            flow = next(generator)

        return flow

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

    def _reassign_inflows(self, flow):
        raise NotImplementedError

    def _get_route_avg_travel_times(self, flow: DynamicFlow, relative: bool = True):
        """
        Computes average travel times for all present s-t pairs.
        If parameter relative is set to True, travel_times are divided by optimal values.
        """
        avg_travel_times = {}
        for (s, t), paths in self._paths.items():
            c_ids = [self._path_to_comm[tuple(p)] for p in paths]
            accum_net_outflow = sum(
                (
                    flow.outflow[e.id]._functions_dict[c_id]
                    for c_id in c_ids
                    for e in t.incoming_edges
                    if c_id in flow.outflow[e.id]._functions_dict
                ),
                start=RightConstant([0.0], [0.0], (0, float("inf"))),
            ).integral()
            accum_net_inflow = sum(
                (
                    inflow
                    for c_id in c_ids
                    for inflow in flow._network.commodities[c_id].sources.values()
                ),
                start=RightConstant([0.0], [0.0], (0, float("inf"))),
            ).integral()
            assert abs(1 - accum_net_inflow(self.horizon)/accum_net_outflow(self.horizon)) < eps
            avg_travel_times[(s,t)] = (
                                      accum_net_inflow.integrate(0.0, self.horizon)
                                      - accum_net_outflow.integrate(0.0, self.horizon)
                              ) / accum_net_inflow(self.horizon)

            if relative: # divide by optimal travel times
                com = self.network.commodities[min(c_ids)]
                opt_avg_travel_time = calculate_optimal_average_travel_time(
                        flow, self.network, self.inflow_horizon, self.horizon, com
                )
                assert opt_avg_travel_time > 0
                avg_travel_times[(s, t)] /= opt_avg_travel_time

        return avg_travel_times

    def run(self, num_iterations, eval_every=10):
        """
        Main cycle
        """

        if self._iter == 0:
            self._initialize_paths()
            flow = self._compute_flow()
            self._flows.append(flow)

            print(f"Initial relative average travel times:")
            for (s, t), avg_travel_time in self._get_route_avg_travel_times(flow, relative=True).items():
                print(f"({s.id} -> {t.id}): {round(avg_travel_time, 4)}")

        while self._iter < num_iterations:
            self._reassign_inflows(flow)
            flow = self._compute_flow()
            self._flows.append(flow)
            self._iter += 1

            if self._iter % eval_every == 0:
                print()
                print(f"Iterations completed: {self._iter}/{num_iterations}")
                print(f"Relative average travel times:")
                for (s, t), avg_travel_time in self._get_route_avg_travel_times(flow, relative=True).items():
                    print(f"({s.id} -> {t.id}): {round(avg_travel_time, 4)}")

        merged_flow = self._flows[-1]
        # combine_commodities_with_same_sink(self.network)
        for paths in self._paths.values():
            commodities = [self._path_to_comm[tuple(p)] for p in paths]
            merged_flow = merge_commodities(merged_flow, self.network, commodities)

        return merged_flow


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


def compute_shortest_paths_over_time(
    earliest_arrival_fcts: Dict[Node, PiecewiseLinear],
    edge_costs: List[PiecewiseLinear],
    source: Node,
    sink: Node,
) -> PathsOverTime:
    """
    Returns the shortest paths from source to sink in the network
    over time.
    """
    shortest_paths_computed_until = 0.0
    identity = PiecewiseLinear(
        [shortest_paths_computed_until],
        [shortest_paths_computed_until],
        1.0,
        1.0,
        (shortest_paths_computed_until, float("inf")),
    )
    edge_exit_times: Dict[Edge, PiecewiseLinear] = {}

    paths_over_time: PathsOverTime = PathsOverTime([], [])

    while shortest_paths_computed_until < float("inf"):
        labelled_path: List[LabelledPathEntry] = []

        path_start = departure = shortest_paths_computed_until
        # We want to find a path that is active starting from time `departure` and that is active for as long as possible (heuristically?).

        # We start at the source and iteratively select the next edge of the path.
        v = source
        while v != sink:
            # Select the next outgoing edge of the current node v:
            # the edge e that is active for the longest period (starting from the arrival/departure time at v `departure`).
            best_edge = None
            best_active_until = None

            for edge in v.outgoing_edges:
                edge_exit_times[edge] = identity.plus(
                    edge_costs[edge.id]
                ).ensure_monotone(True)
                edge_delay = (
                    earliest_arrival_fcts[edge.node_to].compose(edge_exit_times[edge])
                    - earliest_arrival_fcts[v]
                ).simplify()

                # it can happen that delay is negative - just computational issue?

                # edge_delay is (close to) zero at times when the edge is active; otherwise it is positive.
                if edge_delay(departure) > eps:
                    continue

                active_until = edge_delay.next_change_time(departure)

                if best_edge is None or active_until > best_active_until:
                    best_edge = edge
                    best_active_until = active_until

            assert best_edge is not None
            labelled_path.append(
                LabelledPathEntry(best_edge, departure, best_active_until)
            )
            v = best_edge.node_to
            departure = edge_exit_times[best_edge](departure)

        # Compute path_active_until
        rest_of_path_active_until = float("inf")
        for labelled_edge in reversed(labelled_path):
            last_enter_time_st_rest_path_active = edge_exit_times[
                labelled_edge.edge
            ].max_t_below(rest_of_path_active_until)
            rest_of_path_active_until = min(
                last_enter_time_st_rest_path_active, labelled_edge.active_until
            )

        paths_over_time.add_path(
            [label.edge for label in labelled_path],
            Indicator.from_interval(path_start, rest_of_path_active_until)
        )

        assert shortest_paths_computed_until < rest_of_path_active_until

        shortest_paths_computed_until = rest_of_path_active_until
    return paths_over_time


def get_activity_indicator(delay: PiecewiseLinear) -> Indicator:
    """"
    Returns function with value 1 if delay < eps and 0 otherwise
    """

    times = []
    values = []

    is_active = False
    for i in range(len(delay.times)):
        if delay.values[i] < eps and not is_active:
            interval_start = delay.times[i]
            is_active = True
        elif delay.values[i] > eps and is_active:
            interval_end = delay.times[i-1]
            is_active = False
            if interval_end - interval_start > 1000 * eps:  # avoiding too short intervals
                times += [interval_start, interval_end]
                values += [1.0, 0.0]

    if delay.values[-1] < eps:  # assuming last slope for delay can only equal to 0
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


# def compute_exit_time(costs: List[PiecewiseLinear], path: Path) -> PiecewiseLinear:
#     """
#     Computes path exit time as a function
#     """
#     identity = PiecewiseLinear(
#         [0.0], [0.0], first_slope=1, last_slope=1, domain=(0, float("inf"))
#     )
#     path_exit_time = identity
#
#     for edge in path[::-1]:
#         path_exit_time = path_exit_time.compose(
#             identity.plus(costs[edge.id]).ensure_monotone(True)
#         )
#
#     return path_exit_time


def compute_all_shortest_paths(
    earliest_arrival_fcts: Dict[Node, PiecewiseLinear],
    edge_costs: List[PiecewiseLinear],
    source: Node,
    sink: Node,
    inflow_horizon: float,
    activity_threshold: float = 1000*eps
) -> PathsOverTime:
    """
    Constructs all simple s-t paths which are active for longer than activity_threshold
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
                    indicator = get_activity_indicator(delay)
                    if indicator.integral()(inflow_horizon) > activity_threshold:
                        extended_paths_to[e._node_to][0].add_path(path, indicator)
                        extended_paths_to[e._node_to][1].append(exit_time)

        paths_to = {v: (paths_to_v, exit_times)
                    for v, (paths_to_v, exit_times) in extended_paths_to.items()
                    if len(paths_to_v) > 0}

    return shortest_paths


class BetaFlowIterator(BaseFlowIterator):
    alpha_fun: Callable[[float], float] # fraction of inflow to redistribute as function of relative delay
    beta: float # fraction of inflow for uniform redistribution
    approx_inflows: bool
    """
    Inflow redistribution dependent on relative delay 
    """
    def __init__(self,
                 network: Network,
                 reroute_interval: float,
                 horizon: float,
                 inflow_horizon: float,
                 alpha_fun: Callable[[float], float] = lambda d: 0.01,
                 beta: float = 1.0,
                 approx_inflows: bool = True):
        super().__init__(network, reroute_interval, horizon, inflow_horizon)
        self.alpha_fun = alpha_fun
        self.beta = beta
        self.approx_inflows = approx_inflows

    def _compute_new_inflow(self, route, earliest_arrival, costs) -> RightConstant:

        new_inflow = RightConstant([0.0], [0.0], (0, float('inf')))
        opt_travel_time_approx = approximate_linear(earliest_arrival - identity, self.reroute_interval, self.inflow_horizon)

        for path in self._paths[route]:
            path_inflow = self._get_path_inflow(path)

            arrival = identity
            for edge in path[::-1]:
                arrival = arrival.compose(
                    identity.plus(costs[edge.id]).ensure_monotone(True)
                )

            travel_time_approx = approximate_linear(arrival - identity, self.reroute_interval, self.inflow_horizon)
            alpha_vals = [
                self.alpha_fun(tt/opt_tt - 1) if tt > opt_tt else self.alpha_fun(0)
                for tt, opt_tt in zip(travel_time_approx.values, opt_travel_time_approx.values)
            ]
            alpha = RightConstant(travel_time_approx.times, alpha_vals, (0, float('inf')))

            inflow_change = alpha * path_inflow
            new_inflow += inflow_change
            self._set_path_inflow(path, path_inflow - inflow_change)

        return new_inflow.simplify()

    def _assign_new_paths(self, route, shortest_paths, new_inflow):

        uniform_factor = RightConstant.sum(shortest_paths.activity_indicators).invert() # could be not 1, shouldn't be 0

        # active_inflow = sum(
        #     self.network.commodities[com_id].sources[route[0]] * shortest_paths.activity_indicators[i]
        #     for i, com_id in path_commodity.items() if com_id is not None
        # )
        # proportional_factor = 1.0 / active_inflow.integral()(self.horizon) if active_inflow != 0 else 0
        # proportional_factor *= Indicator.from_interval(0, float('inf'))

        for i in range(len(shortest_paths)):
            path = shortest_paths.paths[i]
            current_inflow = self._get_path_inflow(path)

            path_prob = self.beta * shortest_paths.activity_indicators[i] * uniform_factor
            # if com_id is not None:
            #     path_prob += ((1 - self.beta)
            #                   * self.network.commodities[com_id].sources[route[0]].integral()(self.horizon)
            #                   * proportional_factor)

            new_path_inflow = current_inflow + new_inflow * path_prob
            self._set_path_inflow(path, new_path_inflow)

    def _reassign_inflows(self, flow):

        costs = flow.get_edge_costs()

        def process_route(route):
            s, t = route
            earliest_arrivals = bellman_ford(
                t,
                costs,
                self._important_nodes[route],
                0.0,
                float("inf"),
            )

            shortest_paths = compute_all_shortest_paths(earliest_arrivals, costs, s, t, self.inflow_horizon)

            new_inflow = self._compute_new_inflow(route, earliest_arrivals[s], costs)
            self._assign_new_paths(route, shortest_paths, new_inflow)

            if self.approx_inflows:
                for path in self._paths[route]:
                    inflow = self._get_path_inflow(path)
                    self._set_path_inflow(path, inflow.project_to_grid(self.reroute_interval))

        # for route in self._route_users.keys():
        #     process_route(route)

        with ThreadPoolExecutor() as executor:
            executor.map(process_route, self._paths.keys())