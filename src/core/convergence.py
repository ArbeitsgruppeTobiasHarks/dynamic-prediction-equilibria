import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Collection

from core.bellman_ford import bellman_ford
from core.dijkstra import reverse_dijkstra
from core.dynamic_flow import DynamicFlow
from core.graph import Edge, Node
from core.machine_precision import eps
from core.network import Network
from core.path_flow_builder import PathFlowBuilder
from core.predictors.predictor_type import PredictorType
from eval.evaluate import calculate_optimal_average_travel_time
from ml.generate_queues import generate_queues_and_edge_loads
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant
from visualization.to_json import merge_commodities, to_visualization_json

from concurrent.futures import ThreadPoolExecutor


Path = List[Edge]


class PathsOverTime:
    """
    This class represents a set of paths over time.
    A path paths[i] is active until time times[i] (starting from time times[i-1] or -inf).
    """

    times: List[float]
    paths: List[Path]

    def __init__(self):
        self.times = []
        self.paths = []

    def add_path(self, starting_time: float, path: Path):
        self.times.append(starting_time)
        self.paths.append(path)


@dataclass
class LabelledPathEntry:
    """
    This class represents one edge of a labelled path.
    Each such edge is labelled with a starting time and an active_until time.
    """

    edge: Edge
    starting_time: float
    active_until: float

# def evaluate_path(costs: List[PiecewiseLinear], path: Path) -> PiecewiseLinear:
#     """ "
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


class BaseFlowIterator:
    network: Network
    reroute_interval: float
    horizon: float
    num_iterations: int
    _iter: int
    _flows: List[DynamicFlow]
    _route_users: Dict[Tuple[Node, Node], List[int]]  # commodities with common s-t pair
    _inflows: Dict[Tuple[Node, Node], RightConstant]  # route inflows
    _important_nodes: Dict[Tuple[Node, Node], Collection[Node]]

    def __init__(
        self,
        network: Network,
        reroute_interval: float,
        horizon: float,
        num_iterations: int = 100,
    ):
        assert all(len(c.sources) == 1 for c in network.commodities)
        self.network = network
        self.reroute_interval = reroute_interval
        self.horizon = horizon
        self.num_iterations = num_iterations
        self._iter = 0
        self._flows = []
        self._inflows = {}
        self._route_users = {}
        self._important_nodes = {}

        for i, com in enumerate(network.commodities):
            t = com.sink
            s, inflow = next(iter(com.sources.items()))
            self._inflows[(s, t)] = inflow
            self._route_users[(s, t)] = [i]
            self._important_nodes[(s, t)] = (
                    network.graph.get_nodes_reaching(t) & network.graph.get_reachable_nodes(s)
            )

        self._paths = {i: None for i in range(len(network.commodities))}
        assert len(self._route_users) == len(self._paths)

    def _initialize_paths(self):
        """
        Assigns each commodity the shortest path under assumption there are no queues
        """
        for s, t in self._route_users:
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

            for i in self._route_users[(s, t)]:
                self._paths[i] = path

    def _compute_flow(self):
        flow_builder = PathFlowBuilder(self.network, self._paths, self.reroute_interval)
        generator = flow_builder.build_flow()
        flow = next(generator)
        while flow.phi < self.horizon:
            flow = next(generator)

        return flow

    def _reassign_inflows(self, costs):
        raise NotImplementedError

    def _iteration(self):
        """
        Computes current flow and edge costs, then reassigns the inflows
        """

        flow = self._compute_flow()

        costs = [
            PiecewiseLinear(
                flow.queues[e].times,
                [
                    flow._network.travel_time[e] + v / flow._network.capacity[e]
                    for v in flow.queues[e].values
                ],
                flow.queues[e].first_slope / flow._network.capacity[e],
                flow.queues[e].last_slope / flow._network.capacity[e],
                domain=(0.0, float("inf")),
            ).simplify()
            for e in range(len(flow.queues))
        ]

        self._reassign_inflows(costs)

        return flow

    def _avg_travel_times(self, flow, relative=True):
        avg_travel_times = {}
        for (s, t), c_ids in self._route_users.items():
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
            avg_travel_times[(s,t)] = (
                                      accum_net_inflow.integrate(0.0, self.horizon)
                                      - accum_net_outflow.integrate(0.0, self.horizon)
                              ) / accum_net_inflow(self.horizon)

            if relative: # divide by optimal travel times
                com = self.network.commodities[min(c_ids)]
                inflow_horizon = self._inflows[(s,t)].times[-1]
                opt_avg_travel_time = calculate_optimal_average_travel_time(
                        flow, self.network, inflow_horizon, self.horizon, com
                )
                assert opt_avg_travel_time > 0
                avg_travel_times[(s, t)] /= opt_avg_travel_time

        return avg_travel_times

    def run(self, eval_every=10):

        self._initialize_paths()
        flow = self._iteration()
        avg_travel_times = self._avg_travel_times(flow, relative=True)
        self._flows.append(flow)
        self._iter += 1

        print(f"Initial relative average travel times:")
        for (s, t), avg_travel_time in avg_travel_times.items():
            print(f"({s.id} -> {t.id}): {round(avg_travel_time, 4)}")
        print()

        while self._iter < self.num_iterations:
            flow = self._iteration()
            self._flows.append(flow)
            self._iter += 1

            if self._iter % eval_every == 0:
                avg_travel_times = self._avg_travel_times(flow, relative=True)

                print(f"Iterations completed: {self._iter}/{self.num_iterations}")
                print(f"Relative average travel times:")
                for (s,t), avg_travel_time in avg_travel_times.items():
                    print(f"({s.id} -> {t.id}): {round(avg_travel_time, 4)}")
                print()

        merged_flow = self._flows[-1]
        combine_commodities_with_same_sink(self.network)
        for route, commodities in self._route_users.items():
            merged_flow = merge_commodities(merged_flow, self.network, commodities)

        return merged_flow


class AlphaFlowIterator(BaseFlowIterator):
    alpha: float    # convergence rate from (0,1) interval
    approx_inflows: bool    # option to project inflows to regular grid after each iteration
    """
    At each iteration, a constant fraction of inflow is redistributed to optimal path from previous iteration
    """

    def __init__(self,
                 network: Network,
                 reroute_interval: float,
                 horizon: float,
                 num_iterations: int = 100,
                 alpha: float = 0.01,
                 approx_inflows: bool = True):

        super().__init__(network, reroute_interval, horizon, num_iterations)
        assert 0 < alpha < 1
        self.alpha = alpha
        self.approx_inflows = approx_inflows

    def _assign_new_paths(self, route, p_o_t):

        for com_id in self._route_users[route]:
            self.network.commodities[com_id].sources[route[0]] *= 1 - self.alpha

        for i in range(len(p_o_t.paths)):
            new_path = p_o_t.paths[i]
            phi = p_o_t.times[i-1] if i > 0 else 0.0
            phi_next = p_o_t.times[i]

            new_inflow = self._inflows[route].restrict((phi, phi_next)) * self.alpha

            if new_inflow.integral()(self.horizon) > eps:
                already_present = False
                for com_id in self._route_users[route]:
                    if self._paths[com_id] == new_path:
                        already_present = True
                        self.network.commodities[com_id].sources[route[0]] += new_inflow
                        break

                if not already_present:
                    new_com_id = len(self.network.commodities)
                    self._paths[new_com_id] = new_path
                    self._route_users[route].append(new_com_id)
                    self.network.add_commodity(
                        {route[0].id: new_inflow},
                        route[1].id,
                        PredictorType.CONSTANT,
                    )

    def _reassign_inflows(self, costs):

        def process_route(route):
            s, t = route
            earliest_arrivals = bellman_ford(
                t,
                costs,
                self._important_nodes[route],
                0.0,
                float("inf"),
            )
            p_o_t = compute_shortest_paths_over_time(earliest_arrivals, costs, s, t)

            self._assign_new_paths(route, p_o_t)

            if self.approx_inflows:
                for com_id in self._route_users[route]:
                    inflow = self.network.commodities[com_id].sources[route[0]]
                    self.network.commodities[com_id].sources[route[0]] = inflow.project_to_grid(
                        self.reroute_interval).simplify()

        with ThreadPoolExecutor() as executor:
            executor.map(process_route, self._route_users.keys())


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

    paths_over_time: PathsOverTime = PathsOverTime()

    while shortest_paths_computed_until < float("inf"):
        labelled_path: List[LabelledPathEntry] = []

        departure = shortest_paths_computed_until
        # We want to find a path that is active starting from time `departure` and that is active for as long as possible (heuristically?).

        # We start at the source and iteratively select the next edge of the path.
        v = source
        while v != sink:
            # Select the next outgoing edge of the current node v:
            # the edge e that is active for the longest period (starting from the arrival/departure time at v `departure`).
            best_edge = None
            best_edge_active_until = None

            for edge in v.outgoing_edges:
                edge_exit_times[edge] = identity.plus(
                    edge_costs[edge.id]
                ).ensure_monotone(True)
                edge_delay = (
                    earliest_arrival_fcts[edge.node_to].compose(edge_exit_times[edge])
                    - earliest_arrival_fcts[v]
                )
                # edge_delay is (close to) zero at times when the edge is active; otherwise it is positive.
                if edge_delay(departure) > eps:
                    continue

                active_until = edge_delay.next_change_time(departure)
                if best_edge is None or active_until > best_edge_active_until:
                    best_edge = edge
                    best_edge_active_until = active_until

            assert best_edge is not None
            labelled_path.append(
                LabelledPathEntry(best_edge, departure, best_edge_active_until)
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
            rest_of_path_active_until, [label.edge for label in labelled_path]
        )

        assert shortest_paths_computed_until < rest_of_path_active_until

        shortest_paths_computed_until = rest_of_path_active_until
    return paths_over_time