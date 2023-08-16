import os
from typing import Dict, List, Tuple

from core.machine_precision import eps
from core.graph import Node, Edge
from core.network import Network
from core.dynamic_flow import DynamicFlow
from core.predictors.predictor_type import PredictorType
from ml.generate_queues import generate_queues_and_edge_loads
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from core.path_flow_builder import PathFlowBuilder
from visualization.to_json import merge_commodities, to_visualization_json
from core.dijkstra import reverse_dijkstra
from core.bellman_ford import bellman_ford
from utilities.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant
from visualization.to_json import merge_commodities

Path = List[Edge]


def evaluate_path(costs: List[PiecewiseLinear], path: Path) -> PiecewiseLinear:
    """"
    Computes path exit time as a function
    """
    identity = PiecewiseLinear(
        [0.0], [0.0], first_slope=1, last_slope=1, domain=(0, float("inf"))
    )
    path_exit_time = identity

    for edge in path[::-1]:
        path_exit_time = path_exit_time.compose(
            identity.plus(costs[edge.id]).ensure_monotone(True)
        )

    return path_exit_time


def get_shortest_path_at(
        earliest_arrivals: Dict[Node, PiecewiseLinear],
        costs: List[PiecewiseLinear],
        at: float,
        source: Node) -> Path:
    """
    Computes shortest path at given moment
    """
    v = source
    t_arr = earliest_arrivals[source](at)
    path = []
    tau = at
    while tau < t_arr - eps:
        for e in v.outgoing_edges:
            if abs(earliest_arrivals[e._node_to](tau + costs[e.id](tau)) - t_arr) < eps:
                path.append(e)
                v = e._node_to
                tau = tau + costs[e.id](tau)
                break

    return path


class FlowIterator:
    network: Network
    reroute_interval: float
    horizon: float
    num_iterations: int
    alpha: float  # convergence rate from (0,1) interval
    _iter: int
    _flows: List[DynamicFlow]
    _route_users: Dict[Tuple[Node, Node], List[int]]  # commodities with common s-t pair
    _inflows: Dict[Tuple[Node, Node], RightConstant]  # route inflows
    _path_used: Dict[int, Path]  # paths assigned to commodities


    def __init__(
            self,
            network: Network,
            reroute_interval: float,
            horizon: float,
            num_iterations: int = 100,
            alpha: float = 0.1
    ):
        assert all(len(c.sources) == 1 for c in network.commodities)
        assert 0 < alpha < 1
        self.network = network
        self.reroute_interval = reroute_interval
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.alpha = alpha
        self._iter = 0
        self._flows = []
        self._inflows = {}
        self._route_users = {}

        for i, com in enumerate(network.commodities):
            t = com.sink
            for s, inflow in com.sources.items():
                self._inflows.update({(s,t): inflow})
                self._route_users.update({(s,t): [i]})
        self._paths = {i: None for i in range(len(network.commodities))}
        assert len(self._route_users) == len(self._paths)

    def _initialize_paths(self):
        """
        Assigns each commodity the shortest path not regarding the queues
        """
        for s, t in self._route_users:
            dist = reverse_dijkstra(t, self.network.travel_time, self.network.graph.get_nodes_reaching(t))
            v = s
            path = []
            while v != t:
                for e in v.outgoing_edges:
                    if dist[v] - dist[e._node_to] == self.network.travel_time[e.id]:
                        path.append(e)
                        v = e._node_to
                        break

            for i in self._route_users[(s,t)]:
                self._paths[i] = path

    def _compute_flow(self):
        flow_builder = PathFlowBuilder(self.network, self._paths, self.reroute_interval)
        generator = flow_builder.build_flow()
        flow = next(generator)
        while flow.phi < self.horizon:
            flow = next(generator)

        return flow

    def _get_optimal_paths(self, important_nodes, costs, route):
        """
        Computes optimal paths over time for given route (s,t) from cost functions
        """
        s, t = route

        earliest_arrivals = bellman_ford(
            t,
            costs,
            important_nodes,
            0.0,
            float("inf"),
        )

        phi = 0.0
        best_paths = []
        while phi < self.horizon:
            # find optimal path on the next timestep to guarantee improvement
            path = get_shortest_path_at(earliest_arrivals, costs, phi + self.reroute_interval, s)
            best_paths.append((phi, path))

            diff = (evaluate_path(costs, path) - earliest_arrivals[s]).restrict((phi, float('inf'))).simplify()
            if max(diff.values) < eps:
                break
            else:
                assert diff.values[0] < eps
                for idx in range(len(diff.times)-1):
                    if diff.values[idx+1] > 1000*eps:
                        phi = diff.times[idx]
                        break

        return best_paths

    def _assign_new_paths(self, route, new_paths):
        for com_id in self._route_users[route]:
            self.network.commodities[com_id].sources[route[0]] *= (1 - self.alpha)

        for i in range(len(new_paths)):
            phi, new_path = new_paths[i]
            phi_next = new_paths[i+1][0] if i+1 < len(new_paths) else float('inf')

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

    def _iterate(self):
        """
        Performs one iteration
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

        for s, t in self._route_users.keys():
            important_nodes = flow._network.graph.get_nodes_reaching(t)
            best_paths = self._get_optimal_paths(important_nodes, costs, (s, t))
            self._assign_new_paths((s, t), best_paths)

        return flow

    def run(self):

        self._initialize_paths()
        merged_flow: DynamicFlow

        while self._iter < self.num_iterations:

            flow = self._iterate()

            self._flows.append(flow)
            self._iter += 1

            print(f"Iterations completed: {self._iter}/{self.num_iterations};")
            print(f"Average travel times:")
            for (s,t), c_ids in self._route_users.items():
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
                        for inflow in flow._network.commodities[c_id].sources.values()),
                    start=RightConstant([0.0], [0.0], (0, float("inf"))),
                ).integral()
                avg_travel_time = (
                                          accum_net_inflow.integrate(0.0, self.horizon)
                                          - accum_net_outflow.integrate(0.0, self.horizon)
                                  ) / accum_net_inflow(self.horizon)
                print(f"({s.id}, {t.id}): {avg_travel_time}")

        return self._flows[-1]
