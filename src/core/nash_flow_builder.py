from __future__ import annotations

from dataclasses import dataclass
from test.test_interpolate import plot as plot_linear
from test.test_right_constant import plot_many as plot_constant
from typing import Dict, List, Tuple

from core.bellman_ford import bellman_ford
from core.dijkstra import reverse_dijkstra
from core.dynamic_flow import DynamicFlow
from core.graph import Edge, Node
from core.machine_precision import eps
from core.network import Network
from core.network_loader import NetworkLoader
from utilities.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant

Path = List[Edge]
PathInflow = Tuple[Path, RightConstant]


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


class PathInflows:
    _path_inflows: List[Tuple[Path, RightConstant]]

    def __init__(self) -> None:
        self._path_inflows = []

    def of(self, path: Path) -> RightConstant:
        for other_path, inflow in self._path_inflows:
            if other_path == path:
                return inflow
        return RightConstant([0], [0], (0, float("inf")))

    def set(self, path: Path, new_inflow: RightConstant) -> None:
        for i, (other_path, _) in enumerate(self._path_inflows):
            if other_path == path:
                self._path_inflows[i] = (path, new_inflow)
                return
        self._path_inflows.append((path, new_inflow))


class NashFlowBuilder:
    network: Network
    _built: bool

    def __init__(self, network: Network, iterations: int = 10):
        self.network = network
        assert all(len(c.sources) == 1 for c in network.commodities)
        self._built = False
        self._iterations = iterations

    def _get_initial_path_dist(self) -> List[List[PathInflow]]:
        costs = self.network.travel_time

        initial_path_dist: List[List[Tuple[Path, RightConstant]]] = []

        for i, commodity in enumerate(self.network.commodities):
            source = next(iter(commodity.sources.keys()))
            sink = commodity.sink
            dist = reverse_dijkstra(sink, costs, set(self.network.graph.nodes.values()))
            assert source in dist.keys()
            # Find a single shortest path from source to sink
            path = []
            v = source
            while v != sink:
                edge = min(
                    v.outgoing_edges,
                    key=lambda e: dist[e.node_to] + costs[e.id] - dist[v],
                )
                path.append(edge)
                v = edge.node_to

            initial_path_dist.append([(path, commodity.sources[source])])
        return initial_path_dist

    def compute_shortest_paths_over_time(
        self,
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
                # Select the next edge of the 
                best_edge = None
                best_edge_active_until = None

                for edge in v.outgoing_edges:
                    edge_exit_times[edge] = identity.plus(
                        edge_costs[edge.id]
                    ).ensure_monotone(True)
                    edge_delay = (
                        earliest_arrival_fcts[edge.node_to].compose(
                            edge_exit_times[edge]
                        )
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

    def compute_delay(
        self, flow: DynamicFlow, path_dist: List[List[PathInflow]]
    ) -> List[List[Tuple[Path, PathsOverTime, PiecewiseLinear]]]:
        """
        Assumes that the paths have the same source and sink.
        TODO: This function can be optimized (a lot?).
        """

        results = []

        for path_inflows in path_dist:
            result: List[
                Tuple[Path, PathsOverTime, PiecewiseLinear]
            ] = []  # (path, paths_over_time, delay)
            for path, _ in path_inflows:
                od_pair = path[0].node_from, path[-1].node_to
                # Compute earliest arrival times
                costs = [
                    PiecewiseLinear(
                        flow.queues[e].times,
                        [
                            flow._network.travel_time[e]
                            + v / flow._network.capacity[e]
                            for v in flow.queues[e].values
                        ],
                        flow.queues[e].first_slope / flow._network.capacity[e],
                        flow.queues[e].last_slope / flow._network.capacity[e],
                        domain=(0.0, float("inf")),
                    ).simplify()
                    for e in range(len(flow.queues))
                ]
                cur_earliest_arrival_fcts = bellman_ford(
                    od_pair[1],
                    costs,
                    flow._network.graph.get_nodes_reaching(od_pair[1]),
                    0.0,
                    float("inf"),
                )

                cur_best_paths = self.compute_shortest_paths_over_time(
                    cur_earliest_arrival_fcts, costs, od_pair[0], od_pair[1]
                )

                identity = PiecewiseLinear(
                    [0.0], [0.0], first_slope=1, last_slope=1, domain=(0, float("inf"))
                )
                path_exit_time = identity
                for edge in path:
                    path_exit_time = path_exit_time.compose(
                        identity.plus(costs[edge.id]).ensure_monotone(True)
                    )

                path_delay = (
                    path_exit_time - cur_earliest_arrival_fcts[path[0].node_from]
                )

                result.append((path, cur_best_paths, path_delay))
            results.append(result)
        return results

    def iterate(
        self,
        iteration: int,
        prev_flow: DynamicFlow,
        prev_path_dist: List[List[PathInflow]],
    ) -> Tuple[DynamicFlow, List[List[PathInflow]]]:
        assert iteration >= 1

        fraction = 2 / (iteration + 1)
        new_path_dist: List[List[PathInflow]] = []
        results = self.compute_delay(prev_flow, prev_path_dist)
        for i, _ in enumerate(self.network.commodities):
            new_path_inflows = PathInflows()
            result = results[i]
            for path_idx, (path, shortest_paths_over_time, delay) in enumerate(result):
                prev_inflow = prev_path_dist[i][path_idx][1]
                delayed_inflow = prev_inflow * RightConstant.characteristic_of(delay)
                fraction_of_delayed_inflow = (fraction * delayed_inflow).simplify()
                # delayed_inflow is a piecewise constant function that is 0 wherever there is no delay and the original inflow otherwise.
                # We want to shift a fraction of the delayed, original flow to a shortest path.
                new_path_inflow = new_path_inflows.of(path)
                new_path_inflow = (
                    new_path_inflow + prev_inflow - fraction_of_delayed_inflow
                ).simplify()
                new_path_inflows.set(path, new_path_inflow)

                assert (
                    fraction_of_delayed_inflow.times[0]
                    == fraction_of_delayed_inflow.domain[0]
                )
                shortest_path_idx = 0
                for j in range(len(fraction_of_delayed_inflow.times)):
                    if fraction_of_delayed_inflow.values[j] > 0:
                        already_shifted_until = fraction_of_delayed_inflow.times[j]

                        while (
                            j + 1 < len(fraction_of_delayed_inflow.times)
                            and already_shifted_until
                            < fraction_of_delayed_inflow.times[j + 1]
                        ) or already_shifted_until == float("inf"):
                            # Find shortest path at time delayed_inflow.times[j]
                            while (
                                already_shifted_until
                                >= shortest_paths_over_time.times[shortest_path_idx]
                            ):
                                shortest_path_idx += 1

                            shortest_path = shortest_paths_over_time.paths[
                                shortest_path_idx
                            ]
                            shift_until = min(
                                shortest_paths_over_time.times[shortest_path_idx],
                                fraction_of_delayed_inflow.times[j + 1]
                                if j + 1 < len(fraction_of_delayed_inflow.times)
                                else float("inf"),
                            )

                            sh_path_inflow = new_path_inflows.of(shortest_path)

                            shift_times = []
                            shift_values = []
                            if (
                                fraction_of_delayed_inflow.domain[0]
                                < fraction_of_delayed_inflow.times[j]
                            ):
                                shift_times.append(fraction_of_delayed_inflow.domain[0])
                                shift_values.append(0.0)
                            shift_times.append(fraction_of_delayed_inflow.times[j])
                            shift_values.append(fraction_of_delayed_inflow.values[j])
                            if shift_until < float("inf"):
                                shift_times.append(shift_until)
                                shift_values.append(0.0)

                            sh_path_inflow = sh_path_inflow + RightConstant(
                                shift_times,
                                shift_values,
                                fraction_of_delayed_inflow.domain,
                            )
                            new_path_inflows.set(shortest_path, sh_path_inflow)

                            already_shifted_until = shift_until

            new_path_dist.append(new_path_inflows._path_inflows)

        builder = NetworkLoader(
            self.network,
            [
                path_inflow
                for path_inflows in new_path_dist
                for path_inflow in path_inflows
            ],
        ).build_flow()
        flow = next(builder)
        while flow.phi < float("inf"):
            flow = next(builder)

        return flow, new_path_dist

    def build_flow(self) -> Tuple[DynamicFlow, List[List[PathInflow]]]:
        if self._built:
            raise RuntimeError("Flow was already built. Initialize a new builder.")
        self._built = True

        path_dist = self._get_initial_path_dist()
        builder = NetworkLoader(
            self.network,
            [path_inflow for path_inflows in path_dist for path_inflow in path_inflows],
        ).build_flow()
        flow = next(builder)
        while flow.phi < float("inf"):
            flow = next(builder)

        iteration = 1
        while iteration < self._iterations:
            flow, path_dist = self.iterate(iteration, flow, path_dist)
            iteration += 1
        return flow, path_dist
