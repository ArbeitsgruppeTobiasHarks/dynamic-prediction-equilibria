import multiprocessing as mp
from typing import Dict, List, Optional

from core.graph import Edge, Node
from core.machine_precision import eps
from utilities.piecewise_linear import PiecewiseLinear, identity
from utilities.right_constant import Indicator, RightConstant


class Path:
    edges: List[Edge]
    start: Optional[Node]
    end: Optional[Node]

    def __init__(self, edges):
        self.edges = edges
        if len(edges) > 0:
            self.start = edges[0]._node_from
            self.end = edges[-1]._node_to
        else:
            self.start = None
            self.end = None

    def add_edge(self, edge):
        return Path(self.edges + [edge])

    def __len__(self):
        return len(self.edges)

    def __eq__(self, other):
        assert isinstance(other, Path)

        if len(self) != len(other):
            return False
        else:
            return all(e1 == e2 for e1, e2 in zip(self.edges, other.edges))

    def __str__(self):
        return str([e.id for e in self.edges])

    def __hash__(self):
        return hash(tuple(self.edges))


class PathsOverTime(Dict):
    """
    Collection of paths with same source and sink.
    Each path is equipped with intervals of activity represented by indicator
    """

    def __init__(self, paths, activity_indicators):
        super().__init__({p: i for p, i in zip(paths, activity_indicators)})

    def add_path(self, path, activity_indicator):
        if path in self.keys():
            self[path] += activity_indicator
        else:
            self[path] = activity_indicator

    def coverage(self):
        return sum(self.values())


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
    delay: PiecewiseLinear, threshold: float = eps, min_interval_length: float = eps
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
            interval_end = delay.times[i - 1] + offset

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


def compute_path_travel_time(
    path: Path, costs: List[PiecewiseLinear]
) -> PiecewiseLinear:
    path_exit_time = identity

    for edge in path.edges[::-1]:
        path_exit_time = path_exit_time.compose(
            identity.plus(costs[edge.id]).ensure_monotone(True)
        )

    return path_exit_time - identity


def compute_all_active_paths(
    earliest_arrivals: Dict[Node, PiecewiseLinear],
    edge_costs: List[PiecewiseLinear],
    source: Node,
    sink: Node,
    inflow_horizon: float,
    delay_threshold: float = eps,
    min_active_time: float = 1000 * eps,
) -> PathsOverTime:
    """
    Constructs all simple s-t paths which are active for at least min_active_time
    """

    shortest_paths = PathsOverTime([], [])
    subpaths = [
        (Path([]), source, identity, Indicator.from_interval(0.0, float("inf")))
    ]

    while len(subpaths) > 0:
        extended_subpaths = []
        for path, dest, exit_time, indicator in subpaths:
            if dest == sink:
                shortest_paths[path] = indicator
                continue
            for e in dest.outgoing_edges:
                new_dest = e.node_to
                if e in path.edges or new_dest not in earliest_arrivals:
                    continue

                new_path = path.add_edge(e)
                new_exit_time = (identity + edge_costs[e.id]).compose(exit_time)
                delay = (
                    earliest_arrivals[new_dest].compose(new_exit_time)
                    - earliest_arrivals[source]
                )
                new_indicator = get_activity_indicator(delay, threshold=delay_threshold)
                if new_indicator.integral()(inflow_horizon) > min_active_time:
                    extended_subpaths.append(
                        (new_path, new_dest, new_exit_time, new_indicator)
                    )

        subpaths = extended_subpaths

    return shortest_paths


def all_active_paths_parallel(
    earliest_arrivals: Dict[Node, PiecewiseLinear],
    edge_costs: List[PiecewiseLinear],
    source: Node,
    sink: Node,
    inflow_horizon: float,
    delay_threshold: float = eps,
    min_active_time: float = 1000 * eps,
    num_processes: int = mp.cpu_count(),
) -> PathsOverTime:
    """
    Construct all paths with delay below delay_threshold for at least min_active_time.
    Algorithm extends the paths with all possible edges and computes delays in parallel.
    """

    shortest_paths = PathsOverTime(paths=[], activity_indicators=[])
    subpaths = [(Path([]), source, identity, Indicator.from_interval(0, float("inf")))]

    network_data = (
        earliest_arrivals,
        edge_costs,
        source,
        inflow_horizon,
        delay_threshold,
        min_active_time,
    )
    with mp.Pool(processes=num_processes) as pool:
        while subpaths:
            new_subpaths_lists = pool.starmap(
                extend_path, [(path_data, network_data) for path_data in subpaths]
            )

            subpaths = [
                path_data
                for new_subpaths in new_subpaths_lists
                for path_data in new_subpaths
                if path_data[1] != sink
            ]
            shortest_paths.update(
                {
                    path: indicator
                    for new_subpaths in new_subpaths_lists
                    for path, dest, _, indicator in new_subpaths
                    if dest == sink
                }
            )

    return shortest_paths


def extend_path(path_data, network_data):
    path, dest, exit_time, _ = path_data
    (
        earliest_arrivals,
        edge_costs,
        source,
        inflow_horizon,
        delay_threshold,
        min_active_time,
    ) = network_data
    extended = []
    for e in dest.outgoing_edges:
        new_dest = e._node_to
        if (
            e in path.edges or new_dest not in earliest_arrivals
        ):  # avoid loops and wrong paths
            continue

        new_exit_time = (identity + edge_costs[e.id]).compose(exit_time)
        delay = (
            earliest_arrivals[new_dest].compose(new_exit_time)
            - earliest_arrivals[source]
        )
        new_indicator = get_activity_indicator(delay, threshold=delay_threshold)
        if new_indicator.integral()(inflow_horizon) > min_active_time:
            extended.append((path.add_edge(e), new_dest, new_exit_time, new_indicator))

    return extended
