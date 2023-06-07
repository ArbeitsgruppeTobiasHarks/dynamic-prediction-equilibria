import random
from typing import Optional, Tuple, List

from numpy import genfromtxt

from core.network import Network
from core.predictors.predictor_type import PredictorType
from utilities.right_constant import RightConstant


def network_from_csv(path: str) -> Network:
    np_data = genfromtxt(path, delimiter=" ", skip_header=1)
    # subsequent rows are structured as:
    # from_node to_node length_in_meters freeflow_speed_kph num_lanes
    # freeflow_speed_kph is -1 when the value is absent in the data.
    # We default to 20 for now.

    network = Network()
    for i, row in enumerate(np_data):
        freeflow_speed = row[3] if row[3] > 0 else 20.0
        travel_time = row[2] / freeflow_speed
        capacity = row[4] * freeflow_speed

        if row[0] == row[1]:
            print(
                "Edge #{i} has not been added, as it's a loop from and to node #{row[0]}!"
            )
            continue
        network.add_edge(row[0], row[1], travel_time, capacity)
    return network


def add_demands_to_network(
    network: Network,
    demands_path: str,
    inflow_horizon: float,
    use_default_demands: bool = False,
    random_seed: Optional[int] = None,
    upscale: bool = True,
    suppress_log: bool = False,
) -> None:
    if random_seed is None and not use_default_demands:
        raise ValueError(
            "Please either provide a random_seed or set use_default_demands to true."
        )
    if random_seed is not None and use_default_demands:
        raise ValueError(
            "You provided a random seed, but also set the flag to use default demands."
        )
    if random_seed is not None:
        random.seed(random_seed)

    np_data = genfromtxt(demands_path, delimiter=" ")
    removed_rows: List[Tuple[int, int, int]] = []
    for i, row in enumerate(np_data):
        # Filter out unrealistic commodities where the source cannot reach the sink
        source = network.graph.nodes[row[0]]
        sink = network.graph.nodes[row[1]]
        if source not in network.graph.get_nodes_reaching(sink):
            removed_rows.append((i, source.id, sink.id))
        else:
            if use_default_demands:
                # after upscaling: between 20 and 100
                demand = 20 + (row[2] - 10) / 20.0 * 80.0 if upscale else row[2]
            else:
                demand = random.randint(20, 100)
            network.add_commodity(
                row[0],
                row[1],
                RightConstant([0.0, inflow_horizon], [demand, 0.0], (0, float("inf"))),
                PredictorType.CONSTANT,
            )

    if not suppress_log and removed_rows:
        print(
            f"Did not add the following commodities as their source cannot reach their sink #row(source, sink):"
        )
        print(", ".join(map(lambda t: f"#{t[0]}({t[1]}, {t[2]})", removed_rows)))


if __name__ == "__main__":
    network_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/from-kostas/tokyo_tiny.arcs"
    demands_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/from-kostas/tokyo_tiny.demands"
    network = network_from_csv(network_path)
    add_demands_to_network(
        network,
        demands_path,
        use_default_demands=True,
        upscale=True,
        suppress_log=False,
        inflow_horizon=25.0,
    )
    network.remove_unnecessary_nodes()
    network.print_info()
    network.to_file(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/tokyo_tiny/default_demands.pickle"
    )
