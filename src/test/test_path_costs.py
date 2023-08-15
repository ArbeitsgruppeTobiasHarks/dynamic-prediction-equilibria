import os
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from utilities.get_tn_path import get_tn_path

from core.network import Network
from core.predictors.predictor_type import PredictorType
from ml.generate_queues import generate_queues_and_edge_loads
from scenarios.scenario_utils import get_demand_with_inflow_horizon

from core.path_flow_builder import PathFlowBuilder
from utilities.build_with_times import build_with_times
from visualization.to_json import merge_commodities, to_visualization_json
from core.bellman_ford import bellman_ford
from utilities.piecewise_linear import PiecewiseLinear
import matplotlib.pyplot as plt


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    flow_index = 2
    visualization_path = os.path.join(flows_dir, f"flow{flow_index}.vis.json")

    reroute_interval = 0.125
    inflow_horizon = 12.0
    horizon = 100.0
    demand = 2e4

    tn_path = get_tn_path()
    edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
    nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    network.add_commodity(
        {1: get_demand_with_inflow_horizon(demand * 0.4, inflow_horizon)},
        14,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {1: get_demand_with_inflow_horizon(demand * 0.4, inflow_horizon)},
        14,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {1: get_demand_with_inflow_horizon(demand * 0.2, inflow_horizon)},
        14,
        PredictorType.CONSTANT,
    )

    s = network.graph.nodes[1]
    t = network.graph.nodes[14]
    paths_edge_ids = [[1, 5, 9, 33], [1, 6, 35, 33], [0, 3, 14, 10, 9, 33]]
    paths = {i: [network.graph.edges[e_id] for e_id in e_ids] for i, e_ids in enumerate(paths_edge_ids)}

    flow_builder = PathFlowBuilder(network, paths, reroute_interval)
    flow, _ = build_with_times(
        flow_builder, flow_index, reroute_interval, horizon
    )

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

    def evaluate_path(costs, path):
        identity = PiecewiseLinear(
            [0.0], [0.0], first_slope=1, last_slope=1, domain=(0, float("inf"))
        )
        path_exit_time = identity

        for edge in path[::-1]:
            path_exit_time = path_exit_time.compose(
                identity.plus(costs[edge.id]).ensure_monotone(True)
            )

        return path_exit_time

    l0 = evaluate_path(costs, paths[0])
    l1 = evaluate_path(costs, paths[1])
    l2 = evaluate_path(costs, paths[2])

    earliest_arrival_fcts = bellman_ford(
        t,
        costs,
        flow._network.graph.get_nodes_reaching(t),
        0.0,
        float("inf"),
    )

    to_visualization_json(
        visualization_path,
        flow,
        network,
        {
            0: 'green',
            1: 'blue',
            2: 'red'
        },
    )
    print(f"Successfully written visualization to disk!")

    Network.from_file(network_path).print_info()

    plt.plot(l0.times, l0.values, color='green')
    plt.plot(l1.times, l1.values, color='blue')
    plt.plot(l2.times, l2.values, color='red')
    plt.plot(earliest_arrival_fcts[s].times, earliest_arrival_fcts[s].values, color='black')
    plt.plot([0, l0.values[-1]], [earliest_arrival_fcts[s].values[0], earliest_arrival_fcts[s].values[0]+ l0.values[-1]], '--')
    plt.show()


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-test-sioux-falls")

    main()
