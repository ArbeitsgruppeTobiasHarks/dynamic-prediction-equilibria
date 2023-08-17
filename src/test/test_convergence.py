import os

from core.convergence import FlowIterator
from core.network import Network
from core.predictors.predictor_type import PredictorType
from eval.evaluate import calculate_optimal_average_travel_time
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.get_tn_path import get_tn_path
from visualization.to_json import merge_commodities, to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")

    reroute_interval = 0.125
    inflow_horizon = 20.0
    horizon = 100.0
    demand = 5e4

    num_iterations = 100
    alpha = 0.05

    tn_path = get_tn_path()
    edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
    nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    inflow = get_demand_with_inflow_horizon(demand, inflow_horizon)
    network.add_commodity(
        {1: inflow*0.4},
        14,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {5: inflow*0.6},
        23,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {15: inflow},
        3,
        PredictorType.CONSTANT,
    )

    flow_iter = FlowIterator(network, reroute_interval, horizon, num_iterations, alpha)

    last_flow = flow_iter.run()

    merged_flow = last_flow
    combine_commodities_with_same_sink(network)

    for route, commodities in flow_iter._route_users.items():
        merged_flow = merge_commodities(merged_flow, network, commodities)

    print(f"Optimal average travel times:")
    for com in network.commodities:
        s = next(iter(com.sources))
        t = com.sink
        opt_avg_travel_time = calculate_optimal_average_travel_time(
            merged_flow, network, inflow_horizon, horizon, com
        )
        print(f"({s.id}, {t.id}): {opt_avg_travel_time}")


    visualization_path = os.path.join(flows_dir, f"conv_merged_flow.vis.json")
    to_visualization_json(
        visualization_path,
        merged_flow,
        network,
        {0: "green", 1: "blue", 2: "red", 3: "purple", 4: "brown"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-test-sioux-falls")

    main()
