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
    inflow_horizon = 10.0
    horizon = 100.0
    demand = 2e4

    num_iterations = 200
    alpha = 0.01

    tn_path = get_tn_path()
    edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
    nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    inflow = get_demand_with_inflow_horizon(demand, inflow_horizon)
    network.add_commodity(
        {1: inflow},
        14,
        PredictorType.CONSTANT,
    )
    # network.add_commodity(
    #     {5: inflow*0.6},
    #     13,
    #     PredictorType.CONSTANT,
    # )
    # network.add_commodity(
    #     {6: inflow*0.3},
    #     15,
    #     PredictorType.CONSTANT,
    # )

    flow_iter = FlowIterator(network, reroute_interval, horizon, num_iterations, alpha)

    last_flow = flow_iter.run()

    merged_flow = last_flow
    combine_commodities_with_same_sink(network)

    for route, commodities in flow_iter._route_users.items():
        merged_flow = merge_commodities(merged_flow, network, commodities)

    opt_avg_travel_time = calculate_optimal_average_travel_time(
        merged_flow, network, inflow_horizon, horizon, network.commodities[0]
    )
    print(f"Optimal average travel time: {opt_avg_travel_time}")

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
