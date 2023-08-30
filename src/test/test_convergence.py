import os

from core.convergence import AlphaFlowIterator
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
    horizon = 200.0
    demand = 5e4

    num_iterations = 250
    alpha = 0.1
    approx_inflows = True

    tn_path = get_tn_path()
    edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
    nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    inflow = get_demand_with_inflow_horizon(demand, inflow_horizon)
    network.add_commodity(
        {1: inflow*0.3},
        14,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {5: inflow*0.4},
        23,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {15: inflow*0.3},
        3,
        PredictorType.CONSTANT,
    )

    flow_iter = AlphaFlowIterator(network, reroute_interval, horizon, num_iterations, alpha, approx_inflows)

    merged_flow = flow_iter.run(eval_every=10)

    visualization_path = os.path.join(flows_dir, f"conv_merged_flow_approx.vis.json")
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
