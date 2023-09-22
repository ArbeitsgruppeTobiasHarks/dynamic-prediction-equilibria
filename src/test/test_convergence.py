import os
import pickle

from core.convergence import AlphaFlowIterator
from core.predictors.predictor_type import PredictorType
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.get_tn_path import get_tn_path
from visualization.to_json import merge_commodities, to_visualization_json

from utilities.combine_commodities import combine_commodities_with_same_sink


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")

    reroute_interval = 0.125
    inflow_horizon = 20.0
    horizon = 200.0
    demand = 1e5

    num_iterations = 25

    def alpha_fun(delay):
        if delay < 1e-5:
            return 0.0
        elif delay < 1.0:
            return 0.01
        else:
            return 0.1
    delay_threshold = 1e-5
    approx_inflows = True
    evaluate_every = 1

    tn_path = get_tn_path()
    edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
    nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    inflow = get_demand_with_inflow_horizon(demand, inflow_horizon)
    network.add_commodity(
        {1: inflow*0.2},
        14,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {5: inflow*0.4},
        23,
        PredictorType.CONSTANT,
    )
    network.add_commodity(
        {15: inflow*0.4},
        3,
        PredictorType.CONSTANT,
    )

    flow_iter = AlphaFlowIterator(
        network,
        reroute_interval,
        horizon,
        inflow_horizon,
        alpha_fun,
        delay_threshold,
        approx_inflows
    )

    merged_flow, path_metrics = flow_iter.run(num_iterations, evaluate_every)

    metrics_path = os.path.join(flows_dir, f"conv_metrics.pickle")
    with open(f"conv_metrics.pickle", 'wb') as f:
        pickle.dump(path_metrics, f)

    combine_commodities_with_same_sink(network)
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
