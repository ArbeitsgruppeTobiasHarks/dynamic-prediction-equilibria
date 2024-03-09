import os

from core.active_paths import Path
from core.network import Network
from core.path_flow_builder import PathFlowBuilder
from core.predictors.predictor_type import PredictorType
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from ml.generate_queues import generate_queues_and_edge_loads
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.build_with_times import build_with_times
from utilities.get_tn_path import get_tn_path
from visualization.to_json import merge_commodities, to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    flow_index = 0
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

    paths_edge_ids = [[1, 5, 9, 33], [1, 6, 35, 33], [0, 3, 14, 10, 9, 33]]
    paths = {
        i: Path([network.graph.edges[e_id] for e_id in e_ids])
        for i, e_ids in enumerate(paths_edge_ids)
    }

    flow_builder = PathFlowBuilder(network, paths)
    flow, _ = build_with_times(flow_builder, flow_index, reroute_interval, horizon)

    # merged_flow = merge_commodities(
    #     flow, network, range(len(network.commodities))
    # )
    to_visualization_json(
        visualization_path,
        flow,
        network,
        {0: "green", 1: "blue", 2: "red"},
    )
    print(f"Successfully written visualization to {visualization_path}")

    Network.from_file(network_path).print_info()


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-test-sioux-falls")

    main()
