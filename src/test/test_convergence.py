import os
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from utilities.get_tn_path import get_tn_path
from core.network import Network
from core.predictors.predictor_type import PredictorType
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from core.convergence import FlowIterator
from visualization.to_json import merge_commodities, to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")

    reroute_interval = 0.125
    inflow_horizon = 10.0
    horizon = 100.0
    demand = 2e4

    num_iterations = 25
    alpha = 0.05


    tn_path = get_tn_path()
    edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
    nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    inflow = get_demand_with_inflow_horizon(demand, inflow_horizon)
    s = network.graph.nodes[1]
    t = network.graph.nodes[14]

    network.add_commodity(
        {s.id: inflow},
        t.id,
        PredictorType.CONSTANT,
    )

    flow_iter = FlowIterator(network, reroute_interval, horizon, num_iterations, alpha)

    last_flow = flow_iter.run()

    visualization_path = os.path.join(flows_dir, f"conv_last_flow.vis.json")
    to_visualization_json(
        visualization_path,
        last_flow,
        network,
        {
            0: 'green',
            1: 'blue',
            2: 'red',
            3: 'purple',
            4: 'brown'
        }
    )

    # for i, flow in enumerate(flow_iter._flows):
    #     visualization_path = os.path.join(flows_dir, f"conv_flow{i}.vis.json")
    #     to_visualization_json(
    #         visualization_path,
    #         flow,
    #         network,
    #         {
    #             0: 'green',
    #             1: 'blue',
    #             2: 'red',
    #             3: 'purple',
    #             4: 'brown'
    #         },
    #     )
    #     print(f"Successfully written visualization to disk!")

    #Network.from_file(network_path).print_info()


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-test-sioux-falls")

    main()
