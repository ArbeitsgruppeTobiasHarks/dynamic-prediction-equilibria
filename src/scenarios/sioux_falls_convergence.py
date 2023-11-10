import json
import os
import pickle

from core.convergence import AlphaFlowIterator
from core.predictors.predictor_type import PredictorType
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.get_tn_path import get_tn_path
from utilities.json_encoder import JSONEncoder
from utilities.piecewise_linear import PiecewiseLinear
from visualization.to_json import to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)

    run_parameters = dict(
        reroute_interval=0.025,
        horizon=200.0,
        inflow_horizon=20.0,
        alpha_fun=PiecewiseLinear([0.0, 0.01, 5.0], [0.0, 0.0, 0.5], 0.0, 0.0),
        delay_threshold=1e-4,
        min_path_active_time=1e-2,
        approx_inflows=True,
        parallelize=True,
    )
    num_iterations = 500
    log_every = 25

    tn_path = get_tn_path()
    edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
    nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)

    demands = {(1, 4): 1e5, (5, 23): 2e4, (15, 3): 3e4}
    for (s, t), demand in demands.items():
        network.add_commodity(
            {
                s: get_demand_with_inflow_horizon(
                    demand, run_parameters["inflow_horizon"]
                )
            },
            t,
            PredictorType.CONSTANT,
        )

    flow_iter = AlphaFlowIterator(network, **run_parameters)

    merged_flow, merged_network, metrics = flow_iter.run(num_iterations, log_every)

    out_dir = os.path.join(scenario_dir, f"run_{len(os.listdir(scenario_dir))}")
    os.makedirs(out_dir)

    json_path = os.path.join(out_dir, f"run_data.json")
    with open(json_path, "w") as f:
        JSONEncoder().dump(
            {
                "demands": {str(k): v for k, v in demands.items()},
                "parameters": run_parameters,
                "convergence_metrics": metrics,
            },
            f,
        )

    # iterator_path = os.path.join(out_dir, f"flow_iterator.pickle")
    # with open(iterator_path, "wb") as f:
    #     pickle.dump(flow_iter, f, protocol=pickle.HIGHEST_PROTOCOL)

    visualization_path = os.path.join(out_dir, f"merged_flow.vis.json")
    to_visualization_json(
        visualization_path,
        merged_flow,
        merged_network,
        {0: "green", 1: "blue", 2: "red"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-sioux-falls")

    main()
