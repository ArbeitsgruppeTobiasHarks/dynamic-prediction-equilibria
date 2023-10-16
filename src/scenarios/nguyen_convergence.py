import json
import os
import pickle

from core.convergence import AlphaFlowIterator
from core.predictors.predictor_type import PredictorType
from scenarios.nguyen_scenario import build_nguyen_network
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.json_encoder import JSONEncoder
from utilities.piecewise_linear import PiecewiseLinear
from visualization.to_json import to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)

    run_parameters = dict(
        reroute_interval=0.5,
        horizon=100.0,
        inflow_horizon=12.0,
        alpha_fun=PiecewiseLinear([0.0, 0.01, 5.0], [0.0, 0.0, 0.25], 0.0, 0.0),
        delay_threshold=1e-3,
        min_path_active_time=1e-2,
        approx_inflows=True,
        parallelize=False,
    )

    num_iterations = 10000
    log_every = 500

    demands = {(1, 2): 200, (1, 3): 100, (4, 2): 100, (4, 3): 100}
    network = build_nguyen_network()
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

    iterator_path = os.path.join(out_dir, f"flow_iterator.pickle")
    with open(iterator_path, "wb") as f:
        pickle.dump(flow_iter, f, protocol=pickle.HIGHEST_PROTOCOL)

    visualization_path = os.path.join(out_dir, f"merged_flow.vis.json")
    to_visualization_json(
        visualization_path,
        merged_flow,
        merged_network,
        {0: "green", 1: "blue", 2: "red", 3: "purple"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-nguyen")

    main()
