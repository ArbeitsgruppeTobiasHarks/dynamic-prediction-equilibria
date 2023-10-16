import json
import os
import pickle

from core.convergence import AlphaFlowIterator
from core.predictors.predictor_type import PredictorType
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from scenarios.nguyen_scenario import build_nguyen_network
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.get_tn_path import get_tn_path
from utilities.json_encoder import JSONEncoder
from utilities.piecewise_linear import PiecewiseLinear
from utilities.right_constant import RightConstant
from visualization.to_json import merge_commodities, to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)

    run_parameters = dict(
        reroute_interval=0.125,
        horizon=100.0,
        inflow_horizon=12.0,
        alpha_fun=PiecewiseLinear([0.0, 5.0], [0.0, 0.5], 0.0, 0.0),
        delay_threshold=1e-4,
        min_path_active_time=1e-2,
        approx_inflows=True,
        parallelize=False,
    )

    num_iterations = 100
    log_every = 10

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

    json_path = os.path.join(scenario_dir, f"run_data.json")
    with open(json_path, "w") as f:
        JSONEncoder().dump(
            {
                "demands": {str(k): v for k, v in demands.items()},
                "parameters": run_parameters,
                "convergence_metrics": metrics,
            },
            f,
        )

    iterator_path = os.path.join(scenario_dir, f"flow_iterator.pickle")
    with open(iterator_path, "wb") as f:
        pickle.dump(flow_iter, f)

    visualization_path = os.path.join(scenario_dir, f"merged_flow.vis.json")
    to_visualization_json(
        visualization_path,
        merged_flow,
        merged_network,
        {0: "green", 1: "blue", 2: "red", 3: "purple", 4: "brown"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-test-nguyen")

    main()
