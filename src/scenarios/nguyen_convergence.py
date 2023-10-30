import os
import pickle

from core.convergence import AlphaFlowIterator
from core.predictors.predictor_type import PredictorType
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from scenarios.nguyen_scenario import build_nguyen_network
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.get_tn_path import get_tn_path
from visualization.to_json import merge_commodities, to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    out_dir = os.path.join(scenario_dir, f"run_{len(os.listdir(scenario_dir))}")
    os.makedirs(out_dir)

    reroute_interval = 0.1
    inflow_horizon = 12.0
    horizon = 60.0
    demand = 100

    num_iterations = 100

    def alpha_fun(delay):
        return min(0.1 * delay, 0.1)

    delay_threshold = 1e-4
    min_path_active_time = reroute_interval
    approx_inflows = True
    parallelize = False
    log_every = 10

    network = build_nguyen_network()
    for s, t in [(1, 3), (4, 2), (4, 3)]:
        network.add_commodity(
            {s: get_demand_with_inflow_horizon(demand, inflow_horizon)},
            t,
            PredictorType.CONSTANT,
        )

    flow_iter = AlphaFlowIterator(
        network,
        reroute_interval,
        horizon,
        inflow_horizon,
        alpha_fun,
        delay_threshold,
        min_path_active_time,
        approx_inflows,
        parallelize,
    )

    (merged_flow, merged_network), metrics = flow_iter.run(num_iterations, log_every)

    metrics_path = os.path.join(out_dir, f"conv_metrics.pickle")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    visualization_path = os.path.join(out_dir, f"merged_flow.vis.json")
    to_visualization_json(
        visualization_path,
        merged_flow,
        merged_network,
        {0: "green", 1: "blue", 2: "red", 3: "purple", 4: "brown"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-nguyen")

    main()
