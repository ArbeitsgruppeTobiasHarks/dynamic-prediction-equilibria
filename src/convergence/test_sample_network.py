import json
import os
from test.sample_network import build_sample_network
from core.network import Network
from core.predictors.predictor_type import PredictorType
from ml.generate_queues import generate_queues_and_edge_loads
from scenarios.scenario_utils import get_demand_with_inflow_horizon

from convergence.build_path_flows import build_path_flows


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    queues_and_edge_loads_dir = os.path.join(scenario_dir, "queues")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = 0.125
    inflow_horizon = 12.0
    horizon = 60.0
    prediction_interval = 0.5
    past_timesteps = 20
    future_timesteps = 20
    pred_horizon = 20.0
    demand = 4.0
    number_flows = 10
    max_distance = 3

    network = build_sample_network()
    network.add_commodity(
        {0: get_demand_with_inflow_horizon(demand, inflow_horizon)},
        2,
        PredictorType.CONSTANT,
    )
    network.to_file(network_path)
    paths = {0: [([0, 2, 4], 0.10 ), ([1], 0.90)] }
    build_path_flows(
        paths,
        network_path,
        flows_dir,
        inflow_horizon=inflow_horizon,
        number_flows=number_flows,
        horizon=horizon,
        reroute_interval=reroute_interval,
        check_for_optimizations=False,
    )

    generate_queues_and_edge_loads(
        past_timesteps,
        flows_dir,
        queues_and_edge_loads_dir,
        horizon,
        reroute_interval,
        prediction_interval,
    )

    Network.from_file(network_path).print_info()


if __name__ == "__main__":

    def main():
        run_scenario("./out/convergence-test-sample")

    main()
