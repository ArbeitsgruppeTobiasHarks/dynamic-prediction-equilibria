import json
import os
from test.sample_network import build_sample_network

import numpy as np

from core.network import Network
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network_demand
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads
from ml.neighboring_edges import get_neighboring_edges_undirected
from ml.SKFullNetworkModel import train_sk_full_net_model
from ml.SKNeighborhood import train_sk_neighborhood_model
from ml.TFFullNetworkModel import train_tf_full_net_model
from ml.TFNeighborhood import train_tf_neighborhood_model
from scenarios.scenario_utils import get_demand_with_inflow_horizon


def build_nguyen_network():
    path = "./src/scenarios/nguyen_network.csv"
    np_data = np.genfromtxt(path, delimiter=",", skip_header=1)
    # rows are structured as:
    # from_node to_node capacity, free_flow_travel_time

    network = Network()
    for row in np_data:
        network.add_edge(int(row[0]), int(row[1]), row[3] / 60, row[2] * 60)

    network.graph.positions = {
        1: (2, 0),
        12: (4, 0),
        5: (2, 2),
        6: (4, 2),
        8: (8, 2),
        4: (0, 2),
        7: (6, 2),
        9: (2, 4),
        10: (4, 4),
        4: (0, 2),
        7: (6, 2),
        9: (2, 4),
        10: (4, 4),
        11: (6, 4),
        2: (8, 4),
        13: (4, 6),
        3: (6, 6),
    }
    return network


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    tf_full_net_model_path = os.path.join(scenario_dir, "tf-full-net-model")
    tf_neighborhood_models_path = os.path.join(scenario_dir, "tf-neighborhood-models")
    sk_neighborhood_models_path = os.path.join(scenario_dir, "sk-neighborhood-models")
    sk_full_net_model_path = os.path.join(scenario_dir, "sk-full-net-model")
    queues_and_edge_loads_dir = os.path.join(scenario_dir, "queues")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = 0.125
    inflow_horizon = 12.0
    horizon = 60.0
    prediction_interval = 0.5
    past_timesteps = 20
    future_timesteps = 20
    pred_horizon = 20.0
    demand = 100.0
    number_training_flows = 500
    number_eval_flows = 20
    max_distance = 3

    network = build_nguyen_network()
    for s, t in [(1, 2), (1, 3), (4, 2), (4, 3)]:
        network.add_commodity(
            {s: get_demand_with_inflow_horizon(demand, inflow_horizon)},
            t,
            PredictorType.CONSTANT,
        )

    avg_neighborhood_size = np.average(
        [
            len(get_neighboring_edges_undirected(e, max_distance))
            for e in network.graph.edges
        ]
    )
    print(
        f"Avg neighborhood size: {avg_neighborhood_size/len(network.graph.edges)*100}%"
    )
    network.to_file(network_path)
    build_flows(
        network_path,
        flows_dir,
        inflow_horizon=inflow_horizon,
        number_flows=number_training_flows,
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

    build_tf_full_net_predictor = train_tf_full_net_model(
        queues_and_edge_loads_dir,
        past_timesteps,
        future_timesteps,
        reroute_interval,
        prediction_interval,
        horizon,
        network,
        tf_full_net_model_path,
    )

    # build_tf_neighborhood_predictor = train_tf_neighborhood_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps,
    #                                                              reroute_interval, prediction_interval, horizon, network, tf_neighborhood_models_path, max_distance)

    build_sk_full_net_predictor = train_sk_full_net_model(
        queues_and_edge_loads_dir,
        past_timesteps,
        future_timesteps,
        reroute_interval,
        prediction_interval,
        horizon,
        network,
        sk_full_net_model_path,
    )

    # build_sk_neighborhood_predictor = train_sk_neighborhood_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps,
    #                                                              reroute_interval, prediction_interval, horizon, network, sk_neighborhood_models_path, max_distance)

    def build_predictors(network):
        return {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(
                network, pred_horizon, delta=1.0
            ),
            PredictorType.MACHINE_LEARNING_SK_FULL_NET: build_sk_full_net_predictor(
                network
            ),
            # PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: build_sk_neighborhood_predictor(network),
            PredictorType.MACHINE_LEARNING_TF_FULL_NET: build_tf_full_net_predictor(
                network
            ),
            # PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: build_tf_neighborhood_predictor(network),
        }

    # test_mask = train_full_net_model(
    #    queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval, prediction_interval, horizon, network, full_net_model_path)

    # expanded_queues_from_flows_per_edge(network_path, past_timesteps, 1., future_timesteps, flows_dir,
    #                                    expanded_per_edge_dir, horizon, average=False, sample_step=1)

    # train_per_edge_model(network_path, expanded_per_edge_dir, models_per_edge_dir, past_timesteps, future_timesteps)

    eval_network_demand(
        network_path,
        number_eval_flows,
        eval_dir,
        inflow_horizon,
        future_timesteps,
        prediction_interval,
        reroute_interval,
        horizon,
        demand_sigma=None,
        suppress_log=False,
        build_predictors=build_predictors,
        check_for_optimizations=False,
        visualization_config={
            PredictorType.ZERO: ("blue", "$\\hat q^{\\text{Z}}$"),
            PredictorType.CONSTANT: ("red", "$\\hat q^{\\text{C}}$"),
            PredictorType.LINEAR: (
                "{rgb,255:red,0; green,128; blue,0}",
                "$\\hat q^{\\text{L}}$",
            ),
            PredictorType.REGULARIZED_LINEAR: ("orange", "$\\hat q^{\\text{RL}}$"),
            PredictorType.MACHINE_LEARNING_SK_FULL_NET: (
                "black",
                "$\\hat q^{\\text{LR-full}}$",
            ),
            # PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: ("black", "$\\hat q^{\\text{LR-neighboring}}$"),
            PredictorType.MACHINE_LEARNING_TF_FULL_NET: (
                "black",
                "$\\hat q^{\\text{NN-full}}$",
            ),
            # PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: ("black", "$\\hat q^{\\text{NN-neighboring}}$"),
        },
    )

    average_comp_times = []
    for file in os.listdir(eval_dir):
        if file.endswith(".json"):
            path = os.path.join(eval_dir, file)
            with open(path, "r") as file:
                d = json.load(file)
            average_comp_times.append(d["computation_time"])

    avg_comp_time = sum(average_comp_times) / len(average_comp_times)
    print(f"Average computation time: {avg_comp_time}")

    Network.from_file(network_path).print_info()


if __name__ == "__main__":

    def main():
        run_scenario("./out/journal-nguyen")

    main()
