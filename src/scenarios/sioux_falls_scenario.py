import json
import os
import pickle

import numpy as np

from core.dynamic_flow import DynamicFlow
from core.network import Network
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate import evaluate_mean_absolute_error
from eval.evaluate_network import eval_network_demand
from importer.sioux_falls_importer import import_sioux_falls
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads
from ml.neighboring_edges import get_neighboring_edges_undirected
from ml.SKFullNetworkModel import train_sk_full_net_model
from ml.TFFullNetworkModel import train_tf_full_net_model
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.file_lock import wait_for_locks, with_file_lock
from utilities.get_tn_path import get_tn_path


def shallow_evaluate_predictors(
    network_path: str,
    flows_dir: str,
    out_dir: str,
    past_timesteps: int,
    future_timesteps: int,
    horizon: float,
    reroute_interval: float,
    prediction_interval: float,
    build_predictors,
):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        [file for file in os.listdir(flows_dir) if file.endswith(".flow.pickle")]
    )
    for flow_filename in files:
        flow_id = flow_filename[: len(flow_filename) - len(".flow.pickle")]
        out_path = os.path.join(out_dir, f"{flow_id}-shallow-eval.pickle")

        def handle(open_file):
            print(f"Shallow evaluating Flow#{flow_id}...")
            with open(os.path.join(flows_dir, flow_filename), "rb") as flow_file:
                flow: DynamicFlow = pickle.load(flow_file)
            network = Network.from_file(network_path)
            diff = evaluate_mean_absolute_error(
                flow,
                build_predictors(network),
                future_timesteps,
                reroute_interval,
                prediction_interval,
                horizon,
            )
            with open_file("wb") as file:
                pickle.dump(diff, file)

        with_file_lock(out_path, handle)
    wait_for_locks(out_dir)


def run_scenario(edges_tntp_path: str, nodes_tntp_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    tf_full_net_model_path = os.path.join(scenario_dir, "tf-full-network-model")
    tf_neighborhood_models_path = os.path.join(scenario_dir, "tf-neighborhood-models")
    sk_full_net_model_path = os.path.join(scenario_dir, "sk-full-net-model")
    sk_neighborhood_models_path = os.path.join(scenario_dir, "sk-neighborhood-models")
    queues_and_edge_loads_dir = os.path.join(scenario_dir, "queues")
    shallow_eval_dir = os.path.join(scenario_dir, "shallow-eval")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = 0.125
    inflow_horizon = 12.0
    horizon = 60.0
    past_timesteps = 20
    future_timesteps = 20
    prediction_interval = 1.0
    number_training_flows = 500
    number_eval_flows = 20
    pred_horizon = future_timesteps * prediction_interval

    average_demand = 8000

    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    network.add_commodity(
        {1: get_demand_with_inflow_horizon(average_demand, inflow_horizon)},
        14,
        PredictorType.CONSTANT,
    )
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    demand_sigma = min(Network.from_file(network_path).capacity) / 2.0

    network.print_info()

    avg_neighborhood_size = np.average(
        [len(get_neighboring_edges_undirected(e, 3)) for e in network.graph.edges]
    )
    print(
        f"Avg neighborhood size: {avg_neighborhood_size/len(network.graph.edges)*100}%"
    )

    build_flows(
        network_path,
        flows_dir,
        inflow_horizon,
        number_training_flows,
        horizon,
        reroute_interval,
        demand_sigma,
    )

    generate_queues_and_edge_loads(
        past_timesteps,
        flows_dir,
        queues_and_edge_loads_dir,
        horizon,
        reroute_interval,
        prediction_interval,
    )

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
            PredictorType.MACHINE_LEARNING_TF_FULL_NET: build_tf_full_net_predictor(
                network
            ),
        }

    # shallow_evaluate_predictors(network_path, flows_dir, shallow_eval_dir, past_timesteps, future_timesteps,
    #                             horizon, reroute_interval, prediction_interval, build_predictors)

    # expanded_queues_from_flows_per_edge(network_path, past_timesteps, 1., future_timesteps, flows_dir,
    #                                     expanded_per_edge_dir, horizon, average=False, sample_step=1)

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
        demand_sigma=demand_sigma,
        suppress_log=False,
        build_predictors=build_predictors,
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
            PredictorType.MACHINE_LEARNING_TF_FULL_NET: (
                "black",
                "$\\hat q^{\\text{NN-full}}$",
            ),
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
    print(f"Average computing time: {avg_comp_time}")

    Network.from_file(network_path).print_info()


if __name__ == "__main__":

    def main():
        tn_path = get_tn_path()
        edges_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_net.tntp")
        nodes_tntp_path = os.path.join(tn_path, "SiouxFalls/SiouxFalls_node.tntp")
        run_scenario(edges_tntp_path, nodes_tntp_path, "./out/journal-sioux-falls")

    main()
