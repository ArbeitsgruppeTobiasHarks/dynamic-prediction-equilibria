import json
import os
import pickle
from typing import IO

import numpy as np
from core.dynamic_flow import DynamicFlow
from core.network import Network

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.tf_full_net_predictor import TFFullNetPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate import evaluate_prediction_accuracy
from eval.evaluate_network import eval_network_demand
from importer.sioux_falls_importer import import_sioux_falls
from ml.TFFullNetworkModel import train_tf_full_net_model
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads
from utilities.file_lock import wait_for_locks, with_file_lock


def shallow_evaluate_predictors(network_path: str, flows_dir: str, out_dir: str, past_timesteps: int, future_timesteps: int,
                                horizon: float, reroute_interval: float, prediction_interval: float, build_predictors):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([file for file in os.listdir(flows_dir) if file.endswith(".flow.pickle")])
    for flow_filename in files:
        flow_id = flow_filename[: len(flow_filename) - len(".flow.pickle")]
        out_path = os.path.join(out_dir, f"{flow_id}-shallow-eval.pickle")
        def handle(open_file):
            print(f"Shallow evaluating Flow#{flow_id}...")
            with open(os.path.join(flows_dir, flow_filename), "rb") as flow_file:
                flow: DynamicFlow = pickle.load(flow_file)
            network = Network.from_file(network_path)
            diff = evaluate_prediction_accuracy(flow, build_predictors(
                network), future_timesteps, reroute_interval, prediction_interval, horizon)
            with open_file("wb") as file:
                pickle.dump(diff, file)

        with_file_lock(out_path, handle)
    wait_for_locks(out_dir)

def run_scenario(edges_tntp_path: str, nodes_tntp_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    full_net_model_path = os.path.join(scenario_dir, "full-network-model")
    queues_dir = os.path.join(scenario_dir, "queues")
    shallow_eval_dir = os.path.join(scenario_dir, "shallow-eval")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = .125
    inflow_horizon = 12.
    horizon = 60.
    past_timesteps = 20
    future_timesteps = 20
    prediction_interval = 1.

    pred_horizon = future_timesteps * prediction_interval

    network = import_sioux_falls(
        edges_tntp_path, nodes_tntp_path, network_path, inflow_horizon)
    
    build_flows(network_path, flows_dir, inflow_horizon=inflow_horizon,
                number_flows=500, horizon=horizon, reroute_interval=reroute_interval)

    generate_queues_and_edge_loads(
        past_timesteps, flows_dir, queues_dir, horizon, reroute_interval, prediction_interval)

    test_mask = train_tf_full_net_model(queues_dir, past_timesteps, future_timesteps,
                                        reroute_interval, prediction_interval, horizon, network, full_net_model_path)

    def build_predictors(network): return {
        PredictorType.ZERO: ZeroPredictor(network),
        PredictorType.CONSTANT: ConstantPredictor(network),
        PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
        PredictorType.MACHINE_LEARNING: TFFullNetPredictor.from_model(
            network,
            full_net_model_path,
            test_mask,
            past_timesteps,
            future_timesteps,
            step_length=prediction_interval
        )
    }

    shallow_evaluate_predictors(network_path, flows_dir, shallow_eval_dir, past_timesteps, future_timesteps, 
                                horizon, reroute_interval, prediction_interval, build_predictors)
    # expanded_queues_from_flows_per_edge(network_path, past_timesteps, 1., future_timesteps, flows_dir,
    #                                    expanded_per_edge_dir, horizon, average=False, sample_step=1)

    #train_per_edge_model(network_path, expanded_per_edge_dir, models_per_edge_dir, past_timesteps, future_timesteps)

    eval_network_demand(
        network_path,
        10,
        eval_dir,
        inflow_horizon,
        future_timesteps,
        prediction_interval,
        reroute_interval,
        horizon,
        suppress_log=False,
        build_predictors=build_predictors)

    average_comp_times = []
    for file in os.listdir(eval_dir):
        if file.endswith(".json"):
            path = os.path.join(eval_dir, file)
            with open(path, "r") as file:
                d = json.load(file)
            average_comp_times.append(d["comp_time"])

    avg_comp_time = sum(average_comp_times) / len(average_comp_times)
    print(f"Average computing time: {avg_comp_time}")


if __name__ == "__main__":
    def main():
        edges_tntp_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp"
        nodes_tntp_path = "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_node.tntp"
        run_scenario(edges_tntp_path, nodes_tntp_path,
                     "./out/journal-sioux-falls")

    main()
