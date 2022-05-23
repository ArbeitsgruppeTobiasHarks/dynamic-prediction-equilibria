import json
import os

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.full_net_linear_regression_predictor import FullNetLinearRegressionPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.per_edge_linear_regression_predictor import PerEdgeLinearRegressionPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network
from importer.sioux_falls_importer import import_sioux_falls
from ml.SKLearnLinRegFullNetworkModel import train_full_net_model
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads


def run_scenario(edges_tntp_path: str, nodes_tntp_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    expanded_per_edge_dir = os.path.join(scenario_dir, "expanded-queues-per-edge")
    full_net_model_path = os.path.join(scenario_dir, "full-network-model")
    queues_dir = os.path.join(scenario_dir, "queues")
    models_per_edge_dir = os.path.join(scenario_dir, "models-per-edge")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = .25
    inflow_horizon = 25.
    horizon = 75
    past_timesteps = 20
    future_timesteps = 20
    pred_horizon = 20.
    prediction_interval = 1.

    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path, network_path, inflow_horizon)
    build_flows(network_path, flows_dir, inflow_horizon=inflow_horizon, number_flows=50, horizon=horizon, reroute_interval=reroute_interval)

    generate_queues_and_edge_loads(past_timesteps, flows_dir, queues_dir, horizon, step_length=prediction_interval)

    test_mask = train_full_net_model(queues_dir, future_timesteps, past_timesteps, network, full_net_model_path)

    #expanded_queues_from_flows_per_edge(network_path, past_timesteps, 1., future_timesteps, flows_dir,
    #                                    expanded_per_edge_dir, horizon, average=False, sample_step=1)
    
    #train_per_edge_model(network_path, expanded_per_edge_dir, models_per_edge_dir, past_timesteps, future_timesteps)

    eval_network(
        network_path,
        eval_dir,
        inflow_horizon,
        reroute_interval,
        horizon,
        random_commodities=False,
        suppress_log=False,
        build_predictors=lambda network: {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
            PredictorType.MACHINE_LEARNING: FullNetLinearRegressionPredictor.from_model(
                network,
                full_net_model_path,
                test_mask,
                past_timesteps,
                future_timesteps,
                step_length=prediction_interval
            )
        })

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
        run_scenario(edges_tntp_path, nodes_tntp_path, "./out/journal-sioux-falls")

    main()
