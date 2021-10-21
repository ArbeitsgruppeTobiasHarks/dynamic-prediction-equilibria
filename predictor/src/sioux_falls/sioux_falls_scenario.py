import json
import os

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.per_edge_linear_regression_predictor import PerEdgeLinearRegressionPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network
from importer.sioux_falls_importer import import_sioux_falls, DemandsRangeBuilder
from ml.SKLearnLinRegPerEdgeModel import train_per_edge_model
from ml.build_test_flows import build_flows
from ml.generate_queues import expanded_queues_from_flows_per_edge


def run_scenario(tntp_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    expanded_per_edge_dir = os.path.join(scenario_dir, "expanded-queues-per-edge")
    models_per_edge_dir = os.path.join(scenario_dir, "models-per-edge")
    eval_dir = os.path.join(scenario_dir, "eval")

    demands_range_builder: DemandsRangeBuilder = lambda net: (min(net.capacity), max(net.capacity))
    reroute_interval = 1
    inflow_horizon = 25.
    horizon = 100
    past_timesteps = 20
    future_timesteps = 20
    pred_horizon = 20.

    network = import_sioux_falls(tntp_path, network_path, inflow_horizon, demands_range_builder)
    demands_range = demands_range_builder(network)
    build_flows(network_path, flows_dir, number_flows=50, horizon=horizon, reroute_interval=reroute_interval,
                demands_range=demands_range)

    expanded_queues_from_flows_per_edge(network_path, past_timesteps, 1., future_timesteps, flows_dir,
                                        expanded_per_edge_dir, horizon, average=False, sample_step=1)
    train_per_edge_model(network_path, expanded_per_edge_dir, models_per_edge_dir, past_timesteps, future_timesteps)

    eval_network(
        network_path,
        eval_dir,
        inflow_horizon,
        reroute_interval,
        horizon,
        random_commodities=True,
        suppress_log=True,
        build_predictors=lambda network: {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
            PredictorType.MACHINE_LEARNING: PerEdgeLinearRegressionPredictor.from_models(
                network,
                models_per_edge_dir,
                past_timesteps,
                future_timesteps,
                average=False
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
        tntp_path = "/mnt/c/Users/Tür an Tür/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp"
        run_scenario(tntp_path, "../../out/aaai-sioux-falls")

    main()
