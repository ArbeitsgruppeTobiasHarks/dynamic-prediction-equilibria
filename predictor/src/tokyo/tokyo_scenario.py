import os

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.single_linear_regression_predictor import SingleLinearRegressionPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network
from importer.csv_importer import network_from_csv, add_demands_to_network
from ml.single_model import train_single_model
from ml.build_test_flows import build_flows
from ml.generate_queues import expanded_queues_from_flows


def run_scenario(arcs_path: str, demands_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = 2.5
    inflow_horizon = 25.
    horizon = 100
    past_timesteps = 20
    future_timesteps = 20
    pred_horizon = 20.

    os.makedirs(scenario_dir, exist_ok=True)
    network = network_from_csv(
        arcs_path
    )
    add_demands_to_network(
        network,
        demands_path,
        inflow_horizon=inflow_horizon,
        use_default_demands=True
    )
    network.to_file(network_path)
    build_flows(network_path, flows_dir, number_flows=50, horizon=horizon, reroute_interval=reroute_interval)

    expanded_queues_from_flows(network_path, past_timesteps, 1., future_timesteps, flows_dir, scenario_dir, horizon,
                               sample_step=10)
    train_single_model(os.path.join(scenario_dir, "expanded_queues.csv.gz"), scenario_dir, past_timesteps,
                       future_timesteps)
    eval_network(
        network_path,
        eval_dir,
        inflow_horizon,
        reroute_interval,
        horizon,
        random_commodities=True,
        build_predictors=lambda network: {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
            PredictorType.MACHINE_LEARNING: SingleLinearRegressionPredictor.from_scikit_model(
                network,
                os.path.join(scenario_dir, "expanded-model.pickle"),
                past_timesteps,
                future_timesteps
            )
        })
