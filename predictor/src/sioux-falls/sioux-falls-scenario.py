import os

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.expanded_linear_regression_predictor import ExpandedLinearRegressionPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network
from importer.sioux_falls_importer import import_sioux_falls
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues, expanded_queues_from_flows

scenario_dir = "../../out/aaai-sioux-falls"
network_path = os.path.join(scenario_dir, "network.pickle")
flows_dir = os.path.join(scenario_dir, "flows")
models_dir = os.path.join(scenario_dir, "weka/models")
eval_dir = os.path.join(scenario_dir, "eval")


def prepare_scenario():
    import_sioux_falls(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp",
        network_path,
        inflow_horizon=50.
    )
    horizon = 200
    reroute_interval = 1.

    build_flows(network_path, flows_dir, number_flows=200, horizon=horizon, reroute_interval=reroute_interval)
    generate_queues(10, flows_dir, os.path.join(scenario_dir, "queues"), horizon, step_length=2)
    expanded_queues_from_flows(network_path, 10, 2., 10, flows_dir, scenario_dir, horizon, 10)


def eval_scenario():
    pred_horizon = 20.
    eval_network(network_path, eval_dir, build_predictors=lambda network: {
        PredictorType.ZERO: ZeroPredictor(network),
        PredictorType.CONSTANT: ConstantPredictor(network),
        PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=4.),
        PredictorType.MACHINE_LEARNING: ExpandedLinearRegressionPredictor.from_models(network, models_dir)
    },
    check_for_optimizations=False)


if __name__ == "__main__":
    eval_scenario()
