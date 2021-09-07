import os

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.expanded_linear_regression_predictor import ExpandedLinearRegressionPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network
from importer.sioux_falls_importer import import_sioux_falls, DemandsRangeBuilder
from ml.SKLearnLinRegPerEdgeModel import train_per_edge_model
from ml.build_test_flows import build_flows
from ml.generate_queues import expanded_queues_from_flows_per_edge

scenario_dir = "../../out/aaai-sioux-falls"
network_path = os.path.join(scenario_dir, "network.pickle")
flows_dir = os.path.join(scenario_dir, "flows")
weka_models_dir = os.path.join(scenario_dir, "weka/models")
expanded_per_edge_dir = os.path.join(scenario_dir, "avg-expanded-queues-per-edge")
models_per_edge_dir = os.path.join(scenario_dir, "avg-models-per-edge")
queues_dir = os.path.join(scenario_dir, "queues")
eval_dir = os.path.join(scenario_dir, "eval")
sklearn_full_net_model = os.path.join(scenario_dir, "sklearn-full-net-model.pickle")


def prepare_scenario():
    inflow_horizon = 50.
    demands_range_builder: DemandsRangeBuilder = lambda net: (min(net.capacity), max(net.capacity))
    network = import_sioux_falls(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp",
        network_path, inflow_horizon, demands_range_builder
    )
    demands_range = demands_range_builder(network)
    horizon = 200
    reroute_interval = 1.
    build_flows(network_path, flows_dir, number_flows=400, horizon=horizon, reroute_interval=reroute_interval,
                demands_range=demands_range)
    # generate_queues(20, flows_dir, queues_dir, horizon, step_length=1)
    # expanded_queues_from_flows(network_path, 10, 2., 10, flows_dir, scenario_dir, horizon, 10)
    expanded_queues_from_flows_per_edge(network_path, 20, 1., 20, flows_dir, expanded_per_edge_dir, horizon,
                                        sample_step=5)
    train_per_edge_model(network_path, expanded_per_edge_dir, models_per_edge_dir, 20, 20)


def eval_scenario():
    inflow_horizon = 50.
    demands_range_builder: DemandsRangeBuilder = lambda net: (min(net.capacity), max(net.capacity))
    import_sioux_falls(
        "/home/michael/Nextcloud/Universität/2021/softwareproject/data/sioux-falls/SiouxFalls_net.tntp",
        network_path,
        inflow_horizon,
        demands_range_builder
    )
    pred_horizon = 20.
    eval_network(network_path, eval_dir, inflow_horizon, build_predictors=lambda network: {
        PredictorType.ZERO: ZeroPredictor(network),
        PredictorType.CONSTANT: ConstantPredictor(network),
        PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
        PredictorType.MACHINE_LEARNING: ExpandedLinearRegressionPredictor.from_models(network, weka_models_dir)
    },
                 check_for_optimizations=False)


if __name__ == "__main__":
    eval_scenario()
