import json
import os

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.expanded_linear_regression_predictor import (
    ExpandedLinearRegressionPredictor,
)
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network_for_commodities
from importer.csv_importer import add_demands_to_network, network_from_csv
from ml.build_test_flows import build_flows
from ml.generate_queues import expanded_queues_from_flows

# TODO: Potentially rewrite this with the new ML predictors. Otherwise, just remove it.


def run_scenario(arcs_path: str, demands_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = 2.5
    inflow_horizon = 25.0
    horizon = 100
    past_timesteps = 20
    future_timesteps = 20
    pred_horizon = 20.0
    prediction_interval = 1.0

    os.makedirs(scenario_dir, exist_ok=True)
    network = network_from_csv(arcs_path)
    add_demands_to_network(
        network, demands_path, inflow_horizon=inflow_horizon, use_default_demands=True
    )
    network.to_file(network_path)
    build_flows(
        network_path,
        flows_dir,
        inflow_horizon,
        number_flows=50,
        horizon=horizon,
        reroute_interval=reroute_interval,
    )

    expanded_queues_from_flows(
        network_path,
        past_timesteps,
        1.0,
        future_timesteps,
        flows_dir,
        scenario_dir,
        horizon,
        sample_step=10,
    )

    # train_expanded_model(
    #    os.path.join(scenario_dir, "expanded_queues.csv.gz"),
    #    scenario_dir,
    #    past_timesteps,
    #    future_timesteps,
    # )
    eval_network_for_commodities(
        network_path,
        eval_dir,
        inflow_horizon,
        future_timesteps,
        prediction_interval,
        reroute_interval,
        horizon,
        random_commodities=True,
        build_predictors=lambda network: {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(
                network, pred_horizon, delta=1.0
            ),
            PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: ExpandedLinearRegressionPredictor.from_scikit_model(
                network,
                os.path.join(scenario_dir, "expanded-model.pickle"),
                past_timesteps,
                future_timesteps,
            ),
        },
    )


if __name__ == "__main__":

    def main():
        arcs_path = "C:/Users/Michael/Nextcloud/Universität/2021/softwareproject/data/from-kostas/tokyo_tiny.arcs"
        demands_path = "C:/Users/Michael/Nextcloud/Universität/2021/softwareproject/data/from-kostas/tokyo_tiny.demands"
        run_scenario(arcs_path, demands_path, "../../out/aaai-tokyo-scenario")

    main()
    scenario_dir = "../../out/aaai-tokyo-scenario"
    eval_dir = os.path.join(scenario_dir, "eval")

    average_comp_times = []
    for file in os.listdir(eval_dir):
        if file.endswith(".json") and not file.startswith(".lock"):
            path = os.path.join(eval_dir, file)
            with open(path, "r") as file:
                d = json.load(file)
            average_comp_times.append(d["comp_time"])

    avg_comp_time = sum(average_comp_times) / len(average_comp_times)
    print(f"Average computing time: {avg_comp_time}")
