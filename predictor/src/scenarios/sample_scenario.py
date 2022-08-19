import json
import os
from core.network import Network

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.tf_full_net_predictor import TFFullNetPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network_demand
from ml.TFFullNetworkModel import train_tf_full_net_model
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads
from test.sample_network import build_sample_network
from utilities.right_constant import RightConstant


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    full_net_model_path = os.path.join(scenario_dir, "full-network-model")
    queues_and_edge_loads_dir = os.path.join(scenario_dir, "queues")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = .125
    inflow_horizon = 12.
    horizon = 50.
    prediction_interval = .5
    past_timesteps = 20
    future_timesteps = 20
    pred_horizon = 20.
    demand = 4.
    number_training_flows = 500
    number_eval_flows = 20

    network = build_sample_network()
    network.add_commodity(0, 2, RightConstant([inflow_horizon, 0.], [
                          demand, 0.], (0., float('inf'))), PredictorType.CONSTANT)
    network.to_file(network_path)
    build_flows(network_path, flows_dir, inflow_horizon=inflow_horizon, number_flows=number_training_flows, horizon=horizon, reroute_interval=reroute_interval,
                check_for_optimizations=False)

    generate_queues_and_edge_loads(
        past_timesteps, flows_dir, queues_and_edge_loads_dir, horizon, reroute_interval, prediction_interval)

    input_mask, output_mask = train_tf_full_net_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps,
                                        reroute_interval, prediction_interval, horizon, network, full_net_model_path)

    def build_predictors(network): return {
        PredictorType.ZERO: ZeroPredictor(network),
        PredictorType.CONSTANT: ConstantPredictor(network),
        PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
        PredictorType.MACHINE_LEARNING: TFFullNetPredictor.from_model(
            network,
            full_net_model_path,
            input_mask,
            output_mask,
            past_timesteps,
            future_timesteps,
            prediction_interval=prediction_interval
        )
    }

    # test_mask = train_full_net_model(
    #    queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval, prediction_interval, horizon, network, full_net_model_path)

    # expanded_queues_from_flows_per_edge(network_path, past_timesteps, 1., future_timesteps, flows_dir,
    #                                    expanded_per_edge_dir, horizon, average=False, sample_step=1)

    #train_per_edge_model(network_path, expanded_per_edge_dir, models_per_edge_dir, past_timesteps, future_timesteps)

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
        check_for_optimizations=False)

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
        run_scenario("./out/journal-sample")

    main()
