import json
import os
import numpy as np
from core.dynamic_flow import DynamicFlow
from core.network import Network
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network_demand
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from ml.SKFullNetworkModel import train_sk_full_net_model
from ml.SKNeighborhood import train_sk_neighborhood_model
from ml.TFFullNetworkModel import train_tf_full_net_model
from ml.TFNeighborhood import train_tf_neighborhood_model
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads, save_queues_and_edge_loads_for_flow


def run_scenario(edges_tntp_path: str, nodes_tntp_path: str, od_pairs_file_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    tf_full_net_model_path = os.path.join(
        scenario_dir, "tf-full-network-model")
    tf_neighborhood_models_path = os.path.join(
        scenario_dir, "tf-neighborhood-models")
    sk_full_net_model_path = os.path.join(scenario_dir, "sk-full-net-model")
    sk_neighborhood_models_path = os.path.join(
        scenario_dir, "sk-neighborhood-models")
    queues_and_edge_loads_dir = os.path.join(scenario_dir, "queues-and-edge-loads")
    shallow_eval_dir = os.path.join(scenario_dir, "shallow-eval")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = .125
    inflow_horizon = 12.
    horizon = 60.
    past_timesteps = 20
    future_timesteps = 20
    prediction_interval = 1.
    number_training_flows = 500
    number_eval_flows = 20
    max_distance = 3

    pred_horizon = future_timesteps * prediction_interval

    network = import_sioux_falls(edges_tntp_path, nodes_tntp_path)
    add_od_pairs(network, od_pairs_file_path, inflow_horizon)
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    demand_sigma = min(Network.from_file(network_path).capacity) / 8.

    network.print_info()

    def on_flow_computed(flow_id: str, flow: DynamicFlow):
        out_path = os.path.join(
            queues_and_edge_loads_dir, f"{flow_id}-queues-and-edge-loads.npy")
        save_queues_and_edge_loads_for_flow(out_path, past_timesteps, horizon, reroute_interval, prediction_interval,
                                            flow)

    build_flows(network_path, flows_dir, inflow_horizon,
                number_training_flows, horizon, reroute_interval, demand_sigma, check_for_optimizations=False,
                on_flow_computed=on_flow_computed)

    generate_queues_and_edge_loads(
        past_timesteps, flows_dir, queues_and_edge_loads_dir, horizon, reroute_interval, prediction_interval)

    build_tf_full_net_predictor = train_tf_full_net_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps,
                                                          reroute_interval, prediction_interval, horizon, network,
                                                          tf_full_net_model_path)
    build_tf_neighborhood_predictor = train_tf_neighborhood_model(queues_and_edge_loads_dir, past_timesteps,
                                                                  future_timesteps,
                                                                  reroute_interval, prediction_interval, horizon,
                                                                  network, tf_neighborhood_models_path, max_distance)
    build_sk_full_net_predictor = train_sk_full_net_model(queues_and_edge_loads_dir, past_timesteps, future_timesteps,
                                                          reroute_interval, prediction_interval, horizon, network,
                                                          sk_full_net_model_path)
    build_sk_neighborhood_predictor = train_sk_neighborhood_model(queues_and_edge_loads_dir, past_timesteps,
                                                                  future_timesteps,
                                                                  reroute_interval, prediction_interval, horizon,
                                                                  network, sk_neighborhood_models_path, max_distance)

    def build_predictors(network: Network):
        return {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
            PredictorType.MACHINE_LEARNING_TF_FULL_NET: build_tf_full_net_predictor(network),
            PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: build_tf_neighborhood_predictor(network),
            PredictorType.MACHINE_LEARNING_SK_FULL_NET: build_sk_full_net_predictor(network),
            PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: build_sk_neighborhood_predictor(network),
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
            PredictorType.LINEAR: ("{rgb,255:red,0; green,128; blue,0}", "$\\hat q^{\\text{L}}$"),
            PredictorType.REGULARIZED_LINEAR: ("orange", "$\\hat q^{\\text{RL}}$"),
            PredictorType.MACHINE_LEARNING_TF_FULL_NET: ("black", "$\\hat q^{\\text{NN-full}}$"),
            PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: ("black", "$\\hat q^{\\text{NN-neighboring}}$"),
            PredictorType.MACHINE_LEARNING_SK_FULL_NET: ("black", "$\\hat q^{\\text{LR-full}}$"),
            PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: ("black", "$\\hat q^{\\text{LR-neighboring}}$"),
        })

    average_comp_times = []
    for file in os.listdir(eval_dir):
        if file.endswith(".json"):
            path = os.path.join(eval_dir, file)
            with open(path, "r") as file:
                d = json.load(file)
            average_comp_times.append(d["computation_time"])

    avg_comp_time = sum(average_comp_times) / len(average_comp_times)
    print(f"Average computing time: {avg_comp_time}")

    network = Network.from_file(network_path)
    network.print_info()
    print(f"""Average Demand: {np.average([
        source.values[0]
        for c in network.commodities
        for source in c.sources.values()
    ])}""")


if __name__ == "__main__":
    def main():
        edges_tntp_path = os.path.expanduser("~/git/TransportationNetworks/SiouxFalls/SiouxFalls_net.tntp")
        nodes_tntp_path = os.path.expanduser("~/git/TransportationNetworks/SiouxFalls/SiouxFalls_node.tntp")
        od_pairs_csv_path = os.path.expanduser("~/git/TransportationNetworks/SiouxFalls/CSV-data/SiouxFalls_od.csv")
        run_scenario(edges_tntp_path, nodes_tntp_path,
                     od_pairs_csv_path, "./out/journal-sioux-falls-multi-com")


    main()
