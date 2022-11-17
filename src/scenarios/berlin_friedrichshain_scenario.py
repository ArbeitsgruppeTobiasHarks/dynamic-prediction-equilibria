from curses import nonl
import json
from math import pi
import os

from core.dynamic_flow import DynamicFlow
from core.network import Network

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network_demand
from importer.tntp_importer import add_commodities, add_node_positions, import_network, natural_earth_projection
from ml.SKFullNetworkModel import train_sk_full_net_model
from ml.SKNeighborhood import train_sk_neighborhood_model
from ml.TFNeighborhood import train_tf_neighborhood_model
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads, save_queues_and_edge_loads_for_flow
from visualization.to_json import to_visualization_json


def run_scenario(edges_tntp_path: str, trip_tntp_file_path: str, node_tntp_file_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    tf_neighborhood_models_path = os.path.join(
        scenario_dir, "tf-neighborhood-models")
    sk_neighborhood_models_path = os.path.join(
        scenario_dir, "sk-neighborhood-models")
    sk_full_net_path = os.path.join(scenario_dir, "sk-full-net-model")
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
    pred_horizon = future_timesteps * prediction_interval
    max_distance = 3

    network = import_network(edges_tntp_path)

    def visualize_network(network): 
        to_visualization_json(network_path + ".json", DynamicFlow(network), network, {})



    add_node_positions(network, node_tntp_file_path)
    visualize_network(network)
    add_commodities(network, trip_tntp_file_path, inflow_horizon)

    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    demand_sigma = min(Network.from_file(network_path).capacity) / 2.

    network.print_info()

    def on_flow_computed(flow_id: str, flow: DynamicFlow):
        out_path = os.path.join(
            queues_and_edge_loads_dir, f"{flow_id}-queues-and-edge-loads.npy")
        save_queues_and_edge_loads_for_flow(out_path, past_timesteps, horizon, reroute_interval, prediction_interval,
                                            flow)

    build_flows(network_path, flows_dir, inflow_horizon,
                number_training_flows, horizon, reroute_interval, demand_sigma, check_for_optimizations=False,
                on_flow_computed=on_flow_computed, generate_visualization=True, save_dummy=True)

    generate_queues_and_edge_loads(
        past_timesteps, flows_dir, queues_and_edge_loads_dir, horizon, reroute_interval, prediction_interval)

    build_tf_neighborhood_predictor = train_tf_neighborhood_model(
        queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval, prediction_interval, horizon,
        network, tf_neighborhood_models_path, max_distance)

    build_sk_neighborhood_predictor = train_sk_neighborhood_model(
        queues_and_edge_loads_dir, past_timesteps, future_timesteps, reroute_interval, prediction_interval, horizon,
        network, sk_neighborhood_models_path, max_distance)



    tf_predictor = None
    sk_predictor = None

    def build_predictors(network: Network):
        nonlocal tf_predictor
        nonlocal sk_predictor
        if tf_predictor is None:
            tf_predictor = build_tf_neighborhood_predictor(network)
        if sk_predictor is None:
            sk_predictor = build_sk_neighborhood_predictor(network)

        return {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(network, pred_horizon, delta=1.),
            PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: sk_predictor,
            PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: tf_predictor,
        }

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
            PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: ("black", "$\\hat q^{\\text{LR-neighboring}}$"),
            PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: ("black", "$\\hat q^{\\text{NN-neighboring}}$"),
        },
        generate_flow_visualization=False)

    average_comp_times = []
    for file in os.listdir(eval_dir):
        if file.endswith(".json"):
            path = os.path.join(eval_dir, file)
            with open(path, "r") as file:
                d = json.load(file)
            average_comp_times.append(d["computation_time"])

    avg_comp_time = sum(average_comp_times) / len(average_comp_times)
    print(f"Average computing time: {avg_comp_time}")


if __name__ == "__main__":
    def main():
        edges_tntp_path = os.path.expanduser("~/git/TransportationNetworks/Berlin-Friedrichshain/friedrichshain-center_net.tntp")
        nodes_tntp_file_path = os.path.expanduser("~/git/TransportationNetworks/Berlin-Friedrichshain/friedrichshain-center_node.tntp")
        trips_tntp_file_path = os.path.expanduser("~/git/TransportationNetworks/Berlin-Friedrichshain/friedrichshain-center_trips.tntp")
        run_scenario(edges_tntp_path, trips_tntp_file_path,
                     nodes_tntp_file_path, "./out/berlin-friedrichshain")


    main()
