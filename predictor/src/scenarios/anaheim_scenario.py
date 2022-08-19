import json
from math import pi
import os
from core.network import Network

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.tf_full_net_predictor import TFFullNetPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network_demand
from importer.tntp_importer import import_network, natural_earth_projection
from ml.TFFullNetworkModel import train_tf_full_net_model
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads
from scenarios.scenario_utils import get_demand_with_inflow_horizon

def add_node_position_from_geojson(network: Network, geojson_path: str):
    with open(geojson_path) as geojson_file:
        geojson = json.load(geojson_file)
    for row in geojson["features"]:
        id = row["properties"]["id"]
        coordinates = row["geometry"]["coordinates"]
        network.graph.positions[id] = natural_earth_projection(coordinates[1] / 180 * pi, coordinates[0] / 180 * pi)
    assert all(node in network.graph.positions for node in network.graph.nodes.keys())

def run_scenario(edges_tntp_path: str, geojson_path: str, scenario_dir: str):
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
    number_training_flows = 500
    number_eval_flows = 20
    pred_horizon = future_timesteps * prediction_interval

    average_demand = 8000

    network = import_network(edges_tntp_path)
    add_node_position_from_geojson(network, geojson_path)


    network.add_commodity(
        26,
        20,
        get_demand_with_inflow_horizon(average_demand, inflow_horizon),
        PredictorType.CONSTANT
    )
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    demand_sigma = min(Network.from_file(network_path).capacity) / 2.

    network.print_info()

    build_flows(network_path, flows_dir, inflow_horizon,
                number_training_flows, horizon, reroute_interval, demand_sigma, check_for_optimizations=False)

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
            prediction_interval=prediction_interval
        )
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
        build_predictors=build_predictors)

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
        edges_tntp_path = "/home/michael/git/TransportationNetworks/Anaheim/Anaheim_net.tntp"
        geojson_path = "/home/michael/git/TransportationNetworks/Anaheim/anaheim_nodes.geojson"
        run_scenario(edges_tntp_path, geojson_path, "./out/journal-anaheim")

    main()
