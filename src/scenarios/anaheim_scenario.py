import json
from math import pi
import os
from core.network import Network

from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from core.predictors.zero_predictor import ZeroPredictor
from eval.evaluate_network import eval_network_demand
from importer.tntp_importer import import_network, natural_earth_projection
from ml.SKFullNetworkModel import train_sk_full_net_model
from ml.SKNeighborhood import train_sk_neighborhood_model
from ml.TFNeighborhood import train_tf_neighborhood_model
from ml.build_test_flows import build_flows
from ml.generate_queues import generate_queues_and_edge_loads
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.get_tn_path import get_tn_path


def add_node_position_from_geojson(network: Network, geojson_path: str):
    with open(geojson_path) as geojson_file:
        geojson = json.load(geojson_file)
    for row in geojson["features"]:
        id = row["properties"]["id"]
        coordinates = row["geometry"]["coordinates"]
        network.graph.positions[id] = natural_earth_projection(
            coordinates[1] / 180 * pi, coordinates[0] / 180 * pi
        )
    assert all(node in network.graph.positions for node in network.graph.nodes.keys())


def run_scenario(edges_tntp_path: str, geojson_path: str, scenario_dir: str):
    network_path = os.path.join(scenario_dir, "network.pickle")
    flows_dir = os.path.join(scenario_dir, "flows")
    tf_neighborhood_models_path = os.path.join(scenario_dir, "tf-neighborhood-models")
    sk_neighborhood_models_path = os.path.join(scenario_dir, "sk-neighborhood-models")
    sk_full_net_path = os.path.join(scenario_dir, "sk-full-net-model")
    queues_dir = os.path.join(scenario_dir, "queues")
    shallow_eval_dir = os.path.join(scenario_dir, "shallow-eval")
    eval_dir = os.path.join(scenario_dir, "eval")

    reroute_interval = 0.125
    inflow_horizon = 12.0
    horizon = 60.0
    past_timesteps = 20
    future_timesteps = 20
    prediction_interval = 1.0
    number_training_flows = 500
    number_eval_flows = 20
    pred_horizon = future_timesteps * prediction_interval
    max_distance = 3

    average_demand = 8000

    network = import_network(edges_tntp_path)
    add_node_position_from_geojson(network, geojson_path)

    network.add_commodity(
        {26: get_demand_with_inflow_horizon(average_demand, inflow_horizon)},
        20,
        PredictorType.CONSTANT,
    )
    os.makedirs(os.path.dirname(network_path), exist_ok=True)
    network.to_file(network_path)

    demand_sigma = min(Network.from_file(network_path).capacity) / 2.0

    network.print_info()

    build_flows(
        network_path,
        flows_dir,
        inflow_horizon,
        number_training_flows,
        horizon,
        reroute_interval,
        demand_sigma,
        check_for_optimizations=False,
    )

    generate_queues_and_edge_loads(
        past_timesteps,
        flows_dir,
        queues_dir,
        horizon,
        reroute_interval,
        prediction_interval,
    )

    build_tf_neighborhood_predictor = train_tf_neighborhood_model(
        queues_dir,
        past_timesteps,
        future_timesteps,
        reroute_interval,
        prediction_interval,
        horizon,
        network,
        tf_neighborhood_models_path,
        max_distance,
    )

    build_sk_neighborhood_predictor = train_sk_neighborhood_model(
        queues_dir,
        past_timesteps,
        future_timesteps,
        reroute_interval,
        prediction_interval,
        horizon,
        network,
        sk_neighborhood_models_path,
        max_distance,
    )

    sk_predictor = None
    tf_predictor = None

    def build_predictors(network: Network):
        nonlocal sk_predictor
        nonlocal tf_predictor
        if sk_predictor is None:
            sk_predictor = build_sk_neighborhood_predictor(network)
        if tf_predictor is None:
            tf_predictor = build_tf_neighborhood_predictor(network)

        return {
            PredictorType.ZERO: ZeroPredictor(network),
            PredictorType.CONSTANT: ConstantPredictor(network),
            PredictorType.LINEAR: LinearPredictor(network, pred_horizon),
            PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(
                network, pred_horizon, delta=1.0
            ),
            PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: sk_predictor,
            PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: tf_predictor,
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
            PredictorType.LINEAR: (
                "{rgb,255:red,0; green,128; blue,0}",
                "$\\hat q^{\\text{L}}$",
            ),
            PredictorType.REGULARIZED_LINEAR: ("orange", "$\\hat q^{\\text{RL}}$"),
            PredictorType.MACHINE_LEARNING_SK_NEIGHBORHOOD: (
                "black",
                "$\\hat q^{\\text{LR-neighboring}}$",
            ),
            PredictorType.MACHINE_LEARNING_TF_NEIGHBORHOOD: (
                "black",
                "$\\hat q^{\\text{NN-neighboring}}$",
            ),
        },
    )

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
        tn_path = get_tn_path()
        edges_tntp_path = os.path.join(tn_path, "Anaheim/Anaheim_net.tntp")
        geojson_path = os.path.join(tn_path, "Anaheim/anaheim_nodes.geojson")
        run_scenario(edges_tntp_path, geojson_path, "./out/journal-anaheim")

    main()
