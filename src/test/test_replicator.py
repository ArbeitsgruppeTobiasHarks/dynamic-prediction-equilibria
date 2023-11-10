import json
import os
import pickle

from core.network import Network
from core.predictors.predictor_type import PredictorType
from core.replicator import DynamicReplicator
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)

    network = Network()
    network.add_edge(0, 1, 1.0, 2.0)
    network.add_edge(0, 2, 1.0, 1.0)
    network.add_edge(1, 2, 0.5, 0.5)
    network.add_edge(1, 3, 2.0, 1.5)
    network.add_edge(2, 3, 1.0, 1.5)
    network.graph.positions = {0: (0, 1), 1: (1, 2), 2: (1, 0), 3: (2, 1)}

    network.add_commodity(
        {0: RightConstant([0.0], [3.0], (0, float("inf")))}, 3, PredictorType.CONSTANT
    )

    run_parameters = dict(
        reroute_interval=0.01,
        horizon=25.0,
        initial_distribution=[([0, 3], 0.8), ([1, 4], 0.1), ([0, 2, 4], 0.1)],
        replication_window=5.0,
    )
    replicator = DynamicReplicator(network, **run_parameters)

    flow, path_distribution = replicator.run()

    visualization_path = os.path.join(scenario_dir, f"flow.vis.json")
    to_visualization_json(
        visualization_path,
        flow,
        network,
        {0: "green", 1: "blue", 2: "red", 3: "purple", 4: "brown"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/replicator-test")

    main()
