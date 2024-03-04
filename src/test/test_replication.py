import os

from core.network import Network
from core.predictors.predictor_type import PredictorType
from core.replicator import ReplicatorFlowBuilder
from utilities.json_encoder import JSONEncoder
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)

    network = Network()
    network.add_edge(0, 1, 1.0, 3.0)
    network.add_edge(0, 2, 2.0, 2.0)
    network.add_edge(1, 2, 1.0, 1.0)
    network.add_edge(1, 3, 2.0, 2.0)
    network.add_edge(2, 3, 1.0, 3.0)
    network.graph.positions = {0: (0, 1), 1: (1, 2), 2: (1, 0), 3: (2, 1)}

    # network.add_edge(0, 1, 1.0, 2.0)
    # network.add_edge(0, 1, 2.0, 3.0)
    # network.graph.positions = {0: (0, 0), 1: (1, 1)}

    network.add_commodity(
        {0: RightConstant([0.0], [5.0], (0.0, float("inf")))},
        3,
        PredictorType.CONSTANT,
    )

    run_params = dict(
        reroute_interval=0.01,
        horizon=100.0,
        initial_distribution=[([0, 2, 4], 0.4), ([0, 3], 0.3), ([1, 4], 0.3)],
        fitness="neg_proj_tt",
        regularization="logit",
        regularization_decay=1e-1,
        replication_coef=1e-1,
        window_size=0.5,
    )

    replicator = ReplicatorFlowBuilder(network, **run_params)
    flow, dynamics = replicator.run()

    with open(os.path.join(scenario_dir, f"run_data.json"), "w") as f:
        JSONEncoder().dump(
            {
                "run_parameters": run_params,
                "dynamics": dynamics,
                "means": {
                    "fitness": sum(
                        d["fitness"] * d["inflow share"] for d in dynamics.values()
                    ),
                    "travel time": sum(
                        d["travel time"] * d["inflow share"] for d in dynamics.values()
                    ),
                },
            },
            f,
        )

    visualization_path = os.path.join(scenario_dir, f"flow.vis.json")
    to_visualization_json(
        visualization_path,
        flow,
        flow._network,
        {0: "green", 1: "blue", 2: "red", 3: "purple", 4: "brown"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/replication-test")

    main()
