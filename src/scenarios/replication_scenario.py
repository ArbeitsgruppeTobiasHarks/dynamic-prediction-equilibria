import os

from matplotlib import pyplot as plt

from core.network import Network
from core.predictors.predictor_type import PredictorType
from core.replicator import ReplicatorFlowBuilder
from utilities.json_encoder import JSONEncoder
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)

    network = Network()
    network.add_edge(0, 1, 1.0, 2.0)
    network.add_edge(0, 1, 2.0, 3.0)
    network.graph.positions = {0: (0, 0), 1: (1, 1)}

    network.add_commodity(
        {0: RightConstant([0.0], [5.0])},
        1,
        PredictorType.CONSTANT,
    )

    run_params = dict(
        reroute_interval=0.001,
        horizon=250.0,
        initial_distribution=[([0], 0.5), ([1], 0.5)],
        fitness="neg_proj_tt",
        replication_coef=0.1,
        regularization=None,
        regularization_coef=1.0,
        regularization_decay=1e-3,
        window_size=300,
    )

    replicator = ReplicatorFlowBuilder(network, **run_params)
    flow, dynamics = replicator.run()

    out_dir = os.path.join(scenario_dir, f"run_{len(os.listdir(scenario_dir))}")
    os.makedirs(out_dir)

    run_data = {
        "run_parameters": run_params,
        "dynamics": dynamics,
        "means": {
            "fitness": sum(d["fitness"] * d["inflow share"] for d in dynamics.values()),
            "travel time": sum(
                d["travel time"] * d["inflow share"] for d in dynamics.values()
            ),
        },
    }

    with open(os.path.join(out_dir, f"run_data.json"), "w") as f:
        JSONEncoder().dump(run_data, f)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for k, d in run_data["dynamics"].items():
        ax1.plot(d["inflow share"].times, d["inflow share"].values, label=d["path"])
        ax2.plot(d["fitness"].times, d["fitness"].values, label=d["path"])
        ax3.plot(d["travel time"].times, d["travel time"].values, label=d["path"])

    ax2.plot(
        run_data["means"]["fitness"].times,
        run_data["means"]["fitness"].values,
        "--",
        label="mean",
    )
    ax3.plot(
        run_data["means"]["travel time"].times,
        run_data["means"]["travel time"].values,
        "--",
        label="mean",
    )

    ax1.set_ylim(0, 1)
    ax3.set_xlabel("route time")
    ax1.set_ylabel("inflow share")
    ax2.set_ylabel("fitness")
    ax3.set_ylabel("travel time")
    ax2.legend(loc="upper right")

    plt.savefig(os.path.join(out_dir, f"replicator.eps"), format="eps")

    visualization_path = os.path.join(out_dir, f"flow.vis.json")
    to_visualization_json(
        visualization_path,
        flow,
        flow._network,
        {0: "green", 1: "blue", 2: "red", 3: "purple", 4: "brown"},
    )


if __name__ == "__main__":

    def main():
        run_scenario("./out/replication")

    main()
