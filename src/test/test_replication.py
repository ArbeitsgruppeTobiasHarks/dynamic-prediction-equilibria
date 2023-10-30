import os
import pickle

from core.convergence import AlphaFlowIterator
from core.network import Network
from core.predictors.predictor_type import PredictorType
from core.replicator import ReplicatorFlowBuilder
from importer.sioux_falls_importer import add_od_pairs, import_sioux_falls
from scenarios.nguyen_scenario import build_nguyen_network
from scenarios.scenario_utils import get_demand_with_inflow_horizon
from utilities.combine_commodities import combine_commodities_with_same_sink
from utilities.get_tn_path import get_tn_path
from utilities.right_constant import RightConstant
from visualization.to_json import merge_commodities, to_visualization_json


def run_scenario(scenario_dir: str):
    os.makedirs(scenario_dir, exist_ok=True)

    reroute_interval = 0.1
    inflow_horizon = 10.0
    demand = 5

    horizon = 60.0

    network = Network()
    network.add_edge(0, 1, 1.0, 2.0)
    network.add_edge(0, 1, 2.0, 3.0)
    network.graph.positions = {0: (0, 0), 1: (1, 1)}

    network.add_commodity(
        {0: RightConstant([0.0], [5.0])}, 1, PredictorType.CONSTANT,
    )
    initial_distribution = [([0], 0.9), ([1], 0.1)]
    replicator = ReplicatorFlowBuilder(network, reroute_interval, initial_distribution)
    flow = replicator.run(horizon)

    visualization_path = os.path.join(scenario_dir, f"merged_flow.vis.json")
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
