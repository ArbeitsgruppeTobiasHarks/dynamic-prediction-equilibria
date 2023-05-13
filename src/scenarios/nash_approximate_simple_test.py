from typing import List, Tuple
from core.nash_flow_builder import NashFlowBuilder
from core.network import Network
from core.network_loader import NetworkLoader, Path
from core.predictors.predictor_type import PredictorType
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def run_scenario():
    network = Network()
    network.add_edge(0, 1, 10, 10)
    network.add_edge(0, 1, 20, 10)

    network.graph.positions = {
        0: (0, 0),
        1: (1, 0)
    }
    network.add_commodity({ 0: RightConstant([0, 20], [20, 0], (0, float('inf')))}, 1, predictor_type=PredictorType.CONSTANT)

    loader = NashFlowBuilder(network)
    flow, path_dist = loader.build_flow()
    to_visualization_json("./test.json", flow,  network, {0: 'red', 1: 'blue'})


if __name__ == "__main__":
    print("Running scenario.")
    run_scenario()
