from core.dpe_flow_builder import FlowBuilder
from core.network import Network
from core.network_loader import NetworkLoader
from core.predictors.predictor_type import PredictorType
from core.predictors.zero_predictor import ZeroPredictor
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def build_example_nash():
    network = Network()
    # s: 0
    # u: 1
    # v: 2
    # w: 3
    # t: 4
    e1 = network.add_edge(0, 1, 200, 20)
    e2 = network.add_edge(1, 3, 100, 20)
    e3 = network.add_edge(3, 4, 100, 10)
    e4 = network.add_edge(1, 2, 200, 20)
    e5 = network.add_edge(2, 4, 200, 20)

    network.graph.positions = {
        0: (0, 0),
        1: (200, 0),
        2: (400, 0),
        3: (200 - 141.421356237, 141.421356237),
        4: (300, 0),
    }

    loader = NetworkLoader(network, [
        ([e1, e2, e3], RightConstant([0, 300, 500, 700, 850], [20, 0, 20, 0, 0], (0, float('inf')))),
        ([e1, e4, e5], RightConstant([0, 300, 500, 700, 850], [0, 20, 0, 20, 0], (0, float('inf'))))
    ])



    color_by_comm_idx = {0: "red", 1: "blue"}

    horizon = 2000
    generator = loader.build_flow()
    flow = next(generator)
    while flow.phi < horizon:
        flow = next(generator)

    to_visualization_json("./exampleIde.vis.json", flow, network, color_by_comm_idx)


if __name__ == "__main__":
    build_example_nash()
