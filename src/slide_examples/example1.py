from core.flow_builder import FlowBuilder
from core.network import Network
from core.predictors.predictor_type import PredictorType
from core.predictors.zero_predictor import ZeroPredictor
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def build_example1():
    network = Network()
    network.add_edge(0, 1, 200, 20)
    network.add_edge(1, 2, 200, 10)
    network.add_edge(3, 1, 200, 10)
    network.add_commodity(
        {
            0: RightConstant(
                [0, 400, 600, 1200, 1300], [10, 20, 0, 20, 0], domain=(0, float("inf"))
            )
        },
        2,
        PredictorType.ZERO,
    )
    network.add_commodity(
        {3: RightConstant([0, 1200, 1300], [0, 10, 0], domain=(0, float("inf")))},
        2,
        PredictorType.ZERO,
    )

    network.graph.positions = {
        0: (0, 0),
        1: (200, 0),
        2: (400, 0),
        3: (200 - 141.421356237, 141.421356237),
    }

    color_by_comm_idx = {0: "red", 1: "blue"}

    horizon = 2000
    predictors = {PredictorType.ZERO: ZeroPredictor(network)}

    builder = FlowBuilder(network=network, predictors=predictors, reroute_interval=100)
    generator = builder.build_flow()
    flow = next(generator)
    while flow.phi < horizon:
        flow = next(generator)

    to_visualization_json("./example1.vis.json", flow, network, color_by_comm_idx)


if __name__ == "__main__":
    build_example1()
