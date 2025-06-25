from math import ceil
from core.discrete_flow_builder import DiscreteFlowBuilder
from core.network import Network
from core.predictors.constant_predictor import ConstantPredictor
from core.predictors.predictor_type import PredictorType
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def build_example_discrete():
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
        2: (200 + 141.421356237, 141.421356237),
        3: (400, 0),
        4: (600, 0),
    }

    num_points = 50

    times = [0]
    values1 = [20]
    values2 = [0]

    for t in (280, 480, 680, 850):
        for i in range(1, num_points + 1):
            time = t + 40 * i / num_points
            times.append(time)
            if time >= 850:
                values1.append(0.0)
                values2.append(0.0)
                break
            from_high = 20 - 20 * i / num_points
            from_low = 0  + 20 * i / num_points
            if (t - 80) / 200 % 2 == 1:
                values1.append(from_high)
                values2.append(from_low)
            else:
                values1.append(from_low)
                values2.append(from_high)

    network.add_commodity(
        sources={0: RightConstant([0, 800], [20, 0], (0, float('inf')))},
        sink=4,
        predictor_type=PredictorType.CONSTANT
    )

    loader = DiscreteFlowBuilder(network, predictors={PredictorType.CONSTANT: ConstantPredictor(network)}, reroute_interval=1)

    color_by_comm_idx = {0: "red"}

    horizon = 2000
    generator = loader.build_flow()
    flow = next(generator)
    while flow.phi < horizon:
        flow = next(generator)

    to_visualization_json("./exampleDiscrete.vis.json", flow, network, color_by_comm_idx)


if __name__ == "__main__":
    build_example_discrete()
