import json
from core.multi_com_flow_builder import MultiComFlowBuilder
from core.network import Network
from core.predictors.linear_predictor import LinearPredictor
from core.predictors.predictor_type import PredictorType
from core.predictors.zero_predictor import ZeroPredictor
from utilities.right_constant import RightConstant


def build_example3():
    network = Network()
    #
    # 0 -> 1 -> 2
    # |         |
    # 3 -> 4 -> 5
    #
    network.add_edge(0, 1, 200, 20)
    network.add_edge(1, 2, 200, 10)
    network.add_edge(0, 3, 200, 20)
    network.add_edge(3, 4, 200, 20)
    network.add_edge(4, 5, 200, 20)
    network.add_edge(5, 2, 200, 20)
    network.add_commodity(0, 2, RightConstant([-1, 0, 4000], [0, 20, 0]), PredictorType.LINEAR)

    horizon = 2000
    predictors = { PredictorType.LINEAR: LinearPredictor(network, 10000) }

    builder = MultiComFlowBuilder(network=network, predictors=predictors, reroute_interval=1)
    generator = builder.build_flow()
    flow = next(generator)
    while flow.phi < horizon:
        flow = next(generator)
    
    with open("./slides/src/example3FlowData.js", "w") as file:
        file.write("export default ")
        json.dump({
            "inflow": flow.inflow,
            "outflow": flow.outflow,
            "queues": flow.queues
        }, file, default=vars)


if __name__ == '__main__':
    build_example3()