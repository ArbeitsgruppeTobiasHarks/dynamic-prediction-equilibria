from typing import Dict

from dynflows.core.flow_builder import FlowBuilder
from dynflows.core.network import Network
from dynflows.core.predictor import Predictor
from dynflows.core.predictors.predictor_type import PredictorType
from dynflows.core.predictors.zero_predictor import ZeroPredictor
from dynflows.utilities.json_encoder import JSONEncoder
from dynflows.utilities.right_constant import RightConstant


def build_example1():
    network = Network()
    network.add_edge(0, 2, 200, 10)
    network.add_edge(0, 1, 200, 20)
    network.add_edge(1, 2, 200, 20)
    network.add_commodity({0: RightConstant([-1, 0, 1600], [0, 20, 0])}, 2)

    horizon = 2000
    predictors: Dict[PredictorType, Predictor] = {
        PredictorType.ZERO: ZeroPredictor(network)
    }
    predictor_types = [PredictorType.ZERO]

    builder = FlowBuilder(
        network=network,
        predictors=predictors,
        predictor_types=predictor_types,
        reroute_interval=100,
    )
    generator = builder.build_flow()
    flow = next(generator)
    while flow.phi < horizon:
        flow = next(generator)

    with open("./slides/src/example2FlowData.js", "w") as file:
        file.write("export default ")
        JSONEncoder().dump(
            {"inflow": flow.inflow, "outflow": flow.outflow, "queues": flow.queues},
            file,
        )


if __name__ == "__main__":
    build_example1()
