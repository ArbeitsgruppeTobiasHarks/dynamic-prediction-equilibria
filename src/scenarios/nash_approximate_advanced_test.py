from core.nash_flow_builder import NashFlowBuilder
from core.network import Network
from core.predictors.predictor_type import PredictorType
from utilities.right_constant import RightConstant
from visualization.to_json import to_visualization_json


def run_scenario():
    network = Network()
    infinite_capacity = 40
    # Link 1: a -> b; transit time 10; capacity 10
    network.add_edge(0, 1, 10, 10)
    # Link 2: b -> c; transit time 10; capacity infty
    network.add_edge(1, 2, 10, infinite_capacity)
    # Link 3: c -> d; transit time 10; capacity 10
    network.add_edge(2, 3, 10, 10)
    # Link 4: d -> a; transit time 10; capacity infty
    network.add_edge(3, 0, 10, infinite_capacity)
    # Link 5: a -> d; transit time 40; capacity infty
    network.add_edge(0, 3, 40, infinite_capacity)
    # Link 6: c -> b; transit time 40; capacity infty
    network.add_edge(2, 1, 40, infinite_capacity)

    network.graph.positions = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}

    network.add_commodity(
        {0: RightConstant([0, 15], [20, 0], (0, float("inf")))},
        3,
        predictor_type=PredictorType.CONSTANT,
    )
    network.add_commodity(
        {2: RightConstant([0, 15], [20, 0], (0, float("inf")))},
        1,
        predictor_type=PredictorType.CONSTANT,
    )

    loader = NashFlowBuilder(network)
    flow, _ = loader.build_flow()
    to_visualization_json(
        "./test.json", flow, network, {0: "red", 1: "blue", 2: "green", 3: "orange"}
    )


if __name__ == "__main__":
    print("Running scenario.")
    run_scenario()
