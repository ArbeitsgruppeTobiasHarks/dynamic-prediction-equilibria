from test.sample_network import build_sample_network
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison

from core.flow_builder import FlowBuilder
from core.predictor import Predictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from utilities.right_constant import RightConstant


@image_comparison(baseline_images=["reg_linear_1", "reg_linear_2"], extensions=["pdf"])
def test_flow_builder():
    network = build_sample_network()
    predictors: Dict[PredictorType, Predictor] = {
        PredictorType.REGULARIZED_LINEAR: RegularizedLinearPredictor(
            network, horizon=2.0, delta=1.0
        )
    }
    net_inflow = RightConstant([0.0], [3.0], (0, float("inf")))
    network.add_commodity({0: net_inflow}, 2, PredictorType.REGULARIZED_LINEAR)

    reroute_interval = 0.005
    horizon = 100
    flow_builder = FlowBuilder(network, predictors, reroute_interval)
    generator = flow_builder.build_flow()
    flow = None
    while flow is None or flow.phi < horizon:
        flow = next(generator)

    fig, ax = plt.subplots()

    for e in range(len(network.graph.edges)):
        x = np.linspace(0.0, horizon, endpoint=True, num=120)
        ax.plot(
            x,
            [
                (
                    flow.inflow[e]._functions_dict[0](t)
                    if 0 in flow.inflow[e]._functions_dict
                    else 0.0
                )
                for t in x
            ],
            label=str(e),
        )
    ax.legend()
    ax.grid(which="both")
    ax.set_title(f"Inflow, α={reroute_interval}")

    fig, ax = plt.subplots()

    for e in range(len(network.graph.edges)):
        ax.plot(flow.queues[e].times, flow.queues[e].values, label=str(e))
    ax.legend()
    ax.grid(which="both")
    ax.set_title(f"Queue, α={reroute_interval}")
