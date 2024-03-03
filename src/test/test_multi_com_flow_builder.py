import unittest
from test.sample_network import build_sample_network
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from core.dpe_flow_builder import FlowBuilder
from core.predictor import Predictor
from core.predictors.predictor_type import PredictorType
from core.predictors.reg_linear_predictor import RegularizedLinearPredictor
from utilities.right_constant import RightConstant


class TestMultiComFlowBuilder(unittest.TestCase):
    """
    Interesting observations:

    If we take SingleEdgeDistributor together with LinearPredictor (with .005 precision),
    the resulting flow is weirdly glitching away from the expected IDE equilbrium.
    With .0025 precision, the WaterfillingDistribution behaves the same.

    With .003 precision, results are different again for SingleEdgeDistributor...

    LinearPredictor without any regularization seems quite unstable.
    """

    def test_multi_com_flow_builder(self):
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
        for e in range(len(network.graph.edges)):
            x = np.linspace(0.0, horizon, endpoint=True, num=120)
            plt.plot(x, [flow.inflow[e]._functions_dict[0](t) for t in x], label=str(e))
        plt.legend()
        plt.grid(which="both")
        plt.title(f"{PredictorType.REGULARIZED_LINEAR}, α={reroute_interval}")
        plt.show()
        for e in range(len(network.graph.edges)):
            plt.plot(flow.queues[e].times, flow.queues[e].values, label=str(e))
        plt.legend()
        plt.grid(which="both")
        plt.title(f"{PredictorType.REGULARIZED_LINEAR}, α={reroute_interval}")
        plt.show()
