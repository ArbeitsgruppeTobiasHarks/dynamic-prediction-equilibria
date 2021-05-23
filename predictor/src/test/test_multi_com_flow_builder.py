import unittest

import matplotlib.pyplot as plt
import numpy as np

from core.multi_com_flow_builder import MultiComFlowBuilder
from core.reg_linear_predictor import RegularizedLinearPredictor
from core.single_edge_distributor import SingleEdgeDistributor
from core.waterfilling_distributor import WaterfillingDistributor
from test.sample_network import build_sample_network


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
        predictors = [RegularizedLinearPredictor(network)]
        network.add_commodity(0, 2, 3., 0)

        distributor = WaterfillingDistributor(network)
        reroute_interval = 0.005
        horizon = 35
        flow_builder = MultiComFlowBuilder(
            network,
            predictors,
            distributor,
            reroute_interval
        )
        generator = flow_builder.build_flow()
        flow = None
        while flow is None or flow.phi < horizon:
            flow = next(generator)
        for e in range(len(network.graph.edges)):
            x = np.linspace(0., horizon, endpoint=True, num=120)
            plt.plot(x, [flow.inflow[e][0](t) for t in x], label=str(e))
        plt.legend()
        plt.grid(which='both')
        plt.title(f"{predictors[0].type()}, {distributor.type()}, α={reroute_interval}")
        plt.show()
        for e in range(len(network.graph.edges)):
            plt.plot(flow.queues[e].times, flow.queues[e].values, label=str(e))
        plt.legend()
        plt.grid(which='both')
        plt.title(f"{predictors[0].type()}, {distributor.type()}, α={reroute_interval}")
        plt.show()