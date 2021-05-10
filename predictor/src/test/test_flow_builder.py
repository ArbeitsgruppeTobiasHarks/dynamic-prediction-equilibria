import unittest

import matplotlib.pyplot as plt
import numpy as np

from core.flow_builder import FlowBuilder
from core.ide_predictor import IDEPredictor
from test.sample_network import build_sample_network


class TestFlowBuilder(unittest.TestCase):
    def test_flow_builder(self):
        network = build_sample_network()
        predictor = IDEPredictor(network)
        flow_builder = FlowBuilder(network, predictor)
        generator = flow_builder.build_flow()
        flow = None
        while flow is None or flow.times[-1] < 100:
            flow = next(generator)
        queues = np.asarray(flow.queues)
        for e in range(len(network.graph.edges)):
            plt.plot(flow.times, queues[:, e], label=str(e))
        plt.legend()
        plt.show()
