import unittest

import numpy as np

from core.dynamic_flow import PartialDynamicFlow
from test.sample_network import build_sample_network


class TestPartialDynamicFlow(unittest.TestCase):
    def test_queues(self):
        """
        Test whether the queues are calculated correctly.
        """
        network = build_sample_network()
        m = len(network.graph.edges)
        time_step_size = 1
        flow = PartialDynamicFlow(time_step_size, network)

        self.assertTrue(np.array_equal(flow.queues[0], np.zeros(m)), "Queues don't match at time 0!")

        flow.extend(np.array([2, 1, 0, 0, 0]), None)
        self.assertTrue(np.array_equal(flow.queues[1], np.zeros(m)), "Queues don't match at time 1!")

        flow.extend(np.array([0, 0, 2, 0, 4]), None)
        self.assertTrue(np.array_equal(flow.queues[2], np.array([0, 0, 0, 0, 3])))

        flow.extend(np.array([0, 0, 0, 1, 1]), None)
        self.assertTrue(np.array_equal(flow.queues[3], np.array([0, 0, 0, 0, 3])))


if __name__ == '__main__':
    unittest.main()
