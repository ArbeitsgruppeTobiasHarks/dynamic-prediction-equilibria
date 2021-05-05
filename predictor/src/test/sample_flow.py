import unittest

import numpy as np

from core.dynamic_flow import PartialDynamicFlow
from heapq import nsmallest
from test.sample_network import build_sample_network


class TestPartialDynamicFlow(unittest.TestCase):
    def test_queues(self):
        """
        Test whether the queues are calculated correctly.
        """
        network = build_sample_network()
        m = len(network.graph.edges)
        flow = PartialDynamicFlow(network)

        self.assertTrue(np.array_equal(flow.queues[0], np.zeros(m)), "Queues don't match at time 0!")

        time1 = flow.extend(np.array([2, 1, 0, 0, 0]), float('inf'))
        self.assertEqual(time1, 1.)
        self.assertTrue(np.array_equal(flow.queues[1], np.zeros(m)), f"Queues don't match at time {time1}!")
        self.assertEqual(nsmallest(2, flow.change_events), [3.])

        flow.extend(np.array([0, 0, 2, 0, 4]), float('inf'))
        self.assertTrue(np.array_equal(flow.queues[2], np.array([0, 0, 0, 0, 3])))

        flow.extend(np.array([0, 0, 0, 1, 1]), float('inf'))
        self.assertTrue(np.array_equal(flow.queues[3], np.array([0, 0, 0, 0, 3])))


if __name__ == '__main__':
    unittest.main()
