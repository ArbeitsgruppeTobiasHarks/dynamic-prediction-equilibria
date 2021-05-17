import unittest

import numpy as np
from numpy.testing import assert_array_equal

from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow, OutflowChangeEvent
from test.sample_network import build_sample_network


class TestMultiComPartialDynamicFlow(unittest.TestCase):
    def test_queues(self):
        """
        Test whether the queues are calculated correctly.
        """
        network = build_sample_network()
        network.commodities = [network.sink]
        m = len(network.graph.edges)
        n = 1
        flow = MultiComPartialDynamicFlow(network)

        assert_array_equal(flow.queues[0], np.zeros(m), "Queues don't match at time 0!")
        assert_array_equal(flow.curr_outflow, np.zeros((n, m)))

        time1 = flow.extend(np.array([[2, 1, 0, 0, 0]]), float('inf'), stop_at_queue_changes=False)
        self.assertEqual(time1, 1.)
        assert_array_equal(flow.queues[1], np.zeros(m), f"Queues don't match at time {time1}!")
        assert_array_equal(flow.curr_outflow, [[2, 0, 0, 0, 0]])
        self.assertEqual(flow.change_events.sorted(), [OutflowChangeEvent(1, 3, np.array([1]))])

        time2 = flow.extend(np.array([[0, 0, 2, 0, 4]]), float('inf'), stop_at_queue_changes=False)
        self.assertEqual(time2, 2.)
        assert_array_equal(flow.queues[2], np.array([0, 0, 0, 0, 3]))
        assert_array_equal(flow.curr_outflow, [[0, 0, 2, 0, 1]])
        self.assertEqual(flow.change_events.sorted(),
                         [OutflowChangeEvent(1, 3, np.array([1])), OutflowChangeEvent(1, 4, np.array([0]))])

        time3 = flow.extend(np.array([[0, 0, 0, 1, 1]]), float('inf'), stop_at_queue_changes=False)
        self.assertEqual(time3, 3.)
        assert_array_equal(flow.queues[3], np.array([0, 0, 0, 0, 3]))
        assert_array_equal(flow.curr_outflow, [[0, 1, 0, 1, 1]])
        print(flow.change_events.sorted())
        # self.assertEqual(flow.change_events.sorted(), [OutflowChangeEvent(1, 4, np.array([0]))])


if __name__ == '__main__':
    unittest.main()
