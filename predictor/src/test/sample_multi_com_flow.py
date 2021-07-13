import unittest

import numpy as np

from core.multi_com_dynamic_flow import MultiComPartialDynamicFlow
from test.sample_network import build_sample_network
from utilities.piecewise_linear import zero
from utilities.right_constant import RightConstant


class TestMultiComPartialDynamicFlow(unittest.TestCase):
    def test_queues(self):
        """
        Test whether the queues are calculated correctly.
        """
        network = build_sample_network()
        network.add_commodity(0, 2, 3.)

        m = len(network.graph.edges)
        n = 1
        flow = MultiComPartialDynamicFlow(network)

        for i in range(m):
            self.assertTrue(flow.queues[i].equals(zero))
            self.assertTrue(flow.inflow[i][0].equals(RightConstant([-1], [0])))

        edges_changed = flow.extend({0: np.array([2.]), 1: np.array([1.])}, 10)
        self.assertEqual(edges_changed, {0})
        self.assertEqual(flow.phi, 1)
        for i in range(2, m):
            self.assertTrue(flow.queues[i].equals(zero))
            self.assertTrue(flow.inflow[i][0].equals(RightConstant([-1], [0])))

        edges_changed = flow.extend({1: np.array([0.]), 2: np.array([2.]), 4: np.array([4.])}, 10)
        self.assertEqual(flow.phi, 2.)
        self.assertEqual(edges_changed, {2, 4})

        edges_changed = flow.extend({2: np.array([2.]), 4: np.array([4.])}, 10)
        self.assertEqual(flow.phi, 3.)
        self.assertEqual(edges_changed, {1})

        # assert_array_equal(flow.queues[1], np.zeros(m), f"Queues don't match at time {time1}!")
        # assert_array_equal(flow.curr_outflow, [[2, 0, 0, 0, 0]])
        # self.assertEqual(flow.change_events.sorted(), [OutflowChangeEvent(1, 3, np.array([1]))])

        # time2 = flow.extend(np.array([[0, 0, 2, 0, 4]]), float('inf'), stop_at_queue_changes=False)
        # self.assertEqual(time2, 2.)
        # assert_array_equal(flow.queues[2], np.array([0, 0, 0, 0, 3]))
        # assert_array_equal(flow.curr_outflow, [[0, 0, 2, 0, 1]])
        # self.assertEqual(flow.change_events.sorted(),
        #                 [OutflowChangeEvent(1, 3, np.array([1])), OutflowChangeEvent(1, 4, np.array([0]))])

        # time3 = flow.extend(np.array([[0, 0, 0, 1, 1]]), float('inf'), stop_at_queue_changes=False)
        # self.assertEqual(time3, 3.)
        # assert_array_equal(flow.queues[3], np.array([0, 0, 0, 0, 3]))
        # assert_array_equal(flow.curr_outflow, [[0, 1, 0, 1, 1]])
        # print(flow.change_events.sorted())
        # self.assertEqual(flow.change_events.sorted(), [OutflowChangeEvent(1, 4, np.array([0]))])


if __name__ == '__main__':
    unittest.main()
