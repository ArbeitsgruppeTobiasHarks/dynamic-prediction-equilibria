from test.sample_network import build_sample_network

from core.dynamic_flow import DynamicFlow
from src.core.predictors.predictor_type import PredictorType
from utilities.piecewise_linear import zero
from utilities.right_constant import RightConstant


def test_queues():
    """
    Test whether the queues are calculated correctly.
    """
    network = build_sample_network()
    network.add_commodity(
        {0: RightConstant([0.0], [0.0])}, 2, predictor_type=PredictorType.CONSTANT
    )

    m = len(network.graph.edges)
    flow = DynamicFlow(network)

    for i in range(m):
        assert flow.queues[i].equals(zero)
        assert len(flow.inflow[i]._functions_dict) == 0

    edges_changed = flow.extend({0: {0: 2.0}, 1: {0: 1.0}}, 10)
    assert edges_changed == {0}
    assert flow.phi == 1
    for i in range(2, m):
        assert flow.queues[i].equals(zero)
        assert len(flow.inflow[i]._functions_dict) == 0

    edges_changed = flow.extend({1: {}, 2: {0: 2.0}, 4: {0: 4.0}}, 10)
    assert flow.phi == 2.0
    assert edges_changed == {2, 4}

    edges_changed = flow.extend({2: {0: 2.0}, 4: {0: 4.0}}, 10)
    assert flow.phi == 3.0
    assert edges_changed == {1}

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
